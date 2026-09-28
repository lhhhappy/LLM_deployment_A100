#!/usr/bin/env python3
"""Research-only local KDA graphs with explicit, refreshed metadata tables.

The graph key fixes logical/physical rows, sequence count, total 64-token
chunks, and a convolution launch bound. Sequence boundaries, state slots,
prefix flags, input values and dual-snapshot plans change at replay. This is
not arbitrary token-bucket capture: that needs the padded-table kernel guards
and interfaces in upstream #41105. No production kernel is modified here.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace as NS


ROOT = Path(__file__).resolve().parents[2]


def emit(**record):
    print(json.dumps(record), flush=True)


def chunk_count(lengths):
    return sum((n + 63) // 64 for n in lengths)


class HostPlan:
    def __init__(self, lengths, conv_bound):
        self.tokens = sum(lengths)
        self.seqs = len(lengths)
        self.chunks = chunk_count(lengths)
        self.conv_bound = conv_bound
        self.sizes = [self.seqs + 1, self.chunks * 2, self.seqs + 1,
                      self.seqs, self.seqs, self.seqs * 2, self.seqs * 2]

    def pack(self, lengths, slots, prefixes, offsets, destinations):
        assert len(lengths) == self.seqs
        assert all(type(n) is int and 0 < n <= self.conv_bound for n in lengths)
        assert sum(lengths) == self.tokens and chunk_count(lengths) == self.chunks
        assert len(slots) == len(prefixes) == len(offsets) == len(destinations) == self.seqs
        cu, indices, chunks = [0], [], [0]
        for seq, n in enumerate(lengths):
            cu.append(cu[-1] + n)
            count = (n + 63) // 64
            indices.extend(v for chunk in range(count) for v in (seq, chunk))
            chunks.append(chunks[-1] + count)
        packed = [*cu, *indices, *chunks, *slots, *prefixes,
                  *(v for pair in offsets for v in pair),
                  *(v for pair in destinations for v in pair)]
        assert len(packed) == sum(self.sizes)
        return packed


def cpu_checks():
    plan = HostPlan([4095, 2049, 2048], 4096)
    args = ([0, 1, 2], [0, 64, 0], [[64, 128]] * 3, [[4, 5]] * 3)
    for lengths in ([4095, 2049, 2048], [2049, 4095, 2048], [1, 4096, 4095]):
        values = plan.pack(lengths, *args)
        assert values[:4] == [0, lengths[0], sum(lengths[:2]), 8192]
    rejected = 0
    for lengths in ([4096, 2048, 2048], [8192], [4095, 2049, 2047], [0, 4096, 4096]):
        try:
            plan.pack(lengths, *args)
        except AssertionError:
            rejected += 1
    assert rejected == 4
    emit(kind="cpu_complete", passed=True, rejected_topologies=rejected)


def graph_case(torch, module, n, ragged, rounds, inner):
    from sglang.kernels.ops.attention.fla import kda as fla_kda
    from sglang.kernels.ops.attention.fla import chunk_delta_h as delta
    from sglang.kernels.ops.attention.fla import chunk_delta_h_snapshot as snapshots
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    lengths_cases = ([[n // 2 - 1, n // 4 + 1, n // 4],
                      [n // 4 + 1, n // 2 - 1, n // 4],
                      [1, n // 2, n // 2 - 1]] if ragged else [[n], [n], [n]])
    plan = HostPlan(lengths_cases[0], n // 2 if ragged else n)
    seqs, pad, heads, dim = plan.seqs, 7 if ragged else 0, 8, 128
    physical, width = n + pad, heads * dim
    pool_slots = 3 * seqs + 3
    generator = torch.Generator(device="cuda").manual_seed(41105 + n + seqs)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * .2

    layer = NS(layer_id=0, conv_weights=rand(3 * width, 4, dtype=torch.float32),
               bias=None, q_dim=width, k_dim=width, v_dim=width,
               head_q_dim=dim, head_k_dim=dim, head_v_dim=dim,
               A_log=rand(heads, dtype=torch.float32),
               dt_bias=rand(width, dtype=torch.float32), lower_bound=-5.)
    static_raw = torch.zeros(physical, 3 * width, device="cuda", dtype=torch.bfloat16)
    static_gate = torch.zeros(1, physical, width, device="cuda", dtype=torch.bfloat16)
    static_beta = torch.zeros(1, physical, heads, device="cuda", dtype=torch.bfloat16)
    conv = torch.zeros(pool_slots, 3, 3 * width, device="cuda", dtype=torch.bfloat16)
    ssm = torch.zeros(pool_slots, heads, dim, dim, device="cuda", dtype=torch.float32)
    packed = torch.zeros(sum(plan.sizes), device="cuda", dtype=torch.int32)
    cu, indices, offsets, slots, prefixes, snap_offsets, snap_slots = torch.split(packed, plan.sizes)
    indices = indices.view(plan.chunks, 2)
    snap_offsets, snap_slots = snap_offsets.view(seqs, 2), snap_slots.view(seqs, 2)

    def backend(conv_pool, ssm_pool, query, state_slots, prefix_lengths, lengths,
                snapshot_offsets, snapshot_slots):
        fb = NS(forward_mode=ForwardMode.EXTEND, batch_size=seqs,
                extend_seq_lens_cpu=list(lengths), extend_prefix_lens=prefix_lengths,
                ax_kda_snapshot_offsets=snapshot_offsets,
                ax_kda_snapshot_slots=snapshot_slots,
                attn_cp_metadata=None, tbo_parent_token_range=None)
        obj = module.KDAAttnBackend.__new__(module.KDAAttnBackend)
        obj.forward_metadata = NS(query_start_loc=query, mamba_cache_indices=state_slots,
                                  has_mamba_track_mask=False,
                                  _ax_cpu_logical_num_tokens=n)
        obj.req_to_token_pool = NS(mamba2_layer_cache=lambda _: NS(conv=[conv_pool], temporal=ssm_pool))
        obj.kernel_dispatcher = TritonKDAKernel()
        obj.accept_lens_pool = None
        return obj, fb

    captured_backend, captured_batch = backend(
        conv, ssm, cu, slots, prefixes, lengths_cases[0],
        snap_offsets if ragged else None, snap_slots if ragged else None)

    def fill_metadata(values):
        # A fresh pinned allocation avoids rewriting a host block whose previous
        # asynchronous copy is still in flight; PyTorch's host allocator tracks it.
        host = torch.tensor(values, dtype=torch.int32, pin_memory=True)
        packed.copy_(host, non_blocking=True)

    def make_host_metadata(lengths, replay):
        rng = random.Random(n + replay)
        state_slots = rng.sample(range(seqs + 1), seqs)
        prefix_lengths = [128 if (replay + seq) % 2 else 0 for seq in range(seqs)]
        boundaries, destinations = [], []
        for seq, length in enumerate(lengths):
            boundaries.append([(length // 64) * 64, 64] if length >= 64 else [-1, -1])
            destinations.append([seqs + 1 + 2 * seq, seqs + 2 + 2 * seq]
                                if length >= 64 else [-1, -1])
        if replay == 1 and ragged:
            destinations[1][0] = -1
        return state_slots, prefix_lengths, boundaries, destinations

    calls = {"indices": 0, "offsets": 0, "conv": 0}

    @contextmanager
    def explicit_tables():
        def get_indices(query, chunk_size):
            assert query is cu and chunk_size == 64
            calls["indices"] += 1
            return indices

        def get_offsets(query, chunk_size):
            assert query is cu and chunk_size == 64
            calls["offsets"] += 1
            return offsets

        original_conv = module.causal_conv1d_fn

        def bounded_conv(*args, **kwargs):
            assert kwargs["query_start_loc"] is cu
            calls["conv"] += 1
            kwargs["seq_lens_cpu"] = [plan.conv_bound] * seqs
            return original_conv(*args, **kwargs)

        replacements = [(fla_kda, "prepare_chunk_indices", get_indices),
                        (delta, "prepare_chunk_offsets", get_offsets),
                        (snapshots, "prepare_chunk_offsets", get_offsets),
                        (module, "causal_conv1d_fn", bounded_conv)]
        saved = [(owner, name, getattr(owner, name)) for owner, name, _ in replacements]
        try:
            for owner, name, fn in replacements:
                setattr(owner, name, fn)
            yield
        finally:
            for owner, name, fn in saved:
                setattr(owner, name, fn)

    def capture_run():
        return captured_backend.forward_extend(layer, captured_batch, static_raw, static_gate, static_beta)

    def memory():
        free, total = torch.cuda.mem_get_info()
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    free=free, total=total)

    first = make_host_metadata(lengths_cases[0], 0)
    fill_metadata(plan.pack(lengths_cases[0], *first))
    static_raw.copy_(rand(*static_raw.shape))
    static_gate.copy_(rand(*static_gate.shape))
    static_beta.copy_(rand(*static_beta.shape))
    module._AX_KDA_PREFILL_CPU_LENGTH = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with explicit_tables(), torch.cuda.stream(stream):
        for _ in range(3):
            capture_run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    before_capture = memory()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with explicit_tables(), torch.cuda.graph(graph, stream=stream):
        graph_out = capture_run()
    torch.cuda.synchronize()
    after_capture = memory()
    pool_id = tuple(graph.pool())
    allocation_segments = torch.cuda.memory_snapshot()
    pool_segments = [s for s in allocation_segments
                     if tuple(s.get("segment_pool_id", ())) == pool_id]
    pool_bytes = ({k: sum(s.get(k, 0) for s in pool_segments)
                   for k in ("total_size", "allocated_size", "active_size")}
                  if any("segment_pool_id" in s for s in allocation_segments) else None)
    persistent = [static_raw, static_gate, static_beta, conv, ssm, packed]
    emit(kind="graph_capture", logical_rows=n, physical_rows=physical, sequences=seqs,
         chunks=plan.chunks, snapshots=ragged, conv_bound=plan.conv_bound,
         memory_before=before_capture, memory_after=after_capture,
         capture_peak_allocated=torch.cuda.max_memory_allocated(),
         capture_peak_reserved=torch.cuda.max_memory_reserved(),
         private_pool_id=pool_id, private_pool_segments=len(pool_segments),
         allocator_segment_fields=sorted(allocation_segments[0]) if allocation_segments else [],
         private_pool_bytes=pool_bytes, explicit_persistent_bytes=sum(t.numel() * t.element_size() for t in persistent),
         metadata_bytes=packed.numel() * packed.element_size(), hooks=calls,
         scope="one isolated graph; current case only, no all-layer graph pool")
    assert calls["indices"] and calls["offsets"] and calls["conv"]

    old_output = None
    for replay, lengths in enumerate(lengths_cases):
        fields = make_host_metadata(lengths, replay)
        host_values = plan.pack(lengths, *fields)
        raw, gate, beta = rand(*static_raw.shape), rand(*static_gate.shape), rand(*static_beta.shape)
        raw[n:] = float("nan")
        initial_conv = rand(*conv.shape)
        initial_ssm = rand(*ssm.shape, dtype=torch.float32)
        for slot, prefix in zip(fields[0], fields[1]):
            if prefix == 0:
                initial_ssm[slot].zero_()
        conv_ref, ssm_ref = initial_conv.clone(), initial_ssm.clone()
        host_cu = [0]
        for length in lengths:
            host_cu.append(host_cu[-1] + length)
        as_device = lambda value: torch.tensor(value, device="cuda", dtype=torch.int32)
        eager_backend, eager_batch = backend(
            conv_ref, ssm_ref, as_device(host_cu), as_device(fields[0]), as_device(fields[1]), lengths,
            as_device(fields[2]) if ragged else None, as_device(fields[3]) if ragged else None)

        def eager_run(enabled):
            module._AX_KDA_PREFILL_CPU_LENGTH = enabled
            return eager_backend.forward_extend(layer, eager_batch, raw, gate, beta)

        # Warm native cache with a separate cu_seqlens object. Its indices must
        # never come from our captured buffers or tensor-identity cache entries.
        eager_run(False)
        conv_ref.copy_(initial_conv)
        ssm_ref.copy_(initial_ssm)
        expected = eager_run(False).clone()
        static_raw.copy_(raw); static_gate.copy_(gate); static_beta.copy_(beta)
        conv.copy_(initial_conv); ssm.copy_(initial_ssm)
        fill_metadata(host_values)
        graph.replay()
        torch.cuda.synchronize()

        def bits_equal(a, b):
            kind = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
            return torch.equal(a.view(kind), b.view(kind))

        equality = dict(output=bits_equal(graph_out, expected),
                        conv=bits_equal(conv, conv_ref), ssm=bits_equal(ssm, ssm_ref))
        active = set(fields[0])
        if ragged:
            active.update(v for pair in fields[3] for v in pair if v >= 0)
        untouched = sorted(set(range(pool_slots)) - active)
        untouched_ok = (bits_equal(conv[untouched], initial_conv[untouched])
                        and bits_equal(ssm[untouched], initial_ssm[untouched]))
        fresh = old_output is None or not bits_equal(old_output, graph_out)
        finite = bool(torch.isfinite(graph_out).all())
        emit(kind="graph_correctness", rows=n, sequences=seqs, replay=replay,
             lengths=lengths, state_slots=fields[0], prefixes=fields[1], snapshots=ragged,
             snapshot_offsets=fields[2] if ragged else None,
             snapshot_slots=fields[3] if ragged else None,
             bitexact=equality, untouched_slots=untouched, untouched_bitexact=untouched_ok,
             fresh_output=fresh, finite=finite)
        assert all(equality.values()) and untouched_ok and fresh and finite
        old_output = graph_out.clone()

    # Time the last fresh case. Reset pool state outside the measured interval
    # for every arm; repeated calls evolve the state in the same way.
    def graph_staged():
        values = plan.pack(lengths, *fields)
        fill_metadata(values)
        static_raw.copy_(raw); static_gate.copy_(gate); static_beta.copy_(beta)
        graph.replay()

    arms = dict(eager_native=lambda: eager_run(False), eager_cpu=lambda: eager_run(True),
                graph_replay=graph.replay, graph_staged=graph_staged)
    raw_timings = {name: [] for name in arms}
    rng = random.Random(n + seqs)
    for _ in range(rounds):
        order = list(arms)
        rng.shuffle(order)
        for name in order:
            conv.copy_(initial_conv); ssm.copy_(initial_ssm)
            conv_ref.copy_(initial_conv); ssm_ref.copy_(initial_ssm)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            t0 = time.perf_counter()
            for _ in range(inner):
                arms[name]()
            enqueued = time.perf_counter()
            end.record(); end.synchronize()
            raw_timings[name].append(dict(gpu_ms=start.elapsed_time(end) / inner,
                                         host_ms=(enqueued - t0) * 1000 / inner,
                                         wall_ms=(time.perf_counter() - t0) * 1000 / inner))
    emit(kind="graph_timing", rows=n, sequences=seqs, snapshots=ragged, rounds=rounds, inner=inner,
         medians={name: {key: statistics.median(v[key] for v in values)
                         for key in ("gpu_ms", "host_ms", "wall_ms")}
                  for name, values in raw_timings.items()}, samples=raw_timings,
         graph_staged_includes="CPU table build, pinned H2D, full QKV/gate/beta D2D copies, replay",
         excluded="projections, TP communication, scheduler, state reset; no whole-model claim")
    for name in ("eager_cpu", "graph_staged"):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            arms[name]()
            torch.cuda.synchronize()
        counts = {event.key: event.count for event in prof.key_averages()
                  if event.key in ("aten::_local_scalar_dense", "cudaLaunchKernel", "cudaLaunchKernelExC",
                                   "cudaGraphLaunch", "cudaMemcpyAsync", "cudaStreamSynchronize")}
        emit(kind="graph_profile", rows=n, sequences=seqs, arm=name, counts=counts)
        assert counts.get("aten::_local_scalar_dense", 0) == 0
    del graph_out, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--rows", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--inner", type=int, default=4)
    args = parser.parse_args()
    cpu_checks()
    if not args.gpu:
        return
    import torch
    import sglang.srt.layers.attention.linear.kda_backend as module
    expected_source = ROOT / "engine/sglang/srt/layers/attention/linear/kda_backend.py"
    assert Path(module.__file__).resolve() == expected_source.resolve()
    assert torch.cuda.get_device_capability() == (8, 0)
    sources = [Path(__file__), expected_source]
    sources += list((ROOT / "engine/sglang/kernels/ops/attention/fla").glob("*.py"))
    sources.append(ROOT / "engine/sglang/kernels/ops/mamba/causal_conv1d_triton.py")
    emit(kind="environment", device=torch.cuda.get_device_name(), torch=torch.__version__,
         source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         env={k: v for k, v in os.environ.items() if k.startswith(("SGLANG_AX_KDA", "SGLANG_GDN_CHUNK"))},
         scope="local KDA core graphs; explicit fixed topology, dynamic boundaries/slots/prefixes/snapshots")
    original = module._AX_KDA_PREFILL_CPU_LENGTH
    try:
        with torch.inference_mode():
            for rows in args.rows:
                assert rows >= 256 and rows % 256 == 0
                for ragged in (False, True):
                    graph_case(torch, module, rows, ragged, args.rounds, args.inner)
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        module._AX_KDA_PREFILL_CPU_LENGTH = original
    emit(kind="graph_complete", passed=True, graph_cases=len(args.rows) * 2)


if __name__ == "__main__":
    main()
