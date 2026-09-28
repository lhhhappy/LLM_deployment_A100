#!/usr/bin/env python3
"""Prove BV16 startup coverage, guarded dispatch, and state/snapshot raw bits.

Run in a fresh process on one A100. Full KDA core timing belongs to
bench_kda_prepare_sm80.py; this checks the production recurrence wrappers.
"""

from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from unittest import mock

import torch
import triton
import triton.knobs as knobs


ROOT = Path(__file__).resolve().parents[2]
FLAG = "SGLANG_AX_KDA_PREFILL_STATE_BV16"
SOURCES = (
    "engine/sglang/kernels/ops/attention/fla/kda_state_sm80.py",
    "engine/sglang/kernels/ops/attention/fla/chunk_delta_h.py",
    "engine/sglang/kernels/ops/attention/fla/chunk_delta_h_snapshot.py",
    "engine/sglang/kernels/ops/attention/fla/kda.py",
    "engine/sglang/srt/layers/attention/linear/kernels/kda_triton.py",
    "engine/sglang/srt/layers/attention/linear/kda_backend.py",
    "tests/gpu/test_kda_state_sm80.py",
)


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def memory(device):
    free, total = torch.cuda.mem_get_info(device)
    return dict(allocated=torch.cuda.memory_allocated(device),
                reserved=torch.cuda.memory_reserved(device),
                driver_free=free, driver_total=total)


def bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    return torch.equal(actual.view(integer), expected.view(integer))


@contextmanager
def watch_cache():
    runtime, compilation = knobs.runtime, knobs.compilation
    old_jit, old_listener = runtime.jit_cache_hook, compilation.listener
    old_load = runtime.kernel_load_start_hook
    counts = dict(jit_misses=0, disk_hits=0, compiled=0, module_loads=0)

    def jit_hook(*args, **kwargs):
        counts["jit_misses"] += 1
        if old_jit is not None:
            return old_jit(*args, **kwargs)

    def listener(**kwargs):
        counts["disk_hits" if kwargs["cache_hit"] else "compiled"] += 1
        if old_listener is not None:
            return old_listener(**kwargs)

    def load_hook(*args, **kwargs):
        counts["module_loads"] += 1
        if old_load is not None:
            return old_load(*args, **kwargs)

    runtime.jit_cache_hook, compilation.listener = jit_hook, listener
    runtime.kernel_load_start_hook = load_hook
    try:
        yield counts
    finally:
        runtime.jit_cache_hook, compilation.listener = old_jit, old_listener
        runtime.kernel_load_start_hook = old_load


def metadata(lengths, device, dtype):
    n = len(lengths)
    chunks = [(length + 63) // 64 for length in lengths]
    cu = torch.tensor([0] + [sum(lengths[:i + 1]) for i in range(n)],
                      dtype=torch.int32, device=device)
    chunk_indices = torch.tensor(
        [[i, j] for i, count in enumerate(chunks) for j in range(count)],
        dtype=torch.int32, device=device,
    )
    slots = torch.tensor([2] if n == 1 else [2, -1, 0], dtype=dtype, device=device)
    offsets = torch.tensor([[64, length // 64 * 64] for length in lengths],
                           dtype=torch.int64, device=device)
    destinations = torch.arange(3, 3 + 2 * n, dtype=torch.int64, device=device).view(n, 2)
    return cu, chunk_indices, slots, offsets, destinations


def arm_checks(device, helper, delta, module):
    with mock.patch.dict(os.environ, {FLAG: "0"}), mock.patch.object(
        torch.cuda, "get_device_capability", side_effect=AssertionError("CUDA queried while disabled")
    ):
        off = module.TritonKDAKernel()
        assert off.arm_prefill_state(device) is False
    with mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.object(
        torch.cuda, "get_device_capability", side_effect=AssertionError("CUDA queried for CPU")
    ):
        assert module.TritonKDAKernel().arm_prefill_state("cpu") is False
    with mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.object(
        torch.cuda, "get_device_capability", return_value=(9, 0)
    ):
        assert module.TritonKDAKernel().arm_prefill_state(device) is False
    with mock.patch.object(delta, "GDN_CHUNK_H_BV", 16):
        assert helper.warmup_state_sm80(device.index) is False
        with mock.patch.dict(os.environ, {FLAG: "1"}):
            assert module.TritonKDAKernel().arm_prefill_state(device) is False
    emit(kind="arm_fallback", disabled=True, cpu=True, other_sm=True, native_env_override=True)
    return off


def extend_contract(module, kernel, device, expected):
    x = torch.empty((1, 8192, 8, 128), dtype=torch.bfloat16, device=device)
    received = []
    marker = object()

    def fake(**kwargs):
        received.append(kwargs)
        return marker

    with mock.patch.object(module, "chunk_kda", side_effect=fake):
        got = kernel.extend(x, x, x, g=None, beta=None, ssm_states=None,
                            cache_indices=None, query_start_loc=None)
    assert got is marker and len(received) == 1
    assert received[0].get("sm80_bv16", False) is expected
    emit(kind="extend_contract", armed=expected, explicit_state_flag=True)


def case(device, helper, delta, snapshots, lengths, dtype, mode, *, graph=False, guards=False):
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_offsets

    tokens, n = sum(lengths), len(lengths)
    generator = torch.Generator(device=device).manual_seed(tokens + n + 928)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype) * .02

    k, w, u = (rand((1, tokens, 8, 128)) for _ in range(3))
    gk = rand(k.shape, torch.float32) - .05
    cu, chunk_indices, slots, offsets, destinations = metadata(lengths, device, dtype)
    # Non-contiguous slot pitch and unused adjacent layer exercise both the
    # runtime-stride specialization and protection of the complete envelope.
    initial_pool = rand((3 * n + 4, 2, 8, 128, 128), torch.float32)
    native_pool, candidate_pool = initial_pool.clone(), initial_pool.clone()
    export = mode == "snapshot_export"
    wrapper = delta.chunk_gated_delta_rule_fwd_h
    env = {"SGLANG_AX_KDA_DUAL_SNAPSHOT": "1" if mode == "snapshot_no_export" else "0"}
    common = dict(k=k, w=w, u=u, gk=gk, initial_state_indices=slots,
                  cu_seqlens=cu, chunk_indices=chunk_indices, use_exp2=True,
                  snapshot_offsets=offsets if export else None,
                  snapshot_slots=destinations if export else None)
    chunk_offsets = prepare_chunk_offsets(cu, 64)
    nt = len(chunk_indices)

    def allowed(**changes):
        values = dict(k=k, w=w, u=u, g=None, gk=gk,
                      initial_state=candidate_pool[:, 0], indices=slots,
                      cu=cu, chunk_offsets=chunk_offsets, NT=nt,
                      save_new_value=True, use_exp2=True,
                      snapshot_offsets=offsets if export else None,
                      snapshot_slots=destinations if export else None)
        values.update(changes)
        return helper.can_use_state_sm80(**values)

    assert allowed(), (tokens, n, dtype, mode)
    with mock.patch.dict(os.environ, env):
        reference = wrapper(**common, initial_state=native_pool[:, 0])
        torch.cuda.synchronize(device)
        with watch_cache() as counts:
            candidate = wrapper(**common, initial_state=candidate_pool[:, 0], sm80_bv16=True)
            torch.cuda.synchronize(device)
        assert not any(counts.values()), counts
        exact = dict(h=bits(candidate[0], reference[0]),
                     v_new=bits(candidate[1], reference[1]),
                     complete_state_envelope=bits(candidate_pool, native_pool))
        assert all(exact.values()), exact
        emit(kind="serving_signature", tokens=tokens, sequences=n, mode=mode,
             indices_dtype=str(dtype), snapshot_dtype="int64" if export else None,
             nt=nt, nt_bucket=1 if nt <= 128 else 2,
             slot_pitch=candidate_pool[:, 0].stride(0), raw_bits=exact, cache=counts)

        if guards:
            failures = dict(
                not_warmed=False,
                rows=allowed(k=k[:, :4096], w=w[:, :4096], u=u[:, :4096], gk=gk[:, :4096]),
                fp16=allowed(k=k.to(torch.float16)),
                gk_bf16=allowed(gk=gk.to(torch.bfloat16)),
                scalar_gate=allowed(g=torch.empty(1, device=device)),
                no_new_value=allowed(save_new_value=False),
                natural_exp=allowed(use_exp2=False),
                no_state=allowed(initial_state=None),
                state_bf16=allowed(initial_state=candidate_pool[:, 0].to(torch.bfloat16)),
                query_int64=allowed(cu=cu.to(torch.int64)),
                offsets_int32=allowed(chunk_offsets=chunk_offsets.to(torch.int32)),
                wrong_nt=allowed(NT=nt + 4),
                too_many_sequences=allowed(cu=torch.zeros(5, dtype=torch.int32, device=device)),
                snapshot_int32=allowed(snapshot_offsets=offsets.to(torch.int32), snapshot_slots=destinations.to(torch.int32)),
                snapshot_unpaired=allowed(snapshot_offsets=offsets, snapshot_slots=None),
            )
            helper._WARMED_DEVICES.remove(device.index)
            try:
                failures["not_warmed"] = allowed()
            finally:
                helper._WARMED_DEVICES.add(device.index)
            storage = torch.empty(k.numel() + 1, dtype=k.dtype, device=device)
            failures["unaligned"] = allowed(k=storage[1:].view_as(k))
            failures["state_inner_stride"] = allowed(initial_state=candidate_pool[:, 0].transpose(-1, -2))
            with mock.patch.object(delta, "GDN_CHUNK_H_NUM_STAGES", 3):
                failures["native_env_override"] = allowed()
            assert not any(failures.values()), failures

            # Prove an unsupported snapshot dtype invokes the native Autotuner,
            # not only that the pure predicate returns False.
            fake = mock.MagicMock()
            bad_common = dict(common, snapshot_offsets=offsets.to(torch.int32),
                              snapshot_slots=destinations.to(torch.int32))
            with mock.patch.object(snapshots, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_snapshot", fake):
                wrapper(**bad_common, initial_state=candidate_pool[:, 0], sm80_bv16=True)
            fake.__getitem__.assert_called_once()
            fake.fn.__getitem__.assert_not_called()
            emit(kind="shape_fallback", rejected={key: not value for key, value in failures.items()}, native_autotuner=True)

        if not graph:
            return
        candidate_pool.copy_(initial_pool)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph_object = torch.cuda.CUDAGraph()
        with watch_cache() as capture_cache, torch.cuda.graph(graph_object, stream=stream):
            replay_outputs = wrapper(**common, initial_state=candidate_pool[:, 0], sm80_bv16=True)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        assert not any(capture_cache.values()), capture_cache
        for replay in range(2):
            for tensor in (k, w, u):
                tensor.normal_(generator=generator).mul_(.02)
            gk.normal_(generator=generator).mul_(.02).sub_(.05)
            changed = [tokens // 2 + 1, tokens // 4 - 1, tokens // 4] if replay == 0 else lengths
            fresh = metadata(changed, device, dtype)
            cu.copy_(fresh[0])
            chunk_indices.copy_(fresh[1])
            slots.copy_(fresh[2].flip(0))
            offsets.copy_(fresh[3])
            destinations.copy_(fresh[4].flip(0))
            # These are the same graph-captured addresses; recompute the
            # contents explicitly rather than reuse a stale identity cache.
            chunk_offsets.copy_(prepare_chunk_offsets(fresh[0], 64))
            initial_pool.normal_(generator=generator).mul_(.02)
            native_pool.copy_(initial_pool)
            candidate_pool.copy_(initial_pool)
            native_args = dict(common, cu_seqlens=fresh[0], chunk_indices=fresh[1])
            fresh_reference = wrapper(**native_args, initial_state=native_pool[:, 0])
            graph_object.replay()
            torch.cuda.synchronize(device)
            exact = dict(h=bits(replay_outputs[0], fresh_reference[0]),
                         v_new=bits(replay_outputs[1], fresh_reference[1]),
                         complete_state_envelope=bits(candidate_pool, native_pool))
            assert all(exact.values()), (replay, exact)
            emit(kind="fresh_graph", tokens=tokens, replay=replay,
                 changed_boundaries_slots_inputs=True, raw_bits=exact, capture_cache=capture_cache)


def main():
    assert torch.cuda.is_available(), "Run this test on one A100"
    device = torch.device("cuda", torch.cuda.current_device())
    assert torch.cuda.get_device_capability(device) == (8, 0)
    from sglang.kernels.ops.attention.fla import kda_state_sm80 as helper
    from sglang.kernels.ops.attention.fla import chunk_delta_h as delta
    from sglang.kernels.ops.attention.fla import chunk_delta_h_snapshot as snapshots
    from sglang.srt.layers.attention.linear.kernels import kda_triton as module

    emit(kind="environment", torch=torch.__version__, triton=triton.__version__,
         gpu=torch.cuda.get_device_name(device),
         source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES})
    off = arm_checks(device, helper, delta, module)
    extend_contract(module, off, device, False)
    # Mock call records can form cycles retaining the synthetic extend input.
    # Release those before taking the warmup allocation baseline.
    gc.collect()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    memory_before = memory(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with mock.patch.dict(os.environ, {FLAG: "1"}), watch_cache() as counts:
        armed = module.TritonKDAKernel()
        assert armed.arm_prefill_state(device) is True
    torch.cuda.synchronize(device)
    warm_seconds = time.perf_counter() - started
    memory_after = memory(device)
    functions = (delta.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn,
                 snapshots.chunk_gated_delta_rule_fwd_kernel_h_blockdim64_snapshot.fn)
    variants = [list(fn.device_caches[device.index][0].values()) for fn in functions]
    assert list(map(len, variants)) == [4, 8], list(map(len, variants))
    assert all(kernel.function is not None for group in variants for kernel in group)
    after = torch.cuda.memory_allocated(device)
    assert after == before, (before, after)
    emit(kind="startup_warm", counts=counts, loaded_variants=[len(v) for v in variants],
         temporary_peak_bytes=torch.cuda.max_memory_allocated(device) - before,
         retained_tensor_bytes=after - before, seconds=warm_seconds,
         memory_before=memory_before, memory_after=memory_after,
         memory_delta={key: memory_after[key] - memory_before[key] for key in memory_before},
         memory_scope="tensor allocator plus device-wide free delta including module loads; GPU idle")
    with watch_cache() as counts:
        assert helper.warmup_state_sm80(device.index) is True
    assert not any(counts.values()), counts
    extend_contract(module, armed, device, True)
    for dtype in (torch.int32, torch.int64):
        for tokens, lengths in ((8192, [8192]), (16384, [8191, 4097, 4096])):
            for mode in ("normal", "snapshot_no_export", "snapshot_export"):
                case(device, helper, delta, snapshots, lengths, dtype, mode,
                     graph=(dtype == torch.int64 and mode == "snapshot_export" and len(lengths) == 3),
                     guards=(dtype == torch.int32 and mode == "normal" and tokens == 8192))
    # 8k ragged crosses into NT_BUCKET=2; it must reuse the same warm entry.
    case(device, helper, delta, snapshots, [4095, 2049, 2048], torch.int64,
         "snapshot_export", graph=True)
    emit(kind="complete", signature_matrix_cases=12, ragged_8k_case=True, passed=True)


if __name__ == "__main__":
    main()
