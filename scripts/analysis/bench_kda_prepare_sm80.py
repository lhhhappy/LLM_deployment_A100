#!/usr/bin/env python3
"""Compare KDA preparation and recurrence variants on one A100.

The native path copies Q/K/V then normalizes Q/K. Read the original BF16
convolution output directly and write three contiguous planes in one kernel.
This preserves the convolution's BF16 rounding and the original L2 formula.
Default mode reproduces scoped research overrides. --production instead uses
the actual instance arm methods and guarded wrappers, without kernel overrides.
No model projections, TP collectives, or service traffic are included.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace as NS

import torch
import triton
import triton.language as tl

from bench_moe_candidate_sm80 import DeviceTelemetry


@triton.jit(do_not_specialize=['ROWS'])
def prepare_qkv(Q, K, V, OUT, ROWS, STRIDE_T: tl.constexpr,
                HEADS: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    cols = tl.arange(0, D)
    plane = tl.program_id(1)
    src = Q if plane == 0 else (K if plane == 1 else V)
    offsets = (rows // HEADS)[:, None] * STRIDE_T + (rows % HEADS)[:, None] * D + cols[None, :]
    x = tl.load(src + offsets, rows[:, None] < ROWS, 0)
    target = OUT + plane * ROWS * D + rows[:, None] * D + cols[None, :]
    if plane < 2:
        # Same 128-element FP32 reduction and division as l2norm_fwd_kernel.
        xf = x.to(tl.float32)
        variance = tl.sum(xf * xf, axis=1)
        y = xf / tl.sqrt(variance + 1e-6)[:, None]
        tl.store(target, y, rows[:, None] < ROWS)
    else:
        # Preserve V's BF16 NaN payloads too: native V only copies bits.
        tl.store(target, x, rows[:, None] < ROWS)


def emit(**record):
    print(json.dumps(record), flush=True)


def bits(a, b):
    dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return torch.equal(a.view(dtype), b.view(dtype))


def prepared(q, k, v, config):
    assert q.shape == k.shape == v.shape and q.ndim == 4
    assert q.shape[0] == 1 and q.shape[-2:] == (8, 128)
    assert all(t.dtype == torch.bfloat16 and t.stride() == q.stride() for t in (q, k, v))
    assert q.stride(-1) == 1 and q.stride(-2) == 128
    n, heads, d = q.shape[1:]
    out = torch.empty((3, *q.shape), device=q.device, dtype=q.dtype)
    bt, warps = config
    prepare_qkv[(triton.cdiv(n * heads, bt), 3)](
        q, k, v, out, n * heads, q.stride(1), heads, d, bt,
        num_warps=warps, num_stages=3)
    return out[0], out[1], out[2]


@contextmanager
def override(config, state_bv=None):
    from sglang.srt.layers.attention.linear.kernels import kda_triton
    from sglang.kernels.ops.attention.fla import chunk_delta_h as delta
    from sglang.kernels.ops.attention.fla import chunk_delta_h_snapshot as snapshots
    original = kda_triton.chunk_kda

    def with_prepare(*args, **kwargs):
        assert not args and kwargs['use_qk_l2norm_in_kernel']
        q, k, v = prepared(kwargs['q'], kwargs['k'], kwargs['v'], config)
        kwargs.update(q=q, k=k, v=v, use_qk_l2norm_in_kernel=False)
        return original(**kwargs)

    class FixedStateKernel:
        def __init__(self, autotuner):
            assert len(autotuner.configs) == 1
            self.config = autotuner.configs[0]
            assert self.config.kwargs == {'BV': 32}
            self.jit = autotuner.fn

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return self.jit[grid](*args, **kwargs, BV=state_bv,
                                     num_warps=self.config.num_warps,
                                     num_stages=self.config.num_stages,
                                     num_ctas=self.config.num_ctas)
            return launch

    saved = []
    try:
        if config is not None:
            kda_triton.chunk_kda = with_prepare
        if state_bv is not None:
            for owner, name in (
                (delta, 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64'),
                (snapshots, 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64_snapshot'),
            ):
                old = getattr(owner, name)
                saved.append((owner, name, old))
                setattr(owner, name, FixedStateKernel(old))
        yield
    finally:
        kda_triton.chunk_kda = original
        for owner, name, old in saved:
            setattr(owner, name, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', type=int, default=[8192, 16384])
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--inner', type=int, default=8)
    parser.add_argument('--production', action='store_true',
                        help='use the actual armed TritonKDAKernel preparation path')
    args = parser.parse_args()
    from sglang.srt.layers.attention.linear import kda_backend as module
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
    from sglang.kernels.ops.attention.fla import chunk_delta_h as delta
    from sglang.kernels.ops.attention.fla import chunk_delta_h_snapshot as snapshots

    assert torch.cuda.get_device_capability() == (8, 0)
    telemetry = DeviceTelemetry()
    old_flag = module._AX_KDA_PREFILL_CPU_LENGTH
    module._AX_KDA_PREFILL_CPU_LENGTH = True
    state_tiles = [delta.GDN_CHUNK_H_BV, snapshots.GDN_CHUNK_H_BV]
    assert state_tiles == [32, 32], state_tiles
    configs = [(64, 8)]
    from sglang.srt.layers.attention.linear.kernels import kda_triton
    sources = [Path(__file__), Path(module.__file__), Path(kda_triton.__file__)]
    if args.production:
        from sglang.kernels.ops.attention.fla import kda_prepare_sm80 as prepare_module
        from sglang.kernels.ops.attention.fla import kda_state_sm80 as state_module
        from sglang.kernels.ops.attention.fla import kda as fla_kda
        sources += [Path(m.__file__) for m in (prepare_module, state_module, fla_kda, delta, snapshots)]
    emit(kind='environment', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         triton=triton.__version__, arguments=vars(args), native_state_tiles=state_tiles,
         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
         sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         scope='synthetic complete KDA core; no model projections or TP; each timed arm resets once then runs inner stateful calls')
    try:
        with torch.inference_mode():
            for n in args.rows:
                # Preparation itself, including zero vectors and small magnitudes.
                torch.manual_seed(41105928 + n)
                packed = torch.randn(n, 3, 8, 128, device='cuda', dtype=torch.bfloat16)
                packed[::251].zero_()
                packed[1::251].mul_(1e-7)
                # Every BF16 bit pattern in V, including signaling NaNs and -0.
                packed[:64, 2].view(torch.int16).copy_(
                    torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(64, 8, 128))
                q, k, v = (packed[:, i].unsqueeze(0) for i in range(3))
                expected = (l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous()), v.contiguous())
                valid = []
                for cfg in configs:
                    if args.production:
                        prepare_module.warmup_prepare_sm80(torch.cuda.current_device())
                        got = prepare_module.prepare_qkv_if_supported(q, k, v, q.device)
                        assert got is not None
                    else:
                        got = prepared(q, k, v, cfg)
                    exact = [bits(a, b) for a, b in zip(got, expected)]
                    emit(kind='prepare_numerics', rows=n, config=cfg, raw_bits=exact)
                    if all(exact):
                        valid.append(cfg)
                assert valid
                del packed, q, k, v, expected, got

                for ragged in (False, True):
                    lengths = [n // 2 - 1, n // 4 + 1, n // 4] if ragged else [n]
                    seqs, pad, heads, dim = len(lengths), 7 if ragged else 0, 8, 128
                    width, nslots = heads * dim, 3 * seqs + 3
                    generator = torch.Generator(device='cuda').manual_seed(170 + n + seqs)
                    def rand(*shape, dtype=torch.bfloat16):
                        return torch.randn(shape, device='cuda', dtype=dtype, generator=generator) * .2
                    # 171's fused projection returns split views with row gaps.
                    # After causal_conv1d_fn, Q/K/V again share packed stride 3072.
                    raw_envelope = rand(n + pad, 3 * width + (heads if ragged else 0))
                    raw = raw_envelope[:, :3 * width]
                    gate, beta = rand(1, n + pad, width), rand(1, n + pad, heads)
                    raw[n:] = float('nan')
                    cu = torch.tensor([0] + [sum(lengths[:i+1]) for i in range(seqs)], device='cuda', dtype=torch.int32)
                    slots_dtype = torch.int64 if args.production and ragged else torch.int32
                    slots = torch.arange(seqs, device='cuda', dtype=slots_dtype)
                    prefixes = torch.tensor([0 if i % 2 == 0 else 128 for i in range(seqs)], device='cuda', dtype=torch.int32)
                    # An envelope-strided state view exercises the actual slot pitch contract.
                    conv_envelope = rand(nslots, 2, 3, 3 * width)
                    ssm_envelope = rand(nslots, 2, heads, dim, dim, dtype=torch.float32)
                    conv, ssm = conv_envelope[:, 0], ssm_envelope[:, 0]
                    for i in range(seqs):
                        if i % 2 == 0:
                            ssm[i].zero_()
                    initial_conv, initial_ssm = conv_envelope.clone(), ssm_envelope.clone()
                    snapshot_dtype = torch.int64 if args.production else torch.int32
                    snapshot_offsets = torch.tensor([[64, (length // 64)*64] for length in lengths], device='cuda', dtype=snapshot_dtype) if ragged else None
                    snapshot_slots = torch.arange(seqs + 1, 3 * seqs + 1, device='cuda', dtype=snapshot_dtype).view(seqs, 2) if ragged else None
                    layer = NS(layer_id=0, conv_weights=rand(3*width, 4, dtype=torch.float32), bias=None,
                               q_dim=width, k_dim=width, v_dim=width, head_q_dim=dim, head_k_dim=dim, head_v_dim=dim,
                               A_log=rand(heads, dtype=torch.float32), dt_bias=rand(width, dtype=torch.float32), lower_bound=-5.)
                    fb = NS(forward_mode=ForwardMode.EXTEND, batch_size=seqs, extend_seq_lens_cpu=lengths,
                            extend_prefix_lens=prefixes, ax_kda_snapshot_offsets=snapshot_offsets,
                            ax_kda_snapshot_slots=snapshot_slots, attn_cp_metadata=None, tbo_parent_token_range=None)
                    backend = module.KDAAttnBackend.__new__(module.KDAAttnBackend)
                    backend.forward_metadata = NS(query_start_loc=cu, mamba_cache_indices=slots,
                                                  has_mamba_track_mask=False, _ax_cpu_logical_num_tokens=n)
                    backend.req_to_token_pool = NS(mamba2_layer_cache=lambda _: NS(conv=[conv], temporal=ssm))
                    backend.kernel_dispatcher = TritonKDAKernel()
                    backend.accept_lens_pool = None
                    native_kernel = backend.kernel_dispatcher
                    armed_kernels = {}
                    if args.production:
                        from unittest.mock import patch
                        for prep_on, state_on in ((False, False), (True, False), (False, True), (True, True)):
                            kernel = TritonKDAKernel()
                            with patch.dict(os.environ, {
                                'SGLANG_AX_KDA_PREFILL_PREPARE': str(int(prep_on)),
                                'SGLANG_AX_KDA_PREFILL_STATE_BV16': str(int(state_on)),
                            }):
                                assert kernel.arm_prefill_prepare(raw.device) is prep_on
                                assert kernel.arm_prefill_state(raw.device) is state_on
                            armed_kernels[prep_on, state_on] = kernel

                    def run():
                        return backend.forward_extend(layer, fb, raw, gate, beta)
                    def reset():
                        conv_envelope.copy_(initial_conv)
                        ssm_envelope.copy_(initial_ssm)

                    run()  # Warm the unchanged tensor-identity metadata cache.
                    reset()
                    ref_out = run().clone()
                    ref_conv, ref_ssm = conv_envelope.clone(), ssm_envelope.clone()
                    variants = {}
                    arms = {'native': (None, None), 'prepare': (valid[0], None),
                            'state16': (None, 16), 'prepare_state16': (valid[0], 16)}

                    @contextmanager
                    def arm_scope(cfg, bv):
                        if args.production:
                            backend.kernel_dispatcher = armed_kernels[cfg is not None, bv is not None]
                            try:
                                yield
                            finally:
                                backend.kernel_dispatcher = native_kernel
                        else:
                            with override(cfg, bv):
                                yield

                    for name, (cfg, bv) in arms.items():
                        with arm_scope(cfg, bv):
                            reset()
                            if args.production:
                                decisions = []
                                original_check = state_module.can_use_state_sm80
                                def checked_state(*a, **kw):
                                    decision = original_check(*a, **kw)
                                    decisions.append(decision)
                                    return decision
                                with patch.object(state_module, 'can_use_state_sm80', checked_state):
                                    out = run()
                                selected = bv is not None and (n == 16384 or seqs == 1)
                                assert decisions == ([True] if selected else []), (name, decisions)
                                emit(kind='production_dispatch', rows=n, sequences=seqs, variant=name,
                                     state_requested=bv is not None, state_decisions=decisions,
                                     slots_dtype=str(slots.dtype),
                                     snapshot_dtype=str(snapshot_dtype) if ragged else None)
                            else:
                                out = run()
                            exact = dict(output=bits(out, ref_out), conv_envelope=bits(conv_envelope, ref_conv),
                                         ssm_envelope=bits(ssm_envelope, ref_ssm))
                            emit(kind='core_numerics', rows=n, sequences=seqs, variant=name, raw_bits=exact)
                            assert all(exact.values())
                            reset(); torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            before = torch.cuda.memory_allocated()
                            out = run()
                            torch.cuda.synchronize()
                            emit(kind='eager_memory', rows=n, sequences=seqs, variant=name,
                                 peak_delta=torch.cuda.max_memory_allocated()-before,
                                 output_storage_bytes=out.untyped_storage().nbytes(),
                                 output_logical_bytes=out.numel()*out.element_size(),
                                 note='per call allocated peak; already retained graphs excluded from starting allocated')
                            for _ in range(3):
                                run()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                out = run()
                        variants[name] = (graph, out)

                    # Fixed lengths are intentional: this is a kernel comparison,
                    # not the general metadata-refresh graph in the other probe.
                    raw[:n].mul_(-0.43); gate.add_(.11); beta.sub_(.07)
                    slots.copy_(torch.arange(seqs, device='cuda', dtype=slots_dtype).flip(0))
                    reset(); expected = run().clone()
                    ref_conv, ref_ssm = conv_envelope.clone(), ssm_envelope.clone()
                    for name, (graph, out) in variants.items():
                        reset(); graph.replay()
                        exact = dict(output=bits(out, expected), conv_envelope=bits(conv_envelope, ref_conv),
                                     ssm_envelope=bits(ssm_envelope, ref_ssm))
                        emit(kind='fresh_graph_numerics', rows=n, sequences=seqs, variant=name, raw_bits=exact)
                        assert all(exact.values()) and bool(torch.isfinite(out).all())
                    samples = {name: [] for name in variants}
                    rng = random.Random(n + seqs)
                    for repeat in range(args.rounds):
                        order = list(variants); rng.shuffle(order)
                        for name in order:
                            graph = variants[name][0]
                            reset()
                            for _ in range(3): graph.replay()
                            reset(); torch.cuda.synchronize()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            before = telemetry.read()
                            start.record()
                            for _ in range(args.inner): graph.replay()
                            end.record(); end.synchronize()
                            elapsed = start.elapsed_time(end) / args.inner
                            samples[name].append(elapsed)
                            emit(kind='timing_round', rows=n, sequences=seqs, round=repeat, variant=name,
                                 ms=elapsed, telemetry_before=before, telemetry_after=telemetry.read())
                    for name, values in samples.items():
                        gains = [(a-b)/a*100 for a,b in zip(samples['native'], values)]
                        emit(kind='timing', rows=n, sequences=seqs, variant=name,
                             median_ms=statistics.median(values), paired_median_gain_percent=statistics.median(gains),
                             samples=values, paired_gains=gains)
                    # Production presently calls this core eagerly between graph
                    # breaks. Report GPU event span and Python enqueue separately.
                    eager = {name: [] for name in arms}
                    host = {name: [] for name in arms}
                    for repeat in range(args.rounds):
                        order = list(arms); rng.shuffle(order)
                        for name in order:
                            with arm_scope(*arms[name]):
                                reset()
                                for _ in range(3): run()
                                reset(); torch.cuda.synchronize()
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                before = telemetry.read()
                                start.record()
                                host_start = time.perf_counter_ns()
                                for _ in range(args.inner): out = run()
                                host_ms = (time.perf_counter_ns() - host_start) / 1e6 / args.inner
                                end.record(); end.synchronize()
                                ms = start.elapsed_time(end) / args.inner
                            eager[name].append(ms); host[name].append(host_ms)
                            emit(kind='eager_timing_round', rows=n, sequences=seqs, raw_stride=list(raw.stride()),
                                 round=repeat, variant=name, ms=ms, host_ms=host_ms,
                                 telemetry_before=before, telemetry_after=telemetry.read())
                    for name, values in eager.items():
                        gains = [(a-b)/a*100 for a,b in zip(eager['native'], values)]
                        emit(kind='eager_timing', rows=n, sequences=seqs, variant=name,
                             median_ms=statistics.median(values), paired_median_gain_percent=statistics.median(gains),
                             median_host_ms=statistics.median(host[name]), samples=values, host_samples=host[name],
                             paired_gains=gains)
                    for graph, _ in variants.values(): graph.reset()
                    del variants
            emit(kind='complete')
    finally:
        module._AX_KDA_PREFILL_CPU_LENGTH = old_flag
        telemetry.close()


if __name__ == '__main__':
    main()
