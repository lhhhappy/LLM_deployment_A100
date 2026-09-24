#!/usr/bin/env python3
"""172 screening on a real A100: production function, BF16 boundary, graph, full Marlin.

TP8 per-rank dimensions with random weights; not a distributed/model-quality test.
Run with candidate SGLang on PYTHONPATH. Output is JSONL; --full adds 289-expert MoE.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import time

import torch

moe = importlib.import_module('sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe')


def emit(**x):
    print(json.dumps(x, allow_nan=False), flush=True)


def compare(new, ref, label, exact=True):
    finite = torch.isfinite(ref)
    nan_ok = torch.equal(torch.isnan(new), torch.isnan(ref))
    inf_ok = (torch.equal(torch.isposinf(new), torch.isposinf(ref)) and
              torch.equal(torch.isneginf(new), torch.isneginf(ref)))
    d = (new[finite].float() - ref[finite].float()).abs()
    different = ((new != ref) & finite).sum().item()
    result = dict(case=label, different=different, elements=ref.numel(),
                  max_abs=d.max().item() if d.numel() else 0,
                  rel_l2=(d.norm() / ref[finite].float().norm().clamp_min(1e-20)).item(),
                  nan_mask_equal=nan_ok, inf_mask_equal=inf_ok)
    emit(kind='numeric', **result)
    if different:
        ids = torch.nonzero((new != ref) & finite)[:8]
        emit(kind='mismatches', case=label, indices=ids.tolist(),
             new=[float(new[tuple(i)]) for i in ids], ref=[float(ref[tuple(i)]) for i in ids])
    assert nan_ok and inf_ok and (not exact or not different), result
    return result


def activation(x, flag, out=None, limit=10.):
    moe._AX_FUSE_CLAMPED_SWIGLU = flag
    if out is None:
        out = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
    moe.swiglu_limit_func(out, x, limit)
    return out


def measure(fn, repeats=7, inner=20):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, gpu = [], []
    for _ in range(repeats):
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        torch.cuda.synchronize()
        start = time.perf_counter(); a.record()
        for _ in range(inner): fn()
        b.record(); b.synchronize()
        wall.append((time.perf_counter() - start) * 1000 / inner)
        gpu.append(a.elapsed_time(b) / inner)
    return dict(wall_ms=wall, gpu_ms=gpu, wall_p50=statistics.median(wall),
                gpu_p50=statistics.median(gpu),
                extra_allocated=torch.cuda.max_memory_allocated() - live,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())


def numerics():
    # Ensure the comparator rejects changing the sign of infinity.
    try:
        compare(torch.tensor([float('-inf')], device='cuda'),
                torch.tensor([float('inf')], device='cuda'), 'negative_control_inf_sign')
    except AssertionError:
        emit(kind='negative_control', inf_sign_rejected=True)
    else:
        raise AssertionError('comparator accepted an infinity sign change')
    bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    ups = torch.tensor([-10, -1, -.25, 0, .125, 1, 10], device='cuda', dtype=torch.bfloat16)
    for mode in ('all_gate_bits', 'all_up_bits'):
        gate, up = bits[:, None].expand(-1, 7), ups.expand(65536, -1)
        if mode == 'all_up_bits': gate, up = up, gate
        x = torch.cat([gate, up], dim=1)
        compare(activation(x, True), activation(x, False), mode)
    for rows in (0, 1, 33, 37, 63, 65, 513):
        x = torch.randn(rows * 2, 512, device='cuda', dtype=torch.bfloat16)[::2]
        out = torch.empty((rows * 2, 256), device='cuda', dtype=torch.bfloat16)[::2]
        compare(activation(x, True, out), activation(x, False), f'row_stride_{rows}')
    for dtype, stride, limit in ((torch.float16, 1, 10.), (torch.bfloat16, 2, 10.),
                                 (torch.bfloat16, 1, 7.)):
        x = torch.randn(37, 512 * stride, device='cuda', dtype=dtype)[:, ::stride]
        compare(activation(x, True, limit=limit), activation(x, False, limit=limit),
                f'fallback_{dtype}_{stride}_{limit}')
    x = torch.randn(4096, 512, device='cuda', dtype=torch.bfloat16) * 5
    ref = activation(x, False)
    alt = (torch.nn.functional.silu(x[:, :256].float().clamp(max=10)) *
           x[:, 256:].float().clamp(-10, 10)).to(torch.bfloat16)
    count = (alt != ref).sum().item()
    assert count > 0, 'negative control failed to detect removal of BF16 intermediate rounding'
    emit(kind='negative_control', fully_fp32_expression_differences=count)


def bench_activation():
    for tokens in (32, 256, 1024, 4096, 8192, 16384):
        x = torch.randn(tokens * 9, 512, device='cuda', dtype=torch.bfloat16)
        outputs = [torch.empty((tokens * 9, 256), device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        graphs = []
        for i, flag in enumerate((False, True)):
            for _ in range(3): activation(x, flag, outputs[i])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): activation(x, flag, outputs[i])
            graphs.append(graph)
        for _ in range(3):
            x.normal_()
            for g in graphs: g.replay()
            compare(outputs[1], outputs[0], f'graph_mutation_{tokens}')
        for order in ((0, 1), (1, 0)):
            for mode in ('eager', 'graph'):
                for i in order:
                    fn = (lambda i=i: activation(x, bool(i), outputs[i])) if mode == 'eager' else graphs[i].replay
                    emit(kind='activation_cost', tokens=tokens, expanded_rows=tokens * 9,
                         arm=i, order=list(order), mode=mode, **measure(fn))


def full_moe(audit_only=False):
    from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin
    m = torch.nn.Module()
    m.num_experts, m.hidden_size, m.intermediate_size_per_partition = 289, 4096, 256
    m.weight_block_size, m.orig_dtype = [128, 128], torch.bfloat16
    for name, n, k in [('w13', 512, 4096), ('w2', 4096, 256)]:
        w = torch.randn(289, n, k, device='cuda', dtype=torch.bfloat16).float() * .02
        blocks = w.view(289, n // 128, 128, k // 128, 128)
        scale = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-6) / 448.
        q = (blocks / scale[:, :, None, :, None]).reshape_as(w).to(torch.float8_e4m3fn)
        setattr(m, name + '_weight', torch.nn.Parameter(q, requires_grad=False))
        setattr(m, name + '_weight_scale_inv', torch.nn.Parameter(scale, requires_grad=False))
        del w, blocks, q, scale
    prepare_moe_fp8_layer_for_marlin(m, size_k_first=False)
    for tokens in ((256,) if audit_only else (33, 256, 1024, 4096, 8192, 16384)):
        x = torch.randn(tokens, 4096, device='cuda', dtype=torch.bfloat16)
        ids = torch.rand(tokens, 288, device='cuda').topk(8, dim=-1).indices.int()
        ids = torch.cat([ids, torch.full((tokens, 1), 288, device='cuda', dtype=torch.int32)], 1)
        weights = torch.cat([torch.randn(tokens, 8, device='cuda').softmax(-1),
                             torch.ones(tokens, 1, device='cuda')], 1)
        logits = torch.empty(tokens, 289, device='cuda')
        def run(flag):
            moe._AX_FUSE_CLAMPED_SWIGLU = flag
            return moe.fused_marlin_moe(hidden_states=x, w1=m.w13_weight, w2=m.w2_weight,
                w1_scale=m.w13_weight_scale, w2_scale=m.w2_weight_scale, gating_output=logits,
                topk_weights=weights, topk_ids=ids, num_bits=8, fp8_weights=True, clamp_limit=10.)
        # Marlin GEMM's own reduction order must be measured separately from the
        # changed activation. Audit both activations on the SAME GEMM result.
        original_activation = moe.swiglu_limit_func
        def audited_activation(out, inp, limit):
            flag = moe._AX_FUSE_CLAMPED_SWIGLU
            try:
                ref = torch.empty_like(out)
                moe._AX_FUSE_CLAMPED_SWIGLU = False
                original_activation(ref, inp, limit)
                moe._AX_FUSE_CLAMPED_SWIGLU = True
                original_activation(out, inp, limit)
                compare(out, ref, f'activation_on_marlin_output_{tokens}')
                # Preserve the requested arm. Without this copy the labelled
                # reference repeats would both execute the fused activation.
                if not flag:
                    out.copy_(ref)
            finally:
                moe._AX_FUSE_CLAMPED_SWIGLU = flag
        moe.swiglu_limit_func = audited_activation
        try:
            baseline1, baseline2, candidate = run(False), run(False), run(True)
        finally:
            moe.swiglu_limit_func = original_activation
        repeat = compare(baseline2, baseline1, f'full_moe_reference_repeat_{tokens}', exact=False)
        changed = compare(candidate, baseline1, f'full_moe_candidate_{tokens}', exact=False)
        emit(kind='full_moe_numeric_boundary', tokens=tokens, reference_repeat=repeat,
             candidate=changed, activation_same_input_exact=True,
             note='Full MoE discrepancies reported, not accepted by a relaxed tolerance')
        if tokens == 256:
            # Counterfactual: hold the atomic expert-sort result fixed. This is
            # diagnostic only; all timing below restores production routing.
            routing = importlib.import_module('sglang.srt.layers.moe.fused_moe_triton')
            original_align = routing.moe_align_block_size
            cached = []
            def fixed_align(*a, **kw):
                if not cached:
                    cached.extend(t.clone() for t in original_align(*a, **kw))
                return tuple(cached)
            routing.moe_align_block_size = fixed_align
            try:
                fixed_base1, fixed_base2, fixed_candidate = run(False), run(False), run(True)
                compare(fixed_base2, fixed_base1, 'fixed_routing_reference_repeat', exact=False)
                compare(fixed_candidate, fixed_base1, 'fixed_routing_candidate', exact=False)
            finally:
                routing.moe_align_block_size = original_align
        if audit_only:
            continue
        for order in ((False, True), (True, False)):
            for flag in order:
                emit(kind='full_moe_cost', tokens=tokens, arm=int(flag), order=list(order),
                     **measure(lambda: run(flag), repeats=5, inner=5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--numeric-only', action='store_true')
    parser.add_argument('--full-audit-only', action='store_true',
                        help='Only M256 full MoE arm/routing diagnostic; no timing')
    args = parser.parse_args()
    torch.manual_seed(172)
    emit(kind='setup', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         source=moe.__file__, sha256=hashlib.sha256(Path(moe.__file__).read_bytes()).hexdigest(),
         test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         dtype='bfloat16', intermediate_per_tp8_rank=256, actual_tp=1, random_weights=True)
    if args.full_audit_only:
        full_moe(audit_only=True)
        emit(kind='done', full_moe='arm_and_routing_audit_reported')
    else:
        numerics()
        if not args.numeric_only:
            bench_activation()
            if args.full: full_moe()
        emit(kind='done', activation_checks_passed=True,
             full_moe='variance_and_cost_reported' if args.full and not args.numeric_only else 'not_run')
