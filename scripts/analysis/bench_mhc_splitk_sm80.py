#!/usr/bin/env python3
"""Test the large-M mHC prenorm split-K choice, including its final reduction.

Research only. Uses existing production TileLang kernels; no model dispatch
changes. Measures complete post -> prenorm -> Sinkhorn/weighted-sum/RMSNorm.
Each candidate retains the post BF16 residual and folds partial reduction into
the original finalization kernel. Changing the K sum order is not bit exact.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch

from bench_moe_candidate_sm80 import DeviceTelemetry


def emit(**record):
    print(json.dumps(record), flush=True)


def relative_l2(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def reference(residual, fn, scale, base, norm_weight):
    """FP64 reference; mirror the existing fused norm's pre-BF16 square sum."""
    x = residual.double()
    flat = x.flatten(1)
    mix = (flat @ fn.double().T) * torch.rsqrt(flat.square().mean(-1) + 1e-5)[:, None]
    pre = torch.sigmoid(mix[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(mix[:, 4:8] * scale[1] + base[4:8])
    comb = (mix[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4).softmax(-1) + 1e-6
    comb /= comb.sum(-2, keepdim=True) + 1e-6
    for _ in range(19):
        comb /= comb.sum(-1, keepdim=True) + 1e-6
        comb /= comb.sum(-2, keepdim=True) + 1e-6
    weighted = (pre[:, :, None] * x).sum(1)
    inv = torch.rsqrt(weighted.square().mean(-1) + 1e-5)
    output = weighted.bfloat16().double() * inv[:, None] * norm_weight.double()
    return post, comb.flatten(1), output.bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, nargs='+', default=[8192, 16384])
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--inner', type=int, default=16)
    args = parser.parse_args()
    import sglang.kernels.ops.layernorm.mhc as mhc
    import triton

    assert torch.cuda.get_device_capability() == (8, 0)
    telemetry = DeviceTelemetry()
    source = Path(mhc.__file__)
    emit(kind='environment', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         triton=triton.__version__, arguments=vars(args), scope='synthetic complete mHC boundary',
         source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20,
         reference_note='fused norm uses the unrounded weighted-sum square sum, as native code does')
    # (split, token block, hidden block); split=0 is the native large-M branch.
    configs = [(0, 32, 256), (2, 32, 256), (4, 32, 256), (8, 32, 256),
               (16, 32, 256), (4, 64, 256), (4, 32, 128)]
    try:
        with torch.inference_mode():
            for m in args.rows:
                torch.manual_seed(173928 + m)
                residual = torch.randn(m, 4, 4096, device='cuda', dtype=torch.bfloat16)
                hidden = torch.randn(m, 4096, device='cuda', dtype=torch.bfloat16)
                previous_comb = torch.randn(m, 4, 4, device='cuda').softmax(-1)
                previous_post = torch.sigmoid(torch.randn(m, 4, device='cuda')) * 2
                fn = torch.randn(24, 16384, device='cuda') / 128
                scale = torch.tensor([0.25, 0.25, 0.25], device='cuda')
                base = torch.randn(24, device='cuda') * 0.1
                norm_weight = (1 + torch.randn(4096, device='cuda') * 0.1).bfloat16()
                post_residual = torch.empty_like(residual)

                def run_post():
                    mhc.mhc_post_tilelang(previous_comb, residual, previous_post, hidden,
                                          post_residual, 4, 4096)

                run_post()
                ids = torch.linspace(0, m - 1, 32, device='cuda').long().unique()
                oracle = reference(post_residual[ids], fn, scale, base, norm_weight)
                oracle_gemm = post_residual[ids].flatten(1).double() @ fn.double().T
                oracle_sqr = post_residual[ids].flatten(1).double().square().sum(-1)
                variants = {}
                native_errors = None
                for split, bm, bk in configs:
                    name = 'native' if split == 0 else f'split{split}_m{bm}_k{bk}'
                    ns, width = (1, 24) if split == 0 else (split, 32)
                    mul = torch.empty(ns, m, width, device='cuda')
                    sqr = torch.empty(ns, m, device='cuda')
                    post = torch.empty(m, 4, device='cuda')
                    comb = torch.empty(m, 16, device='cuda')
                    output = torch.empty(m, 4096, device='cuda', dtype=torch.bfloat16)
                    if split == 0:
                        def project(mul=mul, sqr=sqr):
                            mhc.mhc_pre_gemm_sqrsum_tilelang(post_residual.flatten(1), fn,
                                mul.squeeze(0), sqr.squeeze(0), 24, 16384)
                    else:
                        kernel, _ = mhc.mhc_pre_gemm_sqrsum_splitk_kernel(24, 16384, split, bm, bk)

                        def project(kernel=kernel, mul=mul, sqr=sqr):
                            kernel(post_residual.flatten(1), fn, mul, sqr)

                    def finish(mul=mul, sqr=sqr, post=post, comb=comb, output=output, ns=ns, width=width):
                        mhc.mhc_pre_big_fuse_with_norm_tilelang(mul, sqr, scale, base,
                            post_residual, post, comb, output, norm_weight, 4096,
                            1e-5, 1e-6, 1e-6, 2.0, 20, 1e-5, ns, 4, width)

                    def full(project=project, finish=finish):
                        run_post()
                        project()
                        finish()

                    full()
                    torch.cuda.synchronize()
                    errors = [relative_l2(t[ids], ref) for t, ref in zip((post, comb, output), oracle)]
                    gemm_error = relative_l2(mul[:, ids, :24].sum(0), oracle_gemm)
                    sqr_error = relative_l2(sqr[:, ids].sum(0), oracle_sqr)
                    finite = all(bool(torch.isfinite(t).all()) for t in (mul, sqr, post, comb, output))
                    emit(kind='numerics', rows=m, variant=name, finite=finite,
                         fp64_relative_l2=errors, gemm_relative_l2=gemm_error, sqr_relative_l2=sqr_error,
                         scratch_bytes=mul.untyped_storage().nbytes() + sqr.untyped_storage().nbytes())
                    assert finite and max(errors) < 0.01 and gemm_error < 0.003 and sqr_error < 1e-5
                    if native_errors is None:
                        native_errors = errors
                    else:
                        assert all(a <= 1.10 * b + 1e-4 for a, b in zip(errors, native_errors))
                    for _ in range(3):
                        full()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        full()
                    # These buffers were allocated outside capture; retain them
                    # explicitly for every graph's lifetime.
                    variants[name] = (graph, post, comb, output, mul, sqr)

                # Fresh data, same captured pointers; all candidate graphs must consume it.
                residual.mul_(0.71)
                hidden.add_(0.37)
                run_post()
                fresh_reference = reference(post_residual[ids], fn, scale, base, norm_weight)
                for name, buffers in variants.items():
                    graph, post, comb, output = buffers[:4]
                    graph.replay()
                    errors = [relative_l2(t[ids], ref) for t, ref in zip((post, comb, output), fresh_reference)]
                    emit(kind='fresh_graph_numerics', rows=m, variant=name, fp64_relative_l2=errors)
                    assert max(errors) < 0.01
                samples = {name: [] for name in variants}
                rng = random.Random(173 + m)
                for iteration in range(args.rounds):
                    names = list(variants)
                    rng.shuffle(names)
                    for name in names:
                        graph = variants[name][0]
                        for _ in range(3):
                            graph.replay()
                        torch.cuda.synchronize()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        before = telemetry.read()
                        start.record()
                        for _ in range(args.inner):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        elapsed = start.elapsed_time(end) / args.inner
                        samples[name].append(elapsed)
                        emit(kind='timing_round', rows=m, round=iteration, variant=name, ms=elapsed,
                             telemetry_before=before, telemetry_after=telemetry.read())
                for name, values in samples.items():
                    gains = [(a-b)/a*100 for a, b in zip(samples['native'], values)]
                    emit(kind='timing', rows=m, variant=name, median_ms=statistics.median(values),
                         paired_median_gain_percent=statistics.median(gains), rounds=values, paired_gains=gains)
                for graph, *_ in variants.values():
                    graph.reset()
                del variants, residual, hidden, post_residual
            emit(kind='complete')
    finally:
        telemetry.close()


if __name__ == '__main__':
    main()
