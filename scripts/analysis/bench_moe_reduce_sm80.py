#!/usr/bin/env python3
"""SM80 MoE weighted-reduction exploration against the real engine kernel.

All candidates preserve the expert order and fp32 FMA expression. Inputs remain
BF16 and router weights FP32. This probe does not change serving dispatch.
Timing alternates candidates with the baseline, flushes L2 before individual
eager events, and separately reports graph replay. Keep all measurements.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def flat_reduce(X, W, Y, M, H: tl.constexpr, TK: tl.constexpr,
                SCALE: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    row, col = offsets // H, offsets % H
    mask = row < M
    acc = tl.full((B,), 0, tl.float32)
    for expert in tl.static_range(TK):
        weight = tl.load(W + row * TK + expert, mask, 0).to(tl.float32)
        if SCALE != 1.0:
            weight = weight * SCALE
        value = tl.load(X + (row * TK + expert) * H + col, mask, 0).to(tl.float32)
        acc += value * weight
    tl.store(Y + offsets, acc, mask)


def emit(**obj):
    print(json.dumps(obj), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows', type=int, nargs='+', default=[128, 1024, 4096, 8192, 16384])
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--calls', type=int, default=7)
    ap.add_argument('--configs', help='JSON list of [block_m, block_k, warps]; block_m=0 selects flat')
    ap.add_argument('--graph', action='store_true')
    args = ap.parse_args()
    from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum_kernel, _heuristic_config
    source=Path(inspect.getfile(_heuristic_config)).resolve()
    assert source.is_relative_to(Path(__file__).resolve().parents[2]/'engine'),source

    configs = json.loads(args.configs) if args.configs else [
        [1, 1024, 4], [1, 2048, 4], [1, 4096, 8], [2, 512, 4], [2, 1024, 4],
        [4, 512, 4], [4, 1024, 4], [4, 1024, 8], [8, 256, 4], [8, 512, 4],
        [8, 512, 8], [8, 1024, 4], [8, 1024, 8], [8, 1024, 16],
        [16, 256, 4], [16, 512, 4], [16, 512, 8], [32, 256, 4], [32, 256, 8],
        [0, 1024, 4], [0, 2048, 4], [0, 4096, 4], [0, 4096, 8],
    ]
    emit(kind='environment', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         triton=triton.__version__, source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         scope='single GPU, BF16 H4096 TK9, synthetic expert output')
    flush = torch.empty(96 << 20, dtype=torch.uint8, device='cuda')
    for rows in args.rows:
        gen = torch.Generator(device='cuda').manual_seed(928 + rows)
        x = torch.randn(rows, 9, 4096, device='cuda', dtype=torch.bfloat16, generator=gen)
        w = torch.rand(rows, 9, device='cuda', generator=gen)
        w /= w.sum(-1, keepdim=True)
        out = torch.empty(rows, 4096, device='cuda', dtype=x.dtype)
        expected = torch.empty_like(out)
        default = _heuristic_config(rows, 9, 4096, 2)

        def run(cfg, dest=out):
            bm, bk, warps, *stages = cfg
            if bm == 0:
                return flat_reduce[(triton.cdiv(rows * 4096, bk),)](
                    x, w, dest, rows, 4096, 9, 2.5, bk, num_warps=warps)
            return moe_fused_mul_sum_kernel[(triton.cdiv(4096, bk), triton.cdiv(rows, bm))](
                x, w, dest, None, None, rows, 9 * 4096, False, False, 9, 4096, 2.5,
                bm, bk, num_warps=warps, num_stages=stages[0] if stages else (4 if bm * bk <= 2048 else 2))

        run(default, expected)
        torch.cuda.synchronize()
        # Independent sampled float64 oracle; bound includes final BF16 rounding
        # and the small FP32 product/reduction errors, including cancellation.
        ids = torch.linspace(0, rows-1, min(16, rows), device='cuda').long().unique()
        xx, ww = x[ids].double(), w[ids].double() * 2.5
        terms = xx * ww[:, :, None]
        ref = terms.sum(1)
        tol = ref.abs() * 2**-8 + terms.abs().sum(1) * 2**-20 + 1e-7
        base_ratio = ((expected[ids].double()-ref).abs()/tol).max().item()
        assert base_ratio <= 1, base_ratio
        valid = [('baseline', list(default))]
        for cfg in configs:
            kernel = run(cfg)
            torch.cuda.synchronize()
            exact = torch.equal(out, expected)
            ratio = ((out[ids].double()-ref).abs()/tol).max().item()
            emit(kind='correctness', rows=rows, config=cfg, exact=exact,
                 oracle_ratio=ratio, baseline_oracle_ratio=base_ratio,
                 registers=kernel.n_regs, spills=kernel.n_spills, shared=kernel.metadata.shared)
            if exact and ratio <= 1:
                valid.append((str(cfg), cfg))
        times = {name: [] for name, _ in valid}
        for _, cfg in valid:
            for _ in range(3): run(cfg)
        rng = random.Random(928 + rows)
        for _ in range(args.rounds):
            rng.shuffle(valid)
            for name, cfg in valid:
                samples=[]
                for _ in range(args.calls):
                    flush.fill_(1)
                    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();run(cfg);b.record();b.synchronize()
                    samples.append(a.elapsed_time(b))
                times[name].append(statistics.median(samples))
        med = {k:statistics.median(v) for k,v in times.items()}
        emit(kind='timing', rows=rows, default=default, rounds_ms=times, medians_ms=med,
             speedups={k:med['baseline']/v for k,v in med.items()})
        if args.graph:
            selected=sorted(med,key=med.get)[:3]
            fns=dict(valid)
            selected=['baseline']+[k for k in selected if k!='baseline']
            graph_times={}
            for name in selected:
                cfg=fns[name]
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):run(cfg)
                samples=[]
                for _ in range(15):
                    flush.fill_(1)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();graph.replay();b.record();b.synchronize()
                    samples.append(a.elapsed_time(b)/8)
                graph_times[name]=samples
                del graph
            emit(kind='graph',rows=rows,per_call_samples_ms=graph_times,
                 note='8 consecutive calls; later calls can hit L2 for small shapes')
        del x,w,out,expected,xx,ww,terms,ref,tol
    emit(kind='complete',rows=args.rows)


if __name__ == '__main__':
    main()
