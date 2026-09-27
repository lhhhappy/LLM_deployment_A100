#!/usr/bin/env python3
"""Explore mHC post on SM80 against the actual TileLang kernel; no serving dispatch.

One Triton program handles a token and a hidden-dimension tile. Keep the four
residual terms in their original order. Numerical equality is checked before
timing; a float64 reference also checks sampled rows independently.
"""
import argparse
import json
import random
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def post_kernel(A, R, C, X, OUT, H: tl.constexpr, B: tl.constexpr, FMA: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * B + tl.arange(0, B)
    out_channel = tl.arange(0, 4)
    x = tl.load(X + row * H + d, d < H, 0).to(tl.float32)
    c = tl.load(C + row * 4 + out_channel)
    if FMA == 2:
        first_r = tl.load(R + row * 4 * H + d, d < H, 0).to(tl.float32)
        first_a = tl.load(A + row * 16 + out_channel)
        acc = tl.fma(c[:, None], x[None, :], first_a[:, None] * first_r[None, :])
        first_channel: tl.constexpr = 1
    else:
        acc = c[:, None] * x[None, :]
        first_channel: tl.constexpr = 0
    for channel in tl.static_range(first_channel, 4):
        residual = tl.load(R + (row * 4 + channel) * H + d, d < H, 0).to(tl.float32)
        mix = tl.load(A + row * 16 + channel * 4 + out_channel)
        if FMA:
            acc = tl.fma(mix[:, None], residual[None, :], acc)
        else:
            acc = acc + mix[:, None] * residual[None, :]
    tl.store(OUT + (row * 4 + out_channel[:, None]) * H + d[None, :], acc, d[None, :] < H)


def emit(**record):
    print(json.dumps(record), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=[33, 128, 512, 2048, 8192, 16384])
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--configs", help="JSON list of [hidden tile, warps, FMA mode: 0=off, 1=residual first, 2=post first]")
    args = ap.parse_args()
    from sglang.kernels.ops.layernorm.mhc import mhc_post_tilelang

    configs = json.loads(args.configs) if args.configs else [
        [b, warps, fma] for b, warps in [(256, 4), (512, 4), (1024, 4), (1024, 8), (2048, 8), (4096, 8)]
        for fma in [0, 1, 2]
    ]
    emit(kind="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         triton=triton.__version__, scope="synthetic mHC post, no serving change")
    flush = torch.empty(96 << 20, device="cuda", dtype=torch.uint8)
    for rows in args.rows:
        gen = torch.Generator(device="cuda").manual_seed(928 + rows)
        a = torch.rand(rows, 4, 4, device="cuda", generator=gen)
        for _ in range(5):
            a = a / a.sum(1, keepdim=True)
            a = a / a.sum(2, keepdim=True)
        c = 2 * torch.rand(rows, 4, device="cuda", generator=gen)
        residual = torch.randn(rows, 4, args.hidden, device="cuda", generator=gen).bfloat16()
        x = torch.randn(rows, args.hidden, device="cuda", generator=gen).bfloat16()
        out = torch.empty_like(residual)
        expected = torch.empty_like(residual)

        def baseline():
            mhc_post_tilelang(a, residual, c, x, expected, 4, args.hidden)

        def candidate(cfg):
            b, warps, fma = cfg
            return post_kernel[(rows, triton.cdiv(args.hidden, b))](
                a, residual, c, x, out, args.hidden, b, fma,
                num_warps=warps, enable_fp_fusion=False)

        baseline()
        ids = torch.linspace(0, rows - 1, min(64, rows), device="cuda").long().unique()
        rr, xx, aa, cc = residual[ids].double(), x[ids].double(), a[ids].double(), c[ids].double()
        ref = cc[:, :, None] * xx[:, None, :]
        magnitude = ref.abs()
        for i in range(4):
            term = aa[:, i, :, None] * rr[:, i, None, :]
            ref += term
            magnitude += term.abs()
        tolerance = 2**-8 * magnitude + 1e-7
        base_ratio = ((expected[ids].double() - ref).abs() / tolerance).max().item()
        assert base_ratio <= 1, base_ratio
        valid = []
        for cfg in configs:
            kernel = candidate(cfg)
            torch.cuda.synchronize()
            exact = torch.equal(out, expected)
            ratio = ((out[ids].double() - ref).abs() / tolerance).max().item()
            max_error = 0.0
            different = 0
            for start in range(0, rows, 128):
                left, right = out[start:start+128], expected[start:start+128]
                different += (left != right).sum().item()
                max_error = max(max_error, (left.float() - right.float()).abs().max().item())
            emit(kind="correctness", rows=rows, hidden=args.hidden, config=cfg, exact=exact,
                 different=different, max_abs=max_error, reference_ratio=ratio,
                 baseline_reference_ratio=base_ratio, registers=kernel.n_regs,
                 spills=kernel.n_spills, shared=kernel.metadata.shared)
            assert ratio <= 1, (cfg, ratio)
            if exact:
                valid.append(tuple(cfg))
        times = {"tilelang": []} | {str(cfg): [] for cfg in valid}
        fns = [("tilelang", baseline)] + [(str(cfg), lambda cfg=cfg: candidate(cfg)) for cfg in valid]
        for _, fn in fns:
            for _ in range(5):
                fn()
        rng = random.Random(928)
        for _ in range(args.rounds):
            rng.shuffle(fns)
            for name, fn in fns:
                observations = []
                for _ in range(args.calls):
                    flush.fill_(1)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    fn()
                    end.record()
                    end.synchronize()
                    observations.append(start.elapsed_time(end))
                times[name].append(statistics.median(observations))
        medians = {name: statistics.median(ts) for name, ts in times.items()}
        emit(kind="timing", rows=rows, hidden=args.hidden, medians_ms=medians, rounds_ms=times,
             speedups={name: medians["tilelang"] / ms for name, ms in medians.items()})
    emit(kind="complete", rows=args.rows)


if __name__ == "__main__":
    main()
