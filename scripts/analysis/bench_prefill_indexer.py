#!/usr/bin/env python3
"""Measure the served 113 SM80 ragged indexer at TP8 per-rank shapes.

This is an operator probe with synthetic activations, not a service score. NQ is
the query row count AFTER 114 sharding; NK counts pooled keys (index_kpool=4).
The baseline includes FP8 unpack, scratch allocation, masking and FP32 output.
Each candidate only changes the existing kernel's launch geometry. No autotune
or shape search is introduced in serving.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import random
import statistics
import time

import torch
import triton


BASE = (2, 128, 32, 4, 4)


def emit(**item):
    print(json.dumps(item, allow_nan=False), flush=True)


def load_source(path):
    spec = importlib.util.spec_from_file_location("probe_indexer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(nq, nk, layout, seed=927):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((nq, 32, 128), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn((nk, 128), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    sc = torch.rand((nk,), generator=gen, device="cuda") + 0.1
    w = torch.rand((nq, 32), generator=gen, device="cuda") - 0.2
    row = torch.arange(nq, device="cuda", dtype=torch.int32)
    if layout == "full":
        ks = torch.zeros_like(row)
        ke = torch.full_like(row, nk)
    elif layout == "causal":
        # One contiguous rank shard in a late 8k/16k chunk, pooled by four.
        ks = torch.zeros_like(row)
        ke = (nk - (nq + 3) // 4 + (row + 1) // 4).clamp(0, nk)
    elif layout == "ragged":
        group = row * 4 // max(nq, 1)
        ks = group * nk // 4
        ke = ((group + 1) * nk // 4).clamp(max=nk)
        ke = torch.where(row % 17 == 0, ks, ke)
    else:
        raise ValueError(layout)
    return q, (k, sc), w, ks, ke


def launch(module, args, cfg):
    q, (k, scale), weights, ks, ke = args
    nq, h, d = q.shape
    nk = k.shape[0]
    bq, bk, group, loop, warps = cfg
    out = torch.empty((nq, nk), device=q.device, dtype=torch.float32)
    qb = torch.empty((nq, h, d), device=q.device, dtype=torch.bfloat16)
    kb = torch.empty((nk, d), device=q.device, dtype=torch.bfloat16)
    module._unpack_prefill[(triton.cdiv(nq * h * d, 1024),)](
        q.view(torch.uint8), qb, nq, h, d, *q.stride(), 1024, num_warps=4)
    module._unpack_prefill[(triton.cdiv(nk * d, 1024),)](
        k.view(torch.uint8), kb, nk, 1, d, k.stride(0), 0, k.stride(1),
        1024, num_warps=4)
    module._prefill[(triton.cdiv(nq, bq) * triton.cdiv(nk, bk * loop),)](
        qb, kb, scale, weights, ks, ke, out, nq, nk, h,
        *qb.stride(), *kb.stride(), scale.stride(0), *weights.stride(),
        ks.stride(0), ke.stride(0), True, bq, bk, h, group, loop,
        num_warps=warps, num_stages=1, enable_fp_fusion=False)
    return out


def elapsed(fn, repeat):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        result = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def configs():
    out = [BASE]
    for bq, bk, loop, warps in (
        (1, 128, 4, 4), (2, 64, 4, 4), (2, 128, 1, 4),
        (2, 128, 2, 4), (2, 128, 8, 4), (2, 256, 2, 4),
        (2, 256, 4, 4), (4, 64, 4, 4), (4, 128, 1, 4),
        (4, 128, 2, 4), (4, 128, 4, 4), (4, 128, 8, 4),
        (4, 256, 2, 4), (4, 256, 4, 8), (8, 64, 2, 4),
        (8, 64, 4, 4), (8, 128, 2, 4), (8, 128, 4, 8),
    ):
        out.append((bq, bk, 32, loop, warps))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--nq", type=int, default=1024)
    parser.add_argument("--nk", type=int, default=32768)
    parser.add_argument("--layout", choices=("full", "causal", "ragged"), default="causal")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--configs", help="JSON list of [BQ,BK,GROUP,LOOP,warps]; default exploratory set")
    opt = parser.parse_args()
    mod = load_source(opt.source)
    choices = [tuple(c) for c in json.loads(opt.configs)] if opt.configs else configs()
    if BASE not in choices:
        choices.insert(0, BASE)
    torch.backends.cuda.matmul.allow_tf32 = False
    emit(kind="environment", torch=torch.__version__, triton=triton.__version__,
         device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
         nq=opt.nq, nk=opt.nk, layout=opt.layout, source=str(opt.source),
         output_mib=opt.nq * opt.nk * 4 / 2**20,
         unpack_mib=(opt.nq * 32 * 128 + opt.nk * 128) * 2 / 2**20)
    args = case(opt.nq, opt.nk, opt.layout)
    reference = mod.fp8_mqa_logits(*args, clean_logits=True)
    valid = []
    for cfg in choices:
        try:
            begin = time.monotonic()
            result = launch(mod, args, cfg)
            torch.cuda.synchronize()
            compile_s = time.monotonic() - begin
            equal = torch.equal(reference, result)
            masks_equal = torch.equal(torch.isfinite(reference), torch.isfinite(result))
            finite = torch.isfinite(reference) & torch.isfinite(result)
            diff = float((reference[finite] - result[finite]).abs().max().item()) if finite.any() else 0.0
            emit(kind="correctness", cfg=cfg, bit_exact=equal, masks_equal=masks_equal,
                 max_abs=diff, first_launch_s=compile_s)
            if equal:
                valid.append(cfg)
            del result, finite
        except Exception as exc:
            emit(kind="rejected", cfg=cfg, error=f"{type(exc).__name__}: {exc}")
    del reference
    samples = {c: [] for c in valid}
    order = list(valid)
    rng = random.Random(927)
    for _ in range(opt.rounds):
        rng.shuffle(order)
        for cfg in order:
            for _ in range(3):
                launch(mod, args, cfg)
            samples[cfg].append(elapsed(lambda: launch(mod, args, cfg), opt.repeats))
    baseline = statistics.median(samples[BASE])
    for cfg in sorted(valid, key=lambda c: statistics.median(samples[c])):
        ms = statistics.median(samples[cfg])
        emit(kind="timing", cfg=cfg, median_ms=ms, speedup=baseline / ms,
             samples_ms=samples[cfg], nq=opt.nq, nk=opt.nk, layout=opt.layout)
    emit(kind="complete", tested=len(choices), bit_exact=len(valid),
         note="Synthetic operator result; no model-output or TP8 service claim")


if __name__ == "__main__":
    main()
