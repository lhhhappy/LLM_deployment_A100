#!/usr/bin/env python3
"""T43 A100 comparison against UNMODIFIED patch 110; JSONL receipts, no model/service.

Copy this file + tests/gpu/kernels/sm80_indexer_112.py + build/p110/sm80_deep_gemm.py
into one private GPU directory. --mode smoke|numeric|bench|graph|all.
Synthetic activations with model shapes, never claimed as model accuracy or SLO.
"""
import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import time

import torch
import triton
import triton.language as tl


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


HERE = Path(__file__).resolve().parent
# In the repo use the canonical sources; on GPU use colocated copies.
SRC = HERE / 'kernels/sm80_indexer_112.py'
REF = HERE.parents[1] / 'build/p110/sm80_deep_gemm.py'
if not SRC.exists():
    SRC, REF = HERE / 'sm80_indexer_112.py', HERE / 'sm80_deep_gemm.py'
new = module('kernels112', SRC)
old = module('oracle110', REF)
FP8 = torch.float8_e4m3fn
H, D, PAGE = 32, 128, 64


def emit(**data):
    print(json.dumps(data, sort_keys=True), flush=True)


def fp8(*shape, factor=0.5):
    return (torch.randn(shape, device='cuda') * factor).clamp(-448, 448).to(FP8)


def decode_case(B, L, N=1, ctx_kind='BN', edges=False, strided=False, h=32, factor=0.5):
    pages = triton.cdiv(L, PAGE)
    nb = max(pages + 7, 8)
    cache = torch.empty((nb, PAGE, 1, D + 4), dtype=torch.uint8, device='cuda')
    vals = fp8(nb, PAGE, D, factor=factor)
    scales = torch.rand((nb, PAGE), device='cuda') + 0.5
    flat = cache.reshape(nb, -1)
    flat[:, :PAGE * D] = vals.view(torch.uint8).reshape(nb, -1)
    flat[:, PAGE * D:] = scales.view(torch.uint8).reshape(nb, -1)
    bt = torch.stack([torch.randperm(nb, device='cuda')[:pages] for _ in range(B)]).int()
    q = fp8(B, N, h, D, factor=factor)
    w = torch.randn((B * N, h), device='cuda')
    # Positive scales, signed head weights as in the 110 tests.
    ctx = torch.full((B, N if ctx_kind == 'BN' else 1), L, device='cuda', dtype=torch.int32)
    if edges:
        lengths = [0, 1, 63, 64, 65, max(0, L - 13), L, L + 97]
        ctx.copy_(torch.tensor([lengths[i % len(lengths)] for i in range(ctx.numel())],
                               device='cuda', dtype=torch.int32).reshape_as(ctx))
        if pages:
            bt[:, 0] = -1
            bt[:, -1] = -1
    if ctx_kind == 'B':
        ctx = ctx[:, 0]
    if strided:
        # Noncontiguous tensors with last dimension contiguous (torch fp8 view requirement).
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        w = w.t().contiguous().t()
        bt = bt.t().contiguous().t()
        if ctx.ndim == 2:
            ctx = ctx.t().contiguous().t()
    return (q, cache, w, ctx, bt, None, L + 17)


def ragged_case(nq, nk, clean, layout='ragged', strided=False, h=32, factor=0.5):
    q, k = fp8(nq, h, D, factor=factor), fp8(nk, D, factor=factor)
    scale = torch.rand(nk, device='cuda') + 0.5
    w = torch.randn(nq, h, device='cuda')
    if layout == 'full':
        ks = torch.zeros(nq, device='cuda', dtype=torch.int32)
        ke = torch.full((nq,), nk, device='cuda', dtype=torch.int32)
    elif layout == 'causal':
        ks = torch.zeros(nq, device='cuda', dtype=torch.int32)
        ke = (torch.arange(nq, device='cuda') + nk - nq + 1).clamp(min=0).int()
    else:
        ks = torch.randint(0, max(1, nk // 2), (nq,), device='cuda', dtype=torch.int32)
        ke = torch.randint(max(1, nk // 2), max(2, nk + 1), (nq,), device='cuda', dtype=torch.int32)
        if nq >= 8:
            ks[:8] = torch.tensor([0, 0, 1, 63, 64, nk, 9, -5], device='cuda')
            ke[:8] = torch.tensor([0, 1, 1, 64, 65, nk, 2, nk + 5], device='cuda')
    if strided:
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        w = w.t().contiguous().t()
        sb = torch.zeros(nk, 2, device='cuda'); sb[:, 0] = scale; scale = sb[:, 0]
    return (q, (k, scale), w, ks, ke), {'clean_logits': clean}


def compare(a, b, label, topk=True):
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
    assert torch.equal(torch.isfinite(a), torch.isfinite(b)), label
    assert torch.equal(torch.isneginf(a), torch.isneginf(b)), label
    assert torch.equal(torch.isnan(a), torch.isnan(b)), label
    # Work row-wise: avoid allocating several [8192,190000] temporary matrices.
    relmax, l2max, absmax, overlaps = 0., 0., 0., []
    for start in range(0, a.shape[0], 16):
        av, bv = a[start:start+16], b[start:start+16]
        valid = torch.isfinite(bv)
        aa, bb = torch.where(valid, av, 0), torch.where(valid, bv, 0)
        diff = aa - bb
        rel = diff.abs().amax(1) / bb.abs().amax(1).clamp_min(1e-12)
        l2 = torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(bb, dim=1).clamp_min(1e-12)
        relmax = max(relmax, rel.max().item())
        l2max = max(l2max, l2.max().item())
        absmax = max(absmax, diff.abs().max().item())
        if topk:
            # Compare only the defined/finite selection domain. K=min(2048, valid length).
            # For decode all columns are defined; zero padding can participate, as in 110.
            kk = min(2048, av.shape[1])
            if kk:
                counts = valid.sum(1).clamp(max=kk)
                ia = av.topk(kk, dim=1).indices
                ib = bv.topk(kk, dim=1).indices.sort(dim=1).values
                at = torch.searchsorted(ib, ia).clamp(max=kk-1)
                match = ib.gather(1, at) == ia
                ranks = torch.arange(kk, device=a.device)[None, :]
                selected = ranks < counts[:, None]
                overlap_rows = (match & selected).sum(1).double() / counts.clamp_min(1)
                overlaps.extend(overlap_rows[counts > 0].tolist())

    overlap = min(overlaps, default=1.)
    result = dict(label=label, shape=list(a.shape), max_row_relative_linf=relmax,
                  max_row_relative_l2=l2max, max_abs=absmax, min_topk_overlap=overlap)
    emit(kind='numeric', **result)
    assert relmax < 1e-2, result
    assert overlap >= .995, result
    return result


@triton.jit
def decode_bits(X, Y, N: tl.constexpr):
    idx = tl.arange(0, 256)
    vals = new._e4m3_to_bf16(tl.load(X + idx))
    tl.store(Y + idx, vals)


def bit_test():
    bits = torch.arange(256, dtype=torch.uint8, device='cuda')
    out = torch.empty(256, dtype=torch.bfloat16, device='cuda')
    decode_bits[(1,)](bits, out, 256)
    ref = bits.view(FP8).to(torch.bfloat16)
    finite = torch.isfinite(ref)
    assert torch.equal(out.view(torch.int16)[finite], ref.view(torch.int16)[finite])
    assert torch.equal(torch.isnan(out), torch.isnan(ref))
    emit(kind='fp8_bits', tested=256, finite_bits_exact=254, nan_class_exact=2)


def numeric(smoke=False):
    bit_test()
    cases = [(2, 129, 2)] if smoke else itertools.product([1, 6, 32], [1024, 32000, 190000], [1, 2])
    for B, L, N in cases:
        for ck in ('B', 'BN'):
            args = decode_case(B, L, N, ck)
            compare(new.fp8_paged_mqa_logits(*args), old.fp8_paged_mqa_logits(*args),
                    f'decode B={B} L={L} N={N} ctx={ck}')
    for L in ([129] if smoke else [0, 1, 63, 64, 65, 1025]):
        args = decode_case(8, L, 2, edges=True, strided=True)
        for clean in (False, True):
            compare(new.fp8_paged_mqa_logits(*args, clean_logits=clean),
                    old.fp8_paged_mqa_logits(*args, clean_logits=clean), f'edge L={L} clean={clean}')
    for nk in ([300] if smoke else [1, 65, 1024, 32000, 190000]):
        for clean in (False, True):
            args, kw = ragged_case(37, nk, clean, strided=True)
            compare(new.fp8_mqa_logits(*args, **kw), old.fp8_mqa_logits(*args, **kw),
                    f'prefill nq=37 nk={nk} clean={clean}')
    if not smoke:
        for factor in (1e-3, 30., 100.):
            args = decode_case(6, 32000, 2, factor=factor)
            compare(new.fp8_paged_mqa_logits(*args), old.fp8_paged_mqa_logits(*args), f'decode scale={factor}')
            args, kw = ragged_case(37, 32000, True, factor=factor)
            compare(new.fp8_mqa_logits(*args, **kw), old.fp8_mqa_logits(*args, **kw), f'prefill scale={factor}')
        for h in (8, 16, 64):
            args = decode_case(2, 1025, 1, h=h)
            compare(new.fp8_paged_mqa_logits(*args), old.fp8_paged_mqa_logits(*args), f'heads decode h={h}')
            args, kw = ragged_case(9, 1025, True, h=h)
            compare(new.fp8_mqa_logits(*args, **kw), old.fp8_mqa_logits(*args, **kw), f'heads prefill h={h}')
        for nq, nk in [(0, 7), (3, 0)]:
            args, kw = ragged_case(nq, nk, True)
            assert torch.equal(new.fp8_mqa_logits(*args, **kw), old.fp8_mqa_logits(*args, **kw))
        args = list(decode_case(6, 1025, 2)); args[-1] = 31
        compare(new.fp8_paged_mqa_logits(*args), old.fp8_paged_mqa_logits(*args), 'output shorter than context')
        args[-1] = 0
        assert torch.equal(new.fp8_paged_mqa_logits(*args), old.fp8_paged_mqa_logits(*args))
        args = list(decode_case(2, 65)); args[0] = args[0].squeeze(1)
        compare(new.fp8_paged_mqa_logits(*args), old.fp8_paged_mqa_logits(*args), '3D query')


def graphs():
    for name in ('decode', 'decode_b', 'prefill', 'prefill_dirty'):
        if name.startswith('decode'):
            args, kw = decode_case(6, 32000, 2, 'B' if name == 'decode_b' else 'BN'), {}
            fn, ref = new.fp8_paged_mqa_logits, old.fp8_paged_mqa_logits
        else:
            args, kw = ragged_case(37, 32000, name != 'prefill_dirty')
            fn, ref = new.fp8_mqa_logits, old.fp8_mqa_logits
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(*args, **kw)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(*args, **kw)
        for step in range(3):
            if name.startswith('decode'):
                # Grow, shrink, empty; mutate the page table and BOTH speculative tokens.
                args[3].copy_(torch.arange(args[3].numel(), device='cuda').reshape_as(args[3]) * (1000 if step == 0 else 37))
                if step == 2:
                    args[3].fill_(0)
                args[4][:, 0] = -1
            else:
                args[3].fill_(step * 20)
                args[4].fill_(30000 if step == 0 else 50 if step == 1 else 0)
            args[0].view(torch.uint8).copy_(fp8(*args[0].shape).view(torch.uint8))
            g.replay()
            torch.cuda.synchronize()
            expected = fn(*args, **kw)
            assert torch.equal(out, expected), (name, step)
            compare(out, ref(*args, **kw), f'graph {name} step={step}')
        emit(kind='graph', name=name, replays=3, eager_bit_exact=True)


def measure(fn, args, kw, reps):
    for _ in range(2):
        z = fn(*args, **kw)
    del z
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); result = fn(*args, **kw); e.record(); e.synchronize()
        samples.append(s.elapsed_time(e))
        del result
    return samples


def graph_measure(fn, args, kw, inner=20, reps=7):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(*args, **kw)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            result = fn(*args, **kw)
    g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); e.synchronize()
        samples.append(s.elapsed_time(e) / inner)
    return samples


def benchmark():
    for L in (32000, 190000):
        args = decode_case(6, L, 1)
        a = measure(old.fp8_paged_mqa_logits, args, {}, 10)
        b = measure(new.fp8_paged_mqa_logits, args, {}, 20)
        emit(kind='bench', name='decode', B=6, N=1, L=L, old_ms=statistics.median(a),
             new_ms=statistics.median(b), speedup=statistics.median(a)/statistics.median(b),
             old_samples=a, new_samples=b)
        a = graph_measure(old.fp8_paged_mqa_logits, args, {})
        b = graph_measure(new.fp8_paged_mqa_logits, args, {})
        emit(kind='bench', name='decode_graph', B=6, N=1, L=L, inner=20,
             old_ms=statistics.median(a), new_ms=statistics.median(b),
             speedup=statistics.median(a)/statistics.median(b), old_samples=a, new_samples=b)
    for nk in (32000, 190000):
        for layout in ('causal', 'ragged'):
            # Real 8192-query shape. Keep full output semantics and compare every row.
            args, kw = ragged_case(8192, nk, True, layout)
            a = old.fp8_mqa_logits(*args, **kw)
            b = new.fp8_mqa_logits(*args, **kw)
            compare(b, a, f'large prefill nq=8192 nk={nk} layout={layout}')
            del a, b
            gc.collect(); torch.cuda.empty_cache()
            a = measure(old.fp8_mqa_logits, args, kw, 3)
            b = measure(new.fp8_mqa_logits, args, kw, 5)
            emit(kind='bench', name='prefill', nq=8192, nk=nk, clean=True, layout=layout,
                 old_ms=statistics.median(a), new_ms=statistics.median(b),
                 speedup=statistics.median(a)/statistics.median(b), old_samples=a, new_samples=b)
            del args
            gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['smoke', 'numeric', 'bench', 'graph', 'all'], default='smoke')
    p.add_argument('--seed', type=int, default=43)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    emit(kind='environment', torch=torch.__version__, triton=triton.__version__,
         device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
         bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
         source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
         oracle_sha256=hashlib.sha256(REF.read_bytes()).hexdigest(), seed=args.seed,
         test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), mode=args.mode)
    if args.mode in ('smoke', 'numeric', 'all'):
        numeric(args.mode == 'smoke')
    if args.mode in ('graph', 'all'):
        graphs()
    if args.mode in ('bench', 'all'):
        benchmark()
    emit(kind='complete', mode=args.mode, status='PASS')


if __name__ == '__main__':
    main()
