#!/usr/bin/env python3
"""Offline T43 tile sweep; no production autotuning/capture-time benchmark."""
import itertools
from pathlib import Path
import sys
import statistics
import torch
import triton
import test_sm80_indexer_112 as t


def run(args, kw, bq, bk, warps):
    q, (k, sc), w, ks, ke = args
    nq, h, d = q.shape
    nk = k.shape[0]
    out = torch.empty(nq, nk, device='cuda')
    t.new._ragged[(triton.cdiv(nq,bq), triton.cdiv(nk,bk))](
        q.view(torch.uint8), k.view(torch.uint8), sc, w, ks, ke, out,
        nq,nk,h,d,*q.stride(),*k.stride(),sc.stride(0),*w.stride(),ks.stride(0),ke.stride(0),
        kw['clean_logits'],bq,bk,h,d,num_warps=warps,enable_fp_fusion=False)
    return out


if len(sys.argv) > 1:
    t.new = t.module('tuned_source', Path(sys.argv[1]))

torch.manual_seed(112)
args,kw=t.ragged_case(1024,32000,True,'causal')
ref=t.new.fp8_mqa_logits(*args,**kw)
for bq,bk,warps in itertools.product([1,2,4,8], [32,64,128], [4,8]):
    try:
        out=run(args,kw,bq,bk,warps)
        assert torch.allclose(out,ref,rtol=1e-3,atol=1e-4)
        def f(*a,**k): return run(a,k,bq,bk,warps)
        times=t.measure(f,args,kw,5)
        t.emit(kind='tune',bq=bq,bk=bk,warps=warps,ms=statistics.median(times),samples=times)
    except Exception as e:
        t.emit(kind='tune_error',bq=bq,bk=bk,warps=warps,error=str(e))
t.emit(kind='complete',mode='tune')
