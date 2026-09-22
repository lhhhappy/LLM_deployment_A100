#!/usr/bin/env python3
"""Compare recorded T43 prototypes (all source files retained under evidence/T43/prototypes)."""
from pathlib import Path
import sys
import statistics
import torch
import triton
import test_sm80_indexer_112 as t

torch.manual_seed(112)
root=Path(sys.argv[1])
mods={n:t.module(n,root/(n+'.py')) for n in (('bits','qmajor') if '--wide' in sys.argv else ('initial','bits','qmajor'))}
args,kw=t.ragged_case(1024,32000,True,'causal')
ref=t.old.fp8_mqa_logits(*args,**kw)
for name,m in mods.items():
    out=m.fp8_mqa_logits(*args,**kw)
    t.compare(out,ref,name)
    for bq,bk,warps in (((4,64,8),(4,64,16),(4,128,16),(8,64,16),(8,128,16)) if '--wide' in sys.argv else ((1,64,4),(2,64,4),(2,128,8),(4,64,8),(4,128,8),(8,32,8))):
        def run(q,kv,w,ks,ke,clean_logits):
            k,sc=kv;nq,h,d=q.shape;nk=k.shape[0]
            out=torch.empty(nq,nk,device='cuda')
            ker=m._ragged[(triton.cdiv(nq,bq),triton.cdiv(nk,bk))](
                q.view(torch.uint8),k.view(torch.uint8),sc,w,ks,ke,out,
                nq,nk,h,d,*q.stride(),*k.stride(),sc.stride(0),*w.stride(),ks.stride(0),ke.stride(0),
                clean_logits,bq,bk,h,d,num_warps=warps,enable_fp_fusion=False)
            run.regs=ker.n_regs;run.spills=ker.n_spills
            return out
        out=run(*args,**kw)
        assert torch.allclose(out,ref,atol=1e-4,rtol=1e-3)
        times=t.measure(run,args,kw,5)
        t.emit(kind='layout',name=name,bq=bq,bk=bk,warps=warps,regs=run.regs,spills=run.spills,
               ms=statistics.median(times),samples=times)
    for L in (32000,190000):
        da=t.decode_case(6,L,1)
        times=t.measure(m.fp8_paged_mqa_logits,da,{},20)
        t.emit(kind='decode_layout',name=name,L=L,ms=statistics.median(times),samples=times)
t.emit(kind='complete',mode='layouts')
