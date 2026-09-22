#!/usr/bin/env python3
"""T44 stage/group/orientation/loop refinement; predecoded buffers timed separately."""
import argparse
import itertools
from pathlib import Path
import torch
import triton
import test_sm80_indexer_112 as t
p=argparse.ArgumentParser();p.add_argument('--nk',type=int,default=190000);p.add_argument('--mode',choices=['layout','loop','group','focused'],default='layout');a=p.parse_args()
m=t.module('prototype_refine',Path('prefill.py'))
torch.manual_seed(44); args,kw=t.ragged_case(8192,a.nk,True,'causal'); pre=m.prepare(args)
out=torch.empty((8192,a.nk),device='cuda'); ref=t.new.fp8_mqa_logits(*args,**kw)
if a.mode=='layout': configs=[dict(bq=bq,bk=bk,warps=warps,qm=qm,group=8,loop=1,stages=1) for bq,bk,warps,qm in itertools.product([2,4,8],[64,128,256],[4,8],[False,True]) if bq*bk<=1024]
if a.mode=='group': configs=[dict(bq=2,bk=bk,warps=4,qm=qm,group=group,loop=1,stages=1) for bk,qm,group in itertools.product([128,256],[False,True],[0,1,2,4,8,16,32,64])]
if a.mode=='loop': configs=[dict(bq=bq,bk=bk,warps=4,qm=qm,group=8,loop=loop,stages=stages) for bq,bk,qm,loop,stages in itertools.product([2,4],[64,128],[False,True],[2,4,8],[1,2,3])]
if a.mode=='focused': configs=[dict(bq=2,bk=bk,warps=4,qm=True,group=group,loop=loop,stages=stages) for bk,group,loop,stages in itertools.product([128,256],[0,8,32],[1,2,4],[1,3])]
for cfg in configs:
 try:
    fn=lambda:m.run(pre,**cfg,out=out)
    fn();torch.cuda.synchronize()
    k=m.run.compiled
    if k.n_spills>0:
        t.emit(kind='skip_spills',**cfg,regs=k.n_regs,spills=k.n_spills,shared=k.metadata.shared);continue
    t.compare(out[:8],ref[:8],str(cfg))
    ms=triton.testing.do_bench(fn,warmup=50,rep=100,quantiles=[.5])
    t.emit(kind='tune',**cfg,ms=ms,nk=a.nk,regs=k.n_regs,spills=k.n_spills,shared=k.metadata.shared,tflops=2*8192*a.nk*32*128/(ms*1e9))
 except Exception as e:t.emit(kind='error',**cfg,error=str(e))
t.emit(kind='complete',status='PASS')
