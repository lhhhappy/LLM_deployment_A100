#!/usr/bin/env python3
"""T44 offline sweep: explicit knobs, no production autotuning; logs JSONL."""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys
import torch
import triton
import test_sm80_indexer_112 as t
p=argparse.ArgumentParser();p.add_argument('--nk',type=int,default=32000);p.add_argument('--nq',type=int,default=8192);p.add_argument('--wide',action='store_true');p.add_argument('--loop',action='store_true');p.add_argument('--path',default='prefill.py');a=p.parse_args()
m=t.module('prototype113',Path(a.path))
torch.manual_seed(44)
args,kw=t.ragged_case(a.nq,a.nk,True,'causal')
pre=m.prepare(args)
out=torch.empty((a.nq,a.nk),device='cuda')
ref=t.new.fp8_mqa_logits(*args,**kw)
t.emit(kind='baseline',nq=a.nq,nk=a.nk,ms=statistics.median(t.measure(t.new.fp8_mqa_logits,args,kw,3)))
configs=itertools.product([2,4,8],[64,128,256],[4,8],[False,True],[8])
if a.wide: configs=itertools.product([2,4,8,16],[64,128,256],[4,8],[True],[0,1,8,32])
if a.loop: configs=itertools.product([2,4,8],[64,128],[4,8],[True],[8])
for bq,bk,warps,prep,group in configs:
  for loop in ([2,4,8] if a.loop else [1]):
    for stages in ([1,2,3] if a.loop else [2]):
      cfg=dict(bq=bq,bk=bk,warps=warps,pre=prep,group=group,loop=loop,stages=stages)
      try:
        inp=pre if prep else args
        fn=lambda: m.run(inp,**cfg,out=out)
        fn();torch.cuda.synchronize()
        if not a.wide and not a.loop: t.compare(out[:16],ref[:16],str(cfg))
        ms=triton.testing.do_bench(fn,warmup=100,rep=200,quantiles=[0.5])
        comp=list(m.kernel.cache[0].values()) if False else None
        t.emit(kind='tune',**cfg,ms=ms,regs=m.run.compiled.n_regs,spills=m.run.compiled.n_spills,shared=m.run.compiled.metadata.shared,tflops=2*a.nq*a.nk*32*128/(ms*1e9))
      except Exception as e: t.emit(kind='error',**cfg,error=str(e))
t.emit(kind='complete',status='PASS')
