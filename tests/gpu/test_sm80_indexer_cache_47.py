#!/usr/bin/env python3
"""T47 actual JIT/compilation counters over unseen random shapes, no service/model.

Warm only finite scalar specialization classes (1 / divisible by 16 / other),
CLEAN, layout, shared/per-token context, and 113 fallback/fast paths. Then 50
unique random NQ/NK/P/batch values, with full real output grids, for BOTH versions.
"""
import hashlib
import itertools
from pathlib import Path
import random
import torch
import triton
import triton.knobs as knobs
import test_sm80_indexer_112 as t

HERE=Path(__file__).resolve().parent
p=HERE/'kernels/sm80_indexer_113.py'
if not p.exists():p=HERE/'sm80_indexer_113.py'
mods={112:t.new,113:t.module('cache113',p)}
counts=dict(jit_misses=0,compiled=0,disk_hits=0)
phase='setup'

def hook(*a,**kw):
    counts['jit_misses']+=1
    t.emit(kind='jit_miss',phase=phase,function=str(kw.get('fn')),key=str(kw.get('compile')))

def listener(**kw):
    counts['disk_hits' if kw['cache_hit'] else 'compiled']+=1


def measured(label,fn,strict=False):
    before=counts.copy();out=fn();torch.cuda.synchronize();del out
    delta={k:counts[k]-before[k] for k in counts}
    t.emit(kind='cache',phase=phase,label=label,**delta)
    if strict:assert not any(delta.values()),(label,delta)


def prefill(nq,nk,clean,strided):
    args,kw=t.ragged_case(nq,nk,clean,strided=strided)
    for n,m in mods.items():
        measured(f'{n} nq={nq} nk={nk} clean={clean} strided={strided}',
                 lambda:m.fp8_mqa_logits(*args,**kw),phase=='random')


def decode(b,n,p,s,shared,strided):
    args=list(t.decode_case(b,p*64,n,'B' if shared else 'BN',strided=strided))
    args[-1]=s
    for version,m in mods.items():
        measured(f'{version} B={b} N={n} P={p} S={s} shared={shared} strided={strided}',
                 lambda:m.fp8_paged_mqa_logits(*args),phase=='random')


def main():
    global phase
    torch.manual_seed(47);rng=random.Random(47)
    knobs.runtime.jit_cache_hook=hook;knobs.compilation.listener=listener
    t.emit(kind='environment',torch=torch.__version__,triton=triton.__version__,seed=47,
           source112=hashlib.sha256(t.SRC.read_bytes()).hexdigest(),source113=hashlib.sha256(p.read_bytes()).hexdigest())
    phase='warmup'
    for nq,nk,clean,strided in itertools.product((1,16,17,32,33),(1,16,17,1024,1025),(False,True),(False,True)):
        prefill(nq,nk,clean,strided)
    for b,n,pages,shared,strided,aligned in itertools.product((1,16,3),(1,2,16),(1,16,17),(False,True),(False,True),(False,True)):
        decode(b,n,pages,pages*64 if aligned else pages*64-1,shared,strided)
    decode(1,1,1,1,False,False)
    warm=counts.copy();phase='random'
    # Sampling without replacement makes each axis contain 50 distinct values.
    nqs=rng.sample(range(1,16385),50);nks=rng.sample(range(1,200001),50)
    pages=rng.sample(range(1,3126),50);batches=rng.sample(range(1,65),50)
    for i,(nq,nk,pg,b) in enumerate(zip(nqs,nks,pages,batches)):
        t.emit(kind='shape',index=i,NQ=nq,NK=nk,P=pg,batch=b)
        prefill(nq,nk,bool(i%2),bool(i%3==0))
        decode(b,(1,2,16)[i%3],pg,nk,bool(i%2),bool(i%3==0))
    # Explicitly revisit the T46 counterexample after finite-class warmup.
    for nq,clean in itertools.product((600,601),(False,True)):
        prefill(nq,2048,clean,False)
    delta={k:counts[k]-warm[k] for k in counts}
    assert not any(delta.values()),delta
    t.emit(kind='complete',status='PASS',random_shapes=50,random_calls=200,t46_regression_calls=8,warmup=warm,random=delta)

if __name__=='__main__':main()
