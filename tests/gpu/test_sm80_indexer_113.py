#!/usr/bin/env python3
"""T44: reuse every T43 case against both immutable 110 and 112, add six large shapes.

All activation data are synthetic. Timings include unpack/allocation and fp32 output.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import statistics
import torch
import triton
import test_sm80_indexer_112 as t

HERE=Path(__file__).resolve().parent
SRC=HERE/'kernels/sm80_indexer_113.py'
if not SRC.exists(): SRC=HERE/'sm80_indexer_113.py'
new=t.module('kernels113',SRC)
old112=t.new
old110=t.old


def bench():
    for nk in (32000,95000,190000):
        for layout in ('causal','ragged'):
            args,kw=t.ragged_case(8192,nk,True,layout)
            out=new.fp8_mqa_logits(*args,**kw)
            for name,ref in [('110',old110),('112',old112)]:
                expected=ref.fp8_mqa_logits(*args,**kw)
                t.compare(out,expected,f'large {layout} nq=8192 nk={nk} versus {name}')
                del expected
            del out
            gc.collect();torch.cuda.empty_cache()
            # Alternate order across repetitions to expose heating/clock drift.
            samples={name:[] for name in ('112','113')}
            for repeat in range(7):
                for name,mod in ([('112',old112),('113',new)] if repeat%2==0 else [('113',new),('112',old112)]):
                    samples[name]+=t.measure(mod.fp8_mqa_logits,args,kw,1)
            oldms=statistics.median(samples['112']);newms=statistics.median(samples['113'])
            valid=(args[4].clamp(0,nk)-args[3].clamp(0,nk)).clamp_min(0).sum().item()
            flops=2*8192*nk*32*128
            t.emit(kind='bench',name='prefill',nq=8192,nk=nk,layout=layout,clean=True,
                   old_ms=oldms,new_ms=newms,speedup=oldms/newms,
                   old_effective_tflops=flops/(oldms*1e9),new_effective_tflops=flops/(newms*1e9),
                   valid_pair_fraction=valid/(8192*nk),valid_pair_tflops=2*valid*32*128/(newms*1e9),
                   samples=samples)
            del args
            gc.collect();torch.cuda.empty_cache()
    for nk in (32000,190000):
        args=t.decode_case(6,nk)
        a=old112.fp8_paged_mqa_logits(*args);b=new.fp8_paged_mqa_logits(*args)
        assert torch.equal(a,b)
        for label,measure in [('eager',lambda f:t.measure(f,args,{},20)),('graph',lambda f:t.graph_measure(f,args,{}))]:
            vals={}
            for name,mod in [('112',old112),('113',new)]:
                vals[name]=measure(mod.fp8_paged_mqa_logits)
            t.emit(kind='bench',name='decode_'+label,nk=nk,old_ms=statistics.median(vals['112']),
                   new_ms=statistics.median(vals['113']),samples=vals,bit_exact=True)


def extra():
    for nq,nk,h in [(1,17,32),(3,129,32),(7,513,32),(31,1023,32),(127,4097,32),
                    (129,95000,32),(65,1025,7),(19,1025,33),(4097,65,32)]:
        for clean in (False,True):
            args,kw=t.ragged_case(nq,nk,clean,strided=True,h=h)
            out=new.fp8_mqa_logits(*args,**kw)
            for name,ref in [('110',old110),('112',old112)]:
                t.compare(out,ref.fp8_mqa_logits(*args,**kw),f'extra nq={nq} nk={nk} h={h} clean={clean} vs {name}')
    # Graph tests in T43 use nq37. Also exercise the large-shape fast path with a tail.
    for clean in (False,True):
        args,kw=t.ragged_case(129,4097,clean)
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3): new.fp8_mqa_logits(*args,**kw)
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out=new.fp8_mqa_logits(*args,**kw)
        for step in range(3):
            args[0].view(torch.uint8).copy_(t.fp8(*args[0].shape).view(torch.uint8))
            args[1][0].view(torch.uint8).copy_(t.fp8(*args[1][0].shape).view(torch.uint8))
            args[1][1].mul_(0.9)
            args[3].fill_(20 if step==1 else 0)
            args[4].fill_(4097 if step==0 else 21 if step==1 else 0)
            g.replay();torch.cuda.synchronize()
            assert torch.equal(out,new.fp8_mqa_logits(*args,**kw))
            t.compare(out,old110.fp8_mqa_logits(*args,**kw),f'extra graph clean={clean} step={step}')
        t.emit(kind='graph',name=f'extra clean={clean}',replays=3,eager_bit_exact=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','numeric','graph','bench','all'],default='all')
    a=p.parse_args();torch.manual_seed(44);torch.backends.cuda.matmul.allow_tf32=False
    t.emit(kind='environment',torch=torch.__version__,triton=triton.__version__,device=torch.cuda.get_device_name(),
           capability=torch.cuda.get_device_capability(),seed=44,mode=a.mode,
           source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
           baseline_sha256=hashlib.sha256(t.SRC.read_bytes()).hexdigest(),
           oracle_sha256=hashlib.sha256(t.REF.read_bytes()).hexdigest(),
           test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           reused_test_sha256=hashlib.sha256(Path(t.__file__).read_bytes()).hexdigest(),
           bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    t.new=new
    for name,ref in [('110',old110),('112',old112)]:
        t.old=ref;t.emit(kind='reference',name=name)
        if a.mode in ('smoke','numeric','all'): t.numeric(a.mode=='smoke')
        if a.mode in ('graph','all'): t.graphs()
    if a.mode in ('numeric','all'): extra()
    if a.mode in ('bench','all'): bench()
    t.emit(kind='complete',status='PASS',mode=a.mode)

if __name__=='__main__': main()
