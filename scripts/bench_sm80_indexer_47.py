#!/usr/bin/env python3
"""T47 paired v1/v2 benchmark; same input, alternating order, original timing helpers."""
import gc
import hashlib
from pathlib import Path
import statistics
import torch
import test_sm80_indexer_112 as t

HERE=Path(__file__).resolve().parent

def source(n,old=False):
    if old:
        p=HERE/f'sm80_indexer_{n}_v1.py'
        if not p.exists(): p=HERE.parent/f'evidence/T47/sm80_indexer_{n}_v1.py'
    else:
        p=HERE/f'kernels/sm80_indexer_{n}.py'
        if not p.exists(): p=HERE/f'sm80_indexer_{n}.py'
    return p

def main():
    torch.manual_seed(47)
    mods={(n,v):t.module(f'bench_{n}_{v}',source(n,v==1)) for n in (112,113) for v in (1,2)}
    t.emit(kind='environment',torch=torch.__version__,device=torch.cuda.get_device_name(),
           sources={f'{n}v{v}':hashlib.sha256(source(n,v==1).read_bytes()).hexdigest() for n,v in mods})
    for nk in (32000,95000,190000):
        for layout in ('causal','ragged'):
            args,kw=t.ragged_case(8192,nk,True,layout)
            for n in (112,113):
                fns={v:mods[n,v].fp8_mqa_logits for v in (1,2)}
                a=fns[1](*args,**kw);b=fns[2](*args,**kw)
                t.compare(b,a,f'{n} v2/v1 nq8192 nk{nk} {layout}')
                del a,b
                samples={1:[],2:[]}
                for repeat in range(7):
                    for v in ((1,2) if repeat%2==0 else (2,1)):
                        samples[v]+=t.measure(fns[v],args,kw,1)
                a,b=[statistics.median(samples[v]) for v in (1,2)]
                t.emit(kind='bench',kernel=n,mode='prefill',nk=nk,layout=layout,v1_ms=a,v2_ms=b,
                       change_percent=(b/a-1)*100,samples=samples)
            del args
            gc.collect();torch.cuda.empty_cache()
    for nk in (32000,190000):
        args=t.decode_case(6,nk)
        for n in (112,113):
            samples={1:[],2:[]}
            for repeat in range(3):
                for v in ((1,2) if repeat%2==0 else (2,1)):
                    samples[v]+=t.graph_measure(mods[n,v].fp8_paged_mqa_logits,args,{})
            a,b=[statistics.median(samples[v]) for v in (1,2)]
            t.emit(kind='bench',kernel=n,mode='decode_graph',nk=nk,v1_ms=a,v2_ms=b,
                   change_percent=(b/a-1)*100,samples=samples)
    t.emit(kind='complete',status='PASS')

if __name__=='__main__':main()
