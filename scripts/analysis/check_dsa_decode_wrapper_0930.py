"""Validate the reviewable default-off wrapper patch, including fallback shapes."""
import importlib.util
import json
import os
from pathlib import Path
import torch
import bench_dsa_decode_0930 as p
from qualify_dsa_decode_0930 import make,selection

spec=importlib.util.spec_from_file_location('candidate_dsa',Path(__file__).parent/'sm80_indexer_kernels_candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def candidate(inp,clean=False,indices=None):
    q,cache,w,ctx,bt,S=inp
    return c.fp8_paged_mqa_logits(q,cache.view(-1,64,1,132),w,ctx,bt,None,S,clean_logits=clean,indices=indices)

rows=[]
for B in [8,24,32,40,48]:
    for L in [16000,64000,250000]:
        for mixed in [False,True]:
            inp=make(B,L,mixed)
            expected=p.baseline(inp)
            for enabled in ['0','1']:
                os.environ['SGLANG_AX_SM80_INDEXER_DECODE_LOOP']=enabled
                got=candidate(inp)
                assert torch.equal(got,expected),(B,L,mixed,enabled)
            a,b=selection(expected,inp),selection(got,inp)
            assert torch.equal(a.sort(-1).values,b.sort(-1).values)
            self_exact=0;candidate_exact=0
            for _ in range(5):
                repeat=selection(expected,inp); other=selection(got,inp)
                self_exact+=int(torch.equal(a,repeat));candidate_exact+=int(torch.equal(a,other))
                assert torch.equal(a.sort(-1).values,repeat.sort(-1).values)
                assert torch.equal(a.sort(-1).values,other.sort(-1).values)
            rows.append(dict(B=B,L=L,mixed=mixed,default_off_bitwise=True,enabled_bitwise=True,topk_sets_equal=True,topk_raw_equal=torch.equal(a,b),baseline_self_raw_equal_5=self_exact,candidate_raw_equal_5=candidate_exact))
for H in [1,4,16,32,64]:
    inp=list(make(8,64000,True,gapped=True,N=3))
    inp[0]=torch.randn(8,3,H,128,device='cuda').to(torch.float8_e4m3fn)
    inp[2]=torch.randn(24,H,device='cuda')
    # Shared per-batch context is broadcast over N.
    inp[3]=inp[3][:,:1]
    for clean in [False,True]:
        got=candidate(inp,clean=clean);expected=p.baseline(inp)
        assert torch.equal(got,expected),(H,clean)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=candidate(inp)
    for val in [65,0,129,3999]:
        inp[3][:]=val;graph.replay();expected=p.baseline(inp)
        torch.cuda.synchronize();assert torch.equal(out,expected),(H,val)
    rows.append(dict(H=H,N=3,broadcast_context=True,clean_ignored=True,graph_bitwise=True))
inp=make(8,64000,False)
try:candidate(inp,indices=torch.empty(0,device='cuda'))
except AssertionError: rows.append(dict(indices_rejected=True))
else:raise AssertionError('indices contract lost')
for B,S in [(8,0)]:
    inp=list(make(max(1,B),16000,False));inp[-1]=S
    if B==0:
        for i in [0,2,3,4]:inp[i]=inp[i][:0]
    assert candidate(inp).shape==(B,S)
    rows.append(dict(B=B,S=S,empty_contract=True))
Path(__file__).with_name('wrapper-results.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(cases=len(rows),all_pass=True)),flush=True)
