"""Actual patched wrapper plus native prebuilt PAGED KPool transformation."""
import importlib.util
import json
import os
from pathlib import Path
import torch
import bench_dsa_decode_0930 as p
from qualify_dsa_decode_0930 import make
from sglang.kernels.ops.moe.kpool_topk_transform import fast_kpool_topk_transform_fused as select

spec=importlib.util.spec_from_file_location('candidate_dsa',Path(__file__).parent/'sm80_indexer_kernels_candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
os.environ['SGLANG_AX_SM80_INDEXER_DECODE_LOOP']='1'
rows=[]
for B in [8,24,32,40,48]:
    for L,S in [(16000,None),(64000,None),(250000,None),(250000,262144)]:
        for mixed in [False,True] if S is None else [True]:
            inp=make(B,L,mixed,S=S)
            if S is not None:
                lens=torch.tensor([4000,16000,62500,0,1,63,64,65],device='cuda',dtype=torch.int32)
                inp[3][:,0]=lens[torch.arange(B,device='cuda')%8]
            q,cache,w,ctx,bt,width=inp
            lengths=ctx.reshape(-1);seq=lengths*4+3
            table=torch.arange(L+64,device='cuda',dtype=torch.int32).expand(B,-1).contiguous()
            # Deliberately nonidentity physical token mapping, including holes.
            table=(table*13+torch.arange(B,device='cuda',dtype=torch.int32)[:,None]*1000000)
            table[:,::113]=-1
            kv=cache.view(-1,64,1,132)
            def scorer():return c.fp8_paged_mqa_logits(q,kv,w,ctx,bt,None,width)
            def top(logits):return select(logits,lengths,4,2048,page_table=table,seq_lens=seq)
            ref=p.baseline(inp);cand=scorer();assert torch.equal(ref,cand)
            a,b=top(ref),top(cand)
            set_equal=torch.equal(a.sort(-1).values,b.sort(-1).values)
            self_sets=0;self_raw=0
            for _ in range(5):
                repeat=top(ref)
                self_raw+=int(torch.equal(a,repeat))
                self_sets+=int(torch.equal(a.sort(-1).values,repeat.sort(-1).values))
            score_differences=[]
            if not set_equal:
                for row in range(B):
                    aa=a[row][a[row]>=0];bb=b[row][b[row]>=0]
                    left=aa[~torch.isin(aa,bb)];right=bb[~torch.isin(bb,aa)]
                    if left.numel() or right.numel():
                        lo=((left-row*1000000)//13)//4;ro=((right-row*1000000)//13)//4
                        lv=ref[row,lo];rv=ref[row,ro]
                        score_differences.append(dict(row=row,baseline_only=left.numel(),candidate_only=right.numel(),baseline_scores=lv.unique().tolist(),candidate_scores=rv.unique().tolist()))
            bu,bs=p.timed(lambda:top(p.baseline(inp)),its=32)
            cu,cs=p.timed(lambda:top(scorer()),its=32)
            r=dict(B=B,raw_context=L,mixed=mixed,S=width,baseline_us=bu,candidate_us=cu,speedup=bu/cu,logits_bitwise=True,mapped_topk_set_equal=set_equal,baseline_self_sets_equal_5=self_sets,baseline_self_raw_equal_5=self_raw,tie_score_differences=score_differences,prebuilt_metadata=True,baseline_samples=bs,candidate_samples=cs)
            rows.append(r);print(json.dumps(r),flush=True)
            Path(__file__).with_name('paged-stage-results.json').write_text(json.dumps(rows,indent=2))
print('COMPLETE',flush=True)
