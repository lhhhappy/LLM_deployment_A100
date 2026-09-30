"""Baseline-only physical PAGED topk tie audit; scorer output never changes."""
import json
from pathlib import Path
import torch
from qualify_dsa_decode_0930 import make
import bench_dsa_decode_0930 as p
from sglang.kernels.ops.moe.kpool_topk_transform import fast_kpool_topk_transform_fused as select

B,L,S=24,250000,262144
inp=make(B,L,True,S=S)
lens=torch.tensor([4000,16000,62500,0,1,63,64,65],device='cuda',dtype=torch.int32)
inp[3][:,0]=lens[torch.arange(B,device='cuda')%8]
ref=p.baseline(inp);lengths=inp[3].reshape(-1);seq=lengths*4+3
table=torch.arange(L+64,device='cuda',dtype=torch.int32).expand(B,-1).contiguous()
table=table*13+torch.arange(B,device='cuda',dtype=torch.int32)[:,None]*1000000
table[:,::113]=-1
def run():return select(ref,lengths,4,2048,page_table=table,seq_lens=seq)
a=run();rows=[]
for rep in range(50):
    b=run()
    if torch.equal(a.sort(-1).values,b.sort(-1).values):continue
    for row in range(B):
        if torch.equal(a[row].sort().values,b[row].sort().values):continue
        aa=a[row][a[row]>=0];bb=b[row][b[row]>=0]
        left=aa[~torch.isin(aa,bb)];right=bb[~torch.isin(bb,aa)]
        lo=((left-row*1000000)//13)//4;ro=((right-row*1000000)//13)//4
        lv=ref[row,lo];rv=ref[row,ro]
        length=int(lengths[row]);vals=ref[row,:length];cutoff=torch.topk(vals,min(512,length)).values[-1]
        r=dict(repeat=rep,row=row,length=length,cutoff=float(cutoff),scores_above_cutoff=int((vals>cutoff).sum()),scores_equal_cutoff=int((vals==cutoff).sum()),baseline_only=left.numel(),repeat_only=right.numel(),baseline_scores=lv.unique().tolist(),repeat_scores=rv.unique().tolist(),baseline_physical_only=left[:32].tolist(),repeat_physical_only=right[:32].tolist(),baseline_negative_slots=int((a[row]<0).sum()),repeat_negative_slots=int((b[row]<0).sum()))
        rows.append(r);print(json.dumps(r),flush=True)
    if len(rows)>=10:break
Path(__file__).with_name('topk-tie-diagnostic.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(baseline_only=True,findings=len(rows),input_unchanged=True)),flush=True)
