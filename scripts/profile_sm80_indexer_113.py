#!/usr/bin/env python3
"""T44 CUDA profiler and compiler receipt for 112/113; run only on GPU dev box."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
import triton
import test_sm80_indexer_113 as t
p=argparse.ArgumentParser();p.add_argument('outdir');p.add_argument('--nk',type=int,default=190000);a=p.parse_args()
outdir=Path(a.outdir);outdir.mkdir(parents=True,exist_ok=True)
torch.manual_seed(44)
args,kw=t.t.ragged_case(8192,a.nk,True,'causal')
for name,mod in [('112',t.old112),('113',t.new)]:
    for _ in range(3):out=mod.fp8_mqa_logits(*args,**kw)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=True) as prof:
        for _ in range(3): out=mod.fp8_mqa_logits(*args,**kw)
        torch.cuda.synchronize()
    (outdir/f'profile_{name}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    rows=[]
    for evt in prof.key_averages():
        rows.append(dict(name=evt.key,count=evt.count,self_device_time_total_us=evt.self_device_time_total,
                         device_time_total_us=evt.device_time_total,cpu_time_total_us=evt.cpu_time_total,
                         self_device_memory_usage=evt.self_device_memory_usage))
    (outdir/f'profile_{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Chrome trace remains on dev machine; compact JSON/table copied to evidence.
    prof.export_chrome_trace(str(outdir/f'profile_{name}.trace.json'))
receipt=dict(source_sha256=hashlib.sha256(t.SRC.read_bytes()).hexdigest(),nk=a.nk,nq=8192)
(outdir/'profile_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
