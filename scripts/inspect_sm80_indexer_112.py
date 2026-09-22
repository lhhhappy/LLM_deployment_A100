#!/usr/bin/env python3
"""Emit actual sm80 compiler receipts/PTX for final T43 kernels; run on GPU dev box."""
import hashlib
import json
from pathlib import Path
import sys
import torch
import triton
import test_sm80_indexer_112 as t

outdir=Path(sys.argv[1]);outdir.mkdir(exist_ok=True,parents=True)
q,cache,w,ctx,bt,meta,S=t.decode_case(6,32000,1)
B,N,H,D=q.shape;PAGE=64
ctx=ctx.reshape(B,-1);flat=cache.reshape(cache.shape[0],-1)
out=torch.empty((B*N,S),device=q.device)
paged=t.new._paged[(B*N,triton.cdiv(S,PAGE))](
    q.view(torch.uint8),flat,w,ctx,bt,out,N,H,D,bt.shape[1],S,PAGE,
    *q.stride(),*w.stride(),ctx.stride(0),ctx.stride(1) if ctx.shape[1]==N else 0,
    *bt.stride(),flat.stride(0),32,128,num_warps=4,enable_fp_fusion=False)
(q,(k,sc),w,ks,ke),kw=t.ragged_case(37,32000,True)
nq,H,D=q.shape;nk=k.shape[0];bq,bk=2,64
out=torch.empty(nq,nk,device='cuda')
ragged=t.new._ragged[(triton.cdiv(nq,bq),triton.cdiv(nk,bk))](
    q.view(torch.uint8),k.view(torch.uint8),sc,w,ks,ke,out,nq,nk,H,D,
    *q.stride(),*k.stride(),sc.stride(0),*w.stride(),ks.stride(0),ke.stride(0),
    True,bq,bk,H,D,num_warps=4,enable_fp_fusion=False)
receipt={}
for name,kernel in [('paged',paged),('ragged',ragged)]:
    ptx=kernel.asm['ptx']
    assert '.target sm_80' in ptx and '.bf16.bf16' in ptx
    assert '.e4m3' not in ptx
    (outdir/(name+'.ptx')).write_text(ptx)
    receipt[name]=dict(registers=kernel.n_regs,spills=kernel.n_spills,
                       shared=kernel.metadata.shared,ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                       target_sm80=True,bf16_mma=True,fp8_instruction=False)
receipt['source_sha256']=hashlib.sha256(t.SRC.read_bytes()).hexdigest()
(outdir/'compiler_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
