"""Scratch A100 DSA decode engineering probe; no production changes."""
import argparse
import importlib.util
import json
import os
import statistics
from pathlib import Path
import torch
import triton
import triton.language as tl

SOURCE = Path(os.environ.get('DSA_SOURCE', str(Path(__file__).resolve().parents[2] / 'engine/sglang/srt/layers/attention/dsa/sm80_indexer_kernels.py')))
spec = importlib.util.spec_from_file_location('baseline_dsa', SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
decode = base._e4m3_to_bf16

@triton.jit
def uniform(Q,K,W,C,BT,O,N,H:tl.constexpr,D:tl.constexpr,P,S,
            QB,QN,QH,QD,WB,WH,CB,CN,TB,TP,KB,
            HH:tl.constexpr,LOOP:tl.constexpr):
    row=tl.program_id(0); first=tl.program_id(1)*LOOP
    b=row//N; n=row%N
    ctx=tl.load(C+b*CB+n*CN).to(tl.int32)
    tok=tl.arange(0,64); ds=tl.arange(0,128); hs=tl.arange(0,HH)
    if (first<P)&(first*64<ctx):
        q=decode(tl.load(Q+b*QB+n*QN+ds[:,None]*QD+hs[None,:]*QH,hs[None,:]<H,other=0))
        w=tl.load(W+row*WB+hs*WH,hs<H,other=0).to(tl.float32)
        for j in range(LOOP):
            page=first+j; pos=page*64+tok
            result=tl.full((64,),0,tl.float32)
            if (page<P)&(page*64<ctx):
                physical=tl.maximum(tl.load(BT+b*TB+page*TP),0).to(tl.int64)
                kval=decode(tl.load(K+physical*KB+tok[:,None]*D+ds[None,:]))
                dots=tl.dot(kval,q).to(tl.bfloat16).to(tl.float32)
                sp=(K+physical*KB+64*D).to(tl.pointer_type(tl.float32))
                sc=tl.load(sp+tok)
                result=tl.sum(tl.maximum(dots,0.)*w[None,:],1)*sc
                result=tl.where(pos<ctx,result,0.)
            tl.store(O+row.to(tl.int64)*S+pos,result,pos<S)
    else:
        for j in range(LOOP):
            pos=(first+j)*64+tok
            tl.store(O+row.to(tl.int64)*S+pos,0.,pos<S)

@triton.jit
def grouped(Q,K,W,C,BT,O,N,H:tl.constexpr,D:tl.constexpr,P,S,
            QB,QN,QH,QD,WB,WH,CB,CN,TB,TP,KB,
            HH:tl.constexpr,BK:tl.constexpr,LOOP:tl.constexpr):
    row=tl.program_id(0)
    first=tl.program_id(1)*BK*LOOP
    b=row//N; n=row%N
    ctx=tl.load(C+b*CB+n*CN).to(tl.int32)
    ds=tl.arange(0,128); hs=tl.arange(0,HH)
    tok=tl.arange(0,BK)
    if first<ctx:
        q=decode(tl.load(Q+b*QB+n*QN+ds[:,None]*QD+hs[None,:]*QH,
                         hs[None,:]<H,other=0))
        w=tl.load(W+row*WB+hs*WH,hs<H,other=0).to(tl.float32)
        for j in range(LOOP):
            pos=first+j*BK+tok
            page=pos//64
            result=tl.full((BK,),0,tl.float32)
            if first+j*BK<ctx:
                physical=tl.maximum(tl.load(BT+b*TB+page*TP,page<P,other=0),0).to(tl.int64)
                kval=decode(tl.load(K+physical[:,None]*KB+(pos%64)[:,None]*D+ds[None,:],
                                   page[:,None]<P,other=0))
                dots=tl.dot(kval,q).to(tl.bfloat16).to(tl.float32)
                sp=(K+physical*KB+64*D).to(tl.pointer_type(tl.float32))
                sc=tl.load(sp+pos%64,page<P,other=0)
                result=tl.sum(tl.maximum(dots,0.)*w[None,:],1)*sc
                result=tl.where((pos<ctx)&(page<P),result,0.)
            tl.store(O+row.to(tl.int64)*S+pos,result,pos<S)
    else:
        for j in range(LOOP):
            pos=first+j*BK+tok
            tl.store(O+row.to(tl.int64)*S+pos,0.,pos<S)

def invoke(args, out, bk, loop, warps):
    q,cache,w,ctx,bt,S=args
    B,N,H,D=q.shape
    kernel=uniform if bk==0 else grouped
    constexpr=(max(16,triton.next_power_of_2(H)),loop) if bk==0 else (max(16,triton.next_power_of_2(H)),bk,loop)
    kernel[(B*N,triton.cdiv(S,max(64,bk)*loop))](q.view(torch.uint8),cache,w,ctx,bt,out,N,H,D,bt.shape[1],S,
        *q.stride(),*w.stride(),*ctx.stride(),*bt.stride(),cache.stride(0),
        *constexpr,num_warps=warps,num_stages=1,enable_fp_fusion=False)
    return out

def baseline(args):
    q,cache,w,ctx,bt,S=args
    kv=cache.reshape(-1,64,1,132)
    return base.fp8_paged_mqa_logits(q,kv,w,ctx,bt,None,S)

def case(B, raw_ctx, mixed):
    torch.manual_seed(B+raw_ctx)
    S=triton.cdiv(raw_ctx//4,64)*64
    P=S//64
    q=torch.randn(B,1,32,128,device='cuda').to(torch.float8_e4m3fn)
    k=torch.randn(B*P,64*128,device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
    scales=torch.rand(B*P,64,device='cuda')*.5+.5
    cache=torch.cat((k,scales.view(torch.uint8)),1)
    w=torch.randn(B,32,device='cuda')/32
    bt=torch.randperm(B*P,device='cuda').reshape(B,P).int()
    ctx=torch.full((B,1),raw_ctx//4-7,device='cuda',dtype=torch.int32)
    if mixed:
        ctx[::3]=0; ctx[1::3]=S//4+1
        bt[:,min(1,P-1)]=-1
    return q,cache,w,ctx,bt,S

def timed(fn, its=80):
    for _ in range(4): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    vals=[]
    for _ in range(5):
        st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(its): g.replay()
        en.record(); en.synchronize(); vals.append(st.elapsed_time(en)*1000/its)
    return statistics.median(vals),vals

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--full',action='store_true'); ap.add_argument('--uniform',action='store_true')
    args=ap.parse_args()
    variants=[(64,1,4),(128,1,4),(256,1,4),(64,2,4),(64,4,4),(128,2,4),(128,4,4),(256,2,4),(128,1,8),(256,1,8)]
    if args.uniform: variants=[(0,l,w) for l in [2,4,8,16] for w in [2,4]]+[(128,4,4)]
    rows=[]
    shapes=[(B,L,M) for B in ([8,24,32,40,48] if args.full else [8,32,48]) for L in ([16000,64000,250000] if args.full else [16000,250000]) for M in [False,True]]
    for B,L,mixed in shapes:
        inp=case(B,L,mixed); ref=baseline(inp); out=torch.empty_like(ref)
        bu,bs=timed(lambda:baseline(inp))
        for bk,loop,warps in variants:
            fn=lambda:invoke(inp,out,bk,loop,warps)
            try:
                cand=fn(); torch.cuda.synchronize()
                diff=(cand-ref).abs()
                us,ss=timed(fn)
                r=dict(B=B,raw_context=L,mixed=mixed,pooled_width=inp[-1],bk=bk,loop=loop,warps=warps,baseline_us=bu,candidate_us=us,speedup=bu/us,bitwise=torch.equal(cand,ref),max_abs=diff.max().item(),samples=ss,baseline_samples=bs)
            except Exception as e: r=dict(B=B,raw_context=L,mixed=mixed,bk=bk,loop=loop,warps=warps,error=str(e))
            rows.append(r); print(json.dumps(r),flush=True)
            Path(args.output).write_text(json.dumps(rows,indent=2))
        del inp,ref,out
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
