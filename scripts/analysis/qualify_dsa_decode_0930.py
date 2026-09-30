"""Full native KPool topk stage and byte contract qualification for scratch page loops."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import triton
import bench_dsa_decode_0930 as p
from sglang.kernels.ops.moe.kpool_topk_transform import fast_kpool_topk_transform_fused as select

def make(B,L,mixed,S=None,gapped=False,N=1):
    inp=p.case(B,L,mixed)
    q,cache,w,ctx,bt,width=inp
    if S is not None and S>width:
        table=torch.zeros(B,triton.cdiv(S,64),device='cuda',dtype=torch.int32)
        table[:,:bt.shape[1]]=bt
        bt=table; width=S
    if gapped:
        qb=torch.empty(B,N,32,256,device='cuda',dtype=torch.float8_e4m3fn)
        qb[...,::2]=q.expand(B,N,32,128);q=qb[...,::2]
        wb=torch.empty(B*N,64,device='cuda'); wb[:,::2]=w.repeat_interleave(N,0); w=wb[:,::2]
        cb=torch.zeros(B,N*2,device='cuda',dtype=torch.int32);cb[:,::2]=ctx.expand(B,N);ctx=cb[:,::2]
        kb=torch.empty(cache.shape[0],cache.shape[1]+256,device='cuda',dtype=torch.uint8)
        kb[:,:cache.shape[1]]=cache;cache=kb[:,:cache.shape[1]]
        tb=torch.zeros(B,bt.shape[1]*2,device='cuda',dtype=torch.int32);tb[:,::2]=bt;bt=tb[:,::2]
    return q,cache,w,ctx,bt,width

def selection(logits,inp):
    lengths=inp[3].reshape(-1).contiguous()
    seq=lengths*4+3
    offsets=torch.arange(logits.shape[0],device='cuda',dtype=torch.int32)*1000000
    return select(logits,lengths,4,2048,topk_indices_offset=offsets,seq_lens=seq)

def exact(a,b):return bool(torch.equal(a,b))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--loop',type=int,default=4);ap.add_argument('--warps',type=int,default=4);args=ap.parse_args()
    rows=[]
    def save(r):
        rows.append(r); print(json.dumps(r),flush=True);Path(args.output).write_text(json.dumps(rows,indent=2))
    # Native topk JIT compiles before graph timings.
    for B in [8,24,32,40,48]:
        for L in [16000,64000,250000]:
            for mixed in [False,True]:
                inp=make(B,L,mixed);ref=p.baseline(inp);out=torch.empty_like(ref)
                fn=lambda:p.invoke(inp,out,0,args.loop,args.warps)
                cand=fn()
                sa,sb=selection(ref,inp),selection(cand,inp)
                good=exact(cand,ref); topgood=exact(sa.sort(-1).values,sb.sort(-1).values)
                assert good and topgood,(B,L,mixed,'numerical mismatch')
                bu,bs=p.timed(lambda:selection(p.baseline(inp),inp),its=32)
                cu,cs=p.timed(lambda:selection(fn(),inp),its=32)
                save(dict(kind='full_stage',B=B,raw_context=L,mixed=mixed,S=inp[-1],loop=args.loop,warps=args.warps,baseline_us=bu,candidate_us=cu,speedup=bu/cu,logits_bitwise=good,topk_set_equal=topgood,baseline_samples=bs,candidate_samples=cs,scratch_bytes=0))
    # Static graph width substantially larger than live compressed context.
    for B in [8,24,32,40,48]:
        inp=make(B,250000,False,S=262144)
        lens=torch.tensor([4000,16000,62500,0,1,63,64,65],device='cuda',dtype=torch.int32)
        inp[3][:,0]=lens[torch.arange(B,device='cuda')%8]
        ref=p.baseline(inp);out=torch.empty_like(ref);fn=lambda:p.invoke(inp,out,0,args.loop,args.warps)
        assert exact(fn(),ref)
        bu,bs=p.timed(lambda:selection(p.baseline(inp),inp),its=32)
        cu,cs=p.timed(lambda:selection(fn(),inp),its=32)
        save(dict(kind='padded_width_stage',B=B,S=262144,lengths=lens.tolist(),baseline_us=bu,candidate_us=cu,speedup=bu/cu,logits_bitwise=True,baseline_samples=bs,candidate_samples=cs))
    for N in [1,3]:
        inp=make(8,16000,True,S=8193,gapped=True,N=N)
        ref=p.baseline(inp);out=torch.empty_like(ref);fn=lambda:p.invoke(inp,out,0,args.loop,args.warps)
        assert exact(fn(),ref)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):gout=fn()
        for mode in ['original','shrink','empty','changed','grow']:
            if mode=='shrink': inp[3][:]=65
            if mode=='empty':inp[3].zero_()
            if mode=='changed':
                inp[0].copy_(torch.randn_like(inp[0],dtype=torch.float32).to(torch.float8_e4m3fn))
                inp[2].mul_(-.5)
                inp[1][:,64*128:].view(torch.float32).mul_(.75)
                inp[4][:,0]=-7
                inp[3][:]=71
            if mode=='grow':inp[3][:]=3999
            graph.replay();expected=p.baseline(inp);torch.cuda.synchronize()
            assert exact(gout,expected),(N,mode)
            save(dict(kind='dynamic_graph',N=N,mode=mode,logits_bitwise=True,q_stride=inp[0].stride(),w_stride=inp[2].stride(),cache_stride=inp[1].stride(),ctx_stride=inp[3].stride(),bt_stride=inp[4].stride(),width=inp[-1]))
    # Independent FP32 equations for valid page 0 and 1, contract BF16 head boundary.
    inp=make(8,16000,False);out=p.invoke(inp,torch.empty((8,inp[-1]),device='cuda'),0,args.loop,args.warps)
    q,cache,w,ctx,bt,S=inp
    all_reference=[];all_actual=[]
    for b in range(8):
        for page in [0,1,bt.shape[1]-1]:
            physical=max(0,int(bt[b,page]))
            key=cache[physical,:8192].view(torch.float8_e4m3fn).reshape(64,128).float()
            dots=(key@q[b,0].float().T).bfloat16().float()
            scale=cache[physical,8192:].view(torch.float32)
            oracle=(dots.clamp_min(0)*w[b]).sum(-1)*scale
            pos=torch.arange(64,device='cuda')+page*64
            oracle=torch.where(pos<ctx[b,0],oracle,0)
            all_reference.append(oracle);all_actual.append(out[b,page*64:(page+1)*64])
    a,e=torch.cat(all_actual),torch.cat(all_reference);diff=(a-e).abs()
    save(dict(kind='fp32_reference',max_abs=diff.max().item(),relative_l2=(diff.norm()/e.norm()).item(),all_finite=bool(torch.isfinite(a).all())))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
