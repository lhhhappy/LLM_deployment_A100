"""T44 predecoded, query-major and grouped grid experiments."""
import torch
import triton
import triton.language as tl
from sm80_indexer_112 import _e4m3_to_bf16


@triton.jit
def unpack(X, Y, R: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
           XR: tl.constexpr, XH: tl.constexpr, XD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i // (H*D) * XR + (i // D % H) * XH + i % D * XD,
                i < R*H*D, other=0)
    tl.store(Y+i, _e4m3_to_bf16(x), i < R*H*D)


def prepare(args):
    q, (k, s), w, ks, ke = args
    nq, h, d = q.shape; nk = k.shape[0]
    qb = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    kb = torch.empty(k.shape, dtype=torch.bfloat16, device=q.device)
    unpack[(triton.cdiv(q.numel(), 1024),)](q.view(torch.uint8), qb, nq,h,d,*q.stride(),1024)
    unpack[(triton.cdiv(k.numel(), 1024),)](k.view(torch.uint8), kb, nk,1,d,k.stride(0),0,k.stride(1),1024)
    return qb,(kb,s),w,ks,ke


@triton.jit
def kernel(Q,K,SC,W,KS,KE,O, NQ:tl.constexpr,NK:tl.constexpr,H:tl.constexpr,
           QQ:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,KK:tl.constexpr,KD:tl.constexpr,
           SS:tl.constexpr,WQ:tl.constexpr,WH:tl.constexpr,KSS:tl.constexpr,KES:tl.constexpr,
           CLEAN:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,HH:tl.constexpr,
           GROUP:tl.constexpr,PRE:tl.constexpr,LOOP:tl.constexpr,QM:tl.constexpr):
    pid=tl.program_id(0)
    nqtiles=tl.cdiv(NQ,BQ); nktiles=tl.cdiv(NK,BK*LOOP)
    if GROUP == 0:
        pq=pid//nktiles; pk=pid%nktiles
    else:
        group=pid//(GROUP*nktiles)
        first=group*GROUP
        size=tl.minimum(nqtiles-first,GROUP)
        pq=first+pid%size
        pk=(pid%(GROUP*nktiles))//size
    qi=pq*BQ+tl.arange(0,BQ)
    if CLEAN:
        lo=tl.load(KS+qi*KSS,qi<NQ,other=0)
        hi=tl.load(KE+qi*KES,qi<NQ,other=0)
    ds=tl.arange(0,128)
    qh=tl.arange(0,BQ*HH)
    qr=pq*BQ+qh//HH; head=qh%HH
    qv=tl.load(Q+qr[:,None]*QQ+head[:,None]*QH+ds[None,:]*QD,
               (qr[:,None]<NQ)&(head[:,None]<H),other=0)
    if not PRE: qv=_e4m3_to_bf16(qv)
    w=tl.load(W+qr*WQ+head*WH,(qr<NQ)&(head<H),other=0).to(tl.float32)
    for j in range(LOOP):
        ki=(pk*LOOP+j)*BK+tl.arange(0,BK)
        compute=True
        if CLEAN:
            active=(qi<NQ)&(lo<hi)&(lo<(pk*LOOP+j+1)*BK)&(hi>(pk*LOOP+j)*BK)
            compute=tl.sum(active.to(tl.int32),0)>0
        res=tl.full((BQ,BK),float('-inf'),tl.float32)
        if compute:
            kv=tl.load(K+ki[None,:]*KK+ds[:,None]*KD,ki[None,:]<NK,other=0)
            if not PRE: kv=_e4m3_to_bf16(kv)
            if QM:
                dots=tl.dot(qv,kv).to(tl.bfloat16).to(tl.float32)
                weighted=tl.maximum(dots,0.)*w[:,None]
                res=tl.sum(tl.reshape(weighted,(BQ,HH,BK)),1)
            else:
                dots=tl.dot(tl.trans(kv),tl.trans(qv)).to(tl.bfloat16).to(tl.float32)
                weighted=tl.maximum(dots,0.)*w[None,:]
                res=tl.trans(tl.sum(tl.reshape(weighted,(BK,BQ,HH)),2))
            s=tl.load(SC+ki*SS,ki<NK,other=0).to(tl.float32)
            res=res*s[None,:]
            if CLEAN: res=tl.where((ki[None,:]>=lo[:,None])&(ki[None,:]<hi[:,None]),res,float('-inf'))
        tl.store(O+qi[:,None].to(tl.int64)*NK+ki[None,:],res,(qi[:,None]<NQ)&(ki[None,:]<NK))


def run(args, clean=True, bq=4, bk=128, warps=4, stages=2, group=8, pre=True, loop=1, out=None, qm=True):
    q,(k,s),w,ks,ke=args
    nq,h,d=q.shape;nk=k.shape[0]
    if out is None: out=torch.empty((nq,nk),dtype=torch.float32,device=q.device)
    compiled=kernel[(triton.cdiv(nq,bq)*triton.cdiv(nk,bk*loop),)](
        q if pre else q.view(torch.uint8),k if pre else k.view(torch.uint8),s,w,ks,ke,out,
        nq,nk,h,*q.stride(),*k.stride(),s.stride(0),*w.stride(),ks.stride(0),ke.stride(0),clean,
        bq,bk,max(16,triton.next_power_of_2(h)),group,pre,loop,qm,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    run.compiled=compiled
    return out
