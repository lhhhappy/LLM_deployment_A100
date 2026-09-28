#!/usr/bin/env python3
"""Prototype a large-row post + prenorm kernel; no serving integration.

Store the post-mapped BF16 residual (needed later), then reuse that rounded value
for FP32 prenorm products and square sums. The final mixing/Sinkhorn kernel is
unchanged. Compare both intermediate tensors and the complete boundary path.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def post_prenorm(X, R, A, C, FN, R_OUT, MIX, SQR, M,
                 H: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                 SPLITS: tl.constexpr, PRECISION: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    split = tl.program_id(1)
    ks = tl.arange(0, BK)
    ns = tl.arange(0, 32)
    valid = rows < M
    # Mapping convention: old residual channel first, new channel second.
    c0 = tl.load(C + rows * 4 + 0, valid, 0)
    c1 = tl.load(C + rows * 4 + 1, valid, 0)
    c2 = tl.load(C + rows * 4 + 2, valid, 0)
    c3 = tl.load(C + rows * 4 + 3, valid, 0)
    acc = tl.zeros((BM, 32), tl.float32)
    ss = tl.zeros((BM, BK), tl.float32)
    for block in range(H // (SPLITS * BK)):
        cols = split * (H // SPLITS) + block * BK + ks
        x = tl.load(X + rows[:, None] * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        r0 = tl.load(R + (rows[:, None] * 4 + 0) * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        r1 = tl.load(R + (rows[:, None] * 4 + 1) * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        r2 = tl.load(R + (rows[:, None] * 4 + 2) * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        r3 = tl.load(R + (rows[:, None] * 4 + 3) * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        for channel in tl.static_range(4):
            cc = c0 if channel == 0 else (c1 if channel == 1 else (c2 if channel == 2 else c3))
            a0 = tl.load(A + rows * 16 + channel, valid, 0)
            a1 = tl.load(A + rows * 16 + 4 + channel, valid, 0)
            a2 = tl.load(A + rows * 16 + 8 + channel, valid, 0)
            a3 = tl.load(A + rows * 16 + 12 + channel, valid, 0)
            z = tl.fma(cc[:, None], x, a0[:, None] * r0)
            z = tl.fma(a1[:, None], r1, z)
            z = tl.fma(a2[:, None], r2, z)
            z = tl.fma(a3[:, None], r3, z)
            z_bf = z.to(tl.bfloat16)
            z = z_bf.to(tl.float32)
            tl.store(R_OUT + (rows[:, None] * 4 + channel) * H + cols[None, :], z_bf, valid[:, None])
            f = tl.load(FN + ns[None, :] * (4 * H) + channel * H + cols[:, None], ns[None, :] < 24, 0)
            acc = tl.dot(z, f, acc, input_precision=PRECISION)
            ss = tl.fma(z, z, ss)
    sq = tl.sum(ss, 1)
    tl.store(MIX + (split * M + rows[:, None]) * 24 + ns[None, :], acc, valid[:, None] & (ns[None, :] < 24))
    tl.store(SQR + split * M + rows, sq, valid)


@triton.jit
def post_prenorm_compact(X, R, A, C, FN, R_OUT, MIX, SQR, M,
                        H: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                        SPLITS: tl.constexpr, PRECISION: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    split = tl.program_id(1)
    ks = tl.arange(0, BK)
    ns = tl.arange(0, 32)
    valid = rows < M
    acc = tl.zeros((BM, 32), tl.float32)
    ss = tl.zeros((BM, BK), tl.float32)
    # Do not unroll four independent GEMMs into one huge live register set.
    for block in tl.range(H // (SPLITS * BK), num_stages=1, loop_unroll_factor=1):
        cols = split * (H // SPLITS) + block * BK + ks
        x = tl.load(X + rows[:, None] * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        r0 = tl.load(R + (rows[:, None] * 4 + 0) * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        r1 = tl.load(R + (rows[:, None] * 4 + 1) * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        r2 = tl.load(R + (rows[:, None] * 4 + 2) * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        r3 = tl.load(R + (rows[:, None] * 4 + 3) * H + cols[None, :], valid[:, None], 0).to(tl.float32)
        for channel in tl.range(4, num_stages=1, loop_unroll_factor=1):
            cc = tl.load(C + rows * 4 + channel, valid, 0)
            a0 = tl.load(A + rows * 16 + channel, valid, 0)
            a1 = tl.load(A + rows * 16 + 4 + channel, valid, 0)
            a2 = tl.load(A + rows * 16 + 8 + channel, valid, 0)
            a3 = tl.load(A + rows * 16 + 12 + channel, valid, 0)
            z = tl.fma(cc[:, None], x, a0[:, None] * r0)
            z = tl.fma(a1[:, None], r1, z)
            z = tl.fma(a2[:, None], r2, z)
            z = tl.fma(a3[:, None], r3, z)
            zb = z.to(tl.bfloat16)
            z = zb.to(tl.float32)
            tl.store(R_OUT + (rows[:, None] * 4 + channel) * H + cols[None, :], zb, valid[:, None])
            f = tl.load(FN + ns[None, :] * (4 * H) + channel * H + cols[:, None], ns[None, :] < 24, 0)
            if PRECISION == 'bf16x2':
                hi = f.to(tl.bfloat16)
                lo = (f - hi.to(tl.float32)).to(tl.bfloat16)
                acc = tl.dot(zb, lo, acc)
                acc = tl.dot(zb, hi, acc)
            else:
                acc = tl.dot(z, f, acc, input_precision=PRECISION)
            ss = tl.fma(z, z, ss)
    tl.store(MIX + (split * M + rows[:, None]) * 24 + ns[None, :], acc, valid[:, None] & (ns[None, :] < 24))
    tl.store(SQR + split * M + rows, tl.sum(ss, 1), valid)


@triton.jit
def prenorm_only(R, FN, MIX, SQR, M,
                 H: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                 SPLITS: tl.constexpr, PRECISION: tl.constexpr):
    rows=tl.program_id(0).to(tl.int64)*BM+tl.arange(0,BM)
    split=tl.program_id(1)
    ks=tl.arange(0,BK);ns=tl.arange(0,32)
    acc=tl.zeros((BM,32),tl.float32);ss=tl.zeros((BM,BK),tl.float32)
    for block in range(4*H//(SPLITS*BK)):
        cols=split*(4*H//SPLITS)+block*BK+ks
        x=tl.load(R+rows[:,None]*(4*H)+cols[None,:],rows[:,None]<M,0).to(tl.float32)
        f=tl.load(FN+ns[None,:]*(4*H)+cols[:,None],ns[None,:]<24,0)
        acc=tl.dot(x,f,acc,input_precision=PRECISION)
        ss=tl.fma(x,x,ss)
    tl.store(MIX+(split*M+rows[:,None])*24+ns[None,:],acc,(rows[:,None]<M)&(ns[None,:]<24))
    tl.store(SQR+split*M+rows,tl.sum(ss,1),rows<M)


@triton.jit
def partial_sum(MIX, SQR, OUT, SO, M, SPLITS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    n = tl.arange(0, 32)
    acc = tl.zeros((B, 32), tl.float32)
    s = tl.zeros((B,), tl.float32)
    for i in tl.static_range(SPLITS):
        acc += tl.load(MIX + (i * M + row[:, None]) * 24 + n[None, :], (row[:, None] < M) & (n[None, :] < 24), 0)
        s += tl.load(SQR + i * M + row, row < M, 0)
    tl.store(OUT + row[:, None] * 24 + n[None, :], acc, (row[:, None] < M) & (n[None, :] < 24))
    tl.store(SO + row, s, row < M)


def emit(**x): print(json.dumps(x), flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows',type=int,nargs='+',default=[256,1024,4096,8192,16384])
    ap.add_argument('--configs',help='JSON [BM,BK,splits,warps,precision] list')
    ap.add_argument('--rounds',type=int,default=5)
    ap.add_argument('--calls',type=int,default=5)
    ap.add_argument('--prenorm-only',action='store_true',help='Ablation: keep baseline post, replace prenorm only')
    ap.add_argument('--compact',action='store_true',help='Avoid unrolling the four-channel post/GEMM loop')
    args=ap.parse_args()
    from sglang.kernels.ops.layernorm.mhc import mhc_post_tilelang, mhc_pre_gemm_sqrsum_tilelang
    import sglang.kernels.ops.layernorm.mhc as mhc_source
    source=Path(mhc_source.__file__).resolve()
    assert source.is_relative_to(Path(__file__).resolve().parents[2]/'engine'),source
    configs=json.loads(args.configs) if args.configs else [
        [16,64,4,4,'tf32'], [16,128,4,4,'tf32'], [32,64,4,4,'tf32'],
        [16,64,8,4,'tf32'], [16,128,8,4,'tf32'], [32,64,8,4,'tf32'],
        [16,64,4,4,'tf32x3'], [16,128,4,4,'tf32x3'], [32,64,4,4,'tf32x3'],
    ]
    emit(kind='environment',gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
         source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         arguments=vars(args),scope='synthetic mHC boundary, no serving integration; final mixing unchanged')
    flush=torch.empty(96<<20,device='cuda',dtype=torch.uint8)
    H=4096
    for m in args.rows:
        gen=torch.Generator(device='cuda').manual_seed(173+m)
        x=torch.randn(m,H,device='cuda',dtype=torch.bfloat16,generator=gen)
        r=torch.randn(m,4,H,device='cuda',dtype=torch.bfloat16,generator=gen)
        a=torch.rand(m,4,4,device='cuda',generator=gen)
        for _ in range(5):a=a/a.sum(1,keepdim=True);a=a/a.sum(2,keepdim=True)
        c=2*torch.rand(m,4,device='cuda',generator=gen)
        fn=torch.randn(24,4*H,device='cuda',generator=gen)*.01
        rb=torch.empty_like(r);rc=torch.empty_like(r)
        mb=torch.empty(m,24,device='cuda');sb=torch.empty(m,device='cuda')
        mc=torch.empty_like(mb);sc=torch.empty_like(sb)

        def baseline():
            mhc_post_tilelang(a,r,c,x,rb,4,H)
            mhc_pre_gemm_sqrsum_tilelang(rb.view(m,4*H),fn,mb,sb,24,4*H)

        baseline();torch.cuda.synchronize()
        ids=torch.linspace(0,m-1,min(32,m),device='cuda').long().unique()
        ref=rb[ids].double().flatten(1)@fn.double().T
        sq=rb[ids].double().flatten(1).square().sum(1)
        scale=(rb[ids].double().flatten(1).square().sum(1).sqrt()[:,None]*fn.double().square().sum(1).sqrt()[None,:])
        emit(kind='baseline_oracle',rows=m,mix_l2=((mb[ids].double()-ref).norm()/ref.norm()).item(),
             scaled_max=((mb[ids].double()-ref).abs()/scale).max().item(),sq_rel=((sb[ids].double()-sq).abs()/sq).max().item())
        valid=[];bufs={}
        for cfg in configs:
            bm,bk,sp,warps,precision=cfg
            pm=torch.empty(sp,m,24,device='cuda');ps=torch.empty(sp,m,device='cuda')
            def run(cfg=cfg,pm=pm,ps=ps):
                bm,bk,sp,warps,precision=cfg
                if args.prenorm_only:
                    mhc_post_tilelang(a,r,c,x,rc,4,H)
                    kernel=prenorm_only[(triton.cdiv(m,bm),sp)](rc,fn,pm,ps,m,H,bm,bk,sp,precision,
                        num_warps=warps,num_stages=2,enable_fp_fusion=False)
                else:
                    op=post_prenorm_compact if args.compact else post_prenorm
                    kernel=op[(triton.cdiv(m,bm),sp)](x,r,a,c,fn,rc,pm,ps,m,H,bm,bk,sp,precision,
                        num_warps=warps,num_stages=1 if args.compact else 2,enable_fp_fusion=False)
                partial_sum[(triton.cdiv(m,32),)](pm,ps,mc,sc,m,sp,32,num_warps=4)
                return kernel
            kernel=run();torch.cuda.synchronize()
            exact=torch.equal(rb,rc)
            err=((mc[ids].double()-ref).norm()/ref.norm()).item()
            base_err=((mb[ids].double()-ref).norm()/ref.norm()).item()
            sqerr=((sc[ids].double()-sq).abs()/sq).max().item()
            emit(kind='correctness',rows=m,config=cfg,residual_exact=exact,mix_l2=err,
                 baseline_mix_l2=base_err,square_sum_rel=sqerr,
                 mix_vs_base_max=(mc-mb).abs().max().item(),regs=kernel.n_regs,spills=kernel.n_spills,shared=kernel.metadata.shared)
            if exact and err<=max(base_err*1.25,2e-6) and sqerr<2e-5:
                valid.append((str(cfg),run));bufs[str(cfg)]=(pm,ps)
        funcs=[('baseline',baseline)]+valid
        times={k:[] for k,_ in funcs};rng=random.Random(173+m)
        for _,f in funcs:
            for _ in range(3):f()
        for _ in range(args.rounds):
            rng.shuffle(funcs)
            for name,f in funcs:
                samples=[]
                for _ in range(args.calls):
                    flush.fill_(1);start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();f();end.record();end.synchronize();samples.append(start.elapsed_time(end))
                times[name].append(statistics.median(samples))
        med={k:statistics.median(v) for k,v in times.items()}
        emit(kind='timing',rows=m,medians_ms=med,rounds_ms=times,speedups={k:med['baseline']/v for k,v in med.items()})
    emit(kind='complete')


if __name__=='__main__':main()
