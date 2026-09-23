#!/usr/bin/env python3
"""T48 random operator checks using actual 160 candidate kernels, no model/service."""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
import torch
import triton


def emit(**kw):
    print(json.dumps(kw, sort_keys=True), flush=True)


def bootstrap(root):
    # Bypass service package initialization only; do not replace arithmetic.
    for name, rel in [('sglang',''),('sglang.srt','srt'),('sglang.kernels','kernels'),
                      ('sglang.srt.utils','srt/utils')]:
        m=types.ModuleType(name);m.__path__=[str(root/rel)];sys.modules[name]=m
    u=sys.modules['sglang.srt.utils']
    u.is_cpu=u.is_npu=u.is_xpu=u.is_hip=u.is_gfx95_supported=u.cpu_has_amx_support=lambda:False
    u.is_cuda=lambda:True
    u.cdiv=triton.cdiv;u.next_power_of_2=triton.next_power_of_2
    m=types.ModuleType('sglang.srt.utils.common');m.torch_release=(2,13);sys.modules[m.__name__]=m


def compare(a,b,atol=1e-5,rtol=1e-5):
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    return dict(max_abs=float((a.float()-b.float()).abs().max()),
                rel_l2=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)),
                exact=torch.equal(a,b))


def graph_check(fn, inputs, reference=None):
    # fn includes reset copies for mutable state so replay/eager share initial state.
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): outputs=fn()
    if not isinstance(outputs,tuple):outputs=(outputs,)
    for step in range(3):
        for x in inputs:x.copy_(torch.randn_like(x)*.1)
        g.replay();actual=tuple(x.clone() for x in outputs)
        expected=fn();expected=expected if isinstance(expected,tuple) else (expected,)
        for a,b in zip(actual,expected):compare(a,b,0,0)
        if reference is not None:reference(actual)
    return 3


@torch.inference_mode()
def kda(B,T,H=8):
    from sglang.kernels.ops.attention.fla.fused_kda_conv_recurrent_verify import fused_kda_conv_gating_verify
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update as rec
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update as conv
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import scatter_mamba_states_after_mtp_verify as scatter
    D=128;C=3*H*D;S=B*3+2
    raw=torch.randn(B*T,C,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(C,4,device='cuda')*.1
    a=torch.randn(1,B*T,H*D,device='cuda',dtype=torch.bfloat16)*.1
    beta=torch.randn(1,B*T,H,device='cuda',dtype=torch.bfloat16)
    alog=torch.randn(H,device='cuda')*.1;bias=torch.randn(H*D,device='cuda')*.1
    initial=torch.randn(S,H,D,D,device='cuda')*.1
    cs0=torch.randn(S,3,C,device='cuda',dtype=torch.bfloat16)*.1
    slots=torch.arange(1,B+1,device='cuda',dtype=torch.int32)
    scratch_ids=torch.arange(B,device='cuda',dtype=torch.int32)
    cu=torch.arange(B+1,device='cuda',dtype=torch.int32)*T
    ssm=initial.clone();cs=cs0.clone();inter=torch.zeros(B+1,T,H,D,D,device='cuda')
    windows=torch.zeros(B+1,T,3,C,device='cuda',dtype=torch.bfloat16)
    def run(fused):
        ssm.copy_(initial);cs.copy_(cs0);inter.zero_();windows.zero_()
        if fused:
            out=fused_kda_conv_gating_verify(raw,w,None,cs.transpose(-1,-2),slots,windows.transpose(-1,-2),
                scratch_ids,a,beta,alog,bias,ssm,slots,inter,D**-.5,T,H,H,D,D,lower_bound=-5.)
        else:
            proc=conv(raw.view(B,T,C).transpose(1,2),cs.transpose(-1,-2),w,activation='silu',
                      conv_state_indices=slots,intermediate_conv_window=windows.transpose(-1,-2),
                      intermediate_state_indices=scratch_ids).transpose(1,2).reshape(B*T,C)
            q,k,v=[x.reshape(1,B*T,H,D) for x in proc.chunk(3,-1)]
            out=rec(alog,a,bias,1.,20.,q,k,v,beta,ssm,slots,use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu,is_kda=True,lower_bound=-5.,disable_state_update=True,
                    intermediate_states_buffer=inter,intermediate_state_indices=scratch_ids,cache_steps=T)
        return out,inter,windows,cs,ssm
    oracle=tuple(x.clone() for x in run(False))
    assert torch.equal(ssm,initial), 'verify changed committed SSM'
    if T>=3:
        actual=run(True)
        errors=[compare(x,y,3e-5,3e-3) for x,y in zip(actual,oracle)]
        replay=graph_check(lambda:run(True),[raw,a,beta])
    else:
        errors=[];replay=graph_check(lambda:run(False),[raw,a,beta])
    # Every possible accepted length, active + tracking + masked destination.
    run(False)
    src=inter.clone();win=windows.clone()
    for accept in range(1,T+1):
        accepted=torch.full((B,),accept-1,device='cuda',dtype=torch.int32)
        tracks=torch.arange(B+1,2*B+1,device='cuda',dtype=torch.int32);tracks[-1]=-1
        track_steps=torch.zeros(B,device='cuda',dtype=torch.int32)
        cache=types.SimpleNamespace(temporal=ssm.unsqueeze(0),conv=[cs.unsqueeze(0)],
                 intermediate_ssm=inter.unsqueeze(0),intermediate_conv_window=[windows.unsqueeze(0)])
        def commit():
            ssm.copy_(initial);cs.copy_(cs0)
            scatter(cache,slots,accepted,tracks,track_steps)
            return ssm,cs
        commit()
        compare(ssm[slots.long()],src[:B,accept-1],0,0)
        compare(cs[slots.long()],win[:B,accept-1],0,0)
        if B>1:
            compare(ssm[tracks[:-1].long()],src[:B-1,0],0,0)
            compare(cs[tracks[:-1].long()],win[:B-1,0],0,0)
        compare(ssm[-1],initial[-1],0,0)
        compare(cs[-1],cs0[-1],0,0)
        graph_check(commit,[])
    # independent per-token decode on accepted prefix must match verify scratch.
    ssm.copy_(initial);cs.copy_(cs0)
    for i in range(T):
        x=raw.view(B,T,C)[:,i].contiguous()
        proc=conv(x,cs.transpose(-1,-2),w,activation='silu',conv_state_indices=slots)
        q,k,v=[x.reshape(1,B,H,D) for x in proc.chunk(3,-1)]
        rec(alog,a.view(B,T,-1)[:,i].contiguous().unsqueeze(0),bias,1.,20.,q,k,v,
            beta.view(B,T,H)[:,i].contiguous().unsqueeze(0),ssm,slots,
            use_qk_l2norm_in_kernel=True,cu_seqlens=torch.arange(B+1,device='cuda',dtype=torch.int32),
            is_kda=True,lower_bound=-5.)
        compare(ssm[slots.long()],src[:B,i],3e-6,1e-4)
        compare(cs[slots.long()],win[:B,i],0,0)
    emit(kind='kda_verify_commit',B=B,T=T,H=H,errors=errors,graph_replays=replay,accepted_lengths=T,status='PASS')


@torch.inference_mode()
def small():
    from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
    from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm
    B=6;H=4096
    logits=torch.randn(B,154880,device='cuda');pos=torch.zeros(B,device='cuda',dtype=torch.long)
    tokens=torch.zeros(B,3,device='cuda',dtype=torch.long)
    def top():
        pos.zero_();tokens.zero_();p,i=draft_topk1_postprocess(logits,pos,tokens,1)
        return p,i,pos,tokens
    p,i,_,_=top();compare(i,logits.argmax(-1,keepdim=True),0,0)
    graph_check(top,[logits])
    x=torch.randn(B,H,device='cuda',dtype=torch.bfloat16);h=torch.randn_like(x)
    wx=torch.randn(H,device='cuda',dtype=torch.bfloat16);wh=torch.randn_like(wx)
    out=fused_eh_norm(x,h,wx,wh,1e-5)
    def rms(z,w):return ((z.float()*torch.rsqrt(z.float().square().mean(-1,keepdim=True)+1e-5)).to(z.dtype)*w)
    err=compare(out,torch.cat([rms(x,wx),rms(h,wh)],-1),.04,.02)
    graph_check(lambda:fused_eh_norm(x,h,wx,wh,1e-5),[x,h])
    emit(kind='draft_topk1_eh_norm',status='PASS',norm=err)


@torch.inference_mode()
def kpool():
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (update_kpool_write_plan_cuda_graph as plan,
        kpool_write_tail_and_maybe_compress as write)
    from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant
    for T in (2,4,6):
        B=6;D=128;tail_size=4+T
        pool=types.SimpleNamespace(index_kpool=4,tail_extra_slots=T,slots_per_page=64,index_head_dim=D)
        ws=torch.tensor([0,1,3,63,64,255],device='cuda',dtype=torch.int32)
        req=torch.arange(B,device='cuda',dtype=torch.int32)
        pt=torch.stack([torch.arange(1+i*8,1+(i+1)*8,device='cuda') for i in range(B)]).int().repeat_interleave(T,0)
        key=torch.randn(B*T,D,device='cuda',dtype=torch.bfloat16)*.1
        score=torch.randn(B*T,D,device='cuda',dtype=torch.bfloat16)*.1
        tail0=torch.randn(B,tail_size,D,device='cuda',dtype=torch.bfloat16)*.1
        scores0=torch.randn_like(tail0)*.1
        tail=tail0.clone();scores=scores0.clone();ape=torch.randn(4,D,device='cuda')*.1
        buf=torch.zeros(1+B*8,64*132,device='cuda',dtype=torch.uint8)
        ro=torch.empty_like(req);wo=torch.empty_like(ws);base=torch.empty_like(ws)
        loc=torch.empty(B,(T+3)//4,device='cuda',dtype=torch.int64)
        lens=torch.empty(B*T,device='cuda',dtype=torch.int32);plens=torch.empty_like(lens)
        outloc=torch.ones(B*T,device='cuda',dtype=torch.int64);outloc[-T:]=0 # padded graph row
        effective=torch.tensor([T,max(1,T-1),T,1,T,T],device='cuda',dtype=torch.int32)
        def run():
            buf.zero_();tail.copy_(tail0);scores.copy_(scores0)
            plan(ws,req,pt,ro,wo,base,loc,plens,lens,pool_size=4,num_draft_tokens=T,slots_per_page=64)
            write(pool,buf,key,score,tail,scores,ape,ro,wo,base,loc,outloc,T,False,effective)
            return buf,tail,scores,lens,plens
        run()
        expected=(ws[:,None]+torch.arange(1,T+1,device='cuda',dtype=torch.int32)).reshape(-1)
        compare(lens,expected,0,0);compare(plens,expected//4,0,0)
        maxerr=0.
        for i in range(B-1):
            start=int(ws[i]);n=(start+int(effective[i]))//4-start//4
            for step in range(T):
                compare(tail[i,(start+step)%tail_size],key[i*T+step],0,0)
            for j in range(n):
                ix=(int(base[i])+j*4+torch.arange(4,device='cuda'))%tail_size
                z=(torch.softmax(scores[i,ix].float()+ape,dim=0)*tail[i,ix].float()).sum(0).bfloat16().float()
                for stride in (1,2,4,8,16,32,64):
                    v=z.reshape(-1,2,stride);z=torch.stack((v[:,0]+v[:,1],v[:,0]-v[:,1]),1).reshape(-1)
                z=(z*128**-.5).bfloat16().float()
                scale=z.abs().max().clamp_min(1e-4)/448
                ref=(z/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale
                page=int(loc[i,j])//64;slot=int(loc[i,j])%64
                got=buf[page,slot*128:(slot+1)*128].view(torch.float8_e4m3fn).float()
                got=got*buf[page,8192+slot*4:8192+(slot+1)*4].view(torch.float32)
                err=float((got-ref).norm()/ref.norm());assert err<.015,err;maxerr=max(maxerr,err)
        compare(tail[-1],tail0[-1],0,0)
        replay=graph_check(run,[key,score])
        query=torch.randn(B*T,32,128,device='cuda',dtype=torch.bfloat16)*.1
        quant,sc=act_quant(query,128,None)
        assert torch.isfinite(quant.float()).all()
        graph_check(lambda:act_quant(query,128,None),[query])
        emit(kind='kpool_spec_write_plan_quant',T=T,max_rel_l2=maxerr,graph_replays=replay,status='PASS')


@torch.inference_mode()
def dsa():
    # tilelang_kernel imports only this dtype predicate from the large quantization
    # module; BF16 attention below never invokes a quantization replacement.
    mod=types.ModuleType('sglang.kernels.ops.quantization.fp8_kernel')
    mod.is_fp8_fnuz=lambda:False;sys.modules[mod.__name__]=mod
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    for M in (1,6,24,36):
        D=512;H=8;N=4096;K=2112
        q=torch.randn(M,H,D,device='cuda',dtype=torch.bfloat16)*.1
        kv=torch.randn(N,1,D,device='cuda',dtype=torch.bfloat16)*.1
        idx=torch.randint(0,N,(M,1,K),device='cuda',dtype=torch.int32);idx[:,:,2051:]=-1
        def run():return tilelang_sparse_fwd(q=q,kv=kv,indices=idx,sm_scale=D**-.5,d_v=D).reshape(M,H,D)
        def ref():
            keys=kv[idx[:,0,:2051].long(),0].float()
            scores=torch.einsum('mhd,mkd->mhk',q.float(),keys)*D**-.5
            return torch.einsum('mhk,mkd->mhd',scores.softmax(-1),keys)
        error=compare(run().float(),ref(),.001,.02)
        graph_check(run,[q,kv])
        emit(kind='tilelang_dsa',M=M,nvalid=2051,error=error,status='PASS')


@torch.inference_mode()
def acceptance():
    from sgl_kernel import verify_tree_greedy, tree_speculative_sampling_target_only, fast_topk_v2
    from sglang.kernels.ops.speculative.eagle import nextn_mamba_commit_prologue_func
    B=4;T=4;V=128
    cand=torch.arange(B*T,device='cuda',dtype=torch.int64).reshape(B,T)+10
    target=torch.cat((cand[:,1:],torch.full((B,1),77,device='cuda',dtype=torch.int64)),1)
    for i in range(B):target[i,i]=77
    ri=torch.arange(B*T,device='cuda',dtype=torch.int64).reshape(B,T)
    rn=torch.tensor([[1,2,3,-1]]*B,device='cuda',dtype=torch.int64);rs=torch.full_like(rn,-1)
    predict=torch.zeros(B*T,device='cuda',dtype=torch.int32)
    ix=torch.full((B,T),-1,device='cuda',dtype=torch.int32);count=torch.empty(B,device='cuda',dtype=torch.int32)
    probs=torch.nn.functional.one_hot(target,V).float();dp=torch.zeros_like(probs)
    coins=torch.full((B,T),.5,device='cuda');final_coins=torch.full((B,),.5,device='cuda')
    def run(sample=False):
        predict.zero_();ix.fill_(-1);count.zero_();dp.zero_()
        kw=dict(predicts=predict,accept_index=ix,accept_token_num=count,candidates=cand,
                retrive_index=ri,retrive_next_token=rn,retrive_next_sibling=rs)
        if sample:
            tree_speculative_sampling_target_only(**kw,uniform_samples=coins,
                uniform_samples_for_final_sampling=final_coins,target_probs=probs,draft_probs=dp,
                threshold_single=1.,threshold_acc=1.,deterministic=True)
        else:verify_tree_greedy(**kw,target_predict=target)
        return predict,ix,count
    for sample in (False,True):
        run(sample)
        compare(count,torch.arange(B,device='cuda',dtype=torch.int32),0,0)
        for i in range(B):compare(ix[i,:i+1],ri[i,:i+1].int(),0,0)
        graph_check(lambda:run(sample),[])
    accept=torch.arange(1,5,device='cuda',dtype=torch.int32)
    seq=torch.tensor([255,255,254,512],device='cuda',dtype=torch.int64)
    def commit():return nextn_mamba_commit_prologue_func(accept,seq,256,True)
    last,track=commit();compare(last,accept-1,0,0)
    compare(track,torch.tensor([0,0,1,-1],device='cuda',dtype=torch.int32),0,0)
    graph_check(commit,[])
    scores=torch.randn(24,4096,device='cuda');lens=torch.full((24,),4096,device='cuda',dtype=torch.int32)
    def topk():return fast_topk_v2(scores,lens,2048)
    got=topk();ref=scores.topk(2048,-1).indices
    compare(got.sort(-1).values.long(),ref.sort(-1).values,0,0)
    graph_check(lambda:topk().sort(-1).values,[scores])
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import topk_from_pooled_history_logits
    seq_lens=lens*4+3
    def pooled_topk():return topk_from_pooled_history_logits(scores,lens,4,2048,seq_lens=seq_lens)
    got=pooled_topk();groups=scores.topk(512,-1).indices
    expected=(groups[:,:,None]*4+torch.arange(4,device='cuda')).reshape(24,2048)
    compare(got[:,:2048].sort(-1).values.long(),expected.sort(-1).values,0,0)
    compare(got[:,2048:].long(),lens[:,None].long()*4+torch.arange(3,device='cuda'),0,0)
    graph_check(lambda:pooled_topk().sort(-1).values,[scores])
    emit(kind='greedy_target_only_sampling_commit_prologue_dsa_topk',status='PASS',accepted_lengths=[1,2,3,4])


@torch.inference_mode()
def indexer():
    from sglang.srt.layers.attention.dsa.sm80_deep_gemm import fp8_paged_mqa_logits,fp8_mqa_logits
    B=6;N=4;H=32;D=128;L=2048;P=L//64
    def fp8(*shape):return (torch.randn(shape,device='cuda')*.2).to(torch.float8_e4m3fn)
    kvals=fp8(P,64,D);scales=torch.rand(P,64,device='cuda')+.1
    cache=torch.empty(P,64*132,device='cuda',dtype=torch.uint8)
    cache[:,:8192]=kvals.view(torch.uint8).reshape(P,-1);cache[:,8192:]=scales.view(torch.uint8).reshape(P,-1)
    q=fp8(B,N,H,D);w=torch.rand(B*N,H,device='cuda')
    ctx=(torch.arange(B*N,device='cuda',dtype=torch.int32)*67+60).reshape(B,N)
    bt=torch.arange(P,device='cuda',dtype=torch.int32).expand(B,-1).contiguous()
    def run():return fp8_paged_mqa_logits(q,cache.view(P,64,1,132),w,ctx,bt,None,L)
    flatk=kvals.bfloat16().reshape(L,D);qr=q.reshape(B*N,H,D).bfloat16()
    ref=torch.zeros(B*N,L,device='cuda')
    for h in range(H):ref+=(qr[:,h]@flatk.T).float().relu()*w[:,h,None]
    ref*=scales.reshape(1,L);ref.masked_fill_(torch.arange(L,device='cuda')[None,:]>=ctx.reshape(-1,1),0)
    e=compare(run(),ref,.001,.001)
    # Integer-byte graph inputs updated with legal FP8 payloads; graph_check's float
    # randomization is not suitable for byte storage, so use a separate explicit loop.
    for _ in range(3):run()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=run()
    for i in range(3):
        q.view(torch.uint8).copy_(fp8(B,N,H,D).view(torch.uint8));ctx.add_(1)
        g.replay();got=out.clone();compare(got,run(),0,0)
    # Prefill path includes the 113 predecode branch at real H32.
    qq=fp8(64,H,D);ww=torch.rand(64,H,device='cuda');ks=torch.zeros(64,device='cuda',dtype=torch.int32)
    ke=torch.full_like(ks,L);kk=kvals.reshape(L,D);ss=scales.reshape(L)
    def pre():return fp8_mqa_logits(qq,(kk,ss),ww,ks,ke,clean_logits=True)
    pr=torch.zeros(64,L,device='cuda')
    for h in range(H):pr+=(qq[:,h].bfloat16()@flatk.T).float().relu()*ww[:,h,None]
    pr*=ss[None,:];ep=compare(pre(),pr,.002,.001);graph_check(pre,[ww])
    emit(kind='indexer_verify_B6N4_prefill',paged=e,prefill=ep,status='PASS')


@torch.inference_mode()
def mhc():
    import contextlib
    import os
    os.environ['SGLANG_OPT_DEEPGEMM_HC_PRENORM']='0'
    # No collectives in this operator; replace only group/allocation-policy lookups.
    for name,attrs in {
        'sglang.srt.distributed.device_communicators.pynccl_allocator':dict(use_symmetric_memory=lambda *a,**k:contextlib.nullcontext()),
        'sglang.srt.distributed.parallel_state':dict(get_tp_group=lambda:None),
        'sglang.srt.layers.attention.dsa.utils':dict(is_dsa_prefill_cp_round_robin_split=lambda:False),
        'sglang.srt.layers.dp_attention':dict(is_allocation_symmetric=lambda:False),
        'sglang.srt.layers.utils.common':dict(strict_contiguous=lambda t:t.contiguous()),
    }.items():
        mod=types.ModuleType(name);mod.__dict__.update(attrs);sys.modules[name]=mod
    sys.modules['sglang.srt.utils.common'].is_gfx1250_supported=lambda:False
    from sglang.kernels.ops.layernorm.mhc import mhc_pre,mhc_post,_mhc_pre_torch,_mhc_post_torch
    for B in (6,24):
        x=torch.randn(B,4,4096,device='cuda',dtype=torch.bfloat16)*.1
        fn=torch.randn(24,16384,device='cuda')*.001
        scale=torch.ones(3,device='cuda');base=torch.randn(24,device='cuda')*.1
        kw=dict(residual=x,fn=fn,hc_scale=scale,hc_base=base,rms_eps=1e-5,
                hc_pre_eps=1e-6,hc_sinkhorn_eps=1e-6,hc_post_mult_value=2.,sinkhorn_repeat=20)
        def pre():return mhc_pre(**kw)
        got=pre();ref=_mhc_pre_torch(**kw)
        errs=[compare(a,b,.005,.015) for a,b in zip(got,ref)]
        graph_check(pre,[x])
        post,comb,inp=pre()
        def postfn():return mhc_post(inp,x,post,comb)
        e=compare(postfn(),_mhc_post_torch(inp,x,post,comb),.005,.015)
        graph_check(postfn,[inp])
        emit(kind='target_mhc_pre_post',B=B,pre=errs,post=e,status='PASS')


@torch.inference_mode()
def sharing():
    from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState as State
    seed=torch.empty(6,2051,device='cuda',dtype=torch.int32)
    indices=torch.arange(24*2051,device='cuda',dtype=torch.int32).reshape(24,2051)
    sel=torch.tensor([0,5,10,15,16,23],device='cuda')
    spec=types.SimpleNamespace(dsa_topk_indices=None,dsa_seed_topk_capture=seed,dsa_seed_topk_select=sel)
    fb=types.SimpleNamespace(spec_info=spec,reuse_dsa_topk_indices=False,
                            forward_mode=types.SimpleNamespace(is_extend=lambda **kw:True))
    state=State(fb,indices)
    def publish():state.publish();return seed
    publish();compare(seed,indices[sel],0,0)
    graph_check(publish,[])
    spec.dsa_topk_indices=seed
    fb.forward_mode.is_extend=lambda **kw:False
    with State.mtp_iteration(fb,keep_carry_seed=True) as carry:
        assert carry.topk_indices is seed
        for _ in range(2):
            carry=State.from_mtp_carry(fb);assert carry.topk_indices is seed;carry.publish()
    assert spec.dsa_topk_indices is None and not fb.reuse_dsa_topk_indices
    spec.dsa_topk_indices=seed
    try:
        with State.mtp_iteration(fb,keep_carry_seed=True):raise RuntimeError('test cleanup')
    except RuntimeError:pass
    assert spec.dsa_topk_indices is None and not fb.reuse_dsa_topk_indices
    emit(kind='shared_seed_select_publish_carry_finally',width=2051,status='PASS')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--group',default='kda',choices=['kda','small','kpool','dsa','acceptance','indexer','mhc','sharing']);args=p.parse_args()
    assert torch.cuda.get_device_capability()==(8,0)
    bootstrap(args.source.resolve());torch.manual_seed(160)
    emit(kind='environment',torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name())
    if args.group=='kda':
        for b,t,h in [(1,2,8),(1,4,8),(6,4,8),(6,6,8),(1,4,64)]:kda(b,t,h)
    elif args.group=='small':small()
    elif args.group=='kpool':kpool()
    elif args.group=='dsa':dsa()
    elif args.group=='acceptance':acceptance()
    elif args.group=='indexer':indexer()
    elif args.group=='mhc':mhc()
    elif args.group=='sharing':sharing()
    emit(kind='complete',group=args.group,status='PASS')

if __name__=='__main__':main()
