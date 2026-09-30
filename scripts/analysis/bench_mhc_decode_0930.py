#!/usr/bin/env python3
"""Full mHC post -> FFN pre -> output norm probe; no TP collectives.

Imports exact engine kernels. Replaces only distributed allocator/runtime flag
lookups with a disabled symmetric context for an isolated single GPU probe.
Neither kernel arithmetic nor launch tuning is changed.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ['SGLANG_OPT_DEEPGEMM_HC_PRENORM'] = '0'
os.environ['SGLANG_OPT_USE_TILELANG_MHC_PRE'] = '1'
os.environ['SGLANG_OPT_USE_TILELANG_MHC_POST'] = '1'

import torch
from sglang.kernels.ops.layernorm import mhc

mhc.get_tp_group = lambda: None
mhc.is_allocation_symmetric = lambda: False
mhc.use_symmetric_memory = lambda *args, **kwargs: contextlib.nullcontext()
mhc.is_dsa_prefill_cp_round_robin_split = lambda: False


def metrics(actual, expected):
    a, e = actual.float(), expected.float()
    d = (a-e).abs()
    result = {
        'max_abs': d.max().item(), 'rms_abs': d.square().mean().sqrt().item(),
        'rel_l2': (d.square().sum().sqrt()/e.square().sum().sqrt().clamp_min(1e-30)).item(),
        'changed_fraction': (actual != expected).float().mean().item(),
        'finite': bool(torch.isfinite(actual).all()),
    }
    if actual.dtype == torch.bfloat16 and expected.dtype == torch.bfloat16:
        ai = actual.view(torch.int16).int()
        ei = expected.view(torch.int16).int()
        # Ordered magnitude for negatives, including transitions across zero.
        ao = torch.where(ai < 0, -32768-ai, ai)
        eo = torch.where(ei < 0, -32768-ei, ei)
        result['max_bf16_ulp'] = (ao-eo).abs().max().item()
    return result


def reference(x, residual, post, comb, fn, scale, base, norm, case):
    hc, h = residual.shape[-2:]
    # Independent FP32 equations; comb[old,new], explicit BF16 post boundary.
    r = (post*x.float().unsqueeze(1) + torch.einsum('bij,bih->bjh', comb, residual.float())).bfloat16()
    flat = r.flatten(1).float()
    mixes = (flat@fn.t())*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-5)
    pre = torch.sigmoid(mixes[:,:hc]*scale[0]+base[:hc])+1e-6
    new_post = 2*torch.sigmoid(mixes[:,hc:2*hc]*scale[1]+base[hc:2*hc])
    new_comb = torch.softmax(mixes[:,2*hc:].reshape(-1,hc,hc)*scale[2]+base[2*hc:].reshape(hc,hc),-1)+1e-6
    new_comb = new_comb/(new_comb.sum(-2,keepdim=True)+1e-6)
    for _ in range(19):
        new_comb = new_comb/(new_comb.sum(-1,keepdim=True)+1e-6)
        new_comb = new_comb/(new_comb.sum(-2,keepdim=True)+1e-6)
    weighted = (pre.unsqueeze(-1)*r.float()).sum(1)
    rounded = weighted.bfloat16().float()
    if norm is None:
        y = weighted.bfloat16()
        norm_conventional = y
    else:
        # Match current shared finalizer's FP32 denominator/BF16 numerator.
        y = (rounded*torch.rsqrt(weighted.square().mean(-1,keepdim=True)+1e-5)*norm.float()).bfloat16()
        norm_conventional = (rounded*torch.rsqrt(rounded.square().mean(-1,keepdim=True)+1e-5)*norm.float()).bfloat16()
    return r,new_post.unsqueeze(-1),new_comb,y,norm_conventional


def make_case(m, case):
    torch.manual_seed(20260930 + m)
    h,hc = 4096,4
    x = torch.randn(m,h,device='cuda').bfloat16()
    r = torch.randn(m,hc,h,device='cuda').bfloat16()
    # Asymmetric doubly stochastic-ish coefficients, avoiding trivial identity.
    c = torch.softmax(torch.randn(m,hc,hc,device='cuda'),-1)
    for _ in range(20):
        c = c/c.sum(-2,keepdim=True)
        c = c/c.sum(-1,keepdim=True)
    p = (2*torch.sigmoid(torch.randn(m,hc,1,device='cuda'))).float()
    fn = torch.randn(24,16384,device='cuda')/128
    scale = torch.tensor([0.2,0.15,0.1],device='cuda')
    base = torch.linspace(-0.3,0.4,24,device='cuda')
    norm = (1 + 0.1*torch.randn(h,device='cuda')).bfloat16()
    if case == 'nearzero':
        x.mul_(1e-4); r.mul_(1e-4)
    elif case == 'cancel':
        # Close cancellation across routes, still using nontrivial post input.
        r[:,1] = -r[:,0]; r[:,3] = -r[:,2]
        x.mul_(0.01)
        c = torch.full_like(c,0.25)
    elif case == 'zero':
        x.zero_(); r.zero_()
    return x,r,p,c,fn,scale,base,norm


def functions(args, with_norm=True):
    x,r,p,c,fn,scale,base,norm=args
    if not with_norm:
        norm=None
    def baseline():
        cur = mhc.mhc_post(x,r,p,c)
        post,comb,y = mhc.mhc_pre(cur,fn,scale,base,1e-5,1e-6,1e-6,2.0,20,norm_weight=norm,norm_eps=1e-5 if norm is not None else None)
        return cur,post,comb,y
    def fused():
        return mhc.mhc_fused_post_pre(x,r,p,c,fn,scale,base,1e-5,1e-6,1e-6,2.0,20,norm_weight=norm,norm_eps=1e-5 if norm is not None else None)
    return baseline,fused


def timed(fn, mode, repeat=5, iterations=100):
    for _ in range(12):
        fn()
    torch.cuda.synchronize()
    if mode=='graph':
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs=fn()
        run=graph.replay
    else:
        outputs=None
        run=fn
    values=[]
    for _ in range(repeat):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record(); end.synchronize()
        values.append(start.elapsed_time(end)*1000/iterations)
    return {'median_us':statistics.median(values),'samples_us':values}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--sizes',default='24,32,1,4,8,28')
    a=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=Path(mhc.__file__)
    result={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),
            'allocator':'single GPU ordinary CUDA allocator; symmetric context disabled',
            'baseline':'exact mhc_post + mhc_pre splitK32 + big_fuse with norm; deepgemm off',
            'shape':{'hidden':4096,'hc':4,'sinkhorn':20,'eps':1e-6,'rms_eps':1e-5},'timing':{},'correctness':{}}
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('timing','correctness')}),flush=True)
    for m in map(int,a.sizes.split(',')):
        args=make_case(m,'random'); baseline,fused=functions(args)
        print('compile_start',m,flush=True); b,f=baseline(),fused(); torch.cuda.synchronize()
        entry={}
        for mode in ('graph','eager'):
            entry[mode]={'baseline':timed(baseline,mode),'fused':timed(fused,mode)}
            entry[mode]['speedup']=entry[mode]['baseline']['median_us']/entry[mode]['fused']['median_us']
        result['timing'][str(m)]=entry
        print('timing',m,json.dumps(entry),flush=True); save()
        for case in ('random','nearzero','cancel','zero'):
            args=make_case(m,case)
            for with_norm in (True,False):
                base_fn,fuse_fn=functions(args,with_norm)
                b,f=base_fn(),fuse_fn()
                norm=args[-1] if with_norm else None
                refs=reference(*args[:-1],norm,case)
                label=f'M{m}/{case}/norm{int(with_norm)}'
                result['correctness'][label]={}
                for i,name in enumerate(('residual','post','comb','layer_input')):
                    result['correctness'][label][name]={'fused_vs_baseline':metrics(f[i],b[i]),
                                                       'baseline_vs_ref':metrics(b[i],refs[i]),
                                                       'fused_vs_ref':metrics(f[i],refs[i])}
                result['correctness'][label]['baseline_vs_conventional_bf16_norm']=metrics(b[-1],refs[-1])
                result['correctness'][label]['comb_row_error']=(f[2].sum(-1)-1).abs().max().item()
                result['correctness'][label]['comb_col_error']=(f[2].sum(-2)-1).abs().max().item()
                if not all(item['fused_vs_baseline']['finite'] for name,item in result['correctness'][label].items() if isinstance(item,dict) and 'fused_vs_baseline' in item):
                    raise RuntimeError('nonfinite fused output '+label)
        # Verify replay reads updated inputs, outputs match eager and padded rows independent.
        args=make_case(m,'random'); _,fuse_fn=functions(args)
        for _ in range(3): fuse_fn()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): gout=fuse_fn()
        args[0].mul_(0.7); args[1].mul_(1.1); g.replay()
        eager=fuse_fn(); torch.cuda.synchronize()
        result['correctness'][f'M{m}/replay_updated_input']={name:metrics(gg,ee) for name,gg,ee in zip(('residual','post','comb','layer_input'),gout,eager)}
        if m>1:
            active=max(1,m-3); original=tuple(t.clone() for t in eager)
            args[0][active:].fill_(123); args[1][active:].fill_(-37)
            changed=fuse_fn(); torch.cuda.synchronize()
            result['correctness'][f'M{m}/padded_tail_independence']={name:metrics(cc[:active],oo[:active]) for name,cc,oo in zip(('residual','post','comb','layer_input'),changed,original)}
        save(); print('correctness_done',m,flush=True)
        if m==32 and all(result['timing'].get(str(k),{}).get('graph',{}).get('speedup',0)<0.95 for k in (24,32)):
            result['failfast']='M24 and M32 graph boundary >5% slower; no production wiring justified'
            save(); print(result['failfast'],flush=True); break
    print('COMPLETE',str(output),flush=True)


if __name__=='__main__':
    main()
