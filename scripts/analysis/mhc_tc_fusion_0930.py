#!/usr/bin/env python3
"""Isolated SM80 mHC full-boundary Tensor Core tuning; no engine wiring."""
import argparse
import hashlib
import json
from pathlib import Path

from bench_mhc_decode_0930 import make_case, metrics, reference, timed, functions, mhc, torch

tilelang = None
T = None


def fused_stage0_kernel(token_block, split_k):
    """Preserve TC dot; compute each uniquely owned rounded post K tile once."""
    global tilelang,T
    tilelang=mhc._load_tilelang()
    T=mhc.T
    hidden_block=256
    split_size=16384//split_k
    assert split_size % hidden_block == 0
    num_tokens=T.dynamic('num_tokens')
    @tilelang.jit
    def kernel(
        hidden: T.Tensor((num_tokens,4096),T.bfloat16),
        residual: T.Tensor((num_tokens,4,4096),T.bfloat16),
        post: T.Tensor((num_tokens,4),T.float32),
        comb: T.Tensor((num_tokens,4,4),T.float32),
        fn: T.Tensor((24,16384),T.float32),
        current: T.Tensor((num_tokens,16384),T.bfloat16),
        out_partial: T.Tensor((split_k,num_tokens,32),T.float32),
        sqrsum_partial: T.Tensor((split_k,num_tokens),T.float32),
    ):
        with T.Kernel(T.ceildiv(num_tokens,token_block),split_k,threads=128) as (px,bz):
            out_frag=T.alloc_fragment((token_block,32),T.float32)
            sq_part4=T.alloc_fragment((token_block,4),T.float32)
            T.clear(out_frag)
            T.clear(sq_part4)
            k_base=bz*split_size
            for pz in T.Pipelined(split_size//hidden_block,num_stages=2):
                fn_smem=T.alloc_shared((32,hidden_block),T.float32)
                x_f=T.alloc_fragment((token_block,hidden_block),T.float32)
                x_bf=T.alloc_fragment((token_block,hidden_block),T.bfloat16)
                k_begin=k_base+pz*hidden_block
                route=k_begin//4096
                h_begin=k_begin%4096
                T.copy(fn[0,k_begin],fn_smem)
                for i,j in T.Parallel(token_block,hidden_block):
                    token=px*token_block+i
                    h=h_begin+j
                    if token<num_tokens:
                        x_f[i,j]=post[token,route]*hidden[token,h]
                        for old_route in T.serial(4):
                            x_f[i,j]+=comb[token,old_route,route]*residual[token,old_route,h]
                    else:
                        x_f[i,j]=0
                # Explicitly preserve the standalone post BF16 write/read boundary.
                T.copy(x_f,x_bf)
                T.copy(x_bf,current[px*token_block,k_begin])
                T.copy(x_bf,x_f)
                for jj in T.serial(hidden_block//4):
                    for i,j in T.Parallel(token_block,4):
                        value=x_f[i,jj*4+j]
                        sq_part4[i,j]+=value*value
                T.gemm(x_f,fn_smem,out_frag,transpose_A=False,transpose_B=True,clear_accum=False)
            sq_l=T.alloc_fragment((token_block,),T.float32)
            T.reduce_sum(sq_part4,sq_l)
            for i in T.Parallel(token_block):
                token=px*token_block+i
                if token<num_tokens:
                    sqrsum_partial[bz,token]=sq_l[i]
            for i,j in T.Parallel(token_block,32):
                token=px*token_block+i
                if token<num_tokens:
                    out_partial[bz,token,j]=out_frag[i,j]
    return kernel


def tuned_boundary(args, token_block, split_k, *, fused_post=False):
    x,r,p,c,fn,scale,base,norm = args
    m,h = x.shape
    if fused_post:
        stage0=fused_stage0_kernel(token_block,split_k)
    else:
        stage0,_ = mhc.mhc_pre_gemm_sqrsum_splitk_kernel(24,16384,split_k,token_block,256)
    def run():
        cur = torch.empty_like(r) if fused_post else mhc.mhc_post(x,r,p,c)
        mul = torch.empty((split_k,m,32),device=x.device,dtype=torch.float32)
        sqr = torch.empty((split_k,m),device=x.device,dtype=torch.float32)
        if fused_post:
            stage0(x,r,p.reshape(m,4),c,fn,cur.reshape(m,16384),mul,sqr)
        else:
            stage0(cur.reshape(m,16384),fn,mul,sqr)
        post = torch.empty((m,4),device=x.device,dtype=torch.float32)
        comb = torch.empty((m,16),device=x.device,dtype=torch.float32)
        y = torch.empty((m,h),device=x.device,dtype=torch.bfloat16)
        mhc.mhc_pre_big_fuse_with_norm_tilelang(mul,sqr,scale,base,cur,post,comb,y,norm,
            h,1e-5,1e-6,1e-6,2.0,20,1e-5,split_k,4,32)
        return cur,post.reshape(m,4,1),comb.reshape(m,4,4),y
    return run


def numerical(candidate, baseline, ref):
    result={}
    for n,c,b,r in zip(('residual','post','comb','layer_input'),candidate,baseline,ref):
        result[n]={'vs_baseline':metrics(c,b),'vs_reference':metrics(c,r)}
    result['comb_row_error']=(candidate[2].sum(-1)-1).abs().max().item()
    result['comb_col_error']=(candidate[2].sum(-2)-1).abs().max().item()
    result['passes_envelope']=(all(result[n]['vs_baseline']['finite'] for n in ('residual','post','comb','layer_input'))
        and result['residual']['vs_baseline']['max_abs']==0
        and result['layer_input']['vs_baseline']['rel_l2']<0.002
        and result['comb_row_error']<1e-4 and result['comb_col_error']<1e-4)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--sizes',default='24,28,32,40,48')
    p.add_argument('--phase',choices=('tune','fusion'),default='tune')
    a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if a.phase=='fusion':
        probe_fusion(a)
        return
    source=Path(mhc.__file__)
    result={'phase':'existing TC stage0 tuning','source':str(source),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'hidden_block':256,'token_blocks':[16,32],'split_ks':[16,32,64],
        'allocator':'isolated ordinary CUDA; no TP symmetric allocator or collective',
        'rows':{},'winners':{},'diagnostics':{}}
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    def save(): output.write_text(json.dumps(result,indent=2))
    print('metadata',json.dumps({k:v for k,v in result.items() if k not in ('rows','winners','diagnostics')}),flush=True)
    for m in map(int,a.sizes.split(',')):
        args=make_case(m,'random')
        baseline,_=functions(args)
        b=baseline(); torch.cuda.synchronize()
        ref=reference(*args[:-1],args[-1],'random')
        row={'baseline_before':timed(baseline,'graph'),'configs':{}}
        print('M',m,'baseline',json.dumps(row['baseline_before']),flush=True)
        for token_block in (16,32):
            for split_k in (16,32,64):
                label=f'T{token_block}K{split_k}'
                print('compile_start',m,label,flush=True)
                run=tuned_boundary(args,token_block,split_k)
                c=run(); torch.cuda.synchronize()
                n=numerical(c,b,ref)
                entry={'graph':timed(run,'graph'),'numerical':n}
                entry['speedup_before']=row['baseline_before']['median_us']/entry['graph']['median_us']
                row['configs'][label]=entry
                result['rows'][str(m)]=row; save()
                print('M',m,label,'graph_us',entry['graph']['median_us'],'speedup',entry['speedup_before'],'envelope',n['passes_envelope'],flush=True)
        row['baseline_after']=timed(baseline,'graph')
        baseline_us=(row['baseline_before']['median_us']+row['baseline_after']['median_us'])/2
        for entry in row['configs'].values():
            entry['speedup']=baseline_us/entry['graph']['median_us']
        winner=min((k for k,v in row['configs'].items() if v['numerical']['passes_envelope']),key=lambda k:row['configs'][k]['graph']['median_us'])
        result['winners'][str(m)]={'config':winner,'speedup':row['configs'][winner]['speedup'],'candidate_us':row['configs'][winner]['graph']['median_us'],'baseline_us':baseline_us}
        save(); print('winner',m,json.dumps(result['winners'][str(m)]),flush=True)
    # Qualify best M32 tuning on independent challenging cases and graph inputs.
    winner=result['winners'].get('32',next(iter(result['winners'].values())))['config']
    token_block,split_k=map(int,winner[1:].split('K'))
    for case in ('random','nearzero','cancel','zero'):
        args=make_case(32,case); base,_=functions(args)
        run=tuned_boundary(args,token_block,split_k)
        result['diagnostics'][case]=numerical(run(),base(),reference(*args[:-1],args[-1],case))
    args=make_case(32,'random'); run=tuned_boundary(args,token_block,split_k)
    for _ in range(3): run()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): gout=run()
    args[0].mul_(0.7); args[1].mul_(1.1); graph.replay()
    eager=run(); torch.cuda.synchronize()
    result['diagnostics']['replay_updated_input']={n:metrics(g,e) for n,g,e in zip(('residual','post','comb','layer_input'),gout,eager)}
    before=tuple(t.clone() for t in eager)
    args[0][29:].fill_(123); args[1][29:].fill_(-37)
    after=run(); torch.cuda.synchronize()
    result['diagnostics']['padded_tail_independence']={n:metrics(v[:29],b[:29]) for n,v,b in zip(('residual','post','comb','layer_input'),after,before)}
    result['promising_for_post_fusion']=all(result['winners'][str(m)]['speedup']>=1.03 for m in (24,28,32))
    save(); print('COMPLETE',str(output),'promising_for_post_fusion',result['promising_for_post_fusion'],flush=True)


def probe_fusion(a):
    result={'phase':'TC stage0 post-map fusion','source_sha256':hashlib.sha256(Path(mhc.__file__).read_bytes()).hexdigest(),
            'prototype_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'rows':{},'diagnostics':{},'note':'Synthetic coefficients, isolated ordinary allocator; not TP8 ability evidence'}
    output=Path(a.output)
    def save(): output.write_text(json.dumps(result,indent=2))
    # Warm GPU clocks consistently before measured interleaved comparisons.
    args=make_case(32,'random'); warm,_=functions(args)
    for _ in range(1000): warm()
    torch.cuda.synchronize()
    for m in map(int,a.sizes.split(',')):
        args=make_case(m,'random'); base,_=functions(args)
        b=base(); ref=reference(*args[:-1],args[-1],'random')
        row={}
        for split_k in (32,64):
            label=f'T32K{split_k}'
            print('compile_start',m,label,flush=True)
            run=tuned_boundary(args,32,split_k,fused_post=True)
            candidate=run(); torch.cuda.synchronize()
            n=numerical(candidate,b,ref)
            entry={'numerical':n}
            row[label]=entry; result['rows'][str(m)]=row; save()
            if not n['passes_envelope']:
                print('NUMERIC_REJECT',m,label,json.dumps(n),flush=True)
                continue
            pairs=[]
            for _ in range(3):
                # Warmed ABBA mitigates clock ramp and scan-order bias.
                b1=timed(base,'graph',repeat=3,iterations=300)
                f1=timed(run,'graph',repeat=3,iterations=300)
                f2=timed(run,'graph',repeat=3,iterations=300)
                b2=timed(base,'graph',repeat=3,iterations=300)
                pairs.append({'baseline_us':(b1['median_us']+b2['median_us'])/2,'fused_us':(f1['median_us']+f2['median_us'])/2,
                              'raw':{'b1':b1,'f1':f1,'f2':f2,'b2':b2}})
            entry['paired']=pairs
            entry['baseline_us']=sum(v['baseline_us'] for v in pairs)/len(pairs)
            entry['fused_us']=sum(v['fused_us'] for v in pairs)/len(pairs)
            entry['speedup']=entry['baseline_us']/entry['fused_us']
            print('result',m,label,'base_us',entry['baseline_us'],'fused_us',entry['fused_us'],'speedup',entry['speedup'],flush=True)
            save()
        if m==32 and all(v.get('speedup',0)<0.97 for r in result['rows'].values() for v in r.values()):
            result['failfast']='All tested TC fusion variants >3% slower on M24/28/32'
            print(result['failfast'],flush=True); save(); break
    # Qualify the best valid tested variant at M32 beyond random data.
    row=result['rows'].get('32',next(iter(result['rows'].values())))
    eligible={k:v for k,v in row.items() if v['numerical']['passes_envelope']}
    if eligible:
        best=min(eligible,key=lambda k:eligible[k].get('fused_us',float('inf')))
        split_k=int(best.split('K')[1])
        for case in ('random','nearzero','cancel','zero'):
            args=make_case(32,case); base,_=functions(args)
            run=tuned_boundary(args,32,split_k,fused_post=True)
            result['diagnostics'][case]=numerical(run(),base(),reference(*args[:-1],args[-1],case))
        args=make_case(32,'random'); run=tuned_boundary(args,32,split_k,fused_post=True)
        for _ in range(3): run()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): gout=run()
        args[0].mul_(0.7); args[1].mul_(1.1); graph.replay()
        eager=run(); torch.cuda.synchronize()
        result['diagnostics']['replay_updated_input']={n:metrics(g,e) for n,g,e in zip(('residual','post','comb','layer_input'),gout,eager)}
        before=tuple(t.clone() for t in eager)
        args[0][29:].fill_(123); args[1][29:].fill_(-37)
        after=run(); torch.cuda.synchronize()
        result['diagnostics']['padded_tail_independence']={n:metrics(v[:29],b[:29]) for n,v,b in zip(('residual','post','comb','layer_input'),after,before)}
    save(); print('COMPLETE',str(output),flush=True)


if __name__=='__main__': main()
