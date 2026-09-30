#!/usr/bin/env python3
"""Isolated portable TokenSpeed CUDA finalizer probe; exact GLM contracts."""
import argparse
import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess

from bench_mhc_decode_0930 import mhc,torch,make_case,metrics,reference,timed,functions
from mhc_tc_fusion_0930 import numerical


def load_kernel(source,fast_math):
    lib=source.with_suffix('.fast.so' if fast_math else '.so')
    nvcc=Path(os.environ['CUDA_HOME'])/'bin/nvcc'
    cmd=[str(nvcc),'-shared','-Xcompiler','-fPIC','-arch=sm_80','-O3','-lineinfo','--cudart=shared',str(source),'-o',str(lib)]
    if fast_math:
        cmd.insert(1,'-use_fast_math')
    print('compile',json.dumps(cmd),flush=True)
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    print(p.stdout,flush=True)
    p.check_returncode()
    module=ctypes.CDLL(str(lib))
    fn=module.ts_mhc_big_fuse
    fn.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*6+[ctypes.c_float]*3+[ctypes.c_int,ctypes.c_void_p]
    fn.restype=ctypes.c_int
    return fn


def finalizer(fn,args,cur,mul,sqr,block_size,with_norm=True):
    m,h=args[0].shape
    scale,base,norm=args[5:]
    post=torch.empty((m,4),device='cuda',dtype=torch.float32)
    comb=torch.empty((m,16),device='cuda',dtype=torch.float32)
    y=torch.empty((m,h),device='cuda',dtype=torch.bfloat16)
    if fn is None:
        if with_norm:
            mhc.mhc_pre_big_fuse_with_norm_tilelang(mul,sqr,scale,base,cur,post,comb,y,norm,
                h,1e-5,1e-6,1e-6,2.,20,1e-5,mul.shape[0],4,mul.shape[-1])
        else:
            mhc.mhc_pre_big_fuse_tilelang(mul,sqr,scale,base,cur,post,comb,y,
                h,1e-5,1e-6,1e-6,2.,20,mul.shape[0],4,mul.shape[-1])
    else:
        pointers=[mul,sqr,cur,scale,base,post,comb,y,norm if with_norm else None]
        rc=fn(*(ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p() for t in pointers),
            m,h,mul.shape[0],mul.shape[-1],block_size,20,1e-5,1e-6,1e-5,int(with_norm),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc:
            raise RuntimeError('native CUDA launch failed code '+str(rc))
    return cur,post.reshape(m,4,1),comb.reshape(m,4,4),y


@functools.cache
def stage0_kernel(n_splits):
    return mhc.mhc_pre_gemm_sqrsum_splitk_kernel(24,16384,n_splits,32,256)[0]


def prepare(args,n_splits=32):
    x,r,p,c,fn,scale,base,norm=args
    m=x.shape[0]
    cur=mhc.mhc_post(x,r,p,c)
    mul=torch.empty((n_splits,m,32),device='cuda',dtype=torch.float32)
    sqr=torch.empty((n_splits,m),device='cuda',dtype=torch.float32)
    stage0=stage0_kernel(n_splits)
    stage0(cur.reshape(m,16384),fn,mul,sqr)
    return cur,mul,sqr


def full_boundary(fn,args,block_size):
    def run():
        cur,mul,sqr=prepare(args)
        return finalizer(fn,args,cur,mul,sqr,block_size)
    return run


def compare(baseline,candidate,repeat=3):
    pairs=[]
    for _ in range(repeat):
        b1=timed(baseline,'graph',repeat=3,iterations=300)
        c1=timed(candidate,'graph',repeat=3,iterations=300)
        c2=timed(candidate,'graph',repeat=3,iterations=300)
        b2=timed(baseline,'graph',repeat=3,iterations=300)
        pairs.append({'baseline_us':(b1['median_us']+b2['median_us'])/2,
                      'candidate_us':(c1['median_us']+c2['median_us'])/2,
                      'raw':{'b1':b1,'c1':c1,'c2':c2,'b2':b2}})
    b=sum(v['baseline_us'] for v in pairs)/len(pairs)
    c=sum(v['candidate_us'] for v in pairs)/len(pairs)
    return {'baseline_us':b,'candidate_us':c,'speedup':b/c,'pairs':pairs}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--sizes',default='24,28,32,40,48,128,1024')
    p.add_argument('--fast-math',action='store_true',help='match upstream TokenSpeed setup.py -use_fast_math')
    p.add_argument('--block',type=int,choices=(128,256,512),help='qualify one uniform block size')
    a=p.parse_args()
    source=Path(__file__).with_name('ts_mhc_finalizer_0930.cu')
    fn=load_kernel(source,a.fast_math)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    result={'phase':'TokenSpeed portable CUDA mHC finalizer','gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'engine_source_sha256':hashlib.sha256(Path(mhc.__file__).read_bytes()).hexdigest(),
        'compile_fast_math':a.fast_math,
        'adapted_cuda_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'adaptations':['projection leading row width accepts24/32','norm denominator uses unrounded FP32 weighted values','isolated C ABI launches SM80 without PDL'],
        'baseline':'current splitK32 unchanged; only finalizer changed','rows':{},'diagnostics':{}}
    output=Path(a.output)
    def save():output.write_text(json.dumps(result,indent=2))
    for m in map(int,a.sizes.split(',')):
        args=make_case(m,'random');cur,mul,sqr=prepare(args)
        base_stage=lambda:finalizer(None,args,cur,mul,sqr,256)
        baseline,_=functions(args); b=baseline()
        ref=reference(*args[:-1],args[-1],'random')
        row={}
        for block in ((a.block,) if a.block is not None else (128,256,512)):
            print('start',m,'block',block,flush=True)
            stage=lambda:finalizer(fn,args,cur,mul,sqr,block)
            full=full_boundary(fn,args,block)
            c=full();torch.cuda.synchronize()
            n=numerical(c,b,ref)
            entry={'numerical':n}
            row[str(block)]=entry;result['rows'][str(m)]=row;save()
            if not n['passes_envelope']:
                print('NUMERIC_REJECT',m,block,json.dumps(n),flush=True)
                continue
            entry['stage']=compare(base_stage,stage)
            entry['full_boundary']=compare(baseline,full)
            print('result',m,block,'stage',entry['stage']['baseline_us'],entry['stage']['candidate_us'],
                  'full',entry['full_boundary']['baseline_us'],entry['full_boundary']['candidate_us'],
                  'speedup',entry['full_boundary']['speedup'],flush=True)
            save()
    # Choose one constant block size across decode shapes, not per-M routing.
    sizes=[str(m) for m in (24,28,32,40,48) if str(m) in result['rows']]
    blocks=(str(a.block),) if a.block is not None else ('128','256','512')
    viable=[block for block in blocks if all(result['rows'][m][block]['numerical']['passes_envelope'] for m in sizes)]
    if not viable:
        result['rejected']='No constant block size passed numerical envelope'
        save();print('COMPLETE',str(output),flush=True);return
    chosen=max(viable,key=lambda block:min(result['rows'][m][block]['full_boundary']['speedup'] for m in sizes))
    block=int(chosen);result['chosen_constant_block']=block
    for case in ('random','nearzero','cancel','zero'):
        args=make_case(32,case);cur,mul,sqr=prepare(args)
        for norm in (True,False):
            b=finalizer(None,args,cur,mul,sqr,256,norm)
            c=finalizer(fn,args,cur,mul,sqr,block,norm)
            result['diagnostics'][f'{case}/norm{int(norm)}']=numerical(c,b,reference(*args[:-1],args[-1] if norm else None,case))
    # Both projection strides and split counts, including poison padding columns.
    args=make_case(32,'random')
    for nsplit in (32,64):
        cur,mul,sqr=prepare(args,nsplit)
        mul[...,24:].fill_(float('nan'))
        for width in (32,24):
            proj=mul if width==32 else mul[...,:24].contiguous()
            b=finalizer(None,args,cur,proj,sqr,256)
            c=finalizer(fn,args,cur,proj,sqr,block)
            result['diagnostics'][f'splits{nsplit}/stride{width}']=numerical(c,b,reference(*args[:-1],args[-1],'random'))
    args=make_case(32,'random');run=full_boundary(fn,args,block)
    for _ in range(3):run()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):gout=run()
    args[0].mul_(0.7);args[1].mul_(1.1);graph.replay()
    eager=run();torch.cuda.synchronize()
    result['diagnostics']['replay_updated_input']={n:metrics(g,e) for n,g,e in zip(('residual','post','comb','layer_input'),gout,eager)}
    before=tuple(t.clone() for t in eager)
    args[0][29:].fill_(123);args[1][29:].fill_(-37)
    after=run();torch.cuda.synchronize()
    result['diagnostics']['padded_tail_independence']={n:metrics(v[:29],b[:29]) for n,v,b in zip(('residual','post','comb','layer_input'),after,before)}
    save();print('COMPLETE',str(output),'chosen_constant_block',block,flush=True)


if __name__=='__main__':main()
