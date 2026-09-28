#!/usr/bin/env python3
"""Feasibility: combine down projection and FP32 router reduction on A100.

Synthetic true TP8 per-rank shape. No production dispatch, no checkpoint change,
no extra model quantization. The final BF16 GEMM rounding remains present.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import time

import torch
import triton
import triton.language as tl

from prepare_humming_reduce_probe import prepare


@triton.jit
def cast_sum(X,Y,N:tl.constexpr,B:tl.constexpr):
    off=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(Y+off,tl.load(X+off,off<N,0),off<N)


def emit(**r):print(json.dumps(r),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,nargs='+',default=[8192,16384])
    p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--variants',type=int,nargs='+',default=[117,118])
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    assert set(a.variants).issubset({117,118,119,120,121,122})
    emit(kind='environment', arguments=vars(a), gpu=torch.cuda.get_device_name(),
         probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         status='rejected research prototypes; synchronization/error diagnostics hardened after initial timings')
    emit(kind='headers',**prepare(root))
    spec=importlib.util.spec_from_file_location('reference117',root/'tests/gpu/test_fp8_moe_humming_117.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from humming.layer import HummingMethod
    from humming.config import GemmType
    from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
    with contextlib.ExitStack() as stack:
        helper.open_runtime(stack)
        ckpt=helper.make_checkpoint();layer=helper.build_layer(ckpt,True);del ckpt
        core=layer.quant_method.runner_core
        configs=core.get_humming_gemm_configs(GemmType.INDEXED)
        for m in a.rows:
            _,topk=helper.make_inputs(m,173928+m)
            g=torch.Generator(device='cuda').manual_seed(117928+m)
            activation=torch.randn(m*9,256,device='cuda',dtype=torch.bfloat16,generator=g)
            scratch=torch.empty(m*9,4096,device='cuda',dtype=torch.bfloat16)
            expected=torch.empty(m,4096,device='cuda',dtype=torch.bfloat16)
            fp=scratch.view(torch.float32).flatten()[:m*4096]
            auxiliary=torch.zeros(m*9+m*16+1+m*4096//2,device='cuda',dtype=torch.float32)
            auxiliary[:m*9].copy_(topk.topk_weights.flatten())
            locks=auxiliary[m*9:m*9+m*16].view(torch.int32)
            error_word=auxiliary[m*9+m*16:m*9+m*16+1].view(torch.int32)
            out=auxiliary[m*9+m*16+1:].view(torch.bfloat16).view(m,4096)
            _,kw=core._prepare_indexed_gemm_kwargs(topk.topk_ids)
            base_cfg=next(c for lo,hi,c in configs['w2_tuning_config'] if lo<m*9<=hi)
            assert not base_cfg['use_stream_k']
            emit(kind='config',rows=m,config=base_cfg,scratch_mib=scratch.numel()*2/2**20,
                 accumulator_mib=fp.numel()*4/2**20)
            def gemm(cfg=None):
                args=dict(kw)
                if cfg is not None:
                    args['tuning_config']=json.dumps(cfg)
                HummingMethod.forward_layer(layer,activation,outputs=scratch,
                    input_scale=(auxiliary if cfg['raster_group_m']>=119 else topk.topk_weights) if cfg is not None else None,
                    sublayer_name='w2',**args)
            def baseline():
                gemm()
                return moe_fused_mul_sum(scratch.view(m,9,4096),topk.topk_weights,
                                        outputs=out,routed_scaling_factor=2.5)
            baseline();expected.copy_(out)
            ids=torch.linspace(0,m-1,32,device='cuda').long().unique()
            terms=scratch.view(m,9,4096)[ids].double()*(topk.topk_weights[ids].double()*2.5)[:,:,None]
            oracle=terms.sum(1)
            tol=oracle.abs()*2**-8+terms.abs().sum(1)*2**-20+1e-7
            base_err=((expected[ids].double()-oracle).norm()/oracle.norm()).item()
            base_ratio=((expected[ids].double()-oracle).abs()/tol).max().item()
            assert base_ratio<=1
            funcs={'baseline':baseline}
            for raster in a.variants:
                cfg=dict(base_cfg,raster_group_m=raster)
                def fused(cfg=cfg):
                    raster=cfg['raster_group_m']
                    if raster<121:fp.zero_()
                    if raster>=119:
                        locks.zero_()
                        error_word.zero_()
                    gemm(cfg)
                    if raster<121:
                        cast_sum[(triton.cdiv(m*4096,1024),)](fp,out,m*4096,1024,num_warps=4)
                    return out
                fused();torch.cuda.synchronize()
                assert not error_word.any(), 'bounded lock acquisition exhausted'
                if raster>=121:assert (locks==9).all(), 'row not completed exactly nine times'
                else:assert not locks.any(), 'unreleased lock'
                exact_bits=torch.equal(out.view(torch.int16),expected.view(torch.int16))
                if raster>=121:assert exact_bits
                error=((out[ids].double()-oracle).norm()/oracle.norm()).item()
                ratio=((out[ids].double()-oracle).abs()/tol).max().item()
                emit(kind='numerics',rows=m,raster=raster,baseline_l2=base_err,candidate_l2=error,
                     baseline_oracle_ratio=base_ratio,candidate_oracle_ratio=ratio,
                     vs_baseline_l2=helper.rel(out,expected),exact_bits=exact_bits,
                     finite=bool(torch.isfinite(out).all()))
                assert ratio<=1 and error<=base_err*1.05+1e-6 and torch.isfinite(out).all()
                funcs[f'epilogue_{raster}']=fused
            for f in funcs.values():
                for _ in range(5):f()
            deadline=time.monotonic()+1
            while time.monotonic()<deadline:
                for _ in range(8):baseline()
                torch.cuda.synchronize()
            times={k:[] for k in funcs};rng=random.Random(m)
            for _ in range(a.rounds):
                keys=list(funcs);rng.shuffle(keys)
                for key in keys:times[key].append(helper.measure(funcs[key],inner=8,repeats=1))
            emit(kind='timing',rows=m,rounds=times,
                 medians={k:{c:statistics.median(t[c] for t in ts) for c in ts[0]} for k,ts in times.items()})
            for name,f in funcs.items():
                if name=='baseline':continue
                f();torch.cuda.synchronize()
                assert not error_word.any(), 'bounded lock acquisition exhausted after timing'
                if int(name.split('_')[-1])>=121:
                    assert (locks==9).all()
                    assert torch.equal(out.view(torch.int16),expected.view(torch.int16))
                else:assert not locks.any()
        emit(kind='complete')
        torch.distributed.destroy_process_group()


if __name__=='__main__':main()
