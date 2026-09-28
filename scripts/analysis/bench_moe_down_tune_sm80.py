#!/usr/bin/env python3
"""Tune down-GEMM + reduction at actual large-row TP8 shapes, unchanged math.

Keep the indexed row block height so no extra routing sort is needed. Compare
against an FP64 reference using the original FP8 checkpoint and FP32 scales.
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


def emit(**r):print(json.dumps(r),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,nargs='+',default=[8192,16384])
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--sweep',choices=['initial','occupancy'],default='initial')
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('reference117',root/'tests/gpu/test_fp8_moe_humming_117.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from humming.layer import HummingMethod
    from humming.config import GemmType
    from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
    import humming
    emit(kind='environment', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         humming_source=humming.__file__, arguments=vars(a),
         probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with contextlib.ExitStack() as stack:
        helper.open_runtime(stack)
        ckpt=helper.make_checkpoint();layer=helper.build_layer(ckpt,True)
        del ckpt['w13'],ckpt['s13']
        core=layer.quant_method.runner_core
        configs=core.get_humming_gemm_configs(GemmType.INDEXED)
        for m in a.rows:
            _,topk=helper.make_inputs(m,173928+m)
            g=torch.Generator(device='cuda').manual_seed(117928+m)
            activation=torch.randn(m*9,256,device='cuda',dtype=torch.bfloat16,generator=g)
            scratch=torch.empty(m*9,4096,device='cuda',dtype=torch.bfloat16)
            expected=torch.empty(m,4096,device='cuda',dtype=torch.bfloat16)
            out=torch.empty_like(expected)
            _,kw=core._prepare_indexed_gemm_kwargs(topk.topk_ids)
            default=next(c for lo,hi,c in configs['w2_tuning_config'] if lo<m*9<=hi)
            bm=default['block_shape'][0]
            emit(kind='config',rows=m,default=default)
            def run(cfg):
                args=dict(kw,tuning_config=json.dumps(cfg))
                HummingMethod.forward_layer(layer,activation,outputs=scratch,sublayer_name='w2',**args)
                moe_fused_mul_sum(scratch.view(m,9,4096),topk.topk_weights,
                                 outputs=out,routed_scaling_factor=2.5)
                return out
            run(default);expected.copy_(out)
            ids=torch.linspace(0,m-1,16,device='cuda').long().unique()
            routes=topk.topk_ids[ids].cpu().tolist()
            sampled_x=activation.view(m,9,256)[ids].double()
            weights=topk.topk_weights[ids].double()*2.5
            oracle=torch.zeros(len(routes),4096,device='cuda',dtype=torch.float64)
            for row,experts in enumerate(routes):
                for slot,expert in enumerate(experts):
                    w=helper.dequantize(ckpt['w2'][expert:expert+1],ckpt['s2'][expert:expert+1])[0].double()
                    oracle[row] += (w@sampled_x[row,slot])*weights[row,slot]
            base_error=((expected[ids].double()-oracle).norm()/oracle.norm()).item()
            emit(kind='baseline_oracle',rows=m,relative_l2=base_error)
            assert base_error<0.01
            variants={'baseline':default}
            if a.sweep=='initial':
                shapes=[(128,32,bm//2,64,64),(128,64,bm//2,64,64),
                        (256,64,bm//2,64,64),(128,32,bm,64,64),
                        (256,64,bm,64,64),(128,64,bm//2,128,64)]
                schedules=[(2,1),(3,1),(2,2)]
            else:
                shapes=[(128,64,64,64,64),(128,32,128,64,64),
                        (128,64,32,64,64),(128,128,32,64,64),
                        (64,64,64,64,64),(256,64,32,64,64)]
                schedules=[(2,2),(3,2),(4,2),(2,3)]
            for bn,wn,wm,bk,wk in shapes:
                for stages,ctas in schedules:
                    cfg=dict(default,block_shape=[bm,bn,bk],warp_shape=[wm,wn,wk],
                             num_stages=stages,num_ctas_per_sm=ctas)
                    name=f'n{bn}_wn{wn}_wm{wm}_k{bk}_s{stages}_c{ctas}'
                    try:
                        run(cfg);torch.cuda.synchronize()
                    except (AssertionError,RuntimeError,ValueError) as e:
                        # A broken CUDA context invalidates every subsequent result.
                        if any(s in str(e).lower() for s in ('illegal memory', 'device-side assert',
                                                            'unspecified launch failure', 'misaligned address')):
                            raise
                        emit(kind='config_rejected',rows=m,name=name,error=str(e)[-1000:]);continue
                    error=((out[ids].double()-oracle).norm()/oracle.norm()).item()
                    emit(kind='numerics',rows=m,name=name,config=cfg,relative_l2=error,
                         vs_baseline_l2=helper.rel(out,expected),finite=bool(torch.isfinite(out).all()))
                    assert error<=base_error*1.25+1e-5 and torch.isfinite(out).all()
                    variants[name]=cfg
            for cfg in variants.values():
                for _ in range(5):run(cfg)
            end=time.monotonic()+1
            while time.monotonic()<end:
                for _ in range(8):run(default)
                torch.cuda.synchronize()
            times={name:[] for name in variants};rng=random.Random(m)
            for _ in range(a.rounds):
                names=list(variants);rng.shuffle(names)
                for name in names:
                    times[name].append(helper.measure(lambda:run(variants[name]),inner=8,repeats=1)['gpu_ms'])
            emit(kind='timing',rows=m,rounds=times,medians={k:statistics.median(v) for k,v in times.items()})
            # A fresh CUDA graph removes Python enqueue gaps from the same kernels.
            best=sorted(times,key=lambda k:statistics.median(times[k]))[:4]
            graph_times={}
            for name in dict.fromkeys(['baseline']+best):
                graph=torch.cuda.CUDAGraph()
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run(variants[name])
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph,stream=stream):
                    for _ in range(8):run(variants[name])
                graph.replay();torch.cuda.synchronize()
                graph_times[name]=helper.measure(graph.replay,inner=4,repeats=5)['gpu_ms']/8
                del graph
            emit(kind='graph_timing',rows=m,medians=graph_times)
        emit(kind='complete')
        torch.distributed.destroy_process_group()


if __name__=='__main__':main()
