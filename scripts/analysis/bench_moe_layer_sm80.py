#!/usr/bin/env python3
"""Measure real 117 layer, its GPU kernels and internal token tiling.

Synthetic block-FP8 checkpoint; GLM TP8 per-rank shapes on one GPU. This does
not simulate TP8 communication or establish a service/chain result.
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

from bench_moe_stream_sm80 import stream_reduce


def emit(**r):print(json.dumps(r),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',nargs='+',type=int,default=[8192,16384])
    p.add_argument('--tiles',nargs='+',type=int,default=[1024,2048,4096])
    p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    path=root/'tests/gpu/test_fp8_moe_humming_117.py'
    spec=importlib.util.spec_from_file_location('moe117_reference_test',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import humming
    emit(kind='environment',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
         humming_source=humming.__file__,arguments=vars(a),
         probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with contextlib.ExitStack() as stack:
        helper.open_runtime(stack)
        checkpoint=helper.make_checkpoint()
        layer=helper.build_layer(checkpoint,True)
        del checkpoint
        import sglang.srt.layers.moe.moe_runner.humming as runner_module
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        original=runner_module.moe_fused_mul_sum
        verify_reduction=False
        def stream_sum(inputs,topk_weights,outputs=None,topk_ids=None,expert_map=None,
                       routed_scaling_factor=None,is_ep=False):
            if inputs.dtype!=torch.bfloat16 or tuple(inputs.shape[1:])!=(9,4096) or is_ep or expert_map is not None:
                return original(inputs,topk_weights,outputs,topk_ids,expert_map,routed_scaling_factor,is_ep)
            m=inputs.shape[0]
            if outputs is None:outputs=torch.empty((m,4096),device=inputs.device,dtype=inputs.dtype)
            stream_reduce[(m*4,)](inputs,topk_weights,outputs,m,4096,9,
                1.0 if routed_scaling_factor is None else routed_scaling_factor,
                1,1024,'.cg','',0,False,num_warps=4,num_stages=1)
            if verify_reduction:
                reference=original(inputs,topk_weights,topk_ids=topk_ids,
                    routed_scaling_factor=routed_scaling_factor,is_ep=is_ep)
                exact=torch.equal(outputs,reference)
                emit(kind='actual_reduction_identity',rows=m,exact=exact,
                     relative_l2=helper.rel(outputs,reference))
                assert exact
            return outputs
        for m in a.rows:
            x,topk=helper.make_inputs(m,928+m)
            def base():return layer(x,topk)
            def candidate():
                runner_module.moe_fused_mul_sum=stream_sum
                try:return layer(x,topk)
                finally:runner_module.moe_fused_mul_sum=original
            with helper.fixed_alignment():
                b=base();b2=base()
                verify_reduction=True
                c=candidate()
                verify_reduction=False
                exact=torch.equal(b,c)
                repeat=helper.rel(b2,b);error=helper.rel(c,b)
                emit(kind='identity',rows=m,stream_exact=exact,
                     baseline_repeat_l2=repeat,candidate_vs_baseline_l2=error)
                assert error<=repeat*1.25+1e-5
            del b,b2,c
            funcs={'baseline':base,'stream_reduce':candidate}
            for tile in a.tiles:
                parts=[]
                for start in range(0,m,tile):
                    t=StandardTopKOutput(topk_weights=topk.topk_weights[start:start+tile],
                        topk_ids=topk.topk_ids[start:start+tile],router_logits=topk.router_logits[start:start+tile])
                    parts.append((x[start:start+tile],t))
                def tiled(parts=parts):return torch.cat([layer(xx,tt) for xx,tt in parts],dim=0)
                # Different routing alignment may change inherited Humming stream-K rounding.
                out=tiled();ref=base()
                emit(kind='tile_numerics',rows=m,tile=tile,relative_l2=helper.rel(out,ref))
                del out,ref
                funcs[f'tile_{tile}']=tiled
            for name,f in funcs.items():
                for _ in range(4):f()
                emit(kind='memory',rows=m,candidate=name,transient_mib=helper.transient_mb(f))
            end=time.monotonic()+1
            while time.monotonic()<end:
                for _ in range(8):base()
                torch.cuda.synchronize()
            times={name:[] for name in funcs};rng=random.Random(117+m)
            for _ in range(a.rounds):
                names=list(funcs);rng.shuffle(names)
                for name in names:
                    times[name].append(helper.measure(funcs[name],inner=8,repeats=1))
            emit(kind='layer_timing',rows=m,rounds=times,
                 medians={name:{k:statistics.median(t[k] for t in ts) for k in ts[0]} for name,ts in times.items()})
            if a.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):base()
                    torch.cuda.synchronize()
                events=[dict(name=e.key,count=e.count,device_us=e.device_time_total,
                             self_device_us=e.self_device_time_total) for e in prof.key_averages() if e.device_time_total]
                emit(kind='profile',rows=m,events=sorted(events,key=lambda e:e['self_device_us'],reverse=True)[:25])
        emit(kind='complete')


if __name__=='__main__':main()
