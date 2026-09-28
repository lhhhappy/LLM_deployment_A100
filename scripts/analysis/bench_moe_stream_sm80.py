#!/usr/bin/env python3
"""Large-row MoE reduction: cache policy, CTA order and persistent streaming.

Keep serial FP32 expert accumulation and BF16 output exactly as in the engine.
This is a diagnostic, not a dispatch change. Randomized paired timing includes
L2-cold events and graph replay; the graph never fits these inputs in L2.
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
def stream_reduce(X, W, Y, M, H: tl.constexpr, TK: tl.constexpr,
                  SCALE: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                  LOAD: tl.constexpr, STORE: tl.constexpr, ORDER: tl.constexpr,
                  PERSIST: tl.constexpr):
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    nrow = tl.cdiv(M, BM)
    ncol: tl.constexpr = tl.cdiv(H, BK)
    end = nrow * ncol if PERSIST else tl.program_id(0) + 1
    for pid in range(tl.program_id(0), end, tl.num_programs(0)):
        if ORDER == 0:
            row = pid // ncol * BM + rm
            col = pid % ncol * BK + rk
        else:
            row = pid % nrow * BM + rm
            col = pid // nrow * BK + rk
        mask = (row[:, None] < M) & (col[None, :] < H)
        acc = tl.zeros((BM, BK), tl.float32)
        for e in tl.static_range(TK):
            weight = tl.load(W + row * TK + e, row < M, 0).to(tl.float32) * SCALE
            value = tl.load(X + (row[:, None] * TK + e) * H + col[None, :],
                            mask, 0, cache_modifier=LOAD).to(tl.float32)
            acc += value * weight[:, None]
        tl.store(Y + row[:, None] * H + col[None, :], acc, mask, cache_modifier=STORE)


def emit(**r):
    print(json.dumps(r), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', nargs='+', type=int, default=[8192, 16384])
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--calls', type=int, default=7)
    p.add_argument('--configs', help='JSON [BM,BK,warps,load,store,SM_waves,order] list')
    a = p.parse_args()
    from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum_kernel, _heuristic_config
    src = Path(inspect.getfile(_heuristic_config)).resolve()
    assert src.is_relative_to(Path(__file__).resolve().parents[2] / 'engine')
    sms = torch.cuda.get_device_properties('cuda').multi_processor_count
    configs = json.loads(a.configs) if a.configs else [
        [bm, bk, w, load, store, waves, order]
        for bm, bk, w in [(1,512,4), (1,1024,4), (2,1024,4), (8,1024,16)]
        for load, store in [('', ''), ('.cg', ''), ('.cg', '.cs')]
        for waves, order in [(0,0), (0,1), (2,0), (4,0)]
    ]
    emit(kind='environment', gpu=torch.cuda.get_device_name(), sms=sms,
         torch=torch.__version__, triton=triton.__version__, arguments=vars(a),
         source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
         probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    flush = torch.empty(96 << 20, device='cuda', dtype=torch.uint8)
    for m in a.rows:
        g = torch.Generator(device='cuda').manual_seed(117928 + m)
        x = torch.randn(m,9,4096,device='cuda',dtype=torch.bfloat16,generator=g)
        w = torch.rand(m,9,device='cuda',generator=g)
        w /= w.sum(1,keepdim=True)
        expected = torch.empty(m,4096,device='cuda',dtype=x.dtype)
        out = torch.empty_like(expected)
        bm,bk,nw,stages = _heuristic_config(m,9,4096,2)
        def baseline(dest=out):
            return moe_fused_mul_sum_kernel[(triton.cdiv(4096,bk),triton.cdiv(m,bm))](
                x,w,dest,None,None,m,9*4096,False,False,9,4096,2.5,bm,bk,
                num_warps=nw,num_stages=stages)
        baseline(expected)
        valid = {'baseline':baseline}
        for cfg in configs:
            cbm,cbk,cw,load,store,waves,order = cfg
            def f(cfg=cfg):
                cbm,cbk,cw,load,store,waves,order = cfg
                grid = sms * waves if waves else triton.cdiv(m,cbm)*triton.cdiv(4096,cbk)
                return stream_reduce[(grid,)](x,w,out,m,4096,9,2.5,cbm,cbk,load,store,order,waves>0,
                                             num_warps=cw,num_stages=1)
            k=f();torch.cuda.synchronize()
            exact = torch.equal(out,expected)
            emit(kind='correctness',rows=m,config=cfg,exact=exact,registers=k.n_regs,spills=k.n_spills)
            if exact:valid[str(cfg)]=f
        times={k:[] for k in valid}
        rng=random.Random(117928+m)
        for f in valid.values():
            for _ in range(3):f()
        for _ in range(a.rounds):
            names=list(valid);rng.shuffle(names)
            for name in names:
                vals=[]
                for _ in range(a.calls):
                    flush.fill_(1)
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();valid[name]();end.record();end.synchronize()
                    vals.append(start.elapsed_time(end))
                times[name].append(statistics.median(vals))
        med={k:statistics.median(v) for k,v in times.items()}
        emit(kind='timing',rows=m,medians_ms=med,rounds_ms=times,
             nominal_io_bytes=m*4096*2*10+m*9*4)
        selected=['baseline']+[k for k in sorted(med,key=med.get)[:4] if k!='baseline']
        graphs={}
        for name in selected:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):valid[name]()
            graphs[name]=graph
        samples={name:[] for name in selected}
        for _ in range(15):
            rng.shuffle(selected)
            for name in selected:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graphs[name].replay();end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end)/8)
        emit(kind='graph',rows=m,per_call_samples_ms=samples)
    emit(kind='complete')


if __name__=='__main__':main()
