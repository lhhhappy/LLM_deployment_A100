#!/usr/bin/env python3
"""A100 operator-only cold/warm compiler+autotuner receipts; no model/service.

Use an EMPTY task-local TRITON_CACHE_DIR. Inspect actual JITFunction cache-hook
misses and Autotuner._bench calls, plus new disk artifacts, never infer hits
from latency alone. A second process tests persistent artifacts separately.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import torch
import triton
from triton.runtime.jit import JITFunction
from triton.runtime.autotuner import Autotuner


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def bootstrap(root):
    for name, rel in [('sglang',''), ('sglang.srt','srt'), ('sglang.kernels','kernels'),
                      ('sglang.srt.utils','srt/utils'), ('sglang.kernels.jit','kernels/jit')]:
        mod = types.ModuleType(name); mod.__path__ = [str(root/rel)]; sys.modules[name] = mod
    utils = sys.modules['sglang.srt.utils']
    utils.is_cpu = utils.is_npu = utils.cpu_has_amx_support = lambda: False
    utils.cdiv = triton.cdiv
    utils.next_power_of_2 = triton.next_power_of_2
    jit = types.ModuleType('sglang.kernels.jit.utils')
    jit.is_arch_support_pdl = lambda: False
    sys.modules[jit.__name__] = jit
    common = types.ModuleType('sglang.srt.utils.common')
    common.torch_release = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    sys.modules[common.__name__] = common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--persistent', action='store_true')
    args = parser.parse_args()
    cache = Path(os.environ['TRITON_CACHE_DIR'])
    bootstrap(args.source.resolve())
    kda = module('indexer113', args.source/'srt/layers/attention/dsa/sm80_indexer_kernels.py')
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    import triton.knobs as knobs
    counts = {'jit_misses':0, 'bench_calls':0, 'compiled':0, 'disk_hits':0}
    observed = []
    previous_hook = knobs.runtime.jit_cache_hook
    def hook(*a, **kw):
        counts['jit_misses'] += 1
        observed.append({'function':str(kw.get('fn')), 'compile':kw.get('compile')})
        return previous_hook(*a, **kw) if previous_hook else None
    knobs.runtime.jit_cache_hook = hook
    previous_bench = Autotuner._bench
    def bench(self, *a, **kw):
        counts['bench_calls'] += 1
        return previous_bench(self, *a, **kw)
    Autotuner._bench = bench
    previous_listener = knobs.compilation.listener
    def listener(**kw):
        counts['disk_hits' if kw['cache_hit'] else 'compiled'] += 1
        if previous_listener:
            previous_listener(**kw)
    knobs.compilation.listener = listener
    torch.manual_seed(150)
    def fp8(*shape):
        return (torch.randn(shape, device='cuda')*.1).to(torch.float8_e4m3fn)
    def prefill(nq, nk, clean=True):
        q,k = fp8(nq,32,128), fp8(nk,128)
        scales = torch.ones(nk, device='cuda'); w = torch.randn(nq,32,device='cuda')
        ks = torch.zeros(nq,device='cuda',dtype=torch.int32)
        ke = torch.full_like(ks,nk)
        return lambda: kda.fp8_mqa_logits(q,(k,scales),w,ks,ke,clean)
    def decode(batch):
        nk=1024; pages=nk//64
        raw=torch.zeros((pages,64,1,132),device='cuda',dtype=torch.uint8)
        flat=raw.reshape(pages,-1)
        flat[:,:8192]=fp8(pages,64,128).view(torch.uint8).reshape(pages,-1)
        flat[:,8192:]=torch.ones(pages,64,device='cuda').view(torch.uint8).reshape(pages,-1)
        q=fp8(batch,1,32,128); w=torch.ones(batch,32,device='cuda')
        ctx=torch.full((batch,),nk,device='cuda',dtype=torch.int32)
        bt=torch.arange(pages,device='cuda',dtype=torch.int32).repeat(batch,1)
        return lambda: kda.fp8_paged_mqa_logits(q,raw,w,ctx,bt,None,nk)
    def attention(n):
        h=8; d=128
        q,k,v=[torch.randn(1,n,h,d,device='cuda',dtype=torch.bfloat16)*.05 for _ in range(3)]
        gate=torch.randn_like(q)*.01; beta=torch.full((1,n,h),.1,device='cuda')
        state=torch.zeros(3,h,d,d,device='cuda',dtype=torch.float32)
        idx=torch.tensor([1],device='cuda',dtype=torch.int32)
        cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
        alog=torch.zeros(h,device='cuda'); bias=torch.zeros(h*d,device='cuda')
        def call():
            state.zero_()
            return chunk_kda(q,k,v,gate,beta,initial_state=state,initial_state_indices=idx,
                             cu_seqlens=cu,A_log=alog,dt_bias=bias,lower_bound=-5.0)
        return call
    def files():
        return {str(p.relative_to(cache)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in cache.rglob('*') if p.is_file()}
    def measure(label, fn):
        before=counts.copy(); disk=files(); t=time.monotonic()
        fn(); torch.cuda.synchronize()
        now=files()
        row={'case':label, 'seconds':time.monotonic()-t,
             **{k:counts[k]-before[k] for k in counts},
             'cache_files_added':len(now.keys()-disk.keys()),
             'cache_files_changed':sum(now[k]!=v for k,v in disk.items() if k in now)}
        print(json.dumps(row),flush=True)
        return row
    cases=[('112-fallback',prefill(31,1023)), ('113-prefill',prefill(600,2048)),
           ('113-clean-false',prefill(65,1024,False))]
    cases += [(f'112-decode-{b}',decode(b)) for b in (6,7,17,32)]
    cases += [(f'kda-{n}',attention(n)) for n in (65,600,2049,8192,8257)]
    print(json.dumps({'device':torch.cuda.get_device_name(), 'torch':torch.__version__,
                      'triton':triton.__version__, 'persistent':args.persistent,
                      'cache':str(cache), 'initial_cache_files':len(files())}),flush=True)
    cold = [measure(label+'-first',fn) for label,fn in cases]
    warm = [measure(label+'-repeat',fn) for label,fn in cases]
    assert all(r['jit_misses']==r['bench_calls']==r['cache_files_added']==r['cache_files_changed']==0 for r in warm),warm
    novel=measure('113-novel-nq-601',prefill(601,2048))
    # A fresh process may find disk artifacts, but still has JIT in-memory misses.
    assert novel['jit_misses'] > 0, novel
    def serializable(value):
        if isinstance(value, dict):
            return {str(k): serializable(v) for k,v in value.items()}
        if isinstance(value, (tuple,list)):
            return [serializable(v) for v in value]
        return value
    Path('observed_keys_persistent.json' if args.persistent else 'observed_keys_cold.json').write_text(json.dumps(serializable(observed),indent=2,default=str)+'\n')
    print(json.dumps({'status':'PASS', 'same_shape_repeat_count':len(warm),
                      'repeat_jit_misses':sum(r['jit_misses'] for r in warm),
                      'repeat_bench_calls':sum(r['bench_calls'] for r in warm),
                      'first_jit_misses':sum(r['jit_misses'] for r in cold),
                      'first_bench_calls':sum(r['bench_calls'] for r in cold)}),flush=True)


if __name__ == '__main__':
    main()
