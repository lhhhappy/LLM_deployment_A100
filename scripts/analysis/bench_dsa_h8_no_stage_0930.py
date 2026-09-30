#!/usr/bin/env python3
"""Exact candidate selection fragment and real TileLang attention boundary."""
import ast
import importlib.util
import sys
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace as NS

import torch

SOURCE = Path(__file__).with_name("candidate-backend.py")
TILE_SOURCE=Path(__file__).with_name("candidate-tilelang.py")
spec=importlib.util.spec_from_file_location("sglang.kernels.ops.attention.dsa.tilelang_kernel",TILE_SOURCE)
tile_module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=tile_module
spec.loader.exec_module(tile_module)
source = SOURCE.read_text()
tree = ast.parse(source)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekSparseAttnBackend")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_tilelang")
decode = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_decode")
assign = next(n for n in ast.walk(decode) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "reuse_query_view" for t in n.targets))
guard = next(n for n in ast.walk(decode) if isinstance(n, ast.If) and getattr(n, "lineno", 0) == assign.end_lineno + 1)
fragment = "\n".join(source.splitlines()[assign.lineno - 1:guard.end_lineno])
import textwrap
env = dict(torch=torch, _AX_DSA_SPARSE_TRITON=False, _AX_DSA_SPARSE_TRITON_PREFILL=True, _AX_DSA_QUERY_VIEW=True, _is_hip=False)
env["concat_mla_absorb_q_general"] = lambda a, b: torch.cat((a, b), dim=-1)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), env)
exec("def prepare(q_all, q_nope, q_rope, kv_cache, layer):\n" + textwrap.indent(textwrap.dedent(fragment), "    ") + "\n    return q_all\n", env)
owner = NS()
layer = NS(v_head_dim=512)

def same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32), b.contiguous().view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32))

def boundary(enabled, q, kv, idx, lse=False):
    tile_module._AX_SM80_DSA_H8_NO_OUTPUT_STAGE = enabled
    qa = q.contiguous().view(-1, 8, 512)
    qa = env["prepare"](qa, qa[..., :512], qa[..., 512:], kv, layer)
    return env["_forward_tilelang"](owner, qa, kv, 512, idx, 512 ** -0.5, return_lse=lse)

def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    before = torch.cuda.memory_allocated()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out, torch.cuda.memory_allocated() - before

def device_us(fn, calls):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(calls):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / calls

def emit(data):
    print(json.dumps(data), flush=True)

torch.manual_seed(930181)
torch.cuda.set_device(0)
emit(dict(kind="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__, source_sha256=__import__("hashlib").sha256(source.encode()).hexdigest(), geometry="BF16 Q[B,8,512], KV[65536,1,512], topk2051 -> TileLang2112", scope="synthetic real TileLang boundary, not full model/prefill"))
kv = torch.randn(65536, 1, 512, device="cuda", dtype=torch.bfloat16)
qual = []
for batch in (1, 24, 32, 48):
    q_storage = torch.randn(batch + 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    q = q_storage[1:]
    idx = torch.randint(0, len(kv), (batch, 2051), device="cuda", dtype=torch.int32)
    full_idx = idx.clone()
    for row in range(batch):
        length = (2051, 2048, 17, 0)[row % 4] if batch > 1 else 2051
        idx[row, length:] = -1
        if length > 3:
            idx[row, 1] = -1
    mixed_idx = idx.clone()
    original = q.clone()
    a, la = boundary(False, q, kv, idx, True)
    b, lb = boundary(True, q, kv, idx, True)
    assert same_bits(a, b) and same_bits(la, lb)
    assert same_bits(q, original)
    reuse = env["prepare"](q, q, q[..., 512:], kv, layer)
    assert reuse.data_ptr() == q.data_ptr()
    
    graphs = {}
    for enabled in (False, True):
        graphs[enabled] = capture(lambda enabled=enabled: boundary(enabled, q, kv, idx, True))
    assert same_bits(graphs[False][1][0], graphs[True][1][0])
    assert same_bits(graphs[False][1][1], graphs[True][1][1])
    # Stable graph input addresses, mutated Q/KV/indices, shrinking to all empty.
    for mutate in ("inputs_and_partial", "all_empty", "grow"):
        q.add_(0.125)
        kv[:32].add_(0.0625)
        idx.fill_(-1)
        if mutate != "all_empty":
            length = 7 if mutate == "inputs_and_partial" else 2051
            idx[:, :length] = torch.arange(length, device="cuda", dtype=torch.int32) % 65536
        for enabled in (False, True):
            graphs[enabled][0].replay()
        expected = boundary(False, q, kv, idx, True)
        torch.cuda.synchronize()
        for enabled in (False, True):
            assert same_bits(graphs[enabled][1][0], expected[0])
            assert same_bits(graphs[enabled][1][1], expected[1])
    timing = {False: [], True: []}
    del graphs
    idx.copy_(full_idx)
    graphs = {enabled: capture(lambda enabled=enabled: boundary(enabled, q, kv, idx)) for enabled in (False, True)}
    for round_id in range(14):
        for enabled in ((False, True) if round_id % 2 == 0 else (True, False)):
            timing[enabled].append(device_us(graphs[enabled][0].replay, 100))
    eager = {False:[],True:[]}
    idle_wall={False:{'enqueue':[],'complete':[]},True:{'enqueue':[],'complete':[]}}
    for rep in range(100):
        for enabled in ((False,True)if rep%2==0 else(True,False)):
            torch.cuda.synchronize();t=time.perf_counter();boundary(enabled,q,kv,idx);submitted=time.perf_counter();torch.cuda.synchronize();completed=time.perf_counter()
            idle_wall[enabled]['enqueue'].append((submitted-t)*1e6);idle_wall[enabled]['complete'].append((completed-t)*1e6)
    eager={enabled:[device_us(lambda enabled=enabled:boundary(enabled,q,kv,idx),20)for _ in range(5)]for enabled in(False,True)}
    emit({'kind':'idle_wall','batch':batch,'microseconds':{str(k):v for k,v in idle_wall.items()},'medians_us':{str(k):{metric:statistics.median(vals)for metric,vals in measures.items()}for k,measures in idle_wall.items()}})
    baseline, candidate = (statistics.median(timing[enabled]) for enabled in (False, True))
    single = {False: [], True: []}
    events = []
    for i in range(200):
        for enabled in ((False, True) if i % 2 == 0 else (True, False)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[enabled][0].replay()
            end.record()
            events.append((enabled,start,end))
    torch.cuda.synchronize()
    for enabled,start,end in events:
        single[enabled].append(start.elapsed_time(end)*1000)
    emit(dict(kind="decode", batch=batch, exact_output=True, exact_lse=True, full_tilelang_source_sha256=hashlib.sha256(TILE_SOURCE.read_bytes()).hexdigest(), q_unmodified=True, graph_mutation_shrink_grow_exact=True, graph_us={str(k): v for k, v in timing.items()}, graph_median_us={"off":baseline,"on":candidate}, saving_us=baseline-candidate, single_graph_median_us={str(k):statistics.median(v) for k,v in single.items()}, single_graph_p95_us={str(k):sorted(v)[189] for k,v in single.items()}, single_graph_samples={str(k):v for k,v in single.items()}, eager_median_us={str(k):statistics.median(v) for k,v in eager.items()}, graph_allocation_bytes={str(k):graphs[k][2] for k in graphs}, removed_shared_output_bytes=16*512*2))
    idx.copy_(mixed_idx)
    mixed = {False: [], True: []}
    for i in range(10):
        for enabled in ((False, True) if i % 2 == 0 else (True, False)):
            mixed[enabled].append(device_us(graphs[enabled][0].replay, 100))
    emit(dict(kind="mixed_decode", batch=batch, lengths="2051/2048/17/0 repeated, interior negative hole", graph_median_us={str(k):statistics.median(v) for k,v in mixed.items()}, graph_us={str(k):v for k,v in mixed.items()}, saving_us=statistics.median(mixed[False])-statistics.median(mixed[True])))
    del graphs
    # Non-contiguous incoming Q still follows the existing initial contiguous copy.
    q_gap = torch.randn(batch, 8, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2]
    assert same_bits(boundary(False, q_gap, kv, idx), boundary(True, q_gap, kv, idx))
    qual.append(batch)


# Resources/codegen for the two generic factories: no new tile parameters.
for enabled in (False,True):
 kernel=tile_module.sparse_attention_fwd_kernel_v1(8,512,0,2112,sm_scale=512**-.5,return_lse=False,stage_output=not enabled)
 attrs=[k for k in dir(kernel) if 'source' in k or 'resource' in k or 'adapter' in k]
 data={'kind':'kernel_resource','enabled':enabled,'attributes':attrs}
 if hasattr(kernel,'get_kernel_source'):
  code=kernel.get_kernel_source();Path(__file__).with_name('stageoff.cu' if enabled else 'stageon.cu').write_text(code);data['cuda_source_sha256']=hashlib.sha256(code.encode()).hexdigest()
 if hasattr(kernel,'adapter'):data['adapter_type']=str(type(kernel.adapter));data['adapter_attributes']=[k for k in dir(kernel.adapter) if 'source' in k or 'resource' in k or 'path' in k or 'module' in k]
 emit(data)
# Shared factory applies to any eligible shape, so qualify sparse larger-row calls.
for rows in (8192,16384):
 q=torch.randn(rows,8,512,device='cuda',dtype=torch.bfloat16)
 idx=torch.randint(0,len(kv),(rows,2051),device='cuda',dtype=torch.int32)
 lengths=torch.tensor([17,0,2051,2048],device='cuda',dtype=torch.int32).repeat(rows//4)
 idx.masked_fill_(torch.arange(2051,device='cuda').unsqueeze(0)>=lengths.unsqueeze(1),-1)
 a,la=boundary(False,q,kv,idx,True);b,lb=boundary(True,q,kv,idx,True)
 assert same_bits(a,b) and same_bits(la,lb)
 samples={False:[],True:[]}
 for rep in range(4):
  for enabled in ((False,True)if rep%2==0 else(True,False)):
   torch.cuda.synchronize();t=time.perf_counter();boundary(enabled,q,kv,idx);torch.cuda.synchronize();samples[enabled].append((time.perf_counter()-t)*1e6)
 emit({'kind':'large_sparse_shape','rows':rows,'lengths':'17/0/2051/2048','exact_output':True,'exact_lse':True,'wall_complete_us':{str(k):v for k,v in samples.items()},'median_us':{str(k):statistics.median(v)for k,v in samples.items()},'full_model_prefill':False})
 del q,idx,a,b,la,lb
emit(dict(kind='qualification',batches=qual,all_passed=True,scope='exact TileLang generic kernel/source+real backend attention boundary; no model/TP8/chain',flag='SGLANG_AX_SM80_DSA_H8_NO_OUTPUT_STAGE'))
