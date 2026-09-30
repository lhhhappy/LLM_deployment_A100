#!/usr/bin/env python3
"""Exact candidate selection fragment and real TileLang attention boundary."""
import ast
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace as NS

import torch

SOURCE = Path(__file__).with_name("candidate-source.py")
source = SOURCE.read_text()
tree = ast.parse(source)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekSparseAttnBackend")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_tilelang")
decode = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_decode")
assign = next(n for n in ast.walk(decode) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "reuse_query_view" for t in n.targets))
guard = next(n for n in ast.walk(decode) if isinstance(n, ast.If) and getattr(n, "lineno", 0) == assign.end_lineno + 1)
fragment = "\n".join(source.splitlines()[assign.lineno - 1:guard.end_lineno])
import textwrap
env = dict(torch=torch, _AX_DSA_SPARSE_TRITON=False, _AX_DSA_SPARSE_TRITON_PREFILL=True, _AX_DSA_QUERY_VIEW=False, _is_hip=False)
env["concat_mla_absorb_q_general"] = lambda a, b: torch.cat((a, b), dim=-1)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), env)
exec("def prepare(q_all, q_nope, q_rope, kv_cache, layer):\n" + textwrap.indent(textwrap.dedent(fragment), "    ") + "\n    return q_all\n", env)
owner = NS()
layer = NS(v_head_dim=512)

def same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32), b.contiguous().view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32))

def boundary(enabled, q, kv, idx, lse=False):
    env["_AX_DSA_QUERY_VIEW"] = enabled
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
    env["_AX_DSA_QUERY_VIEW"] = False
    copied = env["prepare"](q, q, q[..., 512:], kv, layer)
    assert copied.data_ptr() != q.data_ptr()
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
    eager = {enabled: [device_us(lambda: boundary(enabled, q, kv, idx), 20) for _ in range(5)] for enabled in (False, True)}
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
    emit(dict(kind="decode", batch=batch, exact_output=True, exact_lse=True, q_unmodified=True, graph_mutation_shrink_grow_exact=True, graph_us={str(k): v for k, v in timing.items()}, graph_median_us={"off":baseline,"on":candidate}, saving_us=baseline-candidate, single_graph_median_us={str(k):statistics.median(v) for k,v in single.items()}, single_graph_p95_us={str(k):sorted(v)[189] for k,v in single.items()}, single_graph_samples={str(k):v for k,v in single.items()}, eager_median_us={str(k):statistics.median(v) for k,v in eager.items()}, graph_allocation_bytes={str(k):graphs[k][2] for k in graphs}, q_saved_bytes=q.numel()*q.element_size()))
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

# Static guard fallback checks; no device-value dispatch or host synchronization.
q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
env["_AX_DSA_QUERY_VIEW"] = True
unaligned = torch.randn(2*8*512+1,device="cuda",dtype=torch.bfloat16)[1:].view(2,8,512)
for name, qa, kk, lyr in (("fp16q", q.half(), kv, layer), ("fp8kv", q, kv.to(torch.float8_e4m3fn), layer), ("unaligned", unaligned, kv, layer), ("noncontiguous", torch.randn(2,8,1024,device="cuda",dtype=torch.bfloat16)[...,::2], kv, layer), ("rope_tail", q, kv, NS(v_head_dim=448))):
    out = env["prepare"](qa, qa[..., :lyr.v_head_dim], qa[..., lyr.v_head_dim:], kk, lyr)
    assert out.data_ptr() != qa.data_ptr(), name
assert same_bits(boundary(False, unaligned, kv, idx[:2]), boundary(True, unaligned, kv, idx[:2]))
env["_AX_DSA_SPARSE_TRITON"] = True
assert env["prepare"](q, q, q[...,512:], kv, layer).data_ptr() != q.data_ptr()
env["_AX_DSA_SPARSE_TRITON"] = False
assert env["prepare"](None, q, q[...,512:], kv, layer).shape == q.shape
env["_is_hip"] = True
assert env["prepare"](q, q, q[...,512:], kv, layer).data_ptr() == q.data_ptr()
env["_is_hip"] = False
for rows in (8192, 16384):
    q = torch.randn(rows,8,512,device="cuda",dtype=torch.bfloat16)
    times = [device_us(lambda: torch.cat((q, q[...,512:]), dim=-1), 20) for _ in range(7)]
    emit(dict(kind="cat_only", rows=rows, median_us=statistics.median(times), bytes=q.numel()*q.element_size(), full_prefill=False))
emit(dict(kind="qualification", batches=qual, all_passed=True, fallbacks=["FP16Q","FP8KV","unaligned_contiguous_q","noncontiguous_q_all","separate_rope_tail","all_phase_triton","split_query_none","HIP_existing_view"], scope_exclusions=["model weights", "TP8/DCP collectives", "full prefill"]))
