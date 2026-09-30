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


torch.manual_seed(9301852);torch.cuda.set_device(0)
kv=torch.randn(65536,1,512,device='cuda',dtype=torch.bfloat16)
for batch in (1,32):
 q=torch.randn(batch,8,512,device='cuda',dtype=torch.bfloat16)
 idx=torch.randint(0,len(kv),(batch,2051),device='cuda',dtype=torch.int32)
 for enabled in(False,True):
  for _ in range(5):boundary(enabled,q,kv,idx)
 torch.cuda.synchronize()
 samples={False:{'enqueue':[],'complete':[]},True:{'enqueue':[],'complete':[]}}
 for rep in range(150):
  for enabled in ((False,True)if rep%2==0 else(True,False)):
   torch.cuda.synchronize();t=time.perf_counter();boundary(enabled,q,kv,idx);enqueued=time.perf_counter();torch.cuda.synchronize();done=time.perf_counter()
   samples[enabled]['enqueue'].append((enqueued-t)*1e6);samples[enabled]['complete'].append((done-t)*1e6)
 graphs={enabled:capture(lambda enabled=enabled:boundary(enabled,q,kv,idx))for enabled in(False,True)}
 graph_samples={False:[],True:[]}
 for rep in range(8):
  for enabled in((False,True)if rep%2==0 else(True,False)):graph_samples[enabled].append(device_us(graphs[enabled][0].replay,100))
 emit({'kind':'repeat_boundary','batch':batch,'idle_medians_us':{str(k):{metric:statistics.median(v)for metric,v in a.items()}for k,a in samples.items()},'samples':{str(k):v for k,v in samples.items()},'graph_median_us':{str(k):statistics.median(v)for k,v in graph_samples.items()},'source_sha256':hashlib.sha256(TILE_SOURCE.read_bytes()).hexdigest()})
# CPU exact dispatch test with capability/factory mocks, no unsupported-dtype kernel execution.
state={'cap':(8,0),'dcp':False};calls=[]
def factory(*args,**kw):
 calls.append(kw);return lambda q,kv,idx,lse:q
fn=next(n for n in ast.parse(TILE_SOURCE.read_text()).body if getattr(n,'name','')=='tilelang_sparse_fwd')
ns={'torch':torch,'_is_hip':False,'get_parallel':lambda:NS(dcp_enabled=state['dcp']),'sparse_attention_fwd_kernel_v1':factory,'sparse_attention_fwd_kernel_v2':factory}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'candidate dispatch exactAST','exec'),ns)
original=torch.cuda.get_device_capability;torch.cuda.get_device_capability=lambda device:state['cap']
qual=[]
try:
 for name,flag,heads,width,dv,qdtype,kvtype,cap,dcp in [('flagoff',False,8,512,512,torch.bfloat16,torch.bfloat16,(8,0),False),('eligible',True,8,512,512,torch.bfloat16,torch.bfloat16,(8,0),False),('SM86',True,8,512,512,torch.bfloat16,torch.bfloat16,(8,6),False),('SM90',True,8,512,512,torch.bfloat16,torch.bfloat16,(9,0),False),('H4',True,4,512,512,torch.bfloat16,torch.bfloat16,(8,0),False),('D256',True,8,256,256,torch.bfloat16,torch.bfloat16,(8,0),False),('tail64',True,8,576,512,torch.bfloat16,torch.bfloat16,(8,0),False),('FP16Q',True,8,512,512,torch.float16,torch.bfloat16,(8,0),False),('FP8KV',True,8,512,512,torch.bfloat16,torch.float8_e4m3fn,(8,0),False),('H32_DCP',True,32,512,512,torch.bfloat16,torch.bfloat16,(8,0),True),('H64_existing',False,64,512,512,torch.bfloat16,torch.bfloat16,(8,0),False)]:
  ns['_AX_SM80_DSA_H8_NO_OUTPUT_STAGE']=flag;state.update(cap=cap,dcp=dcp);calls.clear()
  ns['tilelang_sparse_fwd'](torch.empty(1,heads,width,dtype=qdtype),torch.empty(8,1,width,dtype=kvtype),torch.zeros(1,1,64,dtype=torch.int32),width**-.5,d_v=dv)
  kw=calls[0];assert ('stage_output'in kw)==(name in('eligible','H32_DCP','H64_existing')),(name,kw)
  if name=='eligible':assert 'num_stages'not in kw and 'threads'not in kw and 'block_I'not in kw
  qual.append({'case':name,'factory_kwargs':kw})
finally:torch.cuda.get_device_capability=original
emit({'kind':'dispatch_qualification','scope':'CPU exact dispatch AST, mocked cap/context/factory, unsupporteddtype numerics not claimed','cases':qual,'PASS':True})
