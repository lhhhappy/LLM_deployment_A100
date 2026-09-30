"""Actual frozen metadata make/method plus native SM80 logits/TopK/TileLang."""
import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS
import torch

root=Path(__file__).parent
src=(root/'cuda_metadata_probe.py').read_text();tree=ast.parse(src)
nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name=='make']
env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'cuda_metadata_probe.py','exec'),env)
device=torch.device('cuda:0');torch.cuda.set_device(device);env['device']=device
from sglang.srt.layers.attention.dsa.sm80_indexer_kernels import fp8_paged_mqa_logits
from sglang.kernels.ops.moe.kpool_topk_transform import fast_kpool_topk_transform_fused as top
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
WIDTH=1048580
torch.manual_seed(930182)
owners=[]
for enabled in (False,True):
 b,p=env['make'](WIDTH,48); d=b.attn_backend_list[0];d.experimental_kpool_metadata_fusion=enabled; owners.append((b,p,d))
pool=owners[0][1];NP=(WIDTH+63)//64
perm=torch.randperm(NP,device=device,dtype=torch.int32)+1
logical=torch.arange(WIDTH,device=device,dtype=torch.int32)
for req in range(64):pool.req_to_token[req].copy_(perm[(logical//64+req*137)%NP]*64+logical%64)
owners[1][1].req_to_token=pool.req_to_token;owners[1][2].req_to_token=pool.req_to_token
cache=torch.empty(NP+1,64*132,device=device,dtype=torch.uint8)
cache[:,:8192]=torch.randn(NP+1,64*128,device=device).to(torch.float8_e4m3fn).view(torch.uint8)
cache[:,8192:].view(torch.float32).uniform_(0.75,1.25)
kv=torch.randn((NP+1)*64,1,512,device=device,dtype=torch.bfloat16)
results=[]
def emit(d):print(json.dumps(d),flush=True);results.append(d);(root/'fusion-results.json').write_text(json.dumps(results,indent=2)+'\n')
def bits(a,b):return torch.equal(a.contiguous().view(torch.int16 if a.dtype==torch.bfloat16 else torch.int32),b.contiguous().view(torch.int16 if b.dtype==torch.bfloat16 else torch.int32))
def timed(fn,calls=12):
 arr=[]
 for _ in range(5):
  torch.cuda.synchronize();a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);t=time.perf_counter();a.record()
  for _ in range(calls):fn()
  z.record();en=time.perf_counter();z.synchronize();end=time.perf_counter();arr.append(dict(gpu_boundary_us=a.elapsed_time(z)*1000/calls,enqueue_us=(en-t)*1e6/calls,complete_us=(end-t)*1e6/calls))
 return {k:statistics.median(x[k] for x in arr) for k in arr[0]}|dict(samples=arr)
def validmeta(a,b,lens):
 for name in ('cache_seqlens_int32','cu_seqlens_k','dsa_cache_seqlens_int32','dsa_cu_seqlens_k','pooled_cache_seqlens_int32'):
  assert torch.equal(getattr(a,name),getattr(b,name)),name
 for row,L in enumerate(lens):
  for name,n in [('page_table_1',L),('real_page_table',(L+63)//64),('pooled_real_page_table',(L//4+63)//64)]:
   assert torch.equal(getattr(a,name)[row,:n],getattr(b,name)[row,:n]),(name,row,L)
  req=int(view.req_pool_indices[row].item())
  assert torch.equal(a.page_table_1[row,:L],pool.req_to_token[req,:L]),('authoritative_raw',row,L)
  assert torch.equal(b.page_table_1[row,:L],pool.req_to_token[req,:L]),('authoritative_fused_raw',row,L)
 pa,pb=a.kpool_write_plan,b.kpool_write_plan
 for name in ('req','write_start','tail_logical_start'):
  assert torch.equal(getattr(pa,name),getattr(pb,name)),name
 for row,L in enumerate(lens):
  if L>0 and L%4==0:assert torch.equal(pa.write_loc[row],pb.write_loc[row]),('write_loc',row)

emit(dict(kind='environment',gpu=torch.cuda.get_device_name(),width=WIDTH,page=64,kpool=4,topk=2048,scope='Actual metadata backend plus native logits/PAGED topk/BF16 attention, synthetic cache/Q, no model/TP8/KDA scan/cache producer writes',backend_source=str(__import__(DeepseekSparseAttnBackend.__module__,fromlist=['x']).__file__)))
for B in (1,24,32,48):
 lensbase=[0,1,3,4,17,8192,250000,1048575,64,65,256,257]
 view=NS(batch_size=B,forward_mode=ForwardMode.DECODE,actual_forward_mode=ForwardMode.DECODE,req_pool_indices=torch.arange(B,device=device,dtype=torch.int64),seq_lens=torch.zeros(B,device=device,dtype=torch.int64),seq_lens_cpu=torch.zeros(B,dtype=torch.int64),seq_lens_sum=0,num_padding=0,spec_info=None,out_cache_loc=None,mamba_track_indices=torch.arange(B,device=device,dtype=torch.int64)+64)
 qi_storage=torch.randn(B,1,32,256,device=device).to(torch.float8_e4m3fn);qi=qi_storage[...,::2]
 w=torch.randn(B,64,device=device)[:,::2]
 qa=torch.randn(B,8,512,device=device,dtype=torch.bfloat16)
 def metadata(enabled):
  b,p,d=owners[int(enabled)];b.init_forward_metadata_out_graph(view);return d.forward_metadata
 def consume(m,lse=False,common=None):
  logits=fp8_paged_mqa_logits(qi,cache.view(NP+1,64,1,132),w,m.pooled_cache_seqlens_int32.view(B,1),m.pooled_real_page_table,None,m.pooled_real_page_table.shape[1]*64)
  idx=top(logits,m.pooled_cache_seqlens_int32,4,2048,page_table=m.page_table_1,seq_lens=m.cache_seqlens_int32)
  out=DeepseekSparseAttnBackend._forward_tilelang(owners[0][2],qa,kv,512,idx if common is None else common,512**-0.5,return_lse=lse)
  return logits,idx,out
 def stage(enabled):return consume(metadata(enabled),True)
 # make's first call builds capture storage, as at real runner startup. Serving
 # replay must be tested after both captured metadata objects already exist.
 for enabled in (False,True):
  metadata(enabled);metadata(enabled)
 variants=[lensbase[:]] if B==1 else [[lensbase[i%len(lensbase)] for i in range(B)],[8192]*B]
 qual=[]
 for lengths in variants:
  choices=lengths if B==1 else [lengths]
  for ls in choices:
   ls=[ls] if B==1 else ls
   view.seq_lens.copy_(torch.tensor(ls,device=device));view.seq_lens_cpu.copy_(torch.tensor(ls));view.seq_lens_sum=sum(ls)
   ma=metadata(False);mb=metadata(True);validmeta(ma,mb,ls)
   aa,ia,oa=consume(ma,True);ab,ib,ob=consume(mb,True); assert bits(aa,ab)
   # Consumer exactness with the SAME selected tensor removes native atomic order variability.
   _,_,fixed=consume(mb,True,common=ia)
   assert bits(oa[0],fixed[0]) and bits(oa[1],fixed[1])
   seteq=torch.equal(ia.sort(-1).values,ib.sort(-1).values)
   repeat=top(aa,ma.pooled_cache_seqlens_int32,4,2048,page_table=ma.page_table_1,seq_lens=ma.cache_seqlens_int32)
   assert seteq or not torch.equal(ia.sort(-1).values,repeat.sort(-1).values),'candidate-only native selected set mismatch'
   qual.append(dict(lens=ls,live_metadata_exact=True,logits_bitwise=True,common_selected_output_lse_bitwise=True,native_selected_set_equal=seteq,native_raw_order_equal=torch.equal(ia,ib),native_output_bits_equal=bits(oa[0],ob[0]),native_baseline_self_order_equal=torch.equal(ia,repeat)))
 emit(dict(kind='qualification',B=B,cases=qual))
 # Capture actual full-stage refresh and consumers; this is NOT MetadataGlueGraph.
 ls=[lensbase[i%len(lensbase)] for i in range(B)] if B>1 else [8192]
 view.seq_lens.copy_(torch.tensor(ls,device=device));view.seq_lens_cpu.copy_(torch.tensor(ls));view.seq_lens_sum=sum(ls)
 graphs={};outputs={}
 for enabled in (False,True):
  for _ in range(2):stage(enabled)
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):outputs[enabled]=stage(enabled)
  graphs[enabled]=g
 mutations=[]
 for phase,ls in [('short',[0,1,3,4,17][0:1]*B if B==1 else [[0,1,3,4,17][i%5] for i in range(B)]),('empty',[0]*B),('grow',[[8192,250000,1048575][i%3] for i in range(B)])]:
  view.seq_lens.copy_(torch.tensor(ls,device=device));view.seq_lens_cpu.copy_(torch.tensor(ls));view.seq_lens_sum=sum(ls);view.req_pool_indices.copy_(torch.arange(B,device=device).roll(3))
  # Poison captured candidate suffix BEFORE replay. Live values must be refreshed.
  mc=owners[1][2].forward_metadata
  mc.page_table_1.fill_(-999);mc.real_page_table.fill_(-999);mc.pooled_real_page_table.fill_(-999)
  for enabled in (False,True):graphs[enabled].replay()
  torch.cuda.synchronize();validmeta(owners[0][2].forward_metadata,owners[1][2].forward_metadata,ls);assert bits(outputs[False][0],outputs[True][0])
  _,_,common=consume(owners[1][2].forward_metadata,True,common=outputs[False][1])
  assert bits(outputs[False][2][0],common[0]) and bits(outputs[False][2][1],common[1])
  mutations.append(dict(phase=phase,live_metadata_exact=True,logits_bitwise=True,common_selected_output_lse_bitwise=True,native_output_bits_equal=bits(outputs[False][2][0],outputs[True][2][0]),native_lse_bits_equal=bits(outputs[False][2][1],outputs[True][2][1]),poison_suffix_not_consumed=True))
 # Timing uses heterogeneous lengths and independent arbitrary physical pages.
 view.seq_lens.copy_(torch.tensor([8192 if i%2==0 else 250000 for i in range(B)],device=device));view.seq_lens_cpu.copy_(view.seq_lens.cpu())
 eager={str(e):timed(lambda e=e:stage(e)) for e in (False,True)}
 graph={str(e):timed(graphs[e].replay,calls=30) for e in (False,True)}
 meta={str(e):timed(lambda e=e:metadata(e),calls=30) for e in (False,True)}
 emit(dict(kind='performance',B=B,eager_complete_stage=eager,graph_complete_stage=graph,eager_metadata=meta,graph_mutations=mutations))
emit(dict(kind='complete',status='PASS',production_modified=False))
