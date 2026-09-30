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


import sys, textwrap
from sglang.srt.layers.attention.dsa.kpool_fp8_index import kpool_decode_update_and_maybe_write_cache as producer
frozen=(root/'frozen-ca5d646c-backend.py').read_text()
fcls=next(n for n in ast.parse(frozen).body if isinstance(n,ast.ClassDef) and n.name=='DeepseekSparseAttnBackend')
rcls=next(n for n in ast.parse(Path(sys.modules[DeepseekSparseAttnBackend.__module__].__file__).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DeepseekSparseAttnBackend')
fm={n.name:ast.dump(n,include_attributes=False) for n in fcls.body if isinstance(n,ast.FunctionDef)}
rm={n.name:ast.dump(n,include_attributes=False) for n in rcls.body if isinstance(n,ast.FunctionDef)}
changes=[n for n in rm if fm.get(n)!=rm[n]];assert changes==['forward_decode'],changes
fd=next(n for n in fcls.body if isinstance(n,ast.FunctionDef) and n.name=='forward_decode')
a=next(n for n in ast.walk(fd) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reuse_query_view' for t in n.targets))
g=next(n for n in ast.walk(fd) if isinstance(n,ast.If) and n.lineno==a.end_lineno+1)
fragment='\n'.join(frozen.splitlines()[a.lineno-1:g.end_lineno])
qenv=dict(vars(sys.modules[DeepseekSparseAttnBackend.__module__]));qenv['_AX_DSA_QUERY_VIEW']=True
assert not qenv['_AX_DSA_SPARSE_TRITON']
exec('def prepare(q_all,q_nope,q_rope,kv_cache,layer):\n'+textwrap.indent(textwrap.dedent(fragment),'    ')+'\n    return q_all\n',qenv)
# Keep these supplemental results separate from the first raw qualification.
def emit(d):
 print(json.dumps(d),flush=True);results.append(d);(root/'fusion-supplement-results.json').write_text(json.dumps(results,indent=2)+'\n')
def error(a,b):
 af,bf=a.float(),b.float();finite=torch.isfinite(af)&torch.isfinite(bf)
 delta=(af-bf)[finite].abs()
 return dict(bitwise=bits(a,b),finite_count=int(finite.sum().item()),total=a.numel(),finite_pattern_equal=torch.equal(torch.isfinite(af),torch.isfinite(bf)),nan_pattern_equal=torch.equal(torch.isnan(af),torch.isnan(bf)),inf_pattern_equal=torch.equal(torch.isinf(af),torch.isinf(bf)),max_abs=float(delta.max().item()) if delta.numel() else 0.0,mean_abs=float(delta.mean().item()) if delta.numel() else 0.0,rmse=float((delta.square().mean().sqrt()).item()) if delta.numel() else 0.0)
def snapshot_out(x):return tuple(v.clone() for v in x)
def hashes(t):return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()
emit(dict(kind='environment',GPU=torch.cuda.get_device_name(),source_sha256=hashlib.sha256(frozen.encode()).hexdigest(),changed_methods=changes,metadata_attention_ast_exact=True,query_view=True,query_view_fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),producer_source=str(sys.modules[producer.__module__].__file__),width=WIDTH,kpool=4,page=64,topk=2048,topk_v2=os.environ.get('SGLANG_OPT_USE_TOPK_V2')))
producer_pool=NS(index_kpool=4,tail_extra_slots=0,page_size=64,index_head_dim=128,slots_per_page=64)
for B in (1,24,32,48):
 view=NS(batch_size=B,forward_mode=ForwardMode.DECODE,actual_forward_mode=ForwardMode.DECODE,req_pool_indices=torch.arange(B,device=device,dtype=torch.int64),seq_lens=torch.zeros(B,device=device,dtype=torch.int64),seq_lens_cpu=torch.zeros(B,dtype=torch.int64),seq_lens_sum=0,num_padding=0,spec_info=None,out_cache_loc=None,mamba_track_indices=torch.arange(B,device=device,dtype=torch.int64)+64)
 qi=torch.randn(B,1,32,256,device=device).to(torch.float8_e4m3fn)[...,::2]
 w=torch.randn(B,64,device=device)[:,::2];qa=torch.randn(B,8,512,device=device,dtype=torch.bfloat16)
 def metadata(enabled):
  owners[int(enabled)][0].init_forward_metadata_out_graph(view);return owners[int(enabled)][2].forward_metadata
 def stage(enabled):
  m=metadata(enabled)
  logits=fp8_paged_mqa_logits(qi,cache.view(NP+1,64,1,132),w,m.pooled_cache_seqlens_int32.view(B,1),m.pooled_real_page_table,None,m.pooled_real_page_table.shape[1]*64)
  idx=top(logits,m.pooled_cache_seqlens_int32,4,2048,page_table=m.page_table_1,seq_lens=m.cache_seqlens_int32)
  prepared=qenv['prepare'](qa,qa,qa[...,512:],kv,NS(v_head_dim=512));assert prepared.data_ptr()==qa.data_ptr()
  out=DeepseekSparseAttnBackend._forward_tilelang(owners[0][2],prepared,kv,512,idx,512**-0.5,return_lse=True)
  return logits,idx,out
 for e in (False,True):metadata(e);metadata(e)
 view.seq_lens.fill_(8192);view.seq_lens_cpu.fill_(8192)
 graphs={};outputs={}
 for e in (False,True):
  for _ in range(2):stage(e)
  torch.cuda.synchronize();gg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(gg):outputs[e]=stage(e)
  graphs[e]=gg
 phases=[('mixed_short',[0,1,3,4,17]),('empty',[0]),('grow',[8192,250000,1048575]),('partial',[64,65,256,257])]
 for pi,(phase,pattern) in enumerate(phases):
  lens=[pattern[i%len(pattern)] for i in range(B)]
  view.seq_lens.copy_(torch.tensor(lens,device=device));view.seq_lens_cpu.copy_(torch.tensor(lens));view.seq_lens_sum=sum(lens)
  # B1 really changes request here; larger batches also replace every request.
  view.req_pool_indices.copy_((torch.arange(B,device=device)+pi+1)%64)
  mc=owners[1][2].forward_metadata
  mc.page_table_1.fill_(-999);mc.real_page_table.fill_(-999);mc.pooled_real_page_table.fill_(-999)
  reps={False:[],True:[]};sets={False:[],True:[]}
  for e in (False,True):
   for rr in range(6):
    graphs[e].replay();torch.cuda.synchronize()
    reps[e].append(snapshot_out(outputs[e][2]));sets[e].append(outputs[e][1].clone())
  validmeta(owners[0][2].forward_metadata,owners[1][2].forward_metadata,lens)
  assert bits(outputs[False][0],outputs[True][0])
  ref=sets[False][0].sort(-1).values
  selfsets=[torch.equal(ref,z.sort(-1).values) for z in sets[False]]
  crosssets=[torch.equal(ref,z.sort(-1).values) for z in sets[True]]
  assert all(selfsets) and all(crosssets),(B,phase,'full selected set changed')
  selferrors=[dict(output=error(reps[False][0][0],z[0]),lse=error(reps[False][0][1],z[1])) for z in reps[False][1:]]
  crosserrors=[dict(output=error(reps[False][0][0],z[0]),lse=error(reps[False][0][1],z[1])) for z in reps[True]]
  emit(dict(kind='graph_native_qualification',B=B,phase=phase,lens=lens,req_indices=view.req_pool_indices.cpu().tolist(),poison=-999,live_metadata_exact=True,logits_bitwise=True,selected_shape=list(sets[False][0].shape),full_selected_including_padding_self_equal=selfsets,full_selected_including_padding_cross_equal=crosssets,selected_raw_sha256={str(e):[hashes(x) for x in sets[e]] for e in (False,True)},native_baseline_self=selferrors,native_baseline_fusion=crosserrors))
 # Timed graph boundary includes exact181 fragment in both arms; no full-model claim.
 lens=[8192 if i%2==0 else 250000 for i in range(B)]
 view.seq_lens.copy_(torch.tensor(lens,device=device));view.seq_lens_cpu.copy_(torch.tensor(lens));view.seq_lens_sum=sum(lens)
 emit(dict(kind='performance',B=B,query_view=True,graph_complete_stage={str(e):timed(graphs[e].replay,calls=30) for e in (False,True)},eager_complete_stage={str(e):timed(lambda e=e:stage(e)) for e in (False,True)}))
 # Actual compressed-cache producer. Mixed invalid rows have distinct requests;
 # compare full cache plus every tail row, not only slots expected to change.
 initial_cache=cache.clone();initial_k=torch.randn(64,4,128,device=device,dtype=torch.bfloat16);initial_s=torch.randn_like(initial_k)
 key=torch.randn(B,128,device=device,dtype=torch.bfloat16);score=torch.randn_like(key);ape=torch.randn(4,128,device=device,dtype=torch.float32)
 for phase in ('empty','invalid_close','invalid_req','mixed'):
  lens=[0]*B if phase=='empty' else ([4]*B if phase=='invalid_close' else [[0,4,17,64,65][i%5] for i in range(B)])
  view.seq_lens.copy_(torch.tensor(lens,device=device));view.seq_lens_cpu.copy_(torch.tensor(lens));view.seq_lens_sum=sum(lens)
  view.req_pool_indices.copy_(torch.arange(B,device=device)+1)
  positions=torch.tensor([z-1 for z in lens],device=device,dtype=torch.int64)
  loc=torch.tensor([0 if phase!='mixed' or z==0 else 1 for z in lens],device=device,dtype=torch.int64)
  # Invalid close: valid position at a closed pool but loc0 must gate all writes.
  res={};old={}
  producer_req=view.req_pool_indices if phase!='invalid_req' else torch.tensor([-1 if i%2==0 else 64 for i in range(B)],device=device,dtype=torch.int64)
  if phase=='invalid_req':loc.fill_(1)
  for e in (False,True):
   m=metadata(e);buf=initial_cache.clone();tk=initial_k.clone();ts=initial_s.clone()
   before=(buf.clone(),tk.clone(),ts.clone())
   def write():producer(producer_pool,buf,tk,ts,key,score,ape,m.real_page_table,producer_req,positions,view.seq_lens,loc,round_scale=True)
   write();torch.cuda.synchronize()
   if phase!='mixed':assert torch.equal(buf,before[0]) and bits(tk,before[1]) and bits(ts,before[2]),(B,phase,e,'invalid mutated')
   invalid=[i for i,z in enumerate(lens) if z==0 or int(loc[i].item())==0]
   for i in invalid:
    req=i+1;assert bits(tk[req],before[1][req]) and bits(ts[req],before[2][req])
   for i,z in enumerate(lens):
    if phase!='invalid_req' and z>0 and int(loc[i].item())!=0:
     req=i+1;slot=(z-1)%4;assert bits(tk[req,slot],key[i]) and bits(ts[req,slot],score[i])
   allowed=torch.zeros_like(buf,dtype=torch.bool)
   for i,z in enumerate(lens):
    if phase=='mixed' and z>0 and int(loc[i].item())!=0 and (z-1)%4==3:
     poolid=(z-1)//4;page=int(m.real_page_table[i,(poolid//64)*4].item());slot=poolid%64
     allowed[page,slot*128:slot*128+128]=True;allowed[page,8192+slot*4:8192+slot*4+4]=True
   assert torch.equal(buf[~allowed],before[0][~allowed]),(B,phase,e,'cache outside valid close changed')
   # Graph replay producer writes must preserve invalid rows too.
   buf.copy_(before[0]);tk.copy_(before[1]);ts.copy_(before[2]);torch.cuda.synchronize()
   pg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(pg):write()
   pg.replay();torch.cuda.synchronize()
   if phase!='mixed':assert torch.equal(buf,before[0]) and bits(tk,before[1]) and bits(ts,before[2])
   for i in invalid:
    req=i+1;assert bits(tk[req],before[1][req]) and bits(ts[req],before[2][req])
   assert torch.equal(buf[~allowed],before[0][~allowed]),(B,phase,e,'graph cache outside valid close changed')
   res[e]=(buf.clone(),tk.clone(),ts.clone())
  assert torch.equal(res[False][0],res[True][0]) and bits(res[False][1],res[True][1]) and bits(res[False][2],res[True][2]),(B,phase,'producer candidate difference')
  emit(dict(kind='producer_qualification',B=B,phase=phase,lens=lens,out_cache_loc=loc.cpu().tolist(),real_kernel=True,round_scale=True,full_cache_tail_baseline_fusion_bitwise=True,all_invalid_full_buffers_unchanged=phase!='mixed',mixed_invalid_tail_rows_unchanged=True,graph_passed=True,cache_outside_valid_closed_pool_bytes_unchanged=True,invalid_req_guard_test=phase=='invalid_req',active_key_score_written_exact=True))
 del initial_cache,initial_k,initial_s,res,old
emit(dict(kind='complete',status='SUPPLEMENT_ASSERTIONS_PASS',production_modified=False))
