"""Execute exact candidate KDA metadata/extend methods with real conv, stub KDA scan."""
import ast, argparse, hashlib, json, statistics, time
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--qualify-only',action='store_true');args=p.parse_args()
source=Path(args.source); tree=ast.parse(source.read_text())
class Mode:
 def __init__(self,kind='extend'):self.kind=kind
 def is_extend(self):return self.kind=='extend'
 def is_target_verify(self):return self.kind=='verify'
 def is_draft_extend_v2(self):return self.kind=='draft'
mode=Mode(); FM=NS(EXTEND=mode,MIXED=Mode('mixed'))
class Base:
 def init_forward_metadata(self,fb):
  self.forward_metadata=NS(query_start_loc=fb.query_start_loc,mamba_cache_indices=fb.cache_indices,has_mamba_track_mask=False)
class Scan:
 def extend(self,**kw):return kw['q']
selected=[]
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name in ['_ax_kda_extend_prefix_mask','_ax_kda_cpu_logical_tokens','_ax_kda_resolve_logical_tokens']:selected.append(n)
 if isinstance(n,ast.ClassDef) and n.name=='KDAAttnBackend':
  methods=[x for x in n.body if isinstance(x,ast.FunctionDef) and x.name in ['init_forward_metadata','forward_extend']]
  selected.append(ast.ClassDef(name='Backend',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[]))
mod=ast.Module(body=selected,type_ignores=[])
for n in ast.walk(mod):
 if isinstance(n,ast.FunctionDef):n.returns=None;n.decorator_list=[]
 if isinstance(n,ast.arg):n.annotation=None
ast.fix_missing_locations(mod)
g={'torch':torch,'Base':Base,'_AX_KDA_PREFILL_CPU_LENGTH':True,'_AX_KDA_SHARED_PREFIX_MASK':False,'ForwardMode':FM,'get_parallel':lambda:NS(attn_cp_size=1),'causal_conv1d_fn':causal_conv1d_fn}
exec(compile(mod,str(source),'exec'),g); Backend=g['Backend']; rows=[]
def make(B,lengths):
 T=sum(lengths); C=3072; qloc=torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()),device='cuda',dtype=torch.int32)
 fb=NS(batch_size=B,forward_mode=mode,extend_prefix_lens=torch.tensor([0 if i%2==0 else 256*(i+1) for i in range(B)],device='cuda'),extend_prefix_lens_cpu=[0 if i%2==0 else 256*(i+1) for i in range(B)],extend_seq_lens_cpu=lengths,query_start_loc=qloc,cache_indices=torch.arange(B,device='cuda',dtype=torch.int32),ax_kda_snapshot_offsets=None,ax_kda_snapshot_slots=None)
 backend=Backend();backend.kernel_dispatcher=Scan();backend.accept_lens_pool=None
 states=[NS(conv=[torch.randn(B,3,C,device='cuda',dtype=torch.bfloat16)],temporal=torch.empty(B,8,128,128,device='cuda')) for _ in range(34)]
 backend.req_to_token_pool=NS(mamba2_layer_cache=lambda i:states[i])
 layers=[NS(layer_id=i,conv_weights=torch.randn(C,4,device='cuda',dtype=torch.bfloat16)*.05,bias=None,q_dim=1024,k_dim=1024,v_dim=1024,head_q_dim=128,head_k_dim=128,head_v_dim=128,A_log=None,dt_bias=None,lower_bound=None) for i in range(34)]
 x=torch.randn(T,C,device='cuda',dtype=torch.bfloat16); a=torch.randn(1,T,1024,device='cuda',dtype=torch.bfloat16); b=torch.randn(1,T,8,device='cuda',dtype=torch.bfloat16)
 return fb,backend,states,layers,x,a,b
def run(case,enabled,init=True):
 fb,be,st,layers,x,a,b=case;g['_AX_KDA_SHARED_PREFIX_MASK']=enabled
 if init:be.init_forward_metadata(fb)
 return [be.forward_extend(layer,fb,x.clone(),a,b) for layer in layers]
def timeit(fn,repeats=40):
 if args.qualify_only:return 1.0
 for _ in range(3):fn()
 torch.cuda.synchronize(); results=[]
 for _ in range(5):
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record()
  for i in range(repeats):fn()
  e.record();e.synchronize();results.append(s.elapsed_time(e)*1000/repeats)
 return statistics.median(results)
torch.manual_seed(3103)
for B,lens in [(1,[32]),(2,[31,64]),(8,[1,3,16,31,32,64,127,256]),(1,[256]),(2,[256,257]),(8,[256]*8)]:
 case=make(B,lens);fb,be,states,layers,x,a,b=case
 snapshots=[z.conv[0].clone() for z in states]
 ref=run(case,False);refstates=[z.conv[0].clone() for z in states]
 for z,old in zip(states,snapshots):z.conv[0].copy_(old)
 cand=run(case,True)
 assert all(torch.equal(a,b) for a,b in zip(ref,cand));assert all(torch.equal(z.conv[0],v) for z,v in zip(states,refstates))
 masks=[g['_ax_kda_extend_prefix_mask'](be.forward_metadata,fb.extend_prefix_lens) for _ in layers];assert all(t is masks[0] for t in masks)
 base=timeit(lambda:run(case,False));new=timeit(lambda:run(case,True))
 # Repeat prefixes in-place, rebuild once each live round, with changed token lengths.
 for vals in [[256]*B,[0]*B,[0 if i%3 else 4096 for i in range(B)]]:
  fb.extend_prefix_lens.copy_(torch.tensor(vals,device='cuda'));run(case,True)
  assert torch.equal(be.forward_metadata._ax_prefix_mask,fb.extend_prefix_lens>0)
 # Real captured forward uses baseline mask comparisons from static input.
 run(case,True);graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):captured=run(case,True)
 captured_metadata=be.forward_metadata
 for vals in [[0]*B,[256]*B]:
  fb.extend_prefix_lens.copy_(torch.tensor(vals,device='cuda'))
  for z,old in zip(states,snapshots):z.conv[0].copy_(old)
  graph.replay();torch.cuda.synchronize()
  captured_states=[z.conv[0].clone() for z in states]
  assert captured_metadata._ax_prefix_mask_source is None
  for z,old in zip(states,snapshots):z.conv[0].copy_(old)
  expected=run(case,False)
  assert all(torch.equal(a,b) for a,b in zip(captured,expected))
  assert all(torch.equal(z.conv[0],v) for z,v in zip(states,captured_states))
 rows.append(dict(B=B,lengths=lens,bitwise_conv_output=True,bitwise_conv_state=True,shared_objects=34,baseline_us=base,candidate_us=new,speedup_pct=100*(base-new)/base,graph_capture_fallback=True))
 print(rows[-1],flush=True)
 # Capture only the forward body, with a matching eager-prepared mask. It must
 # still record comparisons from static prefix input, not retain that eager mask.
 run(case,True);outside_metadata=be.forward_metadata;g['_AX_KDA_SHARED_PREFIX_MASK']=True
 bodygraph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(bodygraph):body_out=run(case,True,init=False)
 for vals in [[0]*B,[512]*B]:
  fb.extend_prefix_lens.copy_(torch.tensor(vals,device='cuda'))
  for z,old in zip(states,snapshots):z.conv[0].copy_(old)
  bodygraph.replay();torch.cuda.synchronize();body_states=[z.conv[0].clone() for z in states]
  for z,old in zip(states,snapshots):z.conv[0].copy_(old)
  expected=run(case,False)
  assert all(torch.equal(a,b) for a,b in zip(body_out,expected))
  assert all(torch.equal(z.conv[0],v) for z,v in zip(states,body_states))
 rows[-1]['graph_init_outside_capture_bitwise']=True
 run(case,True);original=fb.extend_prefix_lens; replacement=original.clone();replacement.fill_(0)
 assert torch.equal(g['_ax_kda_extend_prefix_mask'](be.forward_metadata,replacement),replacement>0)
 rows[-1]['replacement_tensor_fallback']=True
 # Reinitialize the SAME backend after a different live length/row layout.
 if B>1:
  fb.extend_seq_lens_cpu=list(reversed(lens));fb.query_start_loc=torch.tensor([0]+list(torch.tensor(fb.extend_seq_lens_cpu).cumsum(0).tolist()),device='cuda',dtype=torch.int32)
  for z,old in zip(states,snapshots):z.conv[0].copy_(old)
  expected=run(case,False);expected_states=[z.conv[0].clone() for z in states]
  for z,old in zip(states,snapshots):z.conv[0].copy_(old)
  changed=run(case,True)
  assert all(torch.equal(a,b) for a,b in zip(changed,expected))
  assert all(torch.equal(z.conv[0],v) for z,v in zip(states,expected_states))
 rows[-1]['length_reinit_bitwise']=True
 for kind in ['decode','verify','draft']:
  fb.forward_mode=Mode(kind);g['_AX_KDA_SHARED_PREFIX_MASK']=True;be.init_forward_metadata(fb)
  assert be.forward_metadata._ax_prefix_mask_source is None
 fb.forward_mode=mode
 rows[-1]['decode_verify_draft_no_shared_mask']=True
# Mask-only actual helper cost includes one metadata preparation and 34 method calls.
mask_rows=[]
for B in [1,2,8,48]:
 case=make(B,[1]*B);fb,be,*_=case
 def maskrun(enabled):
  g['_AX_KDA_SHARED_PREFIX_MASK']=enabled;be.init_forward_metadata(fb)
  for _ in range(34):g['_ax_kda_extend_prefix_mask'](be.forward_metadata,fb.extend_prefix_lens)
 t0=timeit(lambda:maskrun(False),100);t1=timeit(lambda:maskrun(True),100)
 mask_rows.append(dict(B=B,baseline_us=t0,candidate_us=t1,saved_us=t0-t1))
if args.qualify_only:
 for row in rows:
  for k in ['baseline_us','candidate_us','speedup_pct']:row.pop(k,None)
 mask_rows=[]
r=dict(qualification_only=args.qualify_only,conv_kernel_width=4,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),scope='Exact AST-extracted KDA init/forward methods, actual Triton conv34 layers at real per-rank width3072; KDA scan stub returning q; not full model.',conv_cases=rows,mask_cases=mask_rows)
Path(args.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
