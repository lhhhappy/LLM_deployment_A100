#!/usr/bin/env python3
"""CPU-only trace interval audit, including overlap and ordered graph nodes."""
import bisect
import collections
import gzip
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/execution-0930/pod/decode-profile'
spec=importlib.util.spec_from_file_location('ledger',ROOT/'scripts/pod/verify/prof_ledger.py')
ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)

def cat(event):
 n=event['name'];c=event.get('cat')
 if c in ('gpu_memcpy','gpu_memset'):return c
 if 'humming<' in n:
  if 'Shape<0u, 512u, 4096u>' in n:return 'moe_humming_up'
  if 'Shape<0u, 4096u, 256u>' in n:return 'moe_humming_down'
  return 'humming_other'
 if n=='_paged':return 'dsa_indexer_baseline'
 if n=='main_kernel' and event['args'].get('grid')==[32,1,1] and event['args'].get('shared memory')==163840:return 'dsa_attention_inferred'
 if 'track_mamba_states' in n:return 'kda_state_checkpoint'
 if 'delta_rule_update' in n or n in ('_causal_conv1d_update_kernel','layer_norm_gated_fwd_kernel'):return 'kda'
 if n in ('_router_triton_kernel','act_and_mul_kernel') or 'count_and_sort_expert_tokens_kernel' in n:return 'moe'
 if n=='_act_quant_kernel' or '_kpool_decode_update' in n:return 'dsa_indexer_aux'
 if 'marlin::Marlin' in n:return 'marlin_linear_unresolved'
 c=ledger.cat_of(n)
 return 'gemm_callsite_unresolved' if c=='gemm' else c

def clipped(events,a,b):
 return [(max(a,x['ts']),min(b,x['ts']+x['dur']),x) for x in events if x['ts']<b and x['ts']+x['dur']>a]

def sweep(intervals,a,b):
 edges=[]
 for i,(s,e,x) in enumerate(intervals):
  edges.extend([(s,1,i),(e,-1,i)])
 edges.sort();active=set();last=a;exclusive=collections.defaultdict(float);alone=collections.defaultdict(float);coactive=collections.defaultdict(float);gaps=[]
 for t,sign,i in edges+[(b,0,-1)]:
  if t>last:
   kinds=sorted({cat(intervals[j][2]) for j in active});dt=t-last
   if not active:gaps.append((last,t))
   elif len(kinds)==1:exclusive[kinds[0]]+=dt
   else:coactive[' + '.join(kinds)]+=dt
   if len(active)==1:alone[kinds[0]]+=dt
  if sign==1:active.add(i)
  elif sign==-1:active.remove(i)
  last=t
 return dict(exclusive_category_us=dict(exclusive),multiple_category_overlap_us=dict(coactive),single_kernel_only_us=dict(alone),no_activity_us=sum(e-s for s,e in gaps),gaps=gaps)

allr=[];ordered=[];names={};rank_steps=[]
for rank in range(8):
 path=next((OUT/'traces').glob(f'*TP-{rank}.trace.json.gz'))
 events=json.load(gzip.open(path))['traceEvents']
 kernels=sorted([x for x in events if x.get('cat')=='kernel'],key=lambda x:x['ts'])
 activity=kernels+[x for x in events if x.get('cat') in ('gpu_memcpy','gpu_memset')]
 gpu_ann=[x for x in events if x.get('cat')=='gpu_user_annotation' and x.get('name','').startswith('step[DECODE')]
 steps=ledger.logical_gpu_steps(gpu_ann);rank_steps.append(steps)
 cpu_steps=sorted([x for x in events if x.get('cat')=='user_annotation' and x.get('name','').startswith('step[DECODE')],key=lambda x:x['ts'])
 launches=sorted([x for x in events if x.get('cat') in ('cuda_runtime','cuda_driver') and x['name']=='cudaGraphLaunch'],key=lambda x:x['ts'])
 result=[]
 for step_id,(a,b,n) in enumerate(steps):
  iv=clipped(activity,a,b);ki=clipped(kernels,a,b);su=sweep(iv,a,b)
  kernelbusy=ledger.union_len([(s,e) for s,e,x in ki]);busy=ledger.union_len([(s,e) for s,e,x in iv]);sums=collections.defaultdict(float);counts=collections.Counter()
  for s,e,x in ki:sums[cat(x)]+=e-s;counts[cat(x)]+=1
  cpu=cpu_steps[step_id]; launch=launches[step_id]
  graph=[x for x in kernels if x['args'].get('correlation')==launch['args']['correlation']]
  gapdetails=[]
  for s,e in sorted(su['gaps'],key=lambda x:x[1]-x[0],reverse=True)[:8]:
   before=[x for x in activity if x['ts']+x['dur']<=s+0.002]
   after=[x for x in activity if x['ts']>=e-0.002]
   prev=max(before,key=lambda x:x['ts']+x['dur']) if before else None
   nxt=min(after,key=lambda x:x['ts']) if after else None
   cpu_overlap=clipped([x for x in events if x.get('cat') in ('cpu_op','cuda_runtime','cuda_driver','user_annotation') and x.get('ph')=='X'],s,e)
   gapdetails.append(dict(start_from_step_us=s-a,duration_us=e-s,previous_activity=prev['name'] if prev else None,next_activity=nxt['name'] if nxt else None,next_graph_node=nxt.get('args',{}).get('graph node id') if nxt else None,next_launch_already_submitted=bool(nxt and nxt.get('args',{}).get('correlation')==launch['args']['correlation'] and launch['ts']+launch['dur']<s),cpu_overlapping_names=sorted({x['name'] for _,_,x in cpu_overlap})[:12]))
  su.pop('gaps')
  result.append(dict(step=step_id,name=n,start_us=a,end_us=b,span_us=b-a,kernel_busy_union_us=kernelbusy,all_gpu_activity_busy_union_us=busy,kernel_duration_sum_us=sum(e-s for s,e,x in ki),category_duration_sums_not_critical_path_us=dict(sums),kernel_counts=dict(counts),cpu_step_span_us=cpu['dur'],graph_launch_cpu_us=launch['dur'],cpu_step_end_vs_gpu_step_end_us=b-(cpu['ts']+cpu['dur']),graph_launch_end_vs_first_graph_kernel_us=min(x['ts'] for x in graph)-(launch['ts']+launch['dur']),largest_unattributed_gaps=gapdetails,**su))
  if rank==0 and step_id==1:
   for x in sorted(graph,key=lambda x:x['ts']):
    if cat(x) in ('moe_humming_up','moe_humming_down','dsa_indexer_baseline','dsa_attention_inferred','comm','kda_state_checkpoint'):
     nm=x['name'];nid=next((key for key,value in names.items() if value['full_name']==nm),None)
     if nid is None:
      nid=str(len(names));names[nid]=dict(full_name=nm,category=cat(x))
     ordered.append(dict(name_id=nid,start_from_step_us=x['ts']-a,duration_us=x['dur'],stream=x['tid'],graph_node_id=x['args'].get('graph node id'),grid=x['args'].get('grid'),block=x['args'].get('block'),shared_memory_bytes=x['args'].get('shared memory'),registers_per_thread=x['args'].get('registers per thread')))
 copies=collections.defaultdict(list)
 for x in events:
  if x.get('cat')=='gpu_memcpy':copies[(x['name'],x['args'].get('bytes'))].append(x['dur'])
 allr.append(dict(rank=rank,trace=path.name,gpu_annotation_count=len(gpu_ann),logical_step_count=len(steps),steps=result,copies=[dict(name=k[0],bytes=k[1],calls=len(v),total_us=sum(v),median_us=statistics.median(v)) for k,v in copies.items()]))

# Rank timestamps share one trace host time domain, but producer clocks are not
# independently calibrated; comparisons are diagnostic, not a dependency DAG.
aligned=[]
for i in range(5):
 ss=[r['steps'][i] for r in allr]
 aligned.append(dict(step=i,earliest_start_us=min(s['start_us'] for s in ss),latest_end_us=max(s['end_us'] for s in ss),rank_envelope_us=max(s['end_us'] for s in ss)-min(s['start_us'] for s in ss),rank_start_spread_us=max(s['start_us'] for s in ss)-min(s['start_us'] for s in ss),rank_end_spread_us=max(s['end_us'] for s in ss)-min(s['end_us'] for s in ss)))
summary=[]
for r in allr:
 ss=r['steps']; summary.append(dict(rank=r['rank'],span_mean_ms=statistics.mean(s['span_us'] for s in ss)/1000,kernel_busy_mean_ms=statistics.mean(s['kernel_busy_union_us'] for s in ss)/1000,all_activity_busy_mean_ms=statistics.mean(s['all_gpu_activity_busy_union_us'] for s in ss)/1000,no_activity_mean_ms=statistics.mean(s['no_activity_us'] for s in ss)/1000,category_exclusive_mean_ms={c:sum(s['exclusive_category_us'].get(c,0) for s in ss)/5000 for c in set().union(*(s['exclusive_category_us'] for s in ss))},multi_category_overlap_mean_ms=sum(sum(s['multiple_category_overlap_us'].values()) for s in ss)/5000,steady_span_median_ms=statistics.median(s['span_us'] for s in ss[1:])/1000))
payload=dict(scope='Read-only CPU audit; warm8187-token prefix, identical requests, B32, five isolated profiled decode forwards. No mixed workload/p95/SLO claim. GPU profiling changes timings.',interval_method='GPU annotations grouped by External id across streams; clipped kernel+memcpy+memset intervals swept per rank. Exclusive categories plus mixed-category overlap plus no-activity exactly partition each wall span. Category duration sums are auxiliary and never called critical-path shares. Cross-rank times are ordinal-aligned diagnostic envelopes, not a full dependency DAG.',classifications='humming dimensions decoded as MoE up/down; _paged baseline DSA indexer; main_kernel inferred DSA BF16 from [32,1,1],163840 bytes and11 calls/step. Captured graph correlation points only to cudaGraphLaunch; no individual Python callsite attribution available. Generic cuBLAS/CUTLASS/Marlin remain unresolved.',summary=summary,ranks=allr,ordinal_rank_envelopes=aligned,ordered_hot_kernels_rank0_step1=ordered,kernel_names=names)
(OUT/'criticalpath-audit.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(summary,indent=2))
