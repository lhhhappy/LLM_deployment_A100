#!/usr/bin/env python3
"""T56 CPU-only reconstruction. No network/engine calls; writes only beside this file.

Run: PYTHONDONTWRITEBYTECODE=1 /tmp/t42-venv/bin/python evidence/T56/attribute.py
Requires numpy and transformers; uses the original harness Renderer and gate selector.
Input hashes bind every output; --render-only produces the prompt-pair evidence.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 's1-dev/harness'))
from s1_common import Renderer, materialize_bodies, load_index, in_ttft_gate
import numpy as np

RAW = ROOT / 'evidence/T53/026_N18_raw.jsonl'
DATA = ROOT / 's1-dev/data/dev-combined-v1'
COHORT = ROOT / 's1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json'
LOG = OUT / '026_server.log'
EXPECTED_RAW = '486f041027abde2a1702e0c19190ce11e7001eb0cc4055d3773aff8c95bfdb59'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def lcp(a, b):
    n = min(len(a), len(b))
    # Many cross-session pairs diverge in the first page. Avoid full-array work.
    for start, stop in ((0, min(n, 1024)), (1024, n)):
        if start >= stop:
            continue
        diff = np.flatnonzero(a[start:stop] != b[start:stop])
        if len(diff):
            return start + int(diff[0])
    return n

def render_pairs(raw):
    bodies = materialize_bodies(str(DATA), [r['req_id'] for r in raw])
    assert len(bodies) == len(raw)
    renderer = Renderer(str(ROOT / 's1-dev/glm_tok'))
    tokens, meta = {}, {}
    for i, r in enumerate(raw):
        rid = r['req_id']
        text = renderer.render(bodies[rid])
        ids = np.asarray(renderer.tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
        assert len(ids) == r['prompt_tokens'] == r['glm_tokens'], (rid, len(ids), r['prompt_tokens'])
        roles = np.flatnonzero(np.isin(ids, [154827,154829])).tolist()
        tokens[rid] = ids
        meta[rid] = dict(raw_line=i+1, prompt_tokens=len(ids), rendered_sha256=hashlib.sha256(text.encode()).hexdigest(),
                         token_i32le_sha256=hashlib.sha256(ids.astype('<i4').tobytes()).hexdigest(),
                         role_positions=roles, role_depth=(roles[-1]//64*64 if roles else 0))
        if (i+1)%100 == 0:
            print('rendered', i+1, flush=True)
    del bodies
    by_chain = {(r['chain_id'], r['idx_in_chain']): r for r in raw}
    pairs=[]
    for r in raw:
        if r['idx_in_chain'] == 0:
            continue
        p=by_chain[(r['chain_id'], r['idx_in_chain']-1)]
        assert p['client_finish_at_s'] <= r['client_dispatch_at_s']
        L=lcp(tokens[p['req_id']], tokens[r['req_id']])
        frozen=r['prompt_tokens']-r['uncached_expected']
        cached=r['cached_tokens']
        ancestors=[]
        for idx in range(r['idx_in_chain']):
            a=by_chain[(r['chain_id'],idx)]
            A=lcp(tokens[a['req_id']],tokens[r['req_id']])
            m=meta[a['req_id']]
            ancestors.append(dict(raw_line=m['raw_line'],idx=idx,lcp=A,cached=a['cached_tokens'],
                                  role=m['role_depth'], role_inside_lcp=m['role_depth']<=A,
                                  prompt_end_floor64=a['prompt_tokens']//64*64,
                                  decode_last_candidate256=(a['prompt_tokens']+a['output_tokens']-1)//256*256))
        cross=[]
        for a in raw:
            if a['session_id']==r['session_id'] or a['client_finish_at_s']>r['t_exec_start_s']:
                continue
            x=lcp(tokens[a['req_id']],tokens[r['req_id']])
            cross.append((x,meta[a['req_id']]['raw_line']))
        cross.sort(reverse=True)
        pairs.append(dict(raw_line=meta[r['req_id']]['raw_line'],previous_line=meta[p['req_id']]['raw_line'],
                          req_id=r['req_id'],chain_id=r['chain_id'],idx=r['idx_in_chain'],edge_type=r['edge_type'],
                          prompt=r['prompt_tokens'],previous_prompt=p['prompt_tokens'],cached=cached,previous_cached=p['cached_tokens'],
                          true_lcp=L,frozen_lcp=frozen,frozen_overestimate=max(0,frozen-L),
                          frozen_positive_loss=max(0,frozen-cached),true_positive_loss=max(0,L-cached),
                          aligned_loss=max(0,L//64*64-cached),beyond_tolerance_loss=max(0,L-cached-64),
                          prior_kv_overlap_gap=max(0,min(L,p['cached_tokens'])-cached),
                          verified_prior_hit_regression=(max(0,p['cached_tokens']-cached) if L>=p['cached_tokens'] else 0),
                          previous_role=meta[p['req_id']]['role_depth'],previous_role_positions=meta[p['req_id']]['role_positions'],
                          cached_at_previous_role=cached==meta[p['req_id']]['role_depth'],
                          cached_at_any_previous_role=cached in [x//64*64 for x in meta[p['req_id']]['role_positions']],
                          prior_ancestors=ancestors,cross_session_best_completed=cross[:3],
                          gap_s=r['client_dispatch_at_s']-p['client_finish_at_s'],
                          recv_to_exec_s=r['t_exec_start_s']-r['t_recv_s'],
                          exec_to_first_s=r['t_first_token_s']-r['t_exec_start_s'],
                          ttft_s=r['ttft_s'],output=r['output_tokens'],tpot_s=r['tpot_s']))
    assert len(pairs)==411
    dump('prompt_metadata.json',meta)
    dump('pairs_rendered.json',pairs)
    return pairs

def classify_pairs(pairs):
    """Location attribution, NOT proof of particular eviction events.

    d applies only when the entire apparent miss is explained within 64 tokens.
    c means zero hit, or a >8192 miss falling wholly inside a verified earlier
    completed OTHER-session common prefix. This is an explicit operational rule.
    Frozen overestimate remains an independent field, including b/c residuals.
    """
    groups=collections.defaultdict(list)
    for p in pairs:
        C=p['cached']; L=p['true_lcp']
        # A formerly used KDA state is comparable ONLY when the new prompt
        # still contains its complete prefix. min(L, old_hit) is just KV overlap:
        # an old state beyond the new fork was never reusable by this request.
        p['prior_kv_overlap_gap']=max(0,min(L,p['previous_cached'])-C)
        p['verified_prior_hit_regression']=max(0,p['previous_cached']-C) if L>=p['previous_cached'] else 0
        cross=p['cross_session_best_completed'][0][0] if p['cross_session_best_completed'] else 0
        if p['true_positive_loss']<=64:
            category='d' if p['frozen_overestimate']>0 else 'a'
        elif C==0 or (p['true_positive_loss']>8192 and C<=cross//64*64):
            category='c'
        else:
            category='b'
        sources=[]
        if C==p['previous_role']:
            sources.append(f"previous_role:{p['previous_line']}@{C}")
        elif p['cached_at_any_previous_role']:
            sources.append(f"earlier_role_in_previous_prompt:{p['previous_line']}@{C}")
        for a in p['prior_ancestors']:
            if a['lcp']<C:
                continue
            if a['role']==C and a['raw_line']!=p['previous_line']:
                sources.append(f"ancestor_role:{a['raw_line']}@{C}")
            if a['cached']==C:
                sources.append(f"previously_used_hit:{a['raw_line']}@{C}")
            if C>a['cached'] and (C-a['cached'])%8192==0:
                sources.append(f"chunk_grid_candidate:{a['raw_line']}:{a['cached']}+{(C-a['cached'])//8192}*8192={C}")
            if C==a['prompt_end_floor64']:
                sources.append(f"prompt_end_candidate:{a['raw_line']}@{C}")
            if C==a['decode_last_candidate256']:
                sources.append(f"decode_grid_candidate:{a['raw_line']}@{C}")
        p.update(category=category,classification_status='VERIFIED' if category in ('a','d') else 'INFERRED',
                 hit_point_depth=C,point_matches=sources or [f'earlier_unidentified_state@{C}'],
                 c_rule_cross_lcp=cross,c_rule_loss_threshold=8192)
        groups[category].append(p)
    summary={k:dict(n=len(ps),prompt_tokens=sum(p['prompt'] for p in ps),true_lcp=sum(p['true_lcp'] for p in ps),
                    cached_tokens=sum(p['cached'] for p in ps),raw_positive_gap=sum(p['true_positive_loss'] for p in ps),
                    aligned_gap=sum(p['aligned_loss'] for p in ps),frozen_overestimate=sum(p['frozen_overestimate'] for p in ps))
             for k,ps in sorted(groups.items())}
    assert sum(g['n'] for g in summary.values())==411
    assert all(p['cached']%64==0 for p in pairs)
    dump('pairs_attributed.json',pairs)
    fields=['raw_line','previous_line','req_id','idx','edge_type','prompt','previous_prompt','cached','previous_cached',
            'true_lcp','frozen_lcp','frozen_overestimate','true_positive_loss','aligned_loss','category',
            'classification_status','hit_point_depth','point_matches','verified_prior_hit_regression','gap_s',
            'recv_to_exec_s','exec_to_first_s','ttft_s']
    with (OUT/'pairs_attributed.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for p in pairs:
            w.writerow({k:('; '.join(p[k]) if k=='point_matches' else p[k]) for k in fields})
    dump('classification_summary.json',summary)
    return summary

def parse_batches():
    batches=[]
    # Keep POSIX line numbers: startup progress bars contain CR, not new lines.
    for line,text in enumerate(LOG.read_bytes().decode().split('\n'),1):
        if 'Prefill batch' not in text:
            continue
        m=re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:[,.](\d+))? TP0\]',text)
        if not m:
            continue
        t=dt.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()
        fields={k:int(v) for k,v in re.findall(r'#([\w-]+): (\d+)',text)}
        rate=float(re.search(r'input throughput \(token/s\): ([\d.]+)',text)[1])
        b=dict(line=line,local_time=m[1],log_second=t,rate=rate,**fields)
        if rate>0:
            b['inter_report_s']=fields['new-token']/rate
            b['rate_rounding_error_s']=max(fields['new-token']/(rate-.005)-b['inter_report_s'],
                                           b['inter_report_s']-fields['new-token']/(rate+.005))
        batches.append(b)
    assert batches
    return batches

def align_time(raw,batches):
    # Independent shape anchors: complete, single-request, one-chunk batches.
    # Rounded input budget is ceil64; log_hit_tokens equals the admitted hit.
    potential=[]
    for i,r in enumerate(raw,1):
        new=r['prompt_tokens']-r['cached_tokens']
        if new>16384 or r['t_first_token_s']-r['t_exec_start_s']>2:
            continue
        for b in batches:
            if b['new-seq']==1 and b['pending-token']==0 and b['cached-token']==r['cached_tokens'] and new<=b['new-token']<new+64:
                delta=b['log_second']-r['t_first_token_s']
                offset=round(delta/3600)*3600
                if -1.05<delta-offset<0.10:
                    potential.append((offset,i,b['line'],delta))
    counts=collections.Counter(x[0] for x in potential)
    offset=counts.most_common(1)[0][0]
    assert counts[offset]>30, counts
    by_raw=collections.defaultdict(list)
    for off,i,line,delta in potential:
        if off==offset:
            by_raw[i].append((line,delta))
    anchors=[]
    for i,vals in by_raw.items():
        if len(vals)==1:
            line,delta=vals[0]
            anchors.append(dict(raw_line=i,log_line=line,log_minus_first_s=delta,
                                first_epoch=raw[i-1]['t_first_token_s'],exec_epoch=raw[i-1]['t_exec_start_s'],
                                recv_epoch=raw[i-1]['t_recv_s']))
    assert len(anchors)>30
    # Throughput measures time between stats calls, NOT GPU duration. Its printed
    # numerator lets us recover subsecond report spacing, even with seconds-only logs.
    # Keep each reconstruction segment short; use whole-second boxes and optional
    # unique first-token anchors (stats emitted after first-token event, ~milliseconds).
    amap={a['log_line']:a for a in anchors}
    segments=[]; current=[];base=0.0;low=-float('inf');high=float('inf')
    def finish():
        if not current:return
        shift=(low+high)/2
        for b,cum in current:
            b['estimated_end']=shift+cum
            b['end_lo']=low+cum
            b['end_hi']=high+cum
            b['time_segment']=len(segments)
        segments.append(dict(first_line=current[0][0]['line'],last_line=current[-1][0]['line'],
                             n=len(current),uncertainty_s=high-low))
    for b in batches:
        second=b['log_second']-offset
        blo,bhi=second,second+1
        if b['line'] in amap:
            a=amap[b['line']]
            # This reconstruction is INFERRED. 50 ms bounds are explicit, and
            # exact first-token events/whole-second log labels are also retained.
            blo=max(blo,a['first_epoch']-.002);bhi=min(bhi,a['first_epoch']+.050)
        delta=b.get('inter_report_s',0)
        nextbase=base+delta if current else 0
        nlo=max(low,blo-nextbase);nhi=min(high,bhi-nextbase)
        if current and (nlo>nhi or len(current)>=100 or delta>5 or b.get('rate_rounding_error_s',0)>.005):
            finish();current=[];base=0;low=-float('inf');high=float('inf');nextbase=0
            nlo,nhi=blo,bhi
        assert nlo<=nhi,(b,blo,bhi)
        current.append((b,nextbase));base=nextbase;low=nlo;high=nhi
    finish()
    for j,b in enumerate(batches):
        b['estimated_interval_start']=b['estimated_end']-b.get('inter_report_s',0)
        b['previous_line']=batches[j-1]['line'] if j else None
    result=dict(local_minus_utc_s=offset,offset_votes=dict(counts),shape_anchors=anchors,
                anchors_n=len(anchors),segments=segments,
                method='shape-matched single complete batch + raw first-token epoch; subsecond estimates from current new/rate = inter-report spacing, constrained by second-resolution log and first-token anchors')
    dump('time_alignment.json',result)
    dump('prefill_batches.json',batches)
    return result

def map_requests(raw,batches):
    """Infer batch membership, then validate against ALL logged sequence/token counts.

    Request first-token events are emitted just before its completed-batch stats.
    Overlap may delay the previous batch's log until another batch is in flight.
    Solve admission subsets, rather than declaring all timestamp overlap membership.
    """
    groups=[]
    for i in sorted(range(len(raw)),key=lambda i:raw[i]['t_first_token_s']):
        if groups and raw[i]['t_first_token_s']-raw[groups[-1][-1]]['t_first_token_s']<.01:
            groups[-1].append(i)
        else:groups.append([i])
    ends={}
    for group in groups:
        ts=max(raw[i]['t_first_token_s'] for i in group)
        candidates=[j for j,b in enumerate(batches) if abs(b['estimated_end']-ts)<.1 and b['new-seq']>=len(group)]
        assert candidates,group
        j=min(candidates,key=lambda j:abs(batches[j]['estimated_end']-ts))
        for i in group:ends[i]=j
    earliest={i:next(j for j in range(ends[i]+1) if batches[j]['estimated_end']>r['t_exec_start_s']+.1)
              for i,r in enumerate(raw)}
    started={};work=collections.Counter();mapping=collections.defaultdict(list)
    checks=[]
    for j in range(min(earliest.values()),max(ends.values())+1):
        b=batches[j]
        ongoing=[i for i in started if ends[i]>=j]
        available=[i for i in earliest if i not in started and earliest[i]<=j<=ends[i]]
        needed=b['new-seq']-len(ongoing)
        choices=[list(xs) for xs in itertools.combinations(available,needed) if needed>=0
                 and sum(raw[i]['cached_tokens'] for i in xs)==b['cached-token']
                 and all(i in xs for i in available if ends[i]==j)] if needed>=0 else []
        assert len(choices)==1,(b['line'],ongoing,available,choices)
        admitted=choices[0]
        for i in admitted:started[i]=j
        members=ongoing+admitted
        assert len(members)==b['new-seq']
        finishing=[i for i in members if ends[i]==j]
        partial=[i for i in members if ends[i]>j]
        assert len(partial)<=1
        amounts={i:((raw[i]['prompt_tokens']-raw[i]['cached_tokens']+63)//64*64)-work[i] for i in finishing}
        rem=b['new-token']-sum(amounts.values())
        assert (partial and rem>0 and rem%64==0) or (not partial and rem==0),(b['line'],members,amounts,rem)
        if partial:amounts[partial[0]]=rem
        assert sum(amounts.values())==b['new-token']
        for i,amount in amounts.items():
            before=raw[i]['cached_tokens']+work[i]
            work[i]+=amount
            depth=min(raw[i]['cached_tokens']+work[i],raw[i]['prompt_tokens'])//64*64
            mapping[i+1].append(dict(log_line=b['line'],new_tokens_budget=amount,prefix_before=before,
                                      chunk_end_floor64=depth,final=ends[i]==j))
        b['raw_members']=[i+1 for i in members]
        b['raw_admitted']=[i+1 for i in admitted]
        b['raw_finished']=[i+1 for i in finishing]
        # First admission starts the current forward; otherwise previous report
        # spacing includes the interleaved decode/host work (kept as a proxy).
        b['prefill_start_estimate']=max(raw[i]['t_exec_start_s'] for i in admitted) if admitted else b['estimated_interval_start']
        b['start_method']='raw_exec_start' if admitted else 'previous_report_proxy'
        if admitted and not partial:
            b['prefill_end_estimate']=max(raw[i]['t_first_token_s'] for i in finishing)
        else:b['prefill_end_estimate']=b['estimated_end']
        b['wall_span_estimate']=max(0,b['prefill_end_estimate']-b['prefill_start_estimate'])
        checks.append(dict(log_line=b['line'],n_members=len(members),new_tokens=sum(amounts.values()),
                           cached_tokens=sum(raw[i]['cached_tokens'] for i in admitted)))
    assert len(started)==722
    assert all(work[i]==(r['prompt_tokens']-r['cached_tokens']+63)//64*64 for i,r in enumerate(raw))
    receipt=dict(status='VERIFIED aggregate constraints; membership INFERRED',requests=722,batches=len(checks),
                 all_new_seq_counts_match=True,all_cached_token_sums_match=True,all_new_token_sums_match=True,
                 all_request_prefill_budgets_match=True,checks=checks)
    dump('request_batch_map.json',mapping)
    dump('batch_mapping_validation.json',receipt)
    return mapping

def decode_baseline(raw):
    lo=min(r['t_recv_s'] for r in raw);hi=max(r['client_finish_at_s'] for r in raw)
    previous=None;had_prefill=False;receipts=[]
    for line,text in enumerate(LOG.read_bytes().decode().split('\n'),1):
        if 'Prefill batch' in text:had_prefill=True
        if 'Decode batch' not in text:continue
        m=re.match(r'\[(.{19}) TP0\]',text)
        if not m:continue
        t=dt.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()
        n=int(re.search(r'#running-req: (\d+)',text)[1])
        rate=float(re.search(r'gen throughput \(token/s\): ([\d.]+)',text)[1])
        if previous and not had_prefill and n==previous['running'] and lo<previous['time'] and t<hi and rate>0:
            receipts.append(dict(line=line,previous_line=previous['line'],running=n,throughput=rate,step_estimate=n/rate))
        previous=dict(line=line,time=t,running=n);had_prefill=False
    by_n={n:float(np.median([x['step_estimate'] for x in receipts if x['running']==n]))
          for n in sorted({x['running'] for x in receipts})}
    dump('decode_baseline.json',dict(status='INFERRED: same running count at adjacent reports with no prefill between; not a GPU timer',
                                    by_running=by_n,samples=receipts))
    return by_n

def refine_points(pairs,mapping):
    for p in pairs:
        hits=[]
        for a in p['prior_ancestors']:
            if a['lcp']<p['cached']:continue
            for event in mapping[a['raw_line']]:
                if event['chunk_end_floor64']==p['cached']:
                    hits.append(dict(ancestor_raw_line=a['raw_line'],**event))
        p['matched_chunk_events']=hits
        if p['cached_at_previous_role']:
            p['point_kind']='previous_last_role'
        elif p['cached_at_any_previous_role']:
            p['point_kind']='previous_earlier_role'
        elif any(a['role']==p['cached'] and a['lcp']>=p['cached'] for a in p['prior_ancestors']):
            p['point_kind']='ancestor_last_role'
        elif any(e['ancestor_raw_line']==p['previous_line'] for e in hits):
            p['point_kind']='previous_chunk_end'
        elif hits:p['point_kind']='ancestor_chunk_end'
        elif p['cached']==p['previous_cached'] and p['cached']>0:
            p['point_kind']='earlier_inherited_hit'
        elif p['cached']==0:p['point_kind']='root_no_hit'
        else:p['point_kind']='earlier_state_unidentified_origin'
    dump('pairs_attributed.json',pairs)
    columns=['raw_line','previous_line','req_id','idx','prompt','cached','true_lcp','frozen_lcp','frozen_overestimate',
             'true_positive_loss','aligned_loss','category','classification_status','point_kind','hit_point_depth',
             'matched_chunk_events','verified_prior_hit_regression','recv_to_exec_s','exec_to_first_s','ttft_s']
    with (OUT/'pairs_attributed.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for p in pairs:w.writerow({k:json.dumps(p[k],ensure_ascii=False) if k=='matched_chunk_events' else p[k] for k in columns})

def overlap(lo,hi,start,end):
    return max(0,min(hi,end)-max(lo,start))

def union_length(intervals):
    total=0;end=-float('inf')
    for lo,hi in sorted(intervals):
        total+=max(0,hi-max(lo,end));end=max(end,hi)
    return total

def detail_entry(b,lo,hi):
    return dict(log_line=b['line'],local_time=b['local_time'],new_tokens=b['new-token'],cached_tokens=b['cached-token'],
                new_seqs=b['new-seq'],raw_members=b['raw_members'],running=b['running-req'],queue=b['queue-req'],
                inter_report_s=b['inter_report_s'],span_start=b['pause_start_estimate'],span_end=b['prefill_end_estimate'],
                pause_estimate_s=b['pause_estimate_s'],overlap_pause_estimate_s=overlap(lo,hi,b['pause_start_estimate'],b['prefill_end_estimate']),
                start_method=b['start_method'],end_lo=b['end_lo'],end_hi=b['end_hi'])

def write_details(raw,pairs,batches):
    measured=[b for b in batches if 'raw_members' in b]
    baseline=decode_baseline(raw)
    for b in measured:
        # Proxy-only continuation report spacing includes at least the scheduled
        # interleaved decode. First admission has its own raw start timestamp.
        nearest=min(baseline,key=lambda n:abs(n-b['running-req']))
        correction=baseline[nearest] if b['start_method']=='previous_report_proxy' and b['running-req']>0 else 0
        b['decode_correction_s']=correction
        b['pause_start_estimate']=min(b['prefill_end_estimate'],b['prefill_start_estimate']+correction)
        b['pause_estimate_s']=max(0,b['prefill_end_estimate']-b['pause_start_estimate'])
    pair_by_line={p['raw_line']:p for p in pairs}
    fast=[];short=[]
    for i,r in enumerate(raw,1):
        if in_ttft_gate(r,'fast_intra') and r['ttft_s']>3:
            p=pair_by_line[i]
            arrival=[detail_entry(b,r['t_recv_s'],r['t_exec_start_s']) for b in measured
                     if b['pause_start_estimate']<=r['t_recv_s']<b['prefill_end_estimate'] and i not in b['raw_members']]
            queue=[detail_entry(b,r['t_recv_s'],r['t_exec_start_s']) for b in measured
                   if overlap(r['t_recv_s'],r['t_exec_start_s'],b['pause_start_estimate'],b['prefill_end_estimate'])>0 and i not in b['raw_members']]
            execution=[detail_entry(b,r['t_exec_start_s'],r['t_first_token_s']) for b in measured if i in b['raw_members']]
            reason='head_of_line' if p['true_positive_loss']<=64 else ('cache_loss_and_head_of_line' if p['recv_to_exec_s']>1 else 'cache_loss')
            fast.append(dict(raw_line=i,req_id=r['req_id'],previous_line=p['previous_line'],cached=p['cached'],true_lcp=p['true_lcp'],
                             true_loss=p['true_positive_loss'],aligned_loss=p['aligned_loss'],frozen_overestimate=p['frozen_overestimate'],
                             recv_epoch=r['t_recv_s'],exec_epoch=r['t_exec_start_s'],first_epoch=r['t_first_token_s'],
                             recv_to_dispatch_s=r['t_admit_s']-r['t_recv_s'],dispatch_to_exec_s=r['t_exec_start_s']-r['t_admit_s'],
                             queue_time_reported_s=r['queue_time_s'],queue_total_s=p['recv_to_exec_s'],execution_s=p['exec_to_first_s'],ttft_s=r['ttft_s'],
                             arrival_batches=arrival,queue_batches=queue,execution_batches=execution,
                             reason_inferred=reason))
        if r['output_tokens']<100 and r['tpot_s']>.1:
            lo=r['client_first_token_at_s'];hi=r['client_finish_at_s']
            events=[detail_entry(b,lo,hi) for b in measured if i not in b['raw_members']
                    and overlap(lo,hi,b['pause_start_estimate'],b['prefill_end_estimate'])>0]
            intervals=[(max(lo,e['span_start']),min(hi,e['span_end'])) for e in events]
            pause=union_length(intervals)
            assert 0<=pause<=hi-lo+1e-6
            # Sensitivity, not confidence bounds: shrink/expand both event edges
            # by 50 ms, without claiming that all sources of error fit this box.
            sensitivity={}
            for name,delta in [('shrink_50ms',.05),('expand_50ms',-.05)]:
                iv=[]
                for b in measured:
                    if i in b['raw_members']:continue
                    start=b['pause_start_estimate']+delta;end=b['prefill_end_estimate']-delta
                    if overlap(lo,hi,start,end)>0:iv.append((max(lo,start),min(hi,end)))
                sensitivity[name]=dict(n=len(iv),pause_union_s=union_length(iv))
            short.append(dict(raw_line=i,req_id=r['req_id'],output_tokens=r['output_tokens'],tpot_s=r['tpot_s'],
                              decode_start=lo,decode_end=hi,decode_window_s=hi-lo,prefill_overlap_count=len(events),
                              pause_estimate_s=pause,residual_wall_s=hi-lo-pause,events=events,sensitivity=sensitivity,
                              status='INFERRED: prefill wall spans, not per-token or CUDA events'))
    dump('fast_10.json',fast);dump('short_tpot_22.json',short)
    def table_events(events):
        lines=['|log行|pod时间|#new-token|#cached-token|raw成员|报告间隔s|批停顿估计s|窗内重叠s|',
               '|---:|---|---:|---:|---|---:|---:|---:|']
        for e in events:
            lines.append(f"|{e['log_line']}|{e['local_time']}|{e['new_tokens']}|{e['cached_tokens']}|{','.join(map(str,e['raw_members']))}|{e['inter_report_s']:.3f}|{e['pause_estimate_s']:.3f}|{e['overlap_pause_estimate_s']:.3f}|")
        return lines
    lines=['# T56 fast 10逐批附录','',
           '数值原始项 VERIFIED；批成员、时段和原因 INFERRED（完整 token 合计校验通过）。日志行号按 LF，保留原始文件的 CR 进度条。',
           '“排队”=recv→exec，含分词/传输/调度；execution=exec→first，含自身分块间隙。批停顿估计方法见 R20 与 attribute.py。','']
    for x in fast:
        lines += [f"## raw {x['raw_line']}",'',f"`{x['req_id']}`；前驱raw{x['previous_line']}。recv {x['recv_epoch']:.6f} / exec {x['exec_epoch']:.6f} / first {x['first_epoch']:.6f}。",
                  f"VERIFIED：TTFT {x['ttft_s']:.6f} = 排队 {x['queue_total_s']:.6f} + 执行 {x['execution_s']:.6f}；LCP {x['true_lcp']}，cached {x['cached']}，真实正差 {x['true_loss']}。",
                  f"INFERRED 原因：{x['reason_inferred']}。",'', '到达瞬间的候选在执行prefill：','']
        lines+=table_events(x['arrival_batches']) if x['arrival_batches'] else ['无其它prefill估计区间覆盖到达时刻；可能处于decode/入口/批间隙。']
        lines+=['','等待窗内其它prefill：','']+table_events(x['queue_batches'])
        lines+=['','自身执行批次（含合批请求）：','']+table_events(x['execution_batches'])+['']
    (OUT/'fast_details.md').write_text('\n'.join(lines)+'\n')
    lines=['# T56 短输出TPOT逐批附录','',
           '每个条目列出decode窗口内所有估计相交的其它请求prefill。原日志时间/token与raw指标 VERIFIED；归属、相交次数、停顿估计 INFERRED。',
           '报告间隔 new/throughput 并非GPU耗时；有新准入使用raw exec_start，续块从报告间隔扣一个同并发decode步。停顿总量按区间并集，避免重叠重复相加。','']
    for x in short:
        lines += [f"## raw {x['raw_line']}",'',f"`{x['req_id']}`；output={x['output_tokens']}，TPOT={x['tpot_s']:.6f}s。",
                  f"decode epoch [{x['decode_start']:.6f}, {x['decode_end']:.6f}]，窗口 {x['decode_window_s']:.3f}s。",
                  f"INFERRED：重叠 {x['prefill_overlap_count']} 批，停顿并集约 {x['pause_estimate_s']:.3f}s，剩余墙钟 {x['residual_wall_s']:.3f}s。",'']
        lines+=table_events(x['events'])+['']
    (OUT/'decode_overlaps.md').write_text('\n'.join(lines)+'\n')
    size_groups=collections.defaultdict(list)
    for b in measured:
        if b['running-req']<=0:continue
        n=b['new-token']
        key='8192' if n==8192 else ('16384' if n==16384 else ('<=4096' if n<=4096 else ('4097..8191' if n<8192 else '8193..16383')))
        size_groups[key].append(b)
    def stats(v):
        return dict(n=len(v),p10=float(np.quantile(v,.1)),p50=float(np.median(v)),p90=float(np.quantile(v,.9)))
    sizes={k:dict(pause_estimate_s=stats([b['pause_estimate_s'] for b in bs]),
                  inter_report_s=stats([b['inter_report_s'] for b in bs]),
                  admission_spans_s=stats([b['pause_estimate_s'] for b in bs if b['start_method']=='raw_exec_start'])
                  if any(b['start_method']=='raw_exec_start' for b in bs) else None)
           for k,bs in size_groups.items()}
    dump('pause_by_batch_size.json',sizes)
    dump('prefill_batches.json',batches)
    return fast,short,sizes

def analyze_log(raw,pairs,inputs):
    batches=parse_batches()
    alignment=align_time(raw,batches)
    mapping=map_requests(raw,batches)
    refine_points(pairs,mapping)
    fast,short,sizes=write_details(raw,pairs,batches)
    manifest=json.loads((OUT/'manifest.json').read_text())
    for p in [LOG,ROOT/'patches/drafts/120-sched-protect-chain-v2.patch',ROOT/'patches/140-kda-dual-snapshot.patch',
              ROOT/'build/base_exact/sglang/srt/managers/scheduler_components/metrics_reporter.py',
              ROOT/'build/base_exact/sglang/srt/managers/scheduler_components/batch_result_processor.py',
              ROOT/'build/base_exact/sglang/srt/managers/scheduler.py',ROOT/'build/base_exact/sglang/srt/observability/req_time_stats.py']:
        manifest['inputs'][str(p.relative_to(ROOT))]=sha(p)
    manifest['script_sha256']=sha(__file__)
    manifest['outputs']={p.name:sha(p) for p in OUT.glob('*') if p.suffix in ('.json','.csv','.md') and p.name!='manifest.json'}
    validation=dict(tests={
        'T56-01':dict(status='pass',requests=722,followup_pairs=411,prompt_lengths_match=722,cohort_positions_match=722,
                      raw_sha256=EXPECTED_RAW,true_positive_gap=sum(p['true_positive_loss'] for p in pairs),
                      aligned_gap=sum(p['aligned_loss'] for p in pairs)),
        'T56-02':dict(status='pass',fast_n=328,fast_over3=len(fast),short_slow=len(short),
                      ttft_additivity_max_error=max(abs(x['ttft_s']-x['queue_total_s']-x['execution_s']) for x in fast)),
        'T56-03':dict(status='pass',log_sha256=sha(LOG),timezone_offset_s=alignment['local_minus_utc_s'],
                      shape_anchors=alignment['anchors_n'],mapped_batches=sum('raw_members' in b for b in batches),
                      all_batch_counts_and_token_sums_match=True),
        'T56-04':dict(status='pass',decode_windows=len(short),overlap_counts=[s['prefill_overlap_count'] for s in short],
                      all_pause_unions_within_window=True,causal_claims_labelled_inferred=True)},
        limitations=['No per-token timestamps, CUDA events, snapshot-slot or radix-eviction events.',
                     'Batch membership validated against all count/token constraints but is reconstructed, not request-ID logging.',
                     'Pause values are inferred wall spans; 50-ms sensitivity is not a confidence interval.'])
    dump('validation.json',validation)
    manifest['outputs']['validation.json']=sha(OUT/'validation.json')
    dump('manifest.json',manifest)
    print('batches',len(batches),'offset',alignment['local_minus_utc_s'],'anchors',alignment['anchors_n'],flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--render-only',action='store_true')
    ap.add_argument('--reuse-render',action='store_true',help='Use SHA-bound local reconstruction cache')
    args=ap.parse_args()
    assert sha(RAW)==EXPECTED_RAW
    raw=[json.loads(l) for l in RAW.read_text().splitlines()]
    assert len(raw)==len({r['req_id'] for r in raw})==722 and all(not r['error'] for r in raw)
    rows,_,_=load_index(str(DATA))
    cohort=json.loads(COHORT.read_text())
    positions={rid:(c['chain_id'],i) for c in cohort['chains'] for i,rid in enumerate(c['req_ids'])}
    assert set(positions)==set(rows)=={r['req_id'] for r in raw}
    assert all(positions[r['req_id']]==(r['chain_id'],r['idx_in_chain']) for r in raw)
    paths=[RAW,COHORT,DATA/'requests.jsonl',DATA/'chains.jsonl',DATA/'bodies/dev-combined-v1.jsonl.gz',
           ROOT/'s1-dev/harness/s1_common.py',ROOT/'s1-dev/harness/s1_loadgen.py',
           *sorted((ROOT/'s1-dev/glm_tok').glob('*'))]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}
    if args.reuse_render:
        assert json.loads((OUT/'render_inputs.json').read_text())==inputs, 'Render inputs changed'
        pairs=json.loads((OUT/'pairs_rendered.json').read_text())
    else:
        pairs=render_pairs(raw)
        dump('render_inputs.json',inputs)
    classify_pairs(pairs)
    fast=[r for r in raw if in_ttft_gate(r,'fast_intra')]
    assert len(fast)==328 and sum(r['ttft_s']>3 for r in fast)==10
    summary=dict(n=722,followups=len(pairs),rendered_lengths_match=722,fast_n=len(fast),fast_over3=10,
                 short_slow_n=sum(r['output_tokens']<100 and (r['tpot_s'] or 0)>0.1 for r in raw),
                 frozen_positive_loss=sum(p['frozen_positive_loss'] for p in pairs),
                 true_positive_loss=sum(p['true_positive_loss'] for p in pairs),
                 aligned_loss=sum(p['aligned_loss'] for p in pairs),
                 frozen_overestimate=sum(p['frozen_overestimate'] for p in pairs),
                 overestimated_pairs=sum(p['frozen_overestimate']>0 for p in pairs),
                 loss_gt64=sum(p['true_positive_loss']>64 for p in pairs))
    dump('render_summary.json',summary)
    inputs[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    dump('manifest.json',dict(inputs=inputs,python=sys.version,packages={k:importlib.metadata.version(k) for k in ('numpy','transformers','tokenizers')},
                              raw_sha256=EXPECTED_RAW))
    print(json.dumps(summary,indent=2),flush=True)
    if not args.render_only:
        analyze_log(raw,pairs,inputs)

if __name__=='__main__':
    main()
