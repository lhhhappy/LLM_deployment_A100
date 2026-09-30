#!/usr/bin/env python3
"""Closed-cohort timing audit. No eligibility/GPU-time inference from raw rows.

Reconstruct earlier arrival censuses from final immutable raw, including requests
excluded by a completed-only window. Pair phases and lifecycle overlaps, keeping
all branch-level and counterfactual service attribution unknown.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/analysis'), str(ROOT / 'scripts'), str(ROOT / 's1-dev/harness')]
import compare_runs as cr
import score_formal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_seconds(intervals):
    total, end = 0., -math.inf
    for a, b in sorted(intervals):
        if b > max(a, end):
            total += b - max(a, end)
        end = max(end, b)
    return total


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def census(rows, at, harness):
    arrived = [r for r in rows if r['t_recv_s'] <= at]
    pending = [r for r in arrived if r['t_first_token_s'] > at]
    result = dict(at_s=at, arrived=len(arrived), server_first_pending=len(pending),
                  before_first_admission=sum(r['t_exec_start_s'] > at for r in pending),
                  after_first_admission=sum(r['t_exec_start_s'] <= at for r in pending),
                  client_unfinished=sum(r['client_dispatch_at_s'] <= at < r['client_finish_at_s'] for r in rows),
                  client_first_pending=sum(r['client_dispatch_at_s'] <= at < r['client_first_token_at_s'] for r in rows),
                  gates={})
    for _, gate, limit in harness.TTFT_GATE_SPECS:
        a = [r for r in arrived if harness.in_ttft_gate(r, gate)]
        p = [r for r in pending if harness.in_ttft_gate(r, gate)]
        overdue = [r for r in p if at - r['t_recv_s'] > limit]
        known_bad = [r for r in a if min(at, r['t_first_token_s'])-r['t_recv_s'] > limit]
        result['gates'][gate] = dict(arrived=len(a), first_pending=len(p), overdue_pending=len(overdue),
            overdue_before_admission=sum(r['t_exec_start_s'] > at for r in overdue),
            observed_bad_among_all_arrived=len(known_bad),
            observed_bad_but_not_completed=sum(r['client_finish_at_s'] > at for r in known_bad),
            overdue_rids=[r['req_id'] for r in overdue])
    return result, pending


def analyze(base, cand, cohort, out, data_root):
    h = score_formal.load_harness()
    ids = {rid for ch in cohort['chains'] for rid in ch['req_ids']}
    assert len(ids)==cohort['n_requests']==sum(len(ch['req_ids']) for ch in cohort['chains'])
    assert hashlib.sha256(json.dumps(cohort['chains'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]==cohort['cohort_sha256']
    B, vb, bs, _, _, ib = cr.load(base, ids, cohort)
    C, vc, cs, pace, _, ic = cr.load(cand, ids, cohort)
    assert ib == ic
    frozen, _, _ = __import__('s1_common').load_index(str(data_root))
    assert set(frozen) == ids
    meta=('prompt_tokens','uncached_expected','max_output_i','phase','chain_id','idx_in_chain','edge_type','replay_gap_ms','effective_replay_gap_ms')
    assert all(all(B[rid].get(k)==C[rid].get(k) for k in meta) for rid in ids)
    for rows in (B, C):
        score_formal.validate_replay_tokens(list(rows.values()), frozen)
        for r in rows.values():
            stamps = [r[k] for k in ('t_recv_s','t_admit_s','t_exec_start_s','t_first_token_s')]
            assert all(math.isfinite(x) for x in stamps) and stamps == sorted(stamps)
            assert abs(r['ttft_s'] - (stamps[-1]-stamps[0])) < 1e-6
            assert r['ttft_source'] == 'server'
            assert 0 <= r['queue_time_s'] <= stamps[2]-stamps[0]
    out.mkdir(parents=True, exist_ok=True)
    report = dict(scope='complete VALID paired N30; timing/overlap diagnostics, not causal attribution',
        rows=len(C), base_raw_sha256=sha(next(base.glob('raw_*.jsonl'))),
        candidate_raw_sha256=sha(next(cand.glob('raw_*.jsonl'))),
        manifest_sha256=sha(data_root/'manifest.json'),
        all_11_gates=json.loads((cand/'score_formal.json').read_text())['estimated']['gates'],
        gates={}, work={}, snapshots=[], cache_cases=[])
    gates_by_id = {rid: [] for rid in C}
    for _, gate, limit in h.TTFT_GATE_SPECS:
        b = [r for r in B.values() if h.in_ttft_gate(r, gate)]
        c = [r for r in C.values() if h.in_ttft_gate(r, gate)]
        assert {r['req_id'] for r in b} == {r['req_id'] for r in c}
        bad_b = {r['req_id'] for r in b if r['ttft_s'] > limit}
        bad_c = {r['req_id'] for r in c if r['ttft_s'] > limit}
        for rid in bad_c:
            gates_by_id[rid].append(gate)
        bad = [C[rid] for rid in bad_c]
        report['gates'][gate] = dict(n=len(c), base_over=len(bad_b), candidate_over=len(bad_c),
            fixed=len(bad_b-bad_c), new=len(bad_c-bad_b),
            base_p95=cr.pct([r['ttft_s'] for r in b], .95), candidate_p95=cr.pct([r['ttft_s'] for r in c], .95),
            bad_queue_seconds=sum(r['queue_time_s'] for r in bad),
            bad_recv_to_admission_seconds=sum(r['t_exec_start_s']-r['t_recv_s'] for r in bad),
            bad_ttft_seconds=sum(r['ttft_s'] for r in bad),
            bad_queue_ge80=sum(r['queue_time_s'] >= .8*r['ttft_s'] for r in bad),
            bad_post_admission_over_limit=sum(r['t_first_token_s']-r['t_exec_start_s'] > limit for r in bad),
            last_bad_arrival_min=max(((r['client_dispatch_at_s']-min(x['client_dispatch_at_s'] for x in C.values()))/60 for r in bad), default=None),
            new_rids=sorted(bad_c-bad_b), fixed_rids=sorted(bad_b-bad_c))
    details = []
    for rid, r in C.items():
        b = B[rid]
        row = dict(req_id=rid, bad_gates='|'.join(gates_by_id[rid]),
            prompt=r['prompt_tokens'], cached=r['cached_tokens'], baseline_cached=b['cached_tokens'],
            uncached=r['prompt_tokens']-r['cached_tokens'], baseline_ttft_s=b['ttft_s'], ttft_s=r['ttft_s'],
            api_s=r['t_admit_s']-r['t_recv_s'],
            dispatch_to_queue_s=r['t_exec_start_s']-r['queue_time_s']-r['t_admit_s'],
            queue_s=r['queue_time_s'], recv_to_admission_s=r['t_exec_start_s']-r['t_recv_s'],
            post_admission_s=r['t_first_token_s']-r['t_exec_start_s'],
            baseline_recv_to_admission_s=b['t_exec_start_s']-b['t_recv_s'],
            baseline_post_admission_s=b['t_first_token_s']-b['t_exec_start_s'],
            tpot_s=r['tpot_s'], baseline_tpot_s=b['tpot_s'])
        assert abs(row['api_s']+row['dispatch_to_queue_s']+row['queue_s']+row['post_admission_s']-r['ttft_s'])<1e-6
        details.append(row)
    write_csv(out/'paired-phases.csv', details)
    bad_details = []
    for row in details:
        if not row['bad_gates']:
            continue
        r = C[row['req_id']]
        overtakers = [x for x in C.values() if r['t_recv_s'] < x['t_recv_s'] and x['t_exec_start_s'] < r['t_exec_start_s']]
        intervals = [(max(r['t_recv_s'], x['t_exec_start_s']), min(r['t_exec_start_s'], x['t_first_token_s'])) for x in overtakers]
        row = dict(row, later_arrivals_admitted_first=len(overtakers),
            other_prefill_lifecycle_union_s=union_seconds(intervals),
            eligible_wait_s='unknown', runnable_bypassed_gpu_service_s='unknown')
        bad_details.append(row)
    write_csv(out/'bad-wait.csv', bad_details)
    report['unique_bad_requests'] = len(bad_details)
    report['bad_with_later_arrival_overtakes'] = sum(r['later_arrivals_admitted_first'] > 0 for r in bad_details)
    report['chain_uncached_bins'] = {}
    for label, low, high in [('0-4096',0,4096),('4097-65536',4097,65536),('65537+',65537,math.inf)]:
        group = [r for r in bad_details if 'chain_start' in r['bad_gates'] and low<=r['uncached']<=high]
        report['chain_uncached_bins'][label] = dict(n=len(group), queue_ge80=sum(r['queue_s']>=.8*r['ttft_s'] for r in group))
    for label, rows, batches in [('base',B,bs),('candidate',C,cs)]:
        times=[r['tpot_s'] for r in rows.values()]
        chain_ends={}
        for r in rows.values():
            chain_ends[r['chain_id']]=max(chain_ends.get(r['chain_id'],0),r['client_finish_at_s'])
        t0=min(r['client_dispatch_at_s'] for r in rows.values())
        report['work'][label]=dict(prompt_tokens=sum(r['prompt_tokens'] for r in rows.values()),
            cached_tokens=sum(r['cached_tokens'] for r in rows.values()),
            uncached_tokens=sum(r['prompt_tokens']-r['cached_tokens'] for r in rows.values()),
            output_tokens=sum(r['output_tokens'] for r in rows.values()),
            prefill_log_batches=len(batches), prefill_log_tokens=sum(x[1] for x in batches),
            single_sequence_pending_batches=sum(x[0]==1 and x[5]>0 for x in batches),
            tpot_mean=sum(times)/len(times),tpot_p95=cr.pct(times,.95),tpot_over100ms=sum(t>.1 for t in times),
            first_dispatch_s=t0,last_finish_s=max(chain_ends.values()),
            duration_s=max(chain_ends.values())-t0,
            remaining_chains_below30_min=(sorted(chain_ends.values())[-30]-t0)/60)
    census_rows=[]
    candidate_t0=report['work']['candidate']['first_dispatch_s']
    for checkpoint in (1,5,10,15,30,45,75,105):
        path=cand.parent/f'check{checkpoint}'/'snapshot.json'
        at=json.loads(path.read_text())['observed_at'] if path.exists() else candidate_t0+60*checkpoint
        elapsed=at-candidate_t0
        item=dict(checkpoint=checkpoint,elapsed_s=elapsed,clock_note='same elapsed time from each run first client dispatch; server timing fields label admission/TTFT, client fields label SSE/completion')
        for label,rows in [('base',B),('candidate',C)]:
            ref=report['work'][label]['first_dispatch_s']+elapsed
            item[label],pending=census(list(rows.values()),ref,h)
            for r in pending:
                census_rows.append(dict(checkpoint=checkpoint,run=label,req_id=r['req_id'],age_s=ref-r['t_recv_s'],
                    before_admission=r['t_exec_start_s']>ref,
                    overdue_gates='|'.join(g for _,g,lim in h.TTFT_GATE_SPECS if h.in_ttft_gate(r,g) and ref-r['t_recv_s']>lim),
                    eventual_ttft_s=r['ttft_s']))
        if path.exists():
            raw=cand.parent/f'check{checkpoint}'/'raw.jsonl'
            old=[json.loads(line) for line in raw.read_text().splitlines() if line]
            assert all(C[r['req_id']]==r for r in old), 'record mutated since checkpoint'
            item['snapshot_unchanged_rows']=len(old)
            item['snapshot_read_race_note']='observed_at precedes raw read; immutable final timestamps are used for this exact-time census'
        report['snapshots'].append(item)
    write_csv(out/'pending-at-checkpoints.csv',census_rows)
    lcp_path=cand.parent/'check45/lcp-cache-cases.json'
    if lcp_path.exists():
        lcp=json.loads(lcp_path.read_text())
        assert sha(lcp_path.parent/'raw.jsonl')==lcp['raw_sha256']
        for case in lcp['results']:
            r,p=C[case['req_id']],C[case['predecessor']]
            assert r['prompt_tokens']==case['prompt_tokens'] and r['cached_tokens']==case['cached_tokens']
            assert p['idx_in_chain']+1==r['idx_in_chain'] and p['chain_id']==r['chain_id']
            overlaps=[]
            for x in C.values():
                lo,hi=max(r['t_exec_start_s'],x['t_recv_s']),min(r['t_first_token_s'],x['t_exec_start_s'])
                if h.in_ttft_gate(x,'chain_start') and hi>lo:
                    overlaps.append(dict(req_id=x['req_id'],ttft_s=x['ttft_s'],wait_lifecycle_overlap_s=hi-lo))
            report['cache_cases'].append(dict(**case,
                predecessor_prompt_tokens=p['prompt_tokens'],predecessor_output_tokens=p['output_tokens'],
                prompt_end_grid_checkpoint=p['prompt_tokens']//256*256,
                next_grid_after_prompt=(p['prompt_tokens']//256+1)*256,
                maximum_prompt_plus_output=p['prompt_tokens']+p['output_tokens'],
                chain_wait_overlaps=overlaps,
                causal_seconds_saved=None,checkpoint_ever_published=None,
                note='grid arithmetic is a candidate checkpoint, not an observed publication; overlap includes decode interleaving and asynchronous work'))
    report['unknowns']=['all actual admission eligibility/rejection intervals in frozen 759a6eb',
        'GPU service delivered while each waiting request was runnable',
        'checkpoint creation/publication/eviction/restore/consumption per node and epoch',
        'per-request marginal batch cost and closed-loop counterfactual improvement']
    report['pace_first_last']=pace[:1]+pace[-1:]
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<12*1024*1024
    print(json.dumps({k:report[k] for k in ('rows','unique_bad_requests','chain_uncached_bins','work')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path)
    p.add_argument('candidate',type=Path)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    analyze(a.base,a.candidate,json.loads(a.cohort.read_text()),a.out,a.data_root)
