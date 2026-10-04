#!/usr/bin/env python3
"""Audit a frozen live DCP snapshot; this never issues a full-run SLO verdict.

Reads every raw/dispatch/metrics/server line. Uses the unmodified harness for
bucket membership and quantiles. Timing is request lifecycle, never GPU time.
Cumulative Prometheus counters keep their labels and their exported units.
"""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score_formal


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def csv_out(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def gates(rows, h):
    result = {'completed': len(rows), 'errors': sum(bool(r.get('error')) for r in rows), 'ttft': {}}
    for _, selector, limit in h.TTFT_GATE_SPECS:
        vals = [r['ttft_s'] for r in rows if not r.get('error') and h.in_ttft_gate(r, selector)]
        result['ttft'][selector] = {'n': len(vals), 'over': sum(v > limit for v in vals),
                                    'p95_s': h.q(vals, .95) if vals else None}
    tp = [r['tpot_s'] for r in rows if not r.get('error') and (r.get('output_tokens') or 0) > 1]
    result['tpot'] = {'n': len(tp), 'mean_s': st.mean(tp) if tp else None,
                      'p95_s': h.q(tp, .95) if tp else None, 'over_0.10': sum(v > .1 for v in tp)}
    return result


def components(r, t0, h):
    stamps = [r[k] for k in ('t_recv_s', 't_admit_s', 't_exec_start_s', 't_first_token_s')]
    if not all(math.isfinite(x) for x in stamps) or stamps != sorted(stamps):
        raise ValueError(f'Invalid lifecycle stamps: {r["req_id"]}')
    if abs(stamps[-1]-stamps[0]-r['ttft_s']) > 1e-5:
        raise ValueError('Server TTFT and lifecycle stamps disagree')
    return dict(req_id=r['req_id'], arrival_s=r['client_dispatch_at_s']-t0,
        gates='|'.join(sel for _, sel, _ in h.TTFT_GATE_SPECS if h.in_ttft_gate(r, sel)),
        bad_gates='|'.join(sel for _, sel, lim in h.TTFT_GATE_SPECS if h.in_ttft_gate(r, sel) and r['ttft_s']>lim),
        prefix_family_id=r.get('prefix_family_id'), prompt=r['prompt_tokens'], cached=r['cached_tokens'],
        ttft_s=r['ttft_s'], api_s=stamps[1]-stamps[0],
        dispatch_to_first_selection_s=stamps[2]-stamps[1],
        selection_to_first_token_s=stamps[3]-stamps[2], queue_s=r.get('queue_time_s'), tpot_s=r.get('tpot_s'))


def audit(args):
    h = score_formal.load_harness(args.harness)
    raw, ledger, metrics = [records(args.snapshot/n) for n in ('raw.jsonl','dispatch.jsonl','metrics.jsonl')]
    meta = json.loads((args.snapshot/'snapshot.json').read_text())
    by_id = {r['req_id']: r for r in raw}
    assert len(by_id) == len(raw), 'Duplicate completed requests'
    dispatch = [r for r in ledger if r['event']=='dispatch']
    assert len({r['req_id'] for r in dispatch}) == len(dispatch), 'Duplicate dispatches'
    assert set(by_id) <= {r['req_id'] for r in dispatch}
    t0 = min(r['client_dispatch_at_s'] for r in dispatch)
    pending = [r for r in dispatch if r['req_id'] not in by_id]
    if args.drained:
        assert not pending, 'A drained window must have every dispatched result'
        assert {r['req_id'] for r in ledger if r['event']=='completed'} == set(by_id)
        assert 'TIMED_DIAGNOSTIC DRAINED' in (args.snapshot/'job.log').read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    detail = [components(r,t0,h) for r in raw if not r.get('error')]
    csv_out(args.output/'requests.csv', detail)
    csv_out(args.output/'bad-requests.csv', [d for d in detail if d['bad_gates']])
    report = dict(scope='OPEN snapshot; completed requests only, no N@SLO verdict or DCP causal claim',
        observed_at_s=meta['observed_at_s'], observed_utc=dt.datetime.fromtimestamp(meta['observed_at_s'],dt.timezone.utc).isoformat(),
        elapsed_min=(meta['observed_at_s']-t0)/60, complete_stats=gates(raw,h),
        dispatched=len(dispatch), pending_or_snapshot_race=len(pending),
        snapshot_read_order='raw before ledger; a newly completed ledger row may be absent from raw',
        timing_semantics={'t_admit_s':'API dispatch finish, not scheduler admission',
                          't_exec_start_s':'first selection into a prefill batch, not kernel start',
                          'selection_to_first_token':'includes chunk interleaving and host work, not pure GPU time'})
    if args.drained:
        report['scope'] = 'DRAINED timed diagnostic, every dispatched ID completed; not the full cohort, no N@SLO verdict or DCP causal claim'
    report['windows'] = []
    for begin in range(0, math.ceil(report['elapsed_min']/5)*300, 300):
        rows=[r for r in raw if begin<=r['client_dispatch_at_s']-t0<begin+300]
        outstanding=[r for r in pending if begin<=r['client_dispatch_at_s']-t0<begin+300]
        report['windows'].append(dict(begin_s=begin,end_s=begin+300,pending=len(outstanding),**gates(rows,h)))
    for name,begin,end in [('opening10s',0,10),('after90s',90,float('inf'))]:
        rows=[r for r in raw if begin<=r['client_dispatch_at_s']-t0<end]
        report[name]=dict(**gates(rows,h), pending=sum(begin<=r['client_dispatch_at_s']-t0<end for r in pending))
    bad_chain=[d for d in detail if 'chain_start' in d['bad_gates']]
    report['chain_misses']=dict(n=len(bad_chain),
        max_arrival_s=max((d['arrival_s'] for d in bad_chain),default=None),
        waiting_over30=sum(d['dispatch_to_first_selection_s']>30 for d in bad_chain),
        selected_over30=sum(d['selection_to_first_token_s']>30 for d in bad_chain),
        sums={k:sum(d[k] for d in bad_chain) for k in ('api_s','dispatch_to_first_selection_s','selection_to_first_token_s')})
    if args.reference:
        base=records(args.reference);bmap={r['req_id']:r for r in base};bt0=min(r['client_dispatch_at_s'] for r in base)
        paired=[]
        for d in detail:
            if d['req_id'] not in bmap:
                continue
            b=components(bmap[d['req_id']],bt0,h)
            assert all(raw_value == bmap[d['req_id']].get(key) for key,raw_value in
                       ((k,by_id[d['req_id']].get(k)) for k in ('prompt_tokens','phase','idx_in_chain','uncached_expected')))
            paired.append(dict(d, **{'baseline_'+k:b[k] for k in ('arrival_s','cached','ttft_s','bad_gates','dispatch_to_first_selection_s','selection_to_first_token_s')}))
        csv_out(args.output/'paired.csv',paired)
        c=[r for r in raw if r['req_id'] in bmap];b=[bmap[r['req_id']] for r in c]
        report['reference_pair']=dict(n=len(c), candidate=gates(c,h),reference=gates(b,h),
            new_chain_misses=[r['req_id'] for r in paired if 'chain_start' in r['bad_gates'] and 'chain_start' not in r['baseline_bad_gates']],
            fixed_chain_misses=[r['req_id'] for r in paired if 'chain_start' not in r['bad_gates'] and 'chain_start' in r['baseline_bad_gates']],
            caveat='Common IDs; dispatch times/caching/interference can differ. Reference is a different engine stack, not DCP-only.')
    sample_detail=[]
    for m in metrics:
        if m['t'] < t0 or 'kv_used_tokens' not in m:
            continue
        cap=sum(m[k] for k in ('kv_used_tokens','kv_available_tokens','kv_evictable_tokens'))
        sample_detail.append(dict(t=m['t'],relative_s=m['t']-t0,capacity=cap,
            active_fraction=m['kv_used_tokens']/cap,free_fraction=m['kv_available_tokens']/cap,
            evictable_fraction=m['kv_evictable_tokens']/cap,queue=m['num_queue_reqs'],
            running=m['num_running_reqs'],throughput=m['gen_throughput'],acceptance=m['spec_accept_length'],
            evicted_exported=m.get('evicted_tokens_total'),host_used=m['hicache_host_used_tokens']))
    csv_out(args.output/'metrics.csv',sample_detail)
    report['sample_windows']=[]
    for begin,end in [(0,90),(90,660),(660,float('inf')),(sample_detail[-1]['relative_s']-111,float('inf'))]:
        rows=[r for r in sample_detail if begin<=r['relative_s']<end]
        if rows:
            report['sample_windows'].append(dict(begin_s=begin,end_s=end if math.isfinite(end) else None,n=len(rows),
                queue_nonzero=sum(r['queue']>0 for r in rows),queue_max=max(r['queue'] for r in rows),
                active_mean=st.mean(r['active_fraction'] for r in rows),active_max=max(r['active_fraction'] for r in rows),
                free_min=min(r['free_fraction'] for r in rows),free_max=max(r['free_fraction'] for r in rows),
                running_mean=st.mean(r['running'] for r in rows)))
    report['counter_samples_last']=[r for r in metrics[-1]['samples'] if any(s in r['name'] for s in ('evict','load_back','backup','dropped'))]
    server=[]; health=[]; mechanisms=[]
    stamp=re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] (.*)')
    fields={'new_tokens':r'#new-token: (\d+)','cached_tokens':r'#cached-token: (\d+)',
            'running':r'#running-req: (\d+)','queue':r'#queue-req: (\d+)',
            'parks':r'parks=(\d+)','relief_rounds':r'relief_rounds=(\d+)'}
    lines=(args.snapshot/'server.log').read_text().splitlines()
    for lineno,line in enumerate(lines,1):
        if any(w in line for w in ('Traceback','OutOfMemory','MISMATCH','ERROR')): health.append({'line':lineno,'text':line})
        if '[ax] mechanisms:' in line: mechanisms.append(line)
        m=stamp.match(line)
        if not m: continue
        timestamp=dt.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()
        if timestamp<t0: continue
        typ='prefill' if 'Prefill batch,' in line else 'decode' if 'Decode batch,' in line else 'policy' if '[ax-124/125]' in line else None
        if typ:
            row=dict(line=lineno,t=timestamp,relative_s=timestamp-t0,kind=typ)
            row.update({key:int(match[1]) if (match:=re.search(pattern,line)) else None for key,pattern in fields.items()})
            server.append(row)
    csv_out(args.output/'server-batches.csv',server)
    pf=[r for r in server if r['kind']=='prefill'];pol=[r for r in server if r['kind']=='policy']
    report['server']=dict(lines_read=len(lines),health_errors=health,mechanisms=mechanisms,
        sampled_prefill_lines=len(pf),prefill_tokens_logged=sum(r['new_tokens'] for r in pf),
        # Logging can be throttled; these are observations, not a complete GPU ledger.
        observed_short_prefills=sum(0<r['new_tokens']<=512 for r in pf),
        observed_2k_to_8k=sum(2048<=r['new_tokens']<=8192 for r in pf),
        prefill_queue_nonzero=sum(r['queue']>0 for r in pf),prefill_queue_max=max(r['queue'] for r in pf),
        first_policy=pol[0] if pol else None,last_policy=pol[-1] if pol else None,
        policy_last_change=next((r for r in reversed(pol) if r['relief_rounds']!=pol[-1]['relief_rounds']),None) if pol else None)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('observed_utc','elapsed_min','complete_stats','dispatched','pending_or_snapshot_race','chain_misses','reference_pair','sample_windows','server') if k in report},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--harness',type=Path,default=score_formal.DEFAULT_HARNESS)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--drained',action='store_true')
    audit(p.parse_args())
