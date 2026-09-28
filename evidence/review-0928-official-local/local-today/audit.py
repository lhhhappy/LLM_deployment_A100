import csv
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
import window_gates as gates
import compare_window as compare

scorer = gates.score_formal.load_harness()
runs = {}
raws = {}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, obj): p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
def csvwrite(p, rows):
    if not rows: return
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def category(r):
    return 'segmented_intra' if r['phase'] == 'intra' and r['idx_in_chain'] == 0 else r['phase']

for tag in 'bcdefjklmnop':
    d, = (ROOT / 'evidence').glob('L130ezn' + tag + '-*')
    fs = list((d / 'N30').glob('raw_*.jsonl'))
    closed = bool(fs)
    p = fs[0] if closed else d / 'window/raw.jsonl'
    rows = gates.load_raw(p)
    compare.compare(rows, rows, scorer)  # Original pairer checks frozen metadata/token contract.
    first = min(r['client_dispatch_at_s'] for r in rows)
    assert all(not r.get('error') and r.get('error_class') is None for r in rows)
    assert all(all(gates.finite(r[k]) for k in ('ttft_s', 'tpot_s', 'client_dispatch_at_s', 'client_finish_at_s')) for r in rows)
    assert all(r['client_dispatch_at_s'] <= r['client_finish_at_s'] and r['tpot_s'] >= 0 for r in rows)
    receipt = dict(path=str(p.relative_to(ROOT)), sha256=sha(p), raw_n=len(rows),
                   unique_ids=len({r['req_id'] for r in rows}), errors=0, contracts='PASS',
                   complete_cohort=False, status='DRAINED_DIAGNOSTIC' if closed else 'OLD_OPEN_SNAPSHOT',
                   dispatch_span_min=(max(r['client_dispatch_at_s'] for r in rows)-first)/60,
                   completed_span_min=(max(r['client_finish_at_s'] for r in rows)-first)/60,
                   stats=gates.stats(rows, scorer))
    if closed:
        meta, = (d / 'N30').glob('run_s1*.json')
        run = json.loads(meta.read_text())
        log = (d/'N30/loadgen.log').read_text()
        timed, = [json.loads(x[len('TIMED_WINDOW '):]) for x in log.splitlines() if x.startswith('TIMED_WINDOW ')]
        assert timed['status']=='DRAINED' and timed['outstanding']==[] and timed['runner_rc']==0
        assert len(rows)==run['n_attempted']==run['dispatched']==timed['n_dispatched']==timed['n_completed']
        assert timed['duration_s']==2400 and not timed['full_cohort_complete']
        assert all(r['client_dispatch_at_s'] < timed['admission_deadline_s'] for r in rows)
        flush = json.loads((d/'N30/flush_evidence.json').read_text())
        assert flush['flush_success'] and flush['flush_http_status']==200
        receipt.update(timed_window=timed, config=run['config'], flush_ok=True,
                       original_level_verdict=json.loads((d/'N30/level_verdict.json').read_text()))
        server = (d/'N30/server.log').read_text()
        receipt['server_log_ax_124m_events'] = server.count('[ax-124m]')
    else:
        snap = json.loads((d/'window/snapshot.json').read_text())
        assert sha(p)==snap['downloaded_raw_sha256']
        assert len(rows)==snap['health']['completed_rows']
        receipt.update(snapshot_observed_at=snap['observed_at'], snapshot_job_state=snap['job_state'],
                       snapshot_age_min=(snap['observed_at']-first)/60,
                       checkpoint=snap['checkpoint'], downloaded_sha_matches=True)
    receipt['chain_bad'] = [dict(req_id=r['req_id'], phase=r['phase'], idx=r['idx_in_chain'],
                               kind=category(r), dispatch_s=r['client_dispatch_at_s']-first,
                               ttft_s=r['ttft_s'], prompt=r['glm_tokens'])
                            for r in rows if 'chain_start' in compare.fail_set(r, scorer)]
    raws[tag] = rows
    runs[tag] = receipt

pairs = {}
for a,b in [('b','c'),('c','l'),('c','k'),('l','m'),('c','m'),('m','n'),('n','o'),
            ('b','o'),('c','o'),('b','d'),('b','f'),('f','e'),('c','j'),('o','p')]:
    ba={r['req_id']:r for r in raws[a]}; ca={r['req_id']:r for r in raws[b]}
    ids=set(ba)&set(ca)
    aa=[r for r in raws[a] if r['req_id'] in ids]; bb=[r for r in raws[b] if r['req_id'] in ids]
    su, de=compare.compare(aa,bb,scorer)
    su['scope']='same-ID completed subset; drained diagnostic' if runs[b]['status']=='DRAINED_DIAGNOSTIC' else 'same-ID completed subset from OLD OPEN snapshot; unfinished unknown'
    su.update(common_n=len(ids), base_raw_n=len(ba), candidate_raw_n=len(ca), base_only=len(ba.keys()-ca.keys()),candidate_only=len(ca.keys()-ba.keys()))
    t0a=min(r['client_dispatch_at_s'] for r in raws[a]);t0b=min(r['client_dispatch_at_s'] for r in raws[b])
    chain=[]
    for d in de:
        r=ca[d['req_id']]; q=ba[d['req_id']]
        d.update(edge_type=r['edge_type'], kind=category(r),base_dispatch_s=q['client_dispatch_at_s']-t0a,candidate_dispatch_s=r['client_dispatch_at_s']-t0b)
        if 'chain_start' in d['baseline_failed'] or 'chain_start' in d['candidate_failed']:
            d['chain_change']='persistent' if 'chain_start' in d['baseline_failed'] and 'chain_start' in d['candidate_failed'] else ('repaired' if 'chain_start' in d['baseline_failed'] else 'new')
            chain.append(d.copy())
        else: d['chain_change']='none'
    su['chain_changes_by_type']={k:dict(Counter(d['chain_change'] for d in chain if d['kind']==k)) for k in sorted({d['kind'] for d in chain})}
    # Fixed baseline cohort avoids reclassifying a request when execution speed shifts dispatch time.
    for name,lo,hi in [('opening_0_60s',0,60),('middle_60_600s',60,600),('steady_ge600s',600,float('inf'))]:
        sel={r['req_id'] for r in aa if lo <= r['client_dispatch_at_s']-t0a < hi}
        bs=[r for r in aa if r['req_id'] in sel];cs=[r for r in bb if r['req_id'] in sel]
        sm,_=compare.compare(bs,cs,scorer) if sel else ({},[])
        su[name]=sm
    csvwrite(OUT/f'{a}-{b}-paired.csv',de)
    csvwrite(OUT/f'{a}-{b}-chain.csv',chain)
    su['chain_details']=chain
    pairs[a+'-'+b]=su

triple=set.intersection(*[{r['req_id'] for r in raws[t]} for t in 'bco'])
ts={t:[r for r in raws[t] if r['req_id'] in triple] for t in 'bco'}
tri=dict(common_n=len(triple), stats={t:gates.stats(rr,scorer) for t,rr in ts.items()})
tri['transitions']={a+'-'+b:compare.compare(ts[a],ts[b],scorer)[0] for a,b in [('b','c'),('c','o'),('b','o')]}
(OUT/'triple-b-c-o-req_ids.txt').write_text('\n'.join(sorted(triple))+'\n')
dump(OUT/'audit.json',dict(method=dict(harness=str(Path(inspect.getfile(scorer)).relative_to(ROOT)),
       compare_window_sha256=sha(ROOT/'scripts/analysis/compare_window.py'),
       window_gates_sha256=sha(ROOT/'scripts/analysis/window_gates.py'),
       classification='original harness; baseline dispatch bins [0,60),[60,600),[600,inf); all diagnostic'),runs=runs,pairs=pairs,triple=tri))
for t,r in runs.items():
    s=r['stats'];print('RUN',t,r['raw_n'],r['status'],{k:(v['n'],v['over']) for k,v in s['gates'].items()},'TPOT',s['tpot'])
for p,s in pairs.items():
    x=s['same_request_baseline'];y=s['candidate']
    print('PAIR',p,s['common_n'],{k:str(x['gates'][k]['over'])+'>'+str(y['gates'][k]['over']) for k in x['gates']},'CHAIN',s['gate_changes']['chain_start'],'TPOTp95',x['tpot']['p95'],y['tpot']['p95'],'TYPES',s['chain_changes_by_type'])
print('TRIPLE',json.dumps(tri['stats'],ensure_ascii=False))
