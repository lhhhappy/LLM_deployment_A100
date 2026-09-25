"""Usage: python3 scripts/analysis/opening_common_ids.py 075=L075-... 076=L076-... (job folders under evidence/).
Opening-probe candidates on one common request-ID set (all runs drained, paired against 074).

Reads evidence/L0xx/opening/paired.csv (reference side = 074, candidate side = the run). Gate membership
uses the harness in_ttft_gate on the frozen labels. Fast misses are split by 074's server-side new tokens
so every run uses the same denominators. Measured numbers only; no projection to an official level.
"""
import csv, json, sys
from pathlib import Path
ROOT = Path('/workspace/Agentic_science_challenge')
sys.path.insert(0, str(ROOT / 's1-dev/harness'))
from s1_common import in_ttft_gate
LIM = {'chain_start': 30.0, 'fast_intra': 3.0, 'overall_intra': 5.0, 'turn_start': 15.0}
lab = {}
for line in open(ROOT / 'data/s1-dev-longchain/requests.jsonl'):
    d = json.loads(line); lab[f"{d['pack']}:{d['view']}:{d['logical_call_id']}"] = d
# position within the chain, as the loadgen's raw idx_in_chain (the harness gate key)
for chain in json.load(open(ROOT / 'data/s1-dev-longchain/cohort.json'))['chains']:
    for i, rid in enumerate(chain['req_ids']):
        lab[rid]['idx_in_chain'] = i
runs = {}
ref = {}
for arg in sys.argv[1:]:
    name, job = arg.split('=')
    rows = {r['req_id']: r for r in csv.DictReader(open(ROOT / f'evidence/{job}/opening/paired.csv'))}
    runs[name] = {k: dict(ttft=float(r['candidate_ttft_s']), tpot=float(r['candidate_tpot_s']) if r['candidate_tpot_s'] else None,
                          new=int(r['candidate_prompt_tokens']) - int(r['candidate_cached_tokens']),
                          err=r['candidate_error'], t=float(r['candidate_client_dispatch_at_s'])) for k, r in rows.items()}
    for k, r in rows.items():
        ref.setdefault(k, dict(ttft=float(r['reference_ttft_s']), tpot=float(r['reference_tpot_s']) if r['reference_tpot_s'] else None,
                               new=int(r['reference_prompt_tokens']) - int(r['reference_cached_tokens']),
                               err=r['reference_error'], t=float(r['reference_client_dispatch_at_s'])))
runs = {'074': ref, **runs}
common = set.intersection(*(set(v) for v in runs.values()))
common = {k for k in common if not any(runs[n][k]['err'] for n in runs)}
print(f'common request IDs across {list(runs)}: {len(common)} (errors excluded: none expected)')
hdr = ['run', 'chain', 'fast', 'overall', 'turn', 'chain<=60s', 'fast<=2k', 'fast2-4k', 'fast>4k', 'tpot>0.10', 'tpot mean', 'tpot p95', 'fix/new chain vs 074']
print(' | '.join(hdr))
t0 = {n: min(runs[n][k]['t'] for k in common) for n in runs}
for n, R in runs.items():
    out = [n]
    miss = {}
    for g, lim in LIM.items():
        ids = [k for k in common if in_ttft_gate(lab[k], g)]
        miss[g] = {k for k in ids if R[k]['ttft'] > lim}
        out.append(f"{len(miss[g])}/{len(ids)}")
    early = [k for k in common if in_ttft_gate(lab[k], 'chain_start') and ref[k]['t'] - t0['074'] < 60]
    out.append(f"{sum(R[k]['ttft'] > 30 for k in early)}/{len(early)}")
    fast = [k for k in common if in_ttft_gate(lab[k], 'fast_intra')]
    for lo, hi in ((0, 2048), (2048, 4096), (4096, 10 ** 9)):
        ids = [k for k in fast if lo < ref[k]['new'] <= hi]
        out.append(f"{sum(R[k]['ttft'] > 3 for k in ids)}/{len(ids)}")
    tp = sorted(R[k]['tpot'] for k in common if R[k]['tpot'] is not None)
    out += [f"{sum(x > 0.10 for x in tp)}/{len(tp)}", f"{1000*sum(tp)/len(tp):.1f}ms", f"{1000*tp[int(0.95*len(tp))]:.1f}ms"]
    refmiss = {k for k in common if in_ttft_gate(lab[k], 'chain_start') and ref[k]['ttft'] > 30}
    out.append(f"{len(refmiss - miss['chain_start'])}/{len(miss['chain_start'] - refmiss)}")
    print(' | '.join(out))
