#!/usr/bin/env python3
"""Per-request chain_start TTFT under the offline lane model (推算): which requests each policy
rescues or newly fails relative to the model's own LPM proxy. Model limits: see lane_model.py."""
import csv, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lane_model as M, sched_wait as SW

pols = ['LPM', 'SRPT', 'SRPT_aging2000', 'SRPT_aging2000_demote', 'EDF_visible_demote']
rows = []
for run in SW.RUNS:
    R, t0 = SW.load(run)
    cap = M.capacity_trace(SW.load_log(run), min(r['recv'] for r in R.values()))
    res = {p: M.simulate(R, cap, p, bypass_short=True) for p in pols}
    for rid, r in R.items():
        if 'chain_start' not in r['gates']:
            continue
        rows.append(dict(run=run, req_id=rid, arr_min=round(r['arr_min'], 2), unc=r['unc'], measured=round(r['ttft_s'], 1),
                         **{p: round(res[p][rid], 1) for p in pols}))
    base = {x['req_id'] for x in rows if x['run'] == run and x['LPM'] > 30}
    for p in pols[1:]:
        bad = {x['req_id'] for x in rows if x['run'] == run and x[p] > 30}
        print(run, p, 'model LPM over', len(base), '->', len(bad), 'rescued', len(base - bad), 'new', len(bad - base),
              'new_unc', sorted(x['unc'] for x in rows if x['run'] == run and x['req_id'] in bad - base))
with open(HERE / 'model-chain-flips.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
