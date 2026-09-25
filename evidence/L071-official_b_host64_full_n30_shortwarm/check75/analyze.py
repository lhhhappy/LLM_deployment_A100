#!/usr/bin/env python3
"""Analyze immutable completed requests only. No engine calls or verdict."""
import csv
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / 'scripts/analysis'), str(ROOT / 's1-dev/harness')]
import admission_triage as triage
import compare_window as cw
import window_gates as wg

receipt = json.loads((OUT / 'transfer.json').read_text())
assert hashlib.sha256((OUT / 'raw.jsonl').read_bytes()).hexdigest() == receipt['raw_sha256']
rows = wg.load_raw(OUT / 'raw.jsonl')
old = wg.load_raw(OUT.parent / 'check45/raw.jsonl')
base_dir = ROOT / 'evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30'
base = wg.load_raw(base_dir / 'raw_s1-dev-longchain_N30_1790280416.jsonl')
h = wg.score_formal.load_harness()
summary, _ = cw.compare(base, rows, h)
by = triage.validate_raw(rows)
bb = triage.validate_raw(base)
old_ids = {r['req_id'] for r in old}
assert old_ids <= by.keys()
assert all(r == by[r['req_id']] for r in old), 'prior snapshot changed'
old_chain = {r['req_id'] for r in old if 'chain_start' in cw.fail_set(r, h)}
gates, details = {}, []
for _, gate, limit in h.TTFT_GATE_SPECS:
    bad = [r for r in rows if gate in cw.fail_set(r, h)]
    gates[gate] = dict(over=len(bad),
        queue_share_of_bad_summed_ttft=sum(r['queue_time_s'] for r in bad)/sum(r['ttft_s'] for r in bad) if bad else None,
        queue_ge80=sum(r['queue_time_s'] >= .8*r['ttft_s'] for r in bad),
        post_admission_over_limit=sum(r['t_first_token_s']-r['t_exec_start_s'] > limit for r in bad))
    for r in bad:
        b = bb[r['req_id']]
        details.append(dict(req_id=r['req_id'], gate=gate, baseline_ttft_s=b['ttft_s'], ttft_s=r['ttft_s'],
            queue_s=r['queue_time_s'], recv_to_admission_s=r['t_exec_start_s']-r['t_recv_s'],
            admission_to_first_s=r['t_first_token_s']-r['t_exec_start_s'],
            baseline_recv_to_admission_s=b['t_exec_start_s']-b['t_recv_s'],
            baseline_admission_to_first_s=b['t_first_token_s']-b['t_exec_start_s'],
            prompt=r['prompt_tokens'], cached=r['cached_tokens'], baseline_cached=b['cached_tokens'],
            newly_observed_since45=r['req_id'] not in old_ids))
new_chain = [d for d in details if d['gate']=='chain_start' and d['req_id'] not in old_chain]
score = json.loads((base_dir / 'score_formal.json').read_text())
limit = score['ttft_estimated']['chain_start(<=30s)']
report = dict(scope='completed requests only; all windows open; not a valid full-run verdict',
    paired=summary, phases=gates, new_chain_since45=new_chain,
    remaining_expected_requests=len(base)-len(rows),
    full_chain_cp_bound=dict(fixed_cohort_n=limit['n'], allowed=limit['allowed_over'],
        observed_bad=gates['chain_start']['over'],
        pass_still_possible=gates['chain_start']['over']<=limit['allowed_over'],
        note='Conditional on the fixed full cohort and unchanged recorded rows; completion/validity still pending. Wald allowance is 30; official CI implementation unknown.'),
    branch_attribution='unknown; frozen engine has no new admission trace',
    server_log_for_new_case='not fetched in this checkpoint; post-admission time is not GPU-only')
(OUT / 'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
with (OUT / 'bad-request-phases.csv').open('w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(details[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(details)
print(json.dumps(dict(phases=gates,new_chain=new_chain,full_chain_cp_bound=report['full_chain_cp_bound']),indent=2))
