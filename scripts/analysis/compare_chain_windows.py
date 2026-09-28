#!/usr/bin/env python3
"""Pair drained diagnostic windows by ID using the unmodified harness buckets.

This reports observations, never a full-cohort verdict or causal attribution.
Receipt IDs must equal raw IDs. Summary.json selects measured raw, not warmup.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def load(level, verdict_path):
    summary = json.loads((level / 'summary.json').read_text())
    raw = level / Path(summary['raw']).name
    data = raw.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    by_id = {r['req_id']: r for r in rows}
    assert len(rows) == len(by_id), 'duplicate request ID'
    assert all(not r.get('error') and not r.get('error_class') for r in rows), 'request errors'
    for r in rows:
        assert all(isinstance(r[k], (int, float)) and math.isfinite(r[k]) and r[k] >= 0
                   for k in ('ttft_s', 'queue_time_s', 'client_dispatch_at_s'))
        if r['output_tokens'] > 1:
            assert isinstance(r['tpot_s'], (int, float)) and math.isfinite(r['tpot_s']) and r['tpot_s'] >= 0
    v = json.loads(verdict_path.read_text())
    assert v['scope'] == 'fixed_duration_diagnostic' and v['status'] == 'DRAINED'
    assert v['n_completed'] == len(rows) and v['errors'] == 0
    assert len(v['dispatched_req_ids']) == len(rows) and set(v['dispatched_req_ids']) == set(by_id)
    assert v['raw_sha256'] == hashlib.sha256(data).hexdigest(), 'receipt does not describe this raw'
    flush = json.loads((level / 'flush_evidence.json').read_text())
    assert flush['flush_success'] and flush['runner_rc'] == 0
    log = (level / 'server.log').read_text()
    events, plans = Counter(), []
    for line in log.splitlines():
        if '[ax-124m] ' in line:
            event = json.loads(line.split('[ax-124m] ', 1)[1])
            events[event['event']] += 1
        if '[ax-124m-plan] ' in line:
            plans.append(json.loads(line.split('[ax-124m-plan] ', 1)[1]))
    return by_id, dict(source=str(level), raw=raw.name, receipt=str(verdict_path),
                       status='DRAINED', full_cohort_complete=False, rows=len(rows), errors=0,
                       flush_success=True, first_dispatch_at_s=v['first_dispatch_at_s'],
                       admission_seconds=v['admission_seconds'], events=dict(events),
                       plan_records=plans, harness_p95=summary['ttft_p95_by_gate'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--before-verdict', type=Path, required=True)
    ap.add_argument('--after-verdict', type=Path, required=True)
    ap.add_argument('--harness-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.harness_dir.resolve()))
    from s1_common import TTFT_GATE_SPECS, in_ttft_gate, q

    before, bs = load(a.before, a.before_verdict)
    after, ats = load(a.after, a.after_verdict)
    assert bs['admission_seconds'] == ats['admission_seconds']
    ids = sorted(before.keys() & after.keys())
    for rid in ids:
        for k in ('phase', 'idx_in_chain', 'glm_tokens', 'uncached_expected', 'max_output_i', 'replay_gap_ms'):
            assert before[rid][k] == after[rid][k], (rid, k, 'frozen data differs')
    result = dict(scope='common_ID_diagnostic', n_common=len(ids), before=bs, after=ats,
                  before_only=sorted(before.keys() - after.keys()), after_only=sorted(after.keys() - before.keys()),
                  bucket_source=str(a.harness_dir / 's1_common.py'), gates={})
    for label, selector, limit in TTFT_GATE_SPECS:
        result['gates'][label] = {}
        for name, data in [('before', before), ('after', after)]:
            values = [data[rid]['ttft_s'] for rid in ids if in_ttft_gate(data[rid], selector)]
            result['gates'][label][name] = dict(n=len(values), over_limit=sum(t > limit for t in values),
                                              p95=q(values, .95))
    result['tpot'] = {}
    for name, data in [('before', before), ('after', after)]:
        values = [data[rid]['tpot_s'] for rid in ids if data[rid]['output_tokens'] > 1]
        result['tpot'][name] = dict(n=len(values), over_010=sum(t > .1 for t in values),
                                   mean=sum(values) / len(values) if values else None, p95=q(values, .95))
    changes = []
    chain_rows = []
    for rid in ids:
        b, c = before[rid], after[rid]
        if not in_ttft_gate(b, 'chain_start'):
            continue
        row = dict(req_id=rid, phase=b['phase'], idx_in_chain=b['idx_in_chain'], edge_type=b['edge_type'],
                   glm_tokens=b['glm_tokens'], uncached_expected=b['uncached_expected'])
        for name, r, s in [('before', b, bs), ('after', c, ats)]:
            row.update({name + '_' + k: r[k] for k in ('ttft_s', 'queue_time_s', 'prompt_tokens', 'cached_tokens')})
            row[name + '_dispatch_s'] = r['client_dispatch_at_s'] - s['first_dispatch_at_s']
            row[name + '_miss'] = r['ttft_s'] > 30
        row['change'] = ('improved' if row['before_miss'] else 'new_miss') if row['before_miss'] != row['after_miss'] else 'same'
        chain_rows.append(row)
        if row['change'] != 'same':
            changes.append(row)
    result['chain_changes'] = changes
    a.out.mkdir(parents=True, exist_ok=True)
    for filename, records in [('paired-chain.csv', chain_rows), ('changed-chain.csv', changes)]:
        with (a.out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(chain_rows[0]) if chain_rows else ['req_id'])
            writer.writeheader()
            writer.writerows(records)
    (a.out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(n_common=len(ids), gates=result['gates'], tpot=result['tpot'],
                         changes=dict(Counter(r['change'] for r in changes)),
                         before_events=bs['events'], after_events=ats['events']), ensure_ascii=False))


if __name__ == '__main__':
    main()
