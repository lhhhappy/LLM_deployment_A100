#!/usr/bin/env python3
"""Complete 069 timing/window audit. CPU only; bounded output, no rescore.

Run from the repository: python3 -B evidence/admission-wait-20260924/analyze.py
The harness's exec_start is the scheduler's first-batch admission mark, not
the first GPU kernel. t_admit is API dispatch completion, not GPU admission.
"""
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts/analysis'), str(ROOT / 's1-dev/harness')]
from compare_runs import GATES, load, pct
from s1_common import in_ttft_gate

LEVEL = ROOT / 'evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30'
cohort = json.loads((ROOT / 'data/s1-dev-longchain/cohort.json').read_text())
ids = [rid for chain in cohort['chains'] for rid in chain['req_ids']]
assert len(ids) == len(set(ids)) == cohort['n_requests']
assert hashlib.sha256(json.dumps(cohort['chains'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16] == cohort['cohort_sha256']
by, verdict, _, _, _, identity = load(LEVEL, set(ids), cohort)
rows = sorted(by.values(), key=lambda r: r['client_dispatch_at_s'])
raw = next(LEVEL.glob('raw_*.jsonl'))
t0 = rows[0]['client_dispatch_at_s']
phases = {}
for r in rows:
    for key in ('t_recv_s', 't_admit_s', 't_exec_start_s', 't_first_token_s',
                'queue_time_s', 'ttft_s', 'client_dispatch_at_s', 'client_finish_at_s', 'tpot_s'):
        assert isinstance(r.get(key), (int, float)) and math.isfinite(r[key]), (r['req_id'], key)
    p = {
        'receive_to_api_dispatch_s': r['t_admit_s'] - r['t_recv_s'],
        'dispatch_to_wait_queue_s': r['t_exec_start_s'] - r['queue_time_s'] - r['t_admit_s'],
        'scheduler_queue_s': r['queue_time_s'],
        'first_admission_to_first_token_s': r['t_first_token_s'] - r['t_exec_start_s'],
    }
    # Keep signed values; reject an invalid partition rather than clipping it.
    assert min(p.values()) >= -1e-5, (r['req_id'], p)
    assert abs(sum(p.values()) - r['ttft_s']) < 1e-5, (r['req_id'], p)
    assert r['client_finish_at_s'] >= r['client_dispatch_at_s']
    phases[r['req_id']] = p


def stats(values):
    return {'n': len(values), 'p50': pct(values, .5), 'p95': pct(values, .95),
            'max': max(values), 'sum': sum(values)} if values else {'n': 0}


def describe(selected):
    heads = [r for r in selected if r['idx_in_chain'] == 0]
    result = {'requests': len(selected), 'gates': {},
              'tpot_gt_100ms': sum(r['tpot_s'] > .1 for r in selected),
              'cohort_heads': len(heads),
              'actual_uncached_per_request': sum(r['prompt_tokens'] - r['cached_tokens'] for r in selected) / len(selected) if selected else None,
              'head_actual_uncached_per_request': sum(r['prompt_tokens'] - r['cached_tokens'] for r in heads) / len(heads) if heads else None}
    for gate, limit in GATES:
        group = [r for r in selected if in_ttft_gate(r, gate)]
        bad = [r for r in group if r['ttft_s'] > limit]
        result['gates'][gate] = {
            'n': len(group), 'over': len(bad),
            'ttft_s': stats([r['ttft_s'] for r in group]),
            'bad_pre_admission_ge80pct': sum(
                r['t_exec_start_s'] - r['t_recv_s'] >= .8 * r['ttft_s'] for r in bad),
            'bad_queue_ge80pct': sum(r['queue_time_s'] >= .8 * r['ttft_s'] for r in bad),
            'bad_post_admission_over_limit': sum(
                r['t_first_token_s'] - r['t_exec_start_s'] > limit for r in bad),
            'bad_last_dispatch_min': max(((r['client_dispatch_at_s'] - t0) / 60 for r in bad), default=None),
            'bad_phase_fraction_of_summed_ttft': {
                k: sum(phases[r['req_id']][k] for r in bad) / sum(r['ttft_s'] for r in bad)
                for k in phases[rows[0]['req_id']]
            } if bad else None,
        }
    return result


result = {
    'scope': '069 complete local N30; arrival windows are descriptive, never partial-run PASS/FAIL',
    'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
    'workload_identity': identity,
    'quantile_method': 'sorted[min(n-1,int(q*n))]; windows not formal scores',
    'timestamp_contract': {
        't_recv_s': 'ASGI entry on server; excludes time before handler/middleware entry',
        't_admit_s': 'API dispatch finish, not engine admission',
        't_exec_start_s': 'set_forward_entry_time after can_run_list chosen, before batch preparation/GPU launch',
        't_first_token_s': 'prefill_finished_time from first streamed meta_info (fallback: response_sent_to_client_ts)',
        'queue_time_s': 'forward_entry_time - wait_queue_entry_time from the first streamed meta_info',
        'dispatch_to_wait_queue_s': 'derived residual; combines IPC/receive/scheduler request processing, not pure network',
    },
    'all': describe(rows),
    'phase_s': {k: stats([p[k] for p in phases.values()]) for k in phases[rows[0]['req_id']]},
    'first30': describe(rows[:30]),
    'first30_dispatch_span_s': rows[29]['client_dispatch_at_s'] - t0,
    'first30_all_cohort_heads': all(r['idx_in_chain'] == 0 for r in rows[:30]),
    'windows': [],
}
for gate, _ in GATES:
    saved = verdict['ttft_gates'][gate]
    current = result['all']['gates'][gate]
    assert (current['n'], current['over']) == (saved['n'], saved['over_limit'])
assert result['all']['tpot_gt_100ms'] == sum(r['tpot_s'] > .1 for r in rows)

ends = {}
for r in rows:
    ends[r['session_id']] = max(ends.get(r['session_id'], 0), r['client_finish_at_s'])
assert len(ends) == len(cohort['chains'])
result['remaining_chains_below30_at_min'] = (sorted(ends.values())[len(ends) - 30] - t0) / 60
for lo, hi in ((0, 1), (1, 10), (10, 40), (40, 80), (80, None)):
    selected = [r for r in rows if lo <= (r['client_dispatch_at_s'] - t0) / 60
                and (hi is None or (r['client_dispatch_at_s'] - t0) / 60 < hi)]
    end = max(r['client_finish_at_s'] for r in rows) if hi is None else t0 + 60 * hi
    begin = t0 + 60 * lo
    inflight_seconds = sum(max(0.0, min(end, r['client_finish_at_s']) -
                               max(begin, r['client_dispatch_at_s'])) for r in rows)
    result['windows'].append({'dispatch_min': [lo, hi], **describe(selected),
                              'mean_client_inflight': inflight_seconds / (end - begin)})
assert sum(w['requests'] for w in result['windows']) == len(rows)
for gate, _ in GATES:
    for k in ('n', 'over'):
        assert sum(w['gates'][gate][k] for w in result['windows']) == result['all']['gates'][gate][k]

bad_rows = []
for r in rows:
    gates = [g for g, limit in GATES if in_ttft_gate(r, g) and r['ttft_s'] > limit]
    if gates:
        bad_rows.append({'req_id': r['req_id'], 'session_id': r['session_id'],
                         'idx_in_chain': r['idx_in_chain'], 'bad_gates': '|'.join(gates),
                         'dispatch_min': (r['client_dispatch_at_s'] - t0) / 60,
                         'ttft_s': r['ttft_s'], **phases[r['req_id']],
                         'actual_uncached': r['prompt_tokens'] - r['cached_tokens']})
result['unique_bad_requests'] = len(bad_rows)
assert len(bad_rows) == 311  # Independently checked in 069's existing case ledger.
csv_buffer = io.StringIO()
writer = csv.DictWriter(csv_buffer, fieldnames=list(bad_rows[0]))
writer.writeheader()
writer.writerows(bad_rows)
outputs = {'summary.json': json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
           'bad-request-phases.csv': csv_buffer.getvalue()}
assert sum(len(s.encode()) for s in outputs.values()) <= 2 * 1024 * 1024
for name, body in outputs.items():
    (OUT / name).write_text(body)
print(json.dumps({'requests': len(rows), 'unique_bad_requests': len(bad_rows),
                  'first30_chain_over': result['first30']['gates']['chain_start']['over'],
                  'remaining_chains_below30_at_min': result['remaining_chains_below30_at_min'],
                  'output_bytes': sum(len(s.encode()) for s in outputs.values())}))
