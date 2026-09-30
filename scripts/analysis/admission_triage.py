#!/usr/bin/env python3
"""Build a first-admission evidence ledger from a complete measured level.

No exclusive root-cause assignment, performance model or GPU saturation inference.
Schema-2 queue-entry timestamps prevent warmup/retraction episodes contaminating
the join. Legacy or missing diagnostics stay unknown. No new server logs needed
to audit historical coverage; collecting actual branch evidence requires a run.
"""
import argparse
from bisect import bisect_right, insort
from collections import Counter
import csv
import hashlib
import io
from itertools import groupby
import json
import math
from pathlib import Path
import sys

from admission_trace import iter_events

ROOT = Path(__file__).resolve().parents[2]
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
TIME_TOLERANCE_S = .01
REASON_GROUPS = {
    'host_eligibility': {'partial_host_restore'},
    'partial_eligibility': {'partial_no_device_prefix', 'partial_long_tail'},
    'token_batch_budget': {'partial_token_budget', 'batch_token_budget'},
    'candidate_kv_budget': {'kv_budget', 'kv_budget_after_lock'},
    'request_slots': {'request_slots', 'request_slots_or_batch_full'},
    'mamba_state_slots': {'kv_scan_mamba_slots'},
    'bounded_scan_eligibility': {'kv_scan_not_complete_device_short', 'kv_scan_ownership_excluded'},
    'full_latch': {'batch_full_latched'},
    'decode_cadence': {'decode_cadence'},
    'scan_stopped_before_request': {'unscanned_after_no_token', 'unscanned_after_other', 'unscanned_after_kv_scan'},
}


def validate_raw(rows):
    by = {}
    for r in rows:
        rid = r.get('req_id')
        if not isinstance(rid, str) or not rid or rid in by:
            raise ValueError('invalid/duplicate raw req_id')
        if r.get('error'):
            raise ValueError(f'raw request error: {rid}')
        for key in ('t_recv_s', 't_exec_start_s', 't_first_token_s', 'queue_time_s', 'ttft_s'):
            value = r.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{rid}: missing/invalid {key}')
        recv, admission, first = (r[k] for k in ('t_recv_s', 't_exec_start_s', 't_first_token_s'))
        if not 0 <= recv <= admission <= first or not 0 <= r['queue_time_s'] <= admission - recv + 1e-5:
            raise ValueError(f'{rid}: inconsistent timestamp order/queue duration')
        if abs(first - recv - r['ttft_s']) > 1e-5:
            raise ValueError(f'{rid}: TTFT differs from server timestamps')
        by[rid] = r
    return by


def overtaking(rows):
    """Strict earlier admission, strict later receipt; ties never count.

    Maintain receipts of already admitted requests. No simulated scheduling or
    assumptions about whether the bypassed request could fit are involved.
    """
    receipts, result = [], {}
    for _, group in groupby(sorted(rows, key=lambda r: r['t_exec_start_s']),
                            key=lambda r: r['t_exec_start_s']):
        group = list(group)
        for r in group:
            boundary = bisect_right(receipts, (r['t_recv_s'], chr(0x10ffff)))
            result[r['req_id']] = {
                'count': len(receipts) - boundary,
                'examples': [rid for _, rid in receipts[boundary:boundary + 3]],
            }
        for r in group:
            insort(receipts, (r['t_recv_s'], r['req_id']))
    return result


def observed_groups(counts):
    return sorted(group for group, reasons in REASON_GROUPS.items()
                  if any(counts.get(reason, 0) > 0 for reason in reasons))


def classify(raw_rows, events, selected_ids=None):
    by = validate_raw(raw_rows)
    selected = set(by) if selected_ids is None else set(selected_ids)
    if not selected <= by.keys():
        raise ValueError('selected ids outside raw population')
    order = overtaking(list(by.values()))
    admitted, pending = {}, {}
    excluded = Counter()
    exhausted = False
    event_count = 0
    for event in events:
        event_count += 1
        if event['event'] == 'budget_exhausted':
            exhausted = True
            continue
        kind = event['event']
        records = [event] if kind == 'admit' else event['sample']
        for row in records:
            rid = row['rid']
            if rid not in by:
                excluded['rid_outside_cohort'] += 1
                continue
            r = by[rid]
            observed, entered = row.get('observed_at_s'), row.get('queue_entry_at_s')
            if observed is None or entered is None:
                excluded['no_episode_timestamps'] += 1
                continue
            expected = r['t_exec_start_s'] - r['queue_time_s']
            if abs(entered - expected) > TIME_TOLERANCE_S:
                excluded['different_queue_episode'] += 1
                continue
            if not r['t_recv_s'] - TIME_TOLERANCE_S <= observed <= r['t_exec_start_s'] + TIME_TOLERANCE_S:
                excluded['outside_first_admission_window'] += 1
                continue
            previous = admitted.get(rid) or pending.get(rid)
            if previous is not None:
                if observed < previous['observed_at_s'] or any(
                    row['decisions'].get(k, 0) < v for k, v in previous['decisions'].items()
                ):
                    raise ValueError(f'{rid}: timestamps/cumulative counters moved backwards')
                before, after = previous.get('decision_intervals'), row.get('decision_intervals')
                if before is not None and after is not None and (
                    after['count'] < before['count'] or any(
                        after[field].get(k, 0.0) + 1e-6 < v
                        for field in ('seconds_by_reason', 'seconds_by_batch')
                        for k, v in before[field].items()
                    )
                ):
                    raise ValueError(f'{rid}: cumulative decision intervals moved backwards')
            if rid in admitted:
                raise ValueError(f'{rid}: duplicate/post-admission record in first episode')
            if kind == 'admit':
                admitted[rid] = row
                pending.pop(rid, None)
            else:
                pending[rid] = row
    rows = []
    scopes, groups = Counter(), Counter()
    for rid in sorted(selected):
        r = by[rid]
        observation = admitted.get(rid) or pending.get(rid)
        scope = 'admission_observed' if rid in admitted else 'snapshot_only' if rid in pending else 'unknown'
        scopes[scope] += 1
        counts = observation['decisions'] if observation else None
        examples = observation.get('examples', {}) if observation else {}
        observed = observed_groups(counts or {})
        groups.update(observed)
        head = examples.get('unscanned_after_no_token', {})
        if (counts or {}).get('unscanned_after_no_token', 0) > 0:
            scan_context = ('rejected_candidate' if head.get('head_added') is False else
                            'admitted_candidate' if head.get('head_added') is True else 'unknown_head_state')
        else:
            scan_context = None
        rows.append({
            'req_id': rid, 'phase': r.get('phase'), 'ttft_s': r['ttft_s'],
            'pre_admission_s': r['t_exec_start_s'] - r['t_recv_s'],
            'scheduler_queue_s': r['queue_time_s'],
            'post_admission_s': r['t_first_token_s'] - r['t_exec_start_s'],
            'diagnostic_scope': scope,
            'observed_groups': observed, 'decision_counts': counts, 'first_reason_examples': examples,
            'decision_intervals': observation.get('decision_intervals') if observation else None,
            'no_token_scan_stop_example': scan_context,
            'skipped_request_feasibility': 'unknown',
            'later_arrivals_admitted_first': order[rid]['count'],
            'overtaking_examples': order[rid]['examples'],
            'ordering_cause': 'unknown', 'compute_saturation': 'unknown',
            'unmapped_reasons': sorted(k for k, v in (counts or {}).items()
                                      if v and not any(k in rs for rs in REASON_GROUPS.values())),
        })
    return {
        'schema': 1, 'population_requests': len(by), 'selected_requests': len(rows),
        'diagnostic_events_in_file': event_count, 'any_process_budget_exhausted': exhausted,
        'excluded_observations': dict(excluded), 'selected_diagnostic_scope': dict(scopes),
        'selected_requests_with_observed_group': dict(groups),
        'selected_requests_overtaken': sum(row['later_arrivals_admitted_first'] > 0 for row in rows),
        'rows': rows,
        'interpretation': [
            'groups overlap; decision counters are not seconds or causal delay shares',
            'observed branch does not prove that removing it improves end-to-end SLO',
            'unscanned-after-no-token does not establish that this skipped request fits',
            'first_reason_examples are one sample per reason, not context for every counter increment',
            'later arrivals admitted first is an observation, not proof of incorrect ordering',
            'GPU saturation always remains unknown without independent device/work evidence',
            'admission observed does not guarantee diagnostics covered the full wait',
            'unknown/unobserved does not mean absence; snapshots and byte-limited streams are partial',
            'only matching first queue episodes are joined; warmup/retractions/unscoped legacy records excluded',
            'decision intervals partition observed wall time by the preceding decision, not persistent blockers or GPU service',
        ],
    }


def write_outputs(result, json_path, csv_path=None, max_bytes=MAX_OUTPUT_BYTES):
    if csv_path is not None and json_path.resolve() == csv_path.resolve():
        raise ValueError('JSON and CSV destinations must differ')
    payloads = {json_path: json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'}
    if csv_path:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(result['rows'][0]) if result['rows'] else ['req_id'],
                                lineterminator='\n')
        writer.writeheader()
        for row in result['rows']:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
        payloads[csv_path] = buf.getvalue()
    # Validate the aggregate bound BEFORE touching either destination.
    if sum(len(body.encode()) for body in payloads.values()) > max_bytes:
        raise ValueError('combined output exceeds byte budget; narrow the selected requests')
    for path, body in payloads.items():
        path.write_text(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('level', type=Path, help='complete fetched level, including original VALID verdict')
    ap.add_argument('--cohort', type=Path, default=ROOT / 'data/s1-dev-longchain/cohort.json')
    ap.add_argument('--log', type=Path, help='default: LEVEL/server.log')
    ap.add_argument('--all-requests', action='store_true', help='default: unique TTFT bad requests only')
    ap.add_argument('--json', type=Path, required=True)
    ap.add_argument('--csv', type=Path)
    args = ap.parse_args()
    from compare_runs import GATES, load
    sys.path.insert(0, str(ROOT / 's1-dev/harness'))
    from s1_common import in_ttft_gate

    cohort = json.loads(args.cohort.read_text())
    ids = [rid for ch in cohort['chains'] for rid in ch['req_ids']]
    if len(ids) != len(set(ids)) or len(ids) != cohort['n_requests']:
        raise ValueError('invalid cohort roster')
    digest = hashlib.sha256(json.dumps(cohort['chains'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    if digest != cohort['cohort_sha256']:
        raise ValueError('cohort identity/content mismatch')
    by, verdict, _, _, _, identity = load(args.level, set(ids), cohort)
    bad_ids, gates = set(), {}
    for gate, limit in GATES:
        group = [r for r in by.values() if in_ttft_gate(r, gate)]
        bad = [r for r in group if r['ttft_s'] > limit]
        if (len(group), len(bad)) != (verdict['ttft_gates'][gate]['n'], verdict['ttft_gates'][gate]['over_limit']):
            raise ValueError(f'{gate}: original verdict and raw selectors disagree')
        bad_ids.update(r['req_id'] for r in bad)
        gates[gate] = {'n': len(group), 'over': len(bad)}
    with (args.log or args.level / 'server.log').open() as f:
        result = classify(list(by.values()), iter_events(f), None if args.all_requests else bad_ids)
    raw = next(args.level.glob('raw_*.jsonl'))
    result['source'] = {'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                        'workload_identity': identity, 'original_verdict_gates_checked': gates}
    write_outputs(result, args.json, args.csv)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'interpretation', 'source')}))


if __name__ == '__main__':
    main()
