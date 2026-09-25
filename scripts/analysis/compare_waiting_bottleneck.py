#!/usr/bin/env python3
"""Compare complete same-ID replays and separate wait from execution at TTFT gates.

Example: python3 scripts/analysis/compare_waiting_bottleneck.py BASE_RAW CAND_RAW OUT_DIR
This is diagnostic: wait does not establish scheduler eligibility or counterfactual gain.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 's1-dev/harness'))
from s1_common import in_ttft_gate


GATES = {'fast_intra': 3.0, 'overall_intra': 5.0, 'chain_start': 30.0}
FIELDS = ('req_id', 'gate', 'base_ttft_s', 'cand_ttft_s', 'base_cached_tokens',
          'cand_cached_tokens', 'cand_uncached_tokens', 'base_recv_admit_s',
          'cand_recv_admit_s', 'base_admit_exec_s', 'cand_admit_exec_s',
          'base_exec_first_s', 'cand_exec_first_s', 'cand_arrival_min')


def read(path):
    rows = [json.loads(line) for line in path.open()]
    by_id = {r['req_id']: r for r in rows}
    if len(rows) != len(by_id) or any(r.get('error') for r in rows):
        raise ValueError(f'incomplete, duplicate, or errored records: {path}')
    return by_id


def segments(r):
    return (r['t_admit_s'] - r['t_recv_s'],
            r['t_exec_start_s'] - r['t_admit_s'],
            r['t_first_token_s'] - r['t_exec_start_s'])


def median(values):
    return statistics.median(values) if values else None


def metrics_by_period(raw_path):
    """Use the existing 10-second samples; these are not continuous-time integrals."""
    root = raw_path.parent
    start = json.loads((root / 'flush_evidence.json').read_text())['flush_finished_s']
    samples = [json.loads(line) for line in (root / 'metrics.jsonl').open()]
    periods = ((0, 10), (10, 30), (30, 60), (60, 120))
    result = {}
    for lo, hi in periods:
        rows = [r for r in samples if lo * 60 <= r['t'] - start < hi * 60]
        if not rows:
            continue
        queue = sorted(r['num_queue_reqs'] for r in rows)
        result[f'{lo}-{hi}m'] = {
            'samples': len(rows),
            'queue_positive_fraction': sum(r['num_queue_reqs'] > 0 for r in rows) / len(rows),
            'queue_p95_sample': queue[int(.95 * (len(queue) - 1))],
            'full_kv_ge_095_fraction': sum(r['full_token_usage'] >= .95 for r in rows) / len(rows),
            'kv_available_median_tokens': median(r['kv_available_tokens'] for r in rows),
            'kv_evictable_median_tokens': median(r['kv_evictable_tokens'] for r in rows),
        }
    measured = [r for r in samples if r['t'] >= start]
    if measured:
        def tier_counts(r):
            return {s['labels'].split('mode="')[1].split('"')[0]: s['value']
                    for s in r['samples']
                    if s['name'] == 'prefill_effective_tokens_total'
                    and 'tp_rank="0"' in s['labels']}
        first, last = tier_counts(measured[0]), tier_counts(measured[-1])
        result['prefill_tier_counter_delta_tp0'] = {k: last[k] - first[k] for k in last}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base_raw', type=Path)
    p.add_argument('cand_raw', type=Path)
    p.add_argument('out_dir', type=Path)
    args = p.parse_args()
    base, cand = read(args.base_raw), read(args.cand_raw)
    if base.keys() != cand.keys():
        raise ValueError('replays must contain exactly the same request IDs')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrival_zero = min(r['client_dispatch_at_s'] for r in cand.values())
    report = {'scope': 'complete same-ID diagnostic; no scheduler eligibility inference',
              'base_raw': str(args.base_raw), 'cand_raw': str(args.cand_raw),
              'request_count': len(base), 'gates': {},
              'metrics_10s_samples': {'base': metrics_by_period(args.base_raw),
                                      'candidate': metrics_by_period(args.cand_raw)}}
    detail = []
    for gate, threshold in GATES.items():
        ids = [rid for rid in base if in_ttft_gate(base[rid], gate)]
        if set(ids) != {rid for rid in cand if in_ttft_gate(cand[rid], gate)}:
            raise ValueError(f'gate membership changed: {gate}')
        old_bad = {rid for rid in ids if base[rid]['ttft_s'] > threshold}
        new_bad = {rid for rid in ids if cand[rid]['ttft_s'] > threshold}
        added, fixed = new_bad - old_bad, old_bad - new_bad
        report['gates'][gate] = {
            'n': len(ids), 'base_over_limit': len(old_bad),
            'cand_over_limit': len(new_bad), 'added': len(added), 'fixed': len(fixed),
            'added_same_cached': sum(base[r]['cached_tokens'] == cand[r]['cached_tokens'] for r in added),
            'added_cached_delta_gt_4096': sum(abs(base[r]['cached_tokens'] - cand[r]['cached_tokens']) > 4096 for r in added),
            'added_median_uncached': median(cand[r]['prompt_tokens'] - cand[r]['cached_tokens'] for r in added),
            'added_median_segment_delta_s': [median(segments(cand[r])[i] - segments(base[r])[i]
                                             for r in added) for i in range(3)],
            'added_wait_majority': sum(cand[r]['t_exec_start_s'] - cand[r]['t_recv_s'] >
                                       cand[r]['ttft_s'] / 2 for r in added),
        }
        for rid in sorted(added):
            a, b = base[rid], cand[rid]
            aa, bb = segments(a), segments(b)
            detail.append(dict(zip(FIELDS, (rid, gate, a['ttft_s'], b['ttft_s'],
                 a['cached_tokens'], b['cached_tokens'], b['prompt_tokens'] - b['cached_tokens'],
                 aa[0], bb[0], aa[1], bb[1], aa[2], bb[2],
                 (b['client_dispatch_at_s'] - arrival_zero) / 60))))
    (args.out_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (args.out_dir / 'new_bad_cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(detail)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
