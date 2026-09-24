#!/usr/bin/env python3
"""Score a drained diagnostic window using original harness statistics.

Validate against the dispatch census and frozen chain prefixes. This is never a
full-cohort VALID verdict. Different candidates may reach different request sets;
report counts, work, whole observed latency, and matched requests separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if not (HERE/'score_formal.py').exists():
    sys.path.insert(0, str(HERE.parents[1]))
import score_formal
from timed_loadgen import write_json


def validate(out, data_root, harness_dir):
    receipt = json.loads((out/'timed_window.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    flush = json.loads((out/'flush_evidence.json').read_text())
    assert receipt['status'] == 'DRAINED' and receipt['scope'] == 'fixed_duration_diagnostic'
    assert flush['flush_success'] is True and flush['runner_rc'] == 0
    assert Path(summary['raw']).name == receipt['raw']
    raw = out/receipt['raw']
    rows = [json.loads(s) for s in raw.read_text().splitlines() if s.strip()]
    ledger_bytes = (out/'dispatch_ledger.jsonl').read_bytes()
    assert hashlib.sha256(ledger_bytes).hexdigest() == receipt['ledger_sha256']
    ledger = [json.loads(s) for s in ledger_bytes.splitlines()]
    dispatched = [e for e in ledger if e['event'] == 'dispatch']
    finished = [e for e in ledger if e['event'] == 'completed']
    assert not any(e['event'] == 'unadmitted_failure' for e in ledger)
    ids = [r['req_id'] for r in rows]
    sent = {e['req_id']:e for e in dispatched}
    done = {e['req_id']:e for e in finished}
    assert len(ids) == len(set(ids)) == len(dispatched) == len(finished)
    assert set(ids) == set(sent) == set(done)
    assert receipt['n_dispatched'] == receipt['n_completed'] == len(ids)
    start, end = receipt['first_dispatch_at_s'], receipt['admission_deadline_s']
    assert start >= flush['flush_finished_s'] and end == start+receipt['duration_s']
    score_formal.load_harness(harness_dir)
    from s1_common import load_index
    index, _, _ = load_index(str(data_root))
    cohort = json.loads((data_root/'cohort.json').read_text())
    expected = {rid for ch in cohort['chains'] for rid in ch['req_ids']}
    assert set(ids) <= expected
    for chain in cohort['chains']:
        observed = [rid for rid in chain['req_ids'] if rid in sent]
        assert observed == chain['req_ids'][:len(observed)], 'non-prefix chain selection'
    for row in rows:
        rid = row['req_id']
        assert row['client_dispatch_at_s'] == sent[rid]['client_dispatch_at_s']
        assert start <= row['client_dispatch_at_s'] < end
        assert row['client_finish_at_s'] == done[rid]['client_finish_at_s']
        assert row['client_finish_at_s'] >= row['client_dispatch_at_s']
        frozen = index[rid]
        for key in ('phase', 'glm_tokens', 'uncached_expected', 'max_output_i', 'replay_gap_ms',
                    'pack', 'view', 'logical_call_id', 'chain_id', 'session_id', 'edge_type'):
            assert row[key] == frozen.get(key), ('metadata mismatch', rid, key)
        assert row['idx_in_chain'] == frozen['_idx_in_chain']
    run = json.loads((out/Path(summary['run']).name).read_text())
    assert run['config']['N'] == summary['n'] == flush['n']
    assert run['config']['cohort_sha256'] == cohort['cohort_sha256']
    return receipt, rows, run


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--harness-dir', type=Path, required=True)
    args=ap.parse_args(argv)
    try:
        receipt, rows, run = validate(args.out, args.data_root, args.harness_dir)
        score = score_formal.score_records(rows, run, score_formal.load_harness(args.harness_dir),
                                           args.data_root/'requests.jsonl')
        score.update(label='fixed_duration_diagnostic', full_cohort_complete=False,
                     note='All admitted requests drained; not a complete-cohort SLO verdict.')
        score['estimated'].update(label='diagnostic_subset', evaluation_status='DIAGNOSTIC_ONLY')
        score['estimated'].pop('passed', None)
        score['dev'].pop('ALL_PASS', None)
        score['dev']['evaluation_status'] = 'DIAGNOSTIC_ONLY'
        t0=receipt['first_dispatch_at_s']
        window=[r for r in rows if t0+600 <= r['client_dispatch_at_s'] < t0+4200]
        # This separate diagnostic has a known admission cutoff. Original
        # steady_window validity is retained, including t_last < W1 failure.
        score['fixed_window_throughput'] = dict(
            label='diagnostic_only', basis='client_dispatch_at_s', start_min=10, end_min=70,
            denominator_minutes=60, n_requests=len(window),
            successful_requests=sum(not r.get('error') for r in window),
            logical_tokens_per_min=sum((r.get('prompt_tokens') or 0)+(r.get('output_tokens') or 0)
                                       for r in window if not r.get('error'))/60,
            decode_tokens_per_min=sum(r.get('output_tokens') or 0 for r in window if not r.get('error'))/60,
            admitted_request_outputs_count_after_drain=True,
            note='Not original steady TPM; includes full output of requests admitted before the cutoff.')
        # Keep the original statistics, including unavailable TPM when the
        # dataset exhausts before 70 minutes. Never invent a filled steady window.
        write_json(args.out/'timed_score.json', score)
        summary = dict(scope='fixed_duration_diagnostic', status='DRAINED',
                       raw=receipt['raw'], n_completed=len(rows), n=run['config']['N'],
                       raw_sha256=hashlib.sha256((args.out/receipt['raw']).read_bytes()).hexdigest(),
                       full_cohort_complete=False, admission_seconds=receipt['duration_s'],
                       first_dispatch_at_s=receipt['first_dispatch_at_s'],
                       admission_deadline_s=receipt['admission_deadline_s'],
                       drained_at_s=receipt['drained_at_s'],
                       completed_by_deadline=sum(r['client_finish_at_s'] <= receipt['admission_deadline_s'] for r in rows),
                       errors=sum(bool(r.get('error')) for r in rows),
                       dispatched_req_ids=sorted(r['req_id'] for r in rows),
                       tpot=score['tpot'], ttft=score['ttft_estimated'])
        write_json(args.out/'timed_verdict.json', summary)
        print('TIMED_DIAGNOSTIC DRAINED',len(rows),'requests; not full-cohort VALID',flush=True)
        return 0
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        write_json(args.out/'timed_verdict.json', dict(status='INVALID', scope='fixed_duration_diagnostic',
                                                      reason=str(exc), full_cohort_complete=False))
        print('TIMED_DIAGNOSTIC INVALID',str(exc),file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.dont_write_bytecode=True
    sys.exit(main())
