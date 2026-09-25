"""Evidence contracts, including real scheduler -> capped log -> scoped ledger."""
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
import admission_trace as parser
import admission_triage as triage
from test_admission_trace import module as collector
from test_sched_protect_chain import Req, make_scheduler, step
from test_tpot_paced_prefill import Clock


def raw(rid='a', recv=10999.0, admit=11010.0, queue=10.0, first=11011.0):
    return {'req_id': rid, 't_recv_s': recv, 't_exec_start_s': admit,
            't_first_token_s': first, 'queue_time_s': queue, 'ttft_s': first - recv}


def event(kind='admit', rid='a', observed=11001.0, entered=11000.0, counts=None, examples=None):
    record = {'rid': rid, 'observed_wait_s': 1.0, 'observed_at_s': observed,
              'queue_entry_at_s': entered, 'decisions': counts if counts is not None else {'partial_host_restore': 2},
              'examples': examples or {}}
    return {'schema': 2, 'event': 'admit', **record} if kind == 'admit' else {
        'schema': 2, 'event': 'waiting', 'queue_size': 1, 'sample': [record]}


def classify(events, rows=None):
    lines = [parser.MARKER + json.dumps(obj) for obj in events]
    return triage.classify(rows or [raw()], parser.iter_events(lines))


def cpu_case(kind):
    """Returns deterministic fake timings plus ACTUAL scheduler branch records."""
    clock, lines = Clock(), []
    env = {'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '4096',
           'SGLANG_AX_SCHED_SHORT_TOKENS': '8192', 'SGLANG_AX_PACE_TPOT': '0',
           'SGLANG_AX_SRPT_AGING': '0', 'SGLANG_AX_ADMISSION_TRACE': '0'}
    if kind == 'host':
        host = Req('host', 4160, cached=8192, host=4096)
        waiting = [host, Req('device', 256, cached=4096)]
        kwargs = {'chunk': Req('partial', 30000)}
    elif kind == 'head':
        waiting = [Req('head', 6000, cached=8192), Req('tail', 256, cached=4096)]
        kwargs = {'available': 5000, 'running': [Req('decode', 1)]}
    elif kind == 'slots':
        waiting = [Req('waiting', 64, cached=4096)]
        kwargs = {'slots': 0, 'running': [Req('decode', 1)]}
    else:
        raise ValueError(kind)
    for req in waiting:
        req.time_stats.wait_queue_entry_time = clock.t
    trace = collector.AxAdmissionTrace(NS(info=lambda fmt, body: lines.append(fmt % body)),
                                       clock=clock.monotonic, to_epoch=lambda t: t + 10000)
    with patch.dict(os.environ, env):
        s, ns = make_scheduler(ROOT / 'engine/sglang', waiting=waiting, **kwargs)
        ns['time'] = clock
        s.enable_hierarchical_cache = True
        s._ax_admission_collector = trace
        clock.t += 1
        scheduling = step(s)
    rows = [raw(r.rid, admit=11001.0, queue=1.0) if r.rid == 'device' else raw(r.rid) for r in waiting]
    result = triage.classify(rows, parser.iter_events(lines))
    return {'kind': 'CPU fakes, not GPU/performance data', 'case': kind,
            'raw': rows, 'log': lines, 'scheduling': scheduling, 'result': result}


class AdmissionTriageTests(unittest.TestCase):
    def test_no_diagnostics_never_means_gpu_saturated_or_wrong_sort(self):
        result = triage.classify([raw('old'), raw('new', recv=11000, admit=11001, queue=1)], [])
        old = next(r for r in result['rows'] if r['req_id'] == 'old')
        self.assertEqual(old['later_arrivals_admitted_first'], 1)
        self.assertEqual(old['ordering_cause'], 'unknown')
        self.assertEqual(old['compute_saturation'], 'unknown')
        self.assertEqual(old['diagnostic_scope'], 'unknown')
        self.assertIsNone(old['decision_counts'])

    def test_first_episode_excludes_warmup_and_retraction(self):
        records = [event(entered=10900, observed=10901), event('waiting'), event('waiting'),
                   event(observed=11002, counts={'partial_host_restore': 3}),
                   event(entered=11020, observed=11021)]
        result = classify(records)
        self.assertEqual(result['rows'][0]['decision_counts'], {'partial_host_restore': 3})
        self.assertEqual(result['excluded_observations'], {'different_queue_episode': 2})

    def test_legacy_same_id_is_not_evidence_for_this_run(self):
        obj = event()
        del obj['queue_entry_at_s']
        del obj['observed_at_s']
        self.assertEqual(classify([obj])['rows'][0]['diagnostic_scope'], 'unknown')

    def test_sampled_requests_do_not_cover_unsampled_queue_tail(self):
        obj = event('waiting')
        obj.update(queue_size=40, truncated=True)
        result = classify([obj], [raw('a'), raw('b')])
        self.assertEqual(result['selected_diagnostic_scope'], {'snapshot_only': 1, 'unknown': 1})

    def test_actual_host_branch_maps_only_to_host_eligibility(self):
        case = cpu_case('host')
        host = next(r for r in case['result']['rows'] if r['req_id'] == 'host')
        self.assertEqual(host['observed_groups'], ['host_eligibility'])
        self.assertEqual(host['first_reason_examples']['partial_host_restore'], {'partial_rid': 'partial'})
        self.assertEqual(host['compute_saturation'], 'unknown')

    def test_actual_head_rejection_and_unscanned_tail_are_distinct(self):
        case = cpu_case('head')
        by = {r['req_id']: r for r in case['result']['rows']}
        self.assertEqual(by['head']['observed_groups'], ['candidate_kv_budget'])
        self.assertEqual(by['tail']['observed_groups'], ['scan_stopped_before_request'])
        self.assertEqual(by['tail']['no_token_scan_stop_example'], 'rejected_candidate')
        self.assertEqual(by['tail']['first_reason_examples']['unscanned_after_no_token']['blocker_rid'], 'head')
        self.assertEqual(by['tail']['skipped_request_feasibility'], 'unknown')

    def test_actual_request_slot_gate_stays_separate(self):
        row = cpu_case('slots')['result']['rows'][0]
        self.assertEqual(row['observed_groups'], ['request_slots'])

    def test_scan_stop_after_success_is_not_rejected_head_example(self):
        obj = event(counts={'unscanned_after_no_token': 1}, examples={
            'unscanned_after_no_token': {'blocker_rid': 'head', 'head_added': True}})
        row = classify([obj])['rows'][0]
        self.assertEqual(row['no_token_scan_stop_example'], 'admitted_candidate')
        self.assertEqual(row['skipped_request_feasibility'], 'unknown')

    def test_compound_causes_and_unmapped_reasons_are_not_forced_into_one_bucket(self):
        row = classify([event(counts={'partial_host_restore': 2, 'batch_full_latched': 3, 'new_reason': 4})])['rows'][0]
        self.assertEqual(row['observed_groups'], ['full_latch', 'host_eligibility'])
        self.assertEqual(row['unmapped_reasons'], ['new_reason'])

    def test_backward_counters_and_duplicate_admit_fail_closed(self):
        for records in ([event('waiting'), event(counts={'partial_host_restore': 1})],
                        [event(), event()], [event('waiting'), event('waiting', observed=11000.5)]):
            with self.assertRaises(ValueError):
                classify(records)

    def test_matching_queue_entry_outside_window_is_excluded(self):
        result = classify([event(observed=11050)])
        self.assertEqual(result['rows'][0]['diagnostic_scope'], 'unknown')
        self.assertEqual(result['excluded_observations'], {'outside_first_admission_window': 1})

    def test_budget_exhaustion_preserves_positive_partial_evidence(self):
        result = classify([event('waiting'), {'event': 'budget_exhausted', 'max_bytes': 512}])
        self.assertTrue(result['any_process_budget_exhausted'])
        self.assertEqual(result['rows'][0]['diagnostic_scope'], 'snapshot_only')
        self.assertEqual(result['rows'][0]['observed_groups'], ['host_eligibility'])

    def test_overtaking_matches_independent_pair_count_with_ties(self):
        rng = random.Random(8)
        rows = []
        for i in range(80):
            recv = rng.randrange(100, 120)
            admit = recv + rng.randrange(0, 20)
            rows.append(raw(str(i), recv=recv, admit=admit, queue=admit-recv, first=admit+1))
        observed = triage.overtaking(rows)
        for r in rows:
            expected = sum(q['t_recv_s'] > r['t_recv_s'] and q['t_exec_start_s'] < r['t_exec_start_s'] for q in rows)
            self.assertEqual(observed[r['req_id']]['count'], expected)

    def test_bad_raw_times_and_duplicates_are_rejected(self):
        for rows in ([raw(), raw()], [raw(queue=1000)], [raw(first=10990)],
                     [{**raw(), 'ttft_s': 99}], [{**raw(), 'error': 'failure'}]):
            with self.assertRaises(ValueError):
                triage.classify(rows, [])

    def test_invalid_diagnostic_context_or_timestamp_rejected(self):
        for obj in (event(observed=float('nan')), event(entered=True),
                    event(examples={'partial_host_restore': {'partial_rid': None}})):
            with self.assertRaises(ValueError):
                classify([obj])

    def test_output_limit_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text('old')
            with self.assertRaises(ValueError):
                triage.write_outputs(classify([event()]), path, max_bytes=10)
            self.assertEqual(path.read_text(), 'old')

    def test_diagnostic_input_has_aggregate_limit(self):
        with self.assertRaises(ValueError):
            list(parser.iter_events([parser.MARKER + json.dumps(event())], max_bytes=10))

    def test_first_reason_example_and_clock_converter_are_preserved(self):
        clock = Clock()
        lines = []
        c = collector.AxAdmissionTrace(NS(info=lambda fmt, body: lines.append(fmt % body)),
                                       clock=clock.monotonic, to_epoch=lambda t: 10000 + t)
        req = Req('a', 64)
        req.time_stats.wait_queue_entry_time = 999
        c.record(req, 'unscanned_after_no_token', {'blocker_rid': 'first', 'head_added': False})
        c.record(req, 'unscanned_after_no_token', {'blocker_rid': 'second', 'head_added': False})
        c.admitted(req)
        obj = list(parser.iter_events(lines))[0]
        self.assertEqual(obj['observed_at_s'], 11000)
        self.assertEqual(obj['queue_entry_at_s'], 10999)
        self.assertEqual(obj['decisions']['unscanned_after_no_token'], 2)
        self.assertEqual(obj['examples']['unscanned_after_no_token']['blocker_rid'], 'first')


if __name__ == '__main__':
    unittest.main()
