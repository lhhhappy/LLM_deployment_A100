"""Decision wall intervals: real collector, fake clocks, no GPU-time claims."""
import copy
import json
import sys
from types import SimpleNamespace as NS
import unittest

from test_admission_trace import module as collector
from test_sched_protect_chain import ROOT, Req

sys.path.insert(0, str(ROOT / 'scripts/analysis'))
import admission_trace as parser


class IntervalTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.lines = []
        self.trace = collector.AxAdmissionTrace(
            NS(info=lambda fmt, body: self.lines.append(fmt % body)),
            clock=lambda: self.now, to_epoch=lambda t: t + 1000,
        )
        self.req = Req('r', 64)
        self.req.time_stats.wait_queue_entry_time = 9.0

    def event(self):
        return list(parser.iter_events(self.lines))[-1]

    def defer(self, reason, kind='decode'):
        self.trace.begin_step([self.req])
        self.trace.record(self.req, reason)
        self.trace.end_step([self.req], kind)

    def test_wall_intervals_and_selection_cpu_time_account_separately(self):
        self.trace.begin_step([self.req])
        self.now = 11
        self.trace.record(self.req, 'decode_cadence')
        self.now = 12
        self.trace.end_step([self.req], 'decode')
        self.now = 14
        self.trace.begin_step([self.req])
        self.now = 15
        self.trace.record(self.req, 'partial_host_restore')
        self.now = 16
        self.trace.end_step([self.req], 'prefill')
        self.now = 21
        self.trace.begin_step([self.req])
        self.now = 22
        self.trace.admitted(self.req)
        row = self.event()
        timing = row['decision_intervals']
        self.assertEqual(row['observed_wait_s'], 12)
        self.assertEqual(timing['seconds_by_reason'], {'decode_cadence': 2, 'partial_host_restore': 5})
        self.assertEqual(timing['seconds_by_batch'], {'decode': 2, 'prefill': 5})
        self.assertEqual(timing['uncovered_s'], 5)
        self.assertEqual(timing['count'], 2)

    def test_many_short_decisions_do_not_outweigh_one_long_interval(self):
        for _ in range(100):
            self.defer('decode_cadence')
            self.now += .001
        self.defer('partial_host_restore', 'prefill')
        self.now += 10
        self.trace.begin_step([self.req])
        self.trace.admitted(self.req)
        row = self.event()
        self.assertEqual(row['decisions'], {'decode_cadence': 100, 'partial_host_restore': 1})
        secs = row['decision_intervals']['seconds_by_reason']
        self.assertAlmostEqual(secs['decode_cadence'], .1)
        self.assertAlmostEqual(secs['partial_host_restore'], 10)

    def test_unobserved_path_and_missing_hook_stay_unknown(self):
        self.trace.begin_step([self.req])
        self.trace.end_step([self.req], 'idle')
        self.now = 12
        self.trace.begin_step([self.req])
        # No end hook for this step: do not carry the old reason across it.
        self.now = 20
        self.trace.begin_step([self.req])
        self.trace.admitted(self.req)
        timing = self.event()['decision_intervals']
        self.assertEqual(timing['seconds_by_reason'], {'unobserved_path': 2})
        self.assertEqual(timing['uncovered_s'], 8)

    def test_snapshot_never_extrapolates_open_interval(self):
        self.defer('decode_cadence')
        self.now = 20
        self.trace.snapshot([self.req])
        timing = self.event()['sample'][0]['decision_intervals']
        self.assertEqual(timing['seconds_by_reason'], {})
        self.assertEqual(timing['uncovered_s'], 10)
        self.trace.begin_step([self.req])
        self.trace.admitted(self.req)
        self.assertEqual(self.event()['decision_intervals']['seconds_by_reason'], {'decode_cadence': 10})

    def test_requeued_episode_cannot_inherit_previous_interval(self):
        self.defer('kv_budget')
        self.now = 20
        self.req.time_stats.wait_queue_entry_time = 20
        self.trace.begin_step([self.req])
        self.trace.admitted(self.req)
        self.assertEqual(self.event()['decision_intervals']['count'], 0)
        self.assertEqual(self.event()['decisions'], {})

    def test_unscanned_does_not_become_resource_failure(self):
        self.defer('unscanned_after_no_token', 'prefill')
        self.now = 11
        self.trace.begin_step([self.req])
        self.trace.admitted(self.req)
        self.assertEqual(self.event()['decision_intervals']['seconds_by_reason'],
                         {'unscanned_after_no_token': 1})

    def test_empty_snapshot_clears_departed_pending_requests(self):
        self.trace.snapshot([self.req])
        self.now = 40
        self.trace.snapshot([])
        result = parser.analyze(self.lines)
        self.assertEqual(result['latest_waiting_snapshot']['queue_size'], 0)
        self.assertEqual(result['rows'], [])
        self.assertEqual(self.event()['observed_at_s'], 1040)

    def test_malformed_time_accounts_fail_closed(self):
        self.defer('decode_cadence')
        self.now += 1
        self.trace.begin_step([self.req])
        self.trace.admitted(self.req)
        row = self.event()
        changes = [
            ('seconds_by_reason', {'decode_cadence': -1}),
            ('seconds_by_reason', {'kv_budget': 1}),
            ('seconds_by_batch', {'prefill': 2}),
            ('seconds_by_batch', {'GPU': 1}),
            ('uncovered_s', float('nan')),
            ('count', 0),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(row)
                bad['decision_intervals'][field] = value
                with self.assertRaises(ValueError):
                    list(parser.iter_events([parser.MARKER + json.dumps(bad)]))
        del row['decision_intervals']
        with self.assertRaises(ValueError):
            list(parser.iter_events([parser.MARKER + json.dumps(row)]))


if __name__ == '__main__':
    unittest.main()
