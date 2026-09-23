#!/usr/bin/env python3
"""Patch 123 (shortest remaining prefill first with aging) on the real scheduler code with CPU fakes.

Build the tree first:
  python3 scripts/patch_stack.py apply build/p123/candidate 000-interface-compliance 101-role-boundary-split \
      106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 120-sched-protect-chain 123-srpt-admission
then: python3 -m unittest discover -s tests -p test_srpt_admission.py
"""
import os
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Req, load_source, make_scheduler, step

P123 = ROOT / 'build/p123/candidate/sglang'


def req(rid, work, cached=0, waited=0.0):
    r = Req(rid, work, cached=cached)
    r.time_stats.wait_queue_entry_time = time.perf_counter() - waited
    return r


class SrptAdmission(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING': '2000', 'SGLANG_AX_SCHED_PROTECT': '1',
                                           'SGLANG_AX_SCHED_COLD_CAP': '2048', 'SGLANG_AX_SCHED_SHORT_TOKENS': '4096'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.ns = load_source(P123)
        self.sort = self.ns['SchedulePolicy']._ax_sort_by_remaining_work

    def order(self, q, held=()):
        self.sort(q, set(held))
        return [r.rid for r in q]

    def test_hit_then_small_cold_then_big_cold(self):
        q = [req('big', 100000), req('small', 3000), req('hit', 500, cached=60000)]
        self.assertEqual(self.order(q), ['hit', 'small', 'big'])

    def test_aging_lets_a_long_waiting_request_through(self):
        # 2000 tokens/s: after 40 s a 100k request still ranks behind a fresh 3k one, after 60 s it goes first
        self.assertEqual(self.order([req('small', 3000), req('big', 100000, waited=40)]), ['small', 'big'])
        self.assertEqual(self.order([req('small', 3000), req('big', 100000, waited=60)]), ['big', 'small'])

    def test_in_batch_prefix_sharing_holdbacks_stay_last(self):
        self.assertEqual(self.order([req('a', 1000), req('b', 2000)], held={'a'}), ['b', 'a'])

    def test_off_when_unset(self):
        with patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING': '0'}):
            self.assertIsNone(self.ns['_ax_srpt_aging']())

    def test_scheduler_admits_the_small_cold_request_first(self):
        s, ns = make_scheduler(P123, waiting=[req('big', 100000), req('small', 3000)])
        s.policy = NS(calc_priority=lambda q, _: ns['SchedulePolicy']._ax_sort_by_remaining_work(q, set()))
        t = step(s)
        self.assertEqual(t['mode'], 'prefill')
        self.assertEqual(t['reqs'][0][0], 'small')


if __name__ == '__main__':
    unittest.main()
