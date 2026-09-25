#!/usr/bin/env python3
"""123 on the working scheduler: ordering, eligibility and resource contracts.

Cache matches, pools and forwards are CPU fakes; no timing/performance prediction.
Run: python3 -m unittest discover -s tests -p test_srpt_admission.py
"""
import os
import random
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Req, load_source, make_scheduler, step
from test_tpot_paced_prefill import Clock, advance

P123 = ROOT / 'engine/sglang'


def req(rid, work, cached=0, waited=0.0):
    r = Req(rid, work, cached=cached)
    r.time_stats.wait_queue_entry_time = time.perf_counter() - waited
    return r


class SrptAdmission(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING': '2000', 'SGLANG_AX_SCHED_PROTECT': '1',
                                           'SGLANG_AX_SCHED_COLD_CAP': '2048', 'SGLANG_AX_SCHED_SHORT_TOKENS': '4096',
                                           'SGLANG_AX_PACE_TPOT': '0', 'SGLANG_AX_ADMISSION_TRACE': '0'})
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


class CurrentSrptContracts(unittest.TestCase):
    """Exercise real calc_priority and scheduler; only prefix-match results are fake."""

    def setUp(self):
        self.env = patch.dict(os.environ, {
            'SGLANG_AX_SRPT_AGING': '2000', 'SGLANG_AX_SCHED_PROTECT': '1',
            'SGLANG_AX_SCHED_COLD_CAP': '4096', 'SGLANG_AX_SCHED_SHORT_TOKENS': '8192',
            'SGLANG_AX_PACE_TPOT': '0', 'SGLANG_AX_ADMISSION_TRACE': '0',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.clock = Clock()

    def request(self, rid, tail, device=0, host=0, waited=0, output=20):
        r = Req(rid, tail + host, cached=device, host=host, output=output)
        # Production match_prefix_for_req counts both tiers (subject to max-prefix
        # clipping); Req's basic fixture counts only device. Leave a nonempty tail.
        r.num_matched_prefix_tokens = device + host
        r.time_stats.wait_queue_entry_time = self.clock.t - waited
        return r

    def scheduler(self, held=(), **kwargs):
        s, ns = make_scheduler(P123, **kwargs)
        ns['time'] = self.clock
        s.enable_hierarchical_cache = True
        cls = ns['SchedulePolicy']
        policy = cls.__new__(cls)
        policy.policy = ns['CacheAwarePolicy'].LPM
        policy.tree_cache = s.tree_cache
        policy.enable_priority_scheduling = False
        policy._compute_prefix_matches = lambda q, p: set(held)
        s.policy = policy
        return s, ns

    def ordered(self, queue, held=()):
        s, _ = self.scheduler(waiting=queue, held=held)
        s.policy.calc_priority(s.waiting_queue, s.running_batch)
        return [r.rid for r in s.waiting_queue]

    def test_off_uses_real_lpm_dispatch_with_host_matches(self):
        for value in ('', '0'):
            with self.subTest(value=value), patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING': value}):
                self.assertEqual(self.ordered([
                    self.request('small', 64),
                    self.request('device', 5000, device=4096),
                    self.request('host', 10000, host=8192),
                ]), ['host', 'device', 'small'])

    def test_host_restore_cost_is_not_in_srpt_score(self):
        self.assertEqual(self.ordered([
            self.request('device', 256, device=4096),
            self.request('host', 64, host=65536),
            self.request('cold', 1024),
        ]), ['host', 'device', 'cold'])

    def test_aged_cold_can_overtake_fresh_host_hit(self):
        self.assertEqual(self.ordered([
            self.request('host', 64, host=65536),
            self.request('old_cold', 100000, waited=60),
        ]), ['old_cold', 'host'])

    def test_sort_does_not_make_host_or_cold_eligible_next_to_partial(self):
        host = self.request('host', 64, device=4096, host=4096)
        cold = self.request('cold', 128)
        device = self.request('device', 256, device=4096)
        s, _ = self.scheduler(chunk=self.request('partial', 30000), waiting=[device, cold, host])
        t = step(s)
        self.assertEqual([r[0] for r in t['reqs']], ['partial', 'device'])
        self.assertEqual(t['waiting'], ['host', 'cold'])
        s.tree_cache.init_load_back.assert_not_called()
        self.assertEqual(t['verdicts'][:2], [('host', 'OTHER'), ('cold', 'OTHER')])

    def test_aging_cannot_preempt_an_active_partial(self):
        old = self.request('old', 10000, waited=100)
        s, _ = self.scheduler(chunk=self.request('partial', 30000), waiting=[old])
        t = step(s)
        self.assertEqual(t['chunk'], 'partial')
        self.assertEqual(t['waiting'], ['old'])
        self.assertEqual([r[0] for r in t['reqs']], ['partial'])

    def test_holdback_and_equal_cost_keep_stable_order(self):
        self.assertEqual(self.ordered([
            self.request('held', 64, waited=100), self.request('first', 256),
            self.request('second', 256),
        ], held={'held'}), ['first', 'second', 'held'])

    def test_retracted_output_counts_as_prefill_work(self):
        resumed = self.request('resumed', 64, device=4096)
        resumed.output_ids = [1] * 4096
        self.assertEqual(self.ordered([resumed, self.request('fresh', 1024)]), ['fresh', 'resumed'])

    def test_missing_or_future_queue_time_adds_no_aging_credit(self):
        missing, future = self.request('missing', 3000), self.request('future', 2000)
        missing.time_stats.wait_queue_entry_time = None
        future.time_stats.wait_queue_entry_time = self.clock.t + 20
        self.assertEqual(self.ordered([missing, future, self.request('short', 64)]),
                         ['short', 'future', 'missing'])

    def test_32_running_requests_still_block_new_admission(self):
        # max_running_requests is resolved into pool/PP limits at initialization;
        # changing the scheduler attribute alone does not resize either resource.
        for pp_cap, free_rows in ((32, 16), (64, 0)):
            with self.subTest(pp_cap=pp_cap, free_rows=free_rows):
                s, ns = self.scheduler(waiting=[self.request('new', 64)], slots=free_rows,
                                      running=[self.request(f'r{i}', 1) for i in range(32)])
                s.max_running_requests = 32
                ns['get_parallel'] = lambda: NS(pp_max_micro_batch_size=pp_cap)
                t = step(s)
                self.assertEqual(t['mode'], 'decode')
                self.assertEqual(t['waiting'], ['new'])
                self.assertEqual(len(t['reqs']), 32)

    def test_srpt_does_not_bypass_kv_rejection(self):
        s, _ = self.scheduler(chunk=self.request('partial', 30000),
                              waiting=[self.request('hit', 4096, device=4096)], available=5000)
        t = step(s)
        self.assertEqual(t['chunk'], 'partial')
        self.assertEqual(t['waiting'], ['hit'])
        self.assertIn(('hit', 'NO_TOKEN'), t['verdicts'])

    def test_122_on_and_off_preserve_one_partial_and_nonnegative_budgets(self):
        for pace in ('0', '0.085'):
            for seed in range(3):
                with self.subTest(pace=pace, seed=seed), patch.dict(os.environ, {'SGLANG_AX_PACE_TPOT': pace}):
                    self.clock = Clock()
                    rng = random.Random(seed)
                    s, _ = self.scheduler(running=[self.request('decoder', 1, output=100)], interval=2)
                    for tick in range(60):
                        arrivals = [self.request(f'q{tick}', rng.choice([64, 1536, 4096, 12000]),
                                                device=rng.choice([0, 4096, 16000]))] if tick < 20 else []
                        t = step(s, arrivals)
                        advance(self.clock, t)
                        if t['mode'] != 'prefill':
                            continue
                        a = s.adders[-1]
                        partials = [r for r in a.can_run_list
                                    if r.extend_range.end < len(r.full_untruncated_fill_ids)]
                        self.assertLessEqual(len(partials), 1)
                        if partials:
                            self.assertIs(partials[0], s.chunked_req)
                        self.assertGreaterEqual(a.rem_input_tokens, 0)
                        self.assertGreaterEqual(a.rem_chunk_tokens, 0)
                        self.assertGreaterEqual(a.rem_total_tokens, 0)


if __name__ == '__main__':
    unittest.main()
