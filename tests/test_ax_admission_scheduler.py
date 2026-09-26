#!/usr/bin/env python3
"""[ax] 124/125 on the real scheduler code (get_next_batch_to_run, PrefillAdder) with CPU fakes.

The working tree's scheduler methods are compiled as in test_sched_protect_chain; the 124/125 methods
and ax_deadline are added. The request-plane broadcast is replaced by a local call (one rank) or by a
shared payload (two simulated ranks). Pools, clocks, forwards and the cache tree are fakes.
Run: python3 -m unittest discover -s tests -p test_ax_admission_scheduler.py
"""
import os
import random
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Req, make_scheduler, step, tree_dir

TREE = ROOT / 'engine/sglang'
BASE = tree_dir('83197755')
PROTECT = {'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '16384',
           'SGLANG_AX_SCHED_SHORT_TOKENS': '8192'}


def scheduler(root=TREE, **kw):
    s, ns = make_scheduler(root, **kw)
    if root == TREE:
        s._ax_rank0_decide = lambda compute: compute()  # one rank; see the two-rank test for the broadcast
    return s, ns


def cold(rid, work, waited=0.0, cached=0):
    r = Req(rid, work, cached=cached)
    r.time_stats.scheduler_recv_time = time.perf_counter() - waited
    r.time_stats.prefill_finished_time = 0.0
    return r


def continuation(rid, done, left):
    r = cold(rid, left, cached=done)
    r.num_matched_prefix_tokens = 0  # a cold chain start being chunked: nothing came from the cache
    return r


class DeadlineOrder(unittest.TestCase):
    def env(self, **extra):
        return patch.dict(os.environ, {**PROTECT, 'SGLANG_AX_DEADLINE_TIERS': '1', **extra})

    def test_small_cold_start_is_admitted_before_a_giant(self):
        with self.env():
            s, _ = scheduler(waiting=[cold('giant', 252000), cold('small', 12000)], budget=16384)
            t = step(s)
        self.assertEqual(t['mode'], 'prefill')
        self.assertEqual(t['reqs'][0], ('small', 0, 12000))

    def test_off_keeps_lpm_order(self):
        with patch.dict(os.environ, PROTECT):
            s, _ = scheduler(waiting=[cold('giant', 252000), cold('small', 12000)], budget=16384)
            t = step(s)
        self.assertEqual(t['reqs'][0][0], 'giant')

    def test_two_ranks_with_different_stamps_apply_the_leader_order(self):
        payload = []
        orders = []
        for rank, small_waited in enumerate((0.0, 29.0)):  # rank 1's stamps would make 'small' hopeless
            with self.env():
                s, _ = scheduler(waiting=[cold('mid', 14000), cold('small', 12000, waited=small_waited)],
                                 budget=16384)

                def decide(compute, rank=rank):
                    if rank == 0:
                        payload.append(compute())
                    return payload[0]

                s._ax_rank0_decide = decide
                t = step(s)
            orders.append([r[0] for r in t['reqs']])
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0][0], 'small')


class Parking(unittest.TestCase):
    def env(self, **extra):
        return patch.dict(os.environ, {**PROTECT, 'SGLANG_AX_DEADLINE_TIERS': '1', **extra})

    def test_long_continuation_yields_one_round_to_a_small_cold_start(self):
        with self.env():
            cont = continuation('cont', 60000, 135000)
            s, _ = scheduler(chunk=cont, waiting=[cold('small', 12000)], budget=16384)
            t = step(s)
            self.assertEqual([r[0] for r in t['reqs']], ['small'])
            self.assertEqual(t['chunk'], 'cont')
            self.assertEqual(cont._ax_parked_rounds, 1)
            self.assertEqual(cont.inflight_middle_chunks, 0)  # 120: only a continuation that runs counts
            # the waiter is served; next time the continuation runs again and its park count resets
            t = step(s)
            while t['mode'] != 'prefill':
                t = step(s)
            self.assertEqual(t['reqs'][0][0], 'cont')
            self.assertEqual(cont._ax_parked_rounds, 0)

    def test_nearly_finished_continuation_is_not_parked(self):
        # A 200k cold start with 196608 tokens computed has 3392 left, not the 200k its admission-time
        # match (0) implies: it finishes this round instead of yielding it.
        with self.env():
            cont = continuation('cont', 196608, 3392)
            s, _ = scheduler(chunk=cont, waiting=[cold('small', 12000)], budget=16384)
            t = step(s)
        self.assertEqual(t['reqs'][0], ('cont', 196608, 200000))

    def test_waiter_above_the_cold_cap_cannot_be_rescued_by_parking(self):
        # the cold cap cuts a new cold request into a partial, which a parked owner forbids
        with self.env(SGLANG_AX_SCHED_COLD_CAP='8192'):
            cont = continuation('cont', 60000, 135000)
            s, _ = scheduler(chunk=cont, waiting=[cold('small', 12000)], budget=16384)
            t = step(s)
        self.assertEqual(t['reqs'][0][0], 'cont')

    def test_waiter_that_does_not_fit_leaves_the_continuation_running(self):
        with self.env():
            cont = continuation('cont', 60000, 135000)
            s, _ = scheduler(chunk=cont, waiting=[cold('big', 40000)], budget=16384)
            t = step(s)
        self.assertEqual(t['reqs'][0][0], 'cont')

    def test_parking_is_bounded_by_rounds(self):
        with self.env(SGLANG_AX_PARK_MAX_ROUNDS='2', SGLANG_AX_PARK_MAX_S='60'):
            cont = continuation('cont', 60000, 135000)
            s, _ = scheduler(chunk=cont, waiting=[cold('s0', 12000)], budget=16384)
            heads = []
            for i in range(1, 8):
                t = step(s, arrivals=[cold(f's{i}', 12000)])
                if t['mode'] == 'prefill':
                    heads.append(t['reqs'][0][0])
        self.assertIn('cont', heads[:3])

    def test_never_two_partials_and_budget_respected(self):
        rng = random.Random(7)
        with self.env():
            cont = continuation('cont', 30000, 200000)
            full = {'cont': len(cont.origin_input_ids)}
            s, _ = scheduler(chunk=cont, waiting=[], budget=16384)
            for i in range(60):
                arrivals = []
                if rng.random() < .6:
                    r = cold(f'r{i}', rng.choice([2000, 9000, 12000, 30000, 90000]))
                    full[r.rid] = len(r.origin_input_ids)
                    arrivals.append(r)
                t = step(s, arrivals=arrivals)
                if t['mode'] != 'prefill':
                    continue
                truncated = [rid for rid, _, end in t['reqs'] if end < full[rid]]
                self.assertLessEqual(len(truncated), 1, t)
                self.assertLessEqual(sum(end - begin for _, begin, end in t['reqs']), 16384, t)


class BacklogRelief(unittest.TestCase):
    def env(self, **extra):
        # the interval tests set the relaxed interval explicitly; it defaults to the configured one
        return patch.dict(os.environ, {**PROTECT, 'SGLANG_AX_BACKLOG_RELIEF': '1',
                                       'SGLANG_AX_BACKLOG_INTERVAL': '1', **extra})

    def test_relieved_prefill_arms_the_relaxed_interval(self):
        with self.env():
            s, _ = scheduler(waiting=[cold(f'c{i}', 120000) for i in range(4)], budget=16384, interval=2)
            s._ax_admission_cfgs()
            s._ax_backlog.rate = 9000.0
            t = step(s)
        self.assertEqual(t['mode'], 'prefill')
        self.assertTrue(s._ax_backlog_relieved)
        self.assertEqual(t['interval'], 1)

    def test_small_backlog_keeps_the_configured_interval(self):
        with self.env():
            s, _ = scheduler(waiting=[cold('c', 12000)], budget=16384, interval=2)
            s._ax_admission_cfgs()
            s._ax_backlog.rate = 9000.0
            t = step(s)
        self.assertFalse(s._ax_backlog_relieved)
        self.assertEqual(t['interval'], 2)

    def test_backlog_counts_what_the_continuation_has_left(self):
        # A 200k cold start with 196608 tokens computed has 3392 left: 0.7 s at 5000 tok/s, not the 40 s its
        # admission-time match (0) implies.
        with self.env():
            s, _ = scheduler(chunk=continuation('cont', 196608, 3392), budget=16384, interval=2)
            s._ax_admission_cfgs()
            s._ax_backlog.rate = 5000.0
            step(s)
        self.assertFalse(s._ax_backlog_relieved)

    def test_rate_charges_each_interval_to_the_prefill_that_ran_in_it(self):
        # An 8192-token prefill, two decode rounds, then a 32-token prefill scheduled 0.8 s after it: the
        # 8192 tokens took those 0.8 s (10240 tok/s); pairing the interval with the new 32 would give 40.
        clock = [0.0]
        with self.env(), patch('time.monotonic', lambda: clock[0]):
            s, _ = scheduler(waiting=[cold('a', 8192)], budget=16384, interval=2)
            trace = [step(s)]
            clock[0] = 0.8
            trace += [step(s, arrivals=[cold('b', 32)]), step(s), step(s)]
        self.assertEqual([t['mode'] for t in trace], ['prefill', 'decode', 'decode', 'prefill'])
        self.assertEqual(trace[-1]['reqs'], [('b', 0, 32)])
        self.assertAlmostEqual(s._ax_backlog.rate, 8192 / 0.8)

    def test_cold_cap_relief_measures_the_rate_with_interval_0(self):
        # Cold-cap-only relief is valid with --prefill-decode-interval 0; the rate must still be sampled
        # (it used to be sampled only where a fixed interval is armed, so relief silently never started).
        with self.env(SGLANG_AX_SCHED_COLD_CAP='4096', SGLANG_AX_BACKLOG_COLD_CAP='8192',
                      SGLANG_AX_BACKLOG_INTERVAL='0'):
            s, _ = scheduler(waiting=[cold(f'c{i}', 120000) for i in range(2)], budget=16384, interval=0)
            s._ax_admission_cfgs()
            prefills = 0
            while prefills < 2:
                prefills += step(s)['mode'] == 'prefill'
        self.assertGreater(s._ax_backlog.rate, 0)

    def test_opening_mode_raises_the_cold_cap_from_the_next_round(self):
        # 120 caps a cold chunk at 4096 while others wait; relieved, the cap is 8192. The relief decided in
        # a round applies to the next round's cap (the plan needs the cap as its round budget).
        with self.env(SGLANG_AX_SCHED_COLD_CAP='4096', SGLANG_AX_BACKLOG_COLD_CAP='8192',
                      SGLANG_AX_BACKLOG_INTERVAL='2'):
            s, _ = scheduler(waiting=[cold(f'c{i}', 120000) for i in range(4)], budget=16384, interval=2)
            s._ax_admission_cfgs()
            s._ax_backlog.rate = 9000.0
            chunks = []
            while len(chunks) < 2:
                t = step(s)
                if t['mode'] == 'prefill':
                    chunks.append(t['reqs'][0])
        self.assertEqual(chunks, [('c0', 0, 4096), ('c0', 4096, 12288)])


class DemandCapUnderRelief(unittest.TestCase):
    # 126 with 125: while relieved, 125's cold cap is the maximum of 126's demand-sized cap, so a waiting short hit
    # keeps its seat beside the cold chunk instead of waiting the whole continuation (run 109: warm turn starts
    # waited 18-23 s while relief ran 8192-token cold chunks with no room left).
    def env(self, **extra):
        return patch.dict(os.environ, {**PROTECT, 'SGLANG_AX_SCHED_COLD_CAP': '4096',
                                       'SGLANG_AX_SCHED_COLD_CAP_MAX': '6144', 'SGLANG_AX_DEADLINE_TIERS': '1',
                                       'SGLANG_AX_BACKLOG_RELIEF': '1', 'SGLANG_AX_BACKLOG_COLD_CAP': '8192',
                                       'SGLANG_AX_BACKLOG_INTERVAL': '2', **extra})

    def hit(self, rid, new):
        return Req(rid, new, cached=65536)  # device prefix hit, `new` uncached tokens

    def run_two_rounds(self, with_126):
        env = self.env() if with_126 else self.env(SGLANG_AX_SCHED_COLD_CAP_MAX='')
        with env:
            # a 30k cold start (short: no parking) and three long ones make the cold backlog large at 9000 tok/s
            s, _ = scheduler(waiting=[cold('c0', 30000)] + [cold(f'c{i}', 120000) for i in (1, 2, 3)],
                             budget=8192, interval=2)
            s._ax_admission_cfgs()
            s._ax_backlog.rate = 9000.0
            first = step(s)
            self.assertEqual(first['mode'], 'prefill')
            self.assertTrue(s._ax_backlog_relieved)  # decided in round one, applied from round two
            t = step(s, arrivals=[self.hit('h', 3000)])
            while t['mode'] != 'prefill':
                t = step(s)
        return first['reqs'], t['reqs']

    def test_relieved_cold_chunk_leaves_the_seat_when_126_is_on(self):
        first, second = self.run_two_rounds(with_126=True)
        self.assertEqual(first, [('c0', 0, 6144)])  # not yet relieved: 126's maximum
        # relieved: 8192 minus the hit's paged need (3008), on the 256 grid -> 5120, and the hit rides along
        self.assertEqual(second, [('c0', 6144, 11264), ('h', 65536, 68536)])

    def test_relieved_cold_chunk_takes_the_whole_round_without_126(self):
        first, second = self.run_two_rounds(with_126=False)
        self.assertEqual(first, [('c0', 0, 4096)])
        self.assertEqual(second, [('c0', 4096, 12288)])  # the hit waits


class FamilyOrder(unittest.TestCase):
    """128 on the real scheduler: the family leader goes before smaller lone heads; riders wait for it."""

    def env(self, **extra):
        return patch.dict(os.environ, {**PROTECT, 'SGLANG_AX_DEADLINE_TIERS': '1',
                                       'SGLANG_AX_DEADLINE_FAMILY': '1', **extra})

    @staticmethod
    def head(rid, shared_blocks, tail_blocks, seed):
        r = cold(rid, (shared_blocks + tail_blocks) * 256)
        r.origin_input_ids = [7] * (shared_blocks * 256) + [1000 + seed] * (tail_blocks * 256)
        r.full_untruncated_fill_ids = r.origin_input_ids[:]
        return r

    def queue(self):
        # arrival order: the 14k and 19k heads first, then a family of four sharing 32k (each 3k of its own)
        return [self.head('h14', 0, 55, 1), self.head('h19', 0, 75, 2),
                self.head('fam0', 128, 12, 10), self.head('fam1', 128, 12, 11),
                self.head('fam2', 128, 12, 12), self.head('fam3', 128, 12, 13)]

    def test_family_leader_is_admitted_before_the_smaller_lone_heads(self):
        with self.env():
            s, _ = scheduler(waiting=self.queue(), budget=16384)
            t = step(s)
        self.assertEqual(t['mode'], 'prefill')
        self.assertTrue(t['reqs'][0][0].startswith('fam'), t['reqs'])  # ranked as 35840 // 4 = 8960 < 14080
        self.assertEqual(sum(r[0].startswith('fam') for r in t['reqs']), 1)  # riders are held, not admitted
        self.assertIn('h14', t['waiting'])

    def test_off_keeps_the_per_request_order(self):
        with patch.dict(os.environ, {**PROTECT, 'SGLANG_AX_DEADLINE_TIERS': '1'}):
            s, _ = scheduler(waiting=self.queue(), budget=16384)
            t = step(s)
        self.assertEqual(t['reqs'][0][0], 'h14')

    def test_family_needs_124(self):
        with patch.dict(os.environ, {**PROTECT, 'SGLANG_AX_DEADLINE_FAMILY': '1'}):
            s, _ = scheduler()
            with self.assertRaisesRegex(ValueError, '128'):
                s._ax_admission_cfgs()


class Refusals(unittest.TestCase):
    # Each refusal is matched on its message, so a config refused for another reason does not pass.
    def test_124_with_123_refuses(self):
        env = {**PROTECT, 'SGLANG_AX_DEADLINE_TIERS': '1', 'SGLANG_AX_SRPT_AGING': '300'}
        with patch.dict(os.environ, env):
            s, _ = scheduler()
            with self.assertRaisesRegex(ValueError, '123'):
                s._ax_admission_cfgs()

    def test_125_without_an_effect_refuses(self):
        with patch.dict(os.environ, {**PROTECT, 'SGLANG_AX_BACKLOG_RELIEF': '1'}):
            s, _ = scheduler(interval=0)
            with self.assertRaisesRegex(ValueError, 'COLD_CAP'):
                s._ax_admission_cfgs()

    def test_125_with_122_refuses(self):
        env = {**PROTECT, 'SGLANG_AX_BACKLOG_RELIEF': '1', 'SGLANG_AX_BACKLOG_COLD_CAP': '8192',
               'SGLANG_AX_PACE_TPOT': '0.085'}
        with patch.dict(os.environ, env):
            s, _ = scheduler(interval=2)
            with self.assertRaisesRegex(ValueError, '122'):
                s._ax_admission_cfgs()

    def test_needs_protection(self):
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_PROTECT': '0', 'SGLANG_AX_DEADLINE_TIERS': '1'}):
            s, _ = scheduler()
            with self.assertRaisesRegex(ValueError, 'protection'):
                s._ax_admission_cfgs()


class OffEqualsBase(unittest.TestCase):
    def trace(self, root, seed):
        rng = random.Random(seed)
        with patch.dict(os.environ, PROTECT):
            s, _ = scheduler(root, waiting=[cold('w0', 30000)], budget=8192, interval=2)
            if root == TREE:
                s._ax_rank0_decide = lambda compute: self.fail('off path must not broadcast')
            out = []
            for i in range(80):
                arrivals = [cold(f'r{i}', rng.choice([500, 3000, 12000, 60000]))] if rng.random() < .5 else []
                t = step(s, arrivals=arrivals)
                out.append((t['mode'], t['reqs'], t['chunk'], t['interval']))
        return out

    def test_decision_sequences_match_the_base(self):
        for seed in range(5):
            self.assertEqual(self.trace(TREE, seed), self.trace(BASE, seed))


if __name__ == '__main__':
    unittest.main()
