#!/usr/bin/env python3
"""[ax] 124/125 decision functions (engine/sglang/srt/managers/ax_deadline.py) on CPU.

Sizes are the first-minute chain starts of run 073/074 (evidence/L073-.../opening/chain.csv).
Run: python3 -m unittest discover -s tests -p test_ax_deadline.py
"""
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ax_deadline', ROOT / 'engine/sglang/srt/managers/ax_deadline.py')
ax = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ax)

CFG = ax.DeadlineConfig()
CHUNK = 8192


def req(rid, prompt, matched=0, output=0):
    return NS(rid=rid, origin_input_ids=[0] * prompt, output_ids=[0] * output, num_matched_prefix_tokens=matched)


class Estimates(unittest.TestCase):
    def test_remaining_counts_output_like_123(self):
        self.assertEqual(ax.remaining_tokens(req('a', 10000, matched=4000, output=5)), 6005)

    def test_cold_is_a_mostly_uncached_prompt(self):
        self.assertTrue(ax.is_cold(req('a', 30000, matched=14000)))
        self.assertFalse(ax.is_cold(req('b', 30000, matched=16000)))

    def test_budget_by_visible_hit_ratio_and_size(self):
        self.assertEqual(ax.budget_s(req('cold', 30000, matched=1000), CFG), 30.0)
        self.assertEqual(ax.budget_s(req('fast', 30000, matched=29000), CFG), 3.0)
        self.assertEqual(ax.budget_s(req('warm', 30000, matched=20000), CFG), 5.0)

    def test_cost_counts_every_chunk(self):
        # 20000 tokens in 8192-token chunks = 3 chunks: 1.27 * (3 * 0.13 + 20000 * 68e-6)
        self.assertAlmostEqual(ax.service_s(20000, CHUNK, CFG), 1.27 * (0.39 + 1.36))
        self.assertEqual(ax.service_s(0, CHUNK, CFG), 0.0)

    def test_252k_cold_start_is_rescuable_only_if_started_at_once(self):
        # 31 chunks: 1.27 * (31 * 0.13 + 252000 * 68e-6) = 26.9 s of service against a 30 s budget
        giant = req('giant', 252000)
        self.assertGreater(ax.slack_s(giant, 252000, 0.0, CHUNK, CFG), 0)
        self.assertLess(ax.slack_s(giant, 252000, 3.0, CHUNK, CFG), 0)


class TierOrder(unittest.TestCase):
    def test_rescuable_small_first_then_hopeless_then_held(self):
        waited = {'giant': 20.0, 'small': 1.0, 'mid': 1.0, 'held': 0.0, 'late_small': 29.0}
        q = [req('giant', 252000), req('held', 12000), req('mid', 71000), req('small', 12000),
             req('late_small', 12000)]
        order = ax.tier_order(q, lambda r: waited[r.rid], {'held'}, CHUNK, CFG)
        # late_small (waited 29 s) cannot make 30 s any more: it joins the hopeless tier by size
        self.assertEqual([r.rid for r in order], ['small', 'mid', 'late_small', 'giant', 'held'])

    def test_stable_for_equal_work(self):
        q = [req('a', 12000), req('b', 12000), req('c', 12000)]
        self.assertEqual([r.rid for r in ax.tier_order(q, lambda r: 0.0, set(), CHUNK, CFG)], ['a', 'b', 'c'])

    def test_cache_hit_shortens_remaining_as_in_123(self):
        q = [req('cold', 20000), req('hit', 60000, matched=59500)]
        self.assertEqual([r.rid for r in ax.tier_order(q, lambda r: 0.0, set(), CHUNK, CFG)], ['hit', 'cold'])

    def test_starved_requests_go_first_oldest_first(self):
        waited = {'small': 1.0, 'giant': 130.0, 'older_giant': 200.0}
        q = [req('small', 12000), req('giant', 252000), req('older_giant', 195000)]
        order = ax.tier_order(q, lambda r: waited[r.rid], set(), CHUNK, CFG)
        self.assertEqual([r.rid for r in order], ['older_giant', 'giant', 'small'])


class Parking(unittest.TestCase):
    def test_long_continuation_yields_to_a_rescuable_waiter_that_fits(self):
        cont, head = req('cont', 195000, matched=60000), req('small', 12000)
        self.assertTrue(ax.should_park(cont, 135000, 10.0, head, 2.0, 16384, 10 ** 6, 0, 0.0, CFG))

    def test_waiter_that_would_become_a_second_partial_does_not_park(self):
        cont, head = req('cont', 195000, matched=60000), req('mid', 34000)
        self.assertFalse(ax.should_park(cont, 135000, 10.0, head, 2.0, 16384, 10 ** 6, 0, 0.0, CFG))

    def test_hopeless_waiter_does_not_park(self):
        cont, head = req('cont', 195000, matched=60000), req('small', 12000)
        self.assertFalse(ax.should_park(cont, 135000, 10.0, head, 29.5, 16384, 10 ** 6, 0, 0.0, CFG))

    def test_short_continuation_that_can_still_make_it_is_not_delayed(self):
        cont, head = req('cont', 40000, matched=10000), req('small', 12000)
        self.assertFalse(ax.should_park(cont, 30000, 1.0, head, 2.0, 16384, 10 ** 6, 0, 0.0, CFG))

    def test_hopeless_short_continuation_yields(self):
        cont, head = req('cont', 40000, matched=10000), req('small', 12000)
        self.assertTrue(ax.should_park(cont, 30000, 40.0, head, 2.0, 16384, 10 ** 6, 0, 0.0, CFG))

    def test_no_kv_room_no_park(self):
        cont, head = req('cont', 195000, matched=60000), req('small', 12000)
        self.assertFalse(ax.should_park(cont, 135000, 10.0, head, 2.0, 16384, 8000, 0, 0.0, CFG))

    def test_wall_clock_bound(self):
        cont, head = req('cont', 195000, matched=60000), req('small', 12000)
        self.assertTrue(ax.should_park(cont, 135000, 10.0, head, 2.0, 16384, 10 ** 6, 1, 1.0, CFG))
        self.assertFalse(ax.should_park(cont, 135000, 10.0, head, 2.0, 16384, 10 ** 6, 1, 2.0, CFG))

    def test_round_limit_and_disable(self):
        cont, head = req('cont', 195000, matched=60000), req('small', 12000)
        self.assertFalse(ax.should_park(cont, 135000, 10.0, head, 2.0, 16384, 10 ** 6, CFG.park_max_rounds, 0.5, CFG))
        off = ax.DeadlineConfig(park_max_rounds=0)
        self.assertFalse(ax.should_park(cont, 135000, 10.0, head, 2.0, 16384, 10 ** 6, 0, 0.0, off))
        self.assertFalse(ax.should_park(cont, 135000, 10.0, None, 0.0, 16384, 10 ** 6, 0, 0.0, CFG))


class Backlog(unittest.TestCase):
    def state(self, **kw):
        return ax.BacklogState(ax.BacklogConfig(**kw))

    def test_hysteresis(self):
        s = self.state()
        s.rate = 9200.0
        self.assertFalse(s.decide(9200 * 20))      # 20 s of backlog: below high
        self.assertTrue(s.decide(9200 * 31))       # above 30 s: relieve
        self.assertTrue(s.decide(9200 * 15))       # between low and high: stay
        self.assertFalse(s.decide(9200 * 9))       # below 10 s: back to normal

    def test_guard_trips_for_good(self):
        s = self.state(max_slow=2)
        s.rate = 9200.0
        self.assertTrue(s.decide(9200 * 40))
        s.note_tpot('a', 0.12)
        s.note_tpot('b', 0.05)
        s.note_tpot('a', 0.13)  # the same request counts once
        self.assertTrue(s.decide(9200 * 40))
        s.note_tpot('c', 0.11)
        self.assertEqual(s.slow, {'a', 'c'})
        self.assertFalse(s.decide(9200 * 40))
        s.note_tpot('a', 0.05)  # recovering later does not un-count it
        self.assertFalse(s.decide(9200 * 40))

    def test_ratio_guard_after_min_seen_holds_until_flush(self):
        s = self.state(max_slow=10 ** 6, max_slow_ratio=0.03, min_seen=100)
        s.rate = 9200.0
        for i in range(100):
            s.note_tpot(f'r{i}', 0.12 if i < 4 else 0.05)
        self.assertFalse(s.decide(9200 * 40))   # 4 of 100 > 3%
        for i in range(100, 200):
            s.note_tpot(f'r{i}', 0.05)
        self.assertFalse(s.decide(9200 * 40))   # 4 of 200 is under 3% again: still off until the flush
        s.reset()
        self.assertEqual((s.slow, s.seen, s.relieved), (set(), set(), False))
        self.assertTrue(s.decide(9200 * 40))    # rate kept, guard cleared

    def test_no_rate_no_relief_smoothing_and_no_sample_across_a_flush(self):
        s = self.state(rate_weight=0.5)
        self.assertFalse(s.decide(10 ** 6))
        s.note_batch(8000, 0.0)
        s.note_batch(12000, 1.0)    # the 8000 tokens took 1 s
        s.note_batch(4000, 2.0)     # the 12000 tokens took 1 s
        self.assertEqual(s.rate, 10000.0)
        s.reset()                   # /flush_cache: the rate is kept, the 4000-token batch's interval dropped
        s.note_batch(4000, 62.0)    # spanning the flush would sample 4000 / 60 s
        self.assertEqual(s.rate, 10000.0)


class Config(unittest.TestCase):
    def test_off_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ax.deadline_config())
            self.assertIsNone(ax.backlog_config(2))

    def test_env_values(self):
        env = {'SGLANG_AX_DEADLINE_TIERS': '1', 'SGLANG_AX_PARK_MAX_ROUNDS': '4',
               'SGLANG_AX_BACKLOG_RELIEF': '1', 'SGLANG_AX_BACKLOG_INTERVAL': '1'}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(ax.deadline_config().park_max_rounds, 4)
            self.assertEqual(ax.backlog_config(2).relaxed_interval, 1)

    def test_cold_cap_alone_keeps_the_configured_interval(self):
        env = {'SGLANG_AX_BACKLOG_RELIEF': '1', 'SGLANG_AX_BACKLOG_COLD_CAP': '8192'}
        with patch.dict(os.environ, env, clear=True):
            cfg = ax.backlog_config(2)
        self.assertEqual((cfg.cold_cap, cfg.relaxed_interval), (8192, 2))

    def test_relief_must_change_something_and_never_add_decode_rounds(self):
        # the interval now defaults to the configured one, so RELIEF alone would do nothing: refused
        with patch.dict(os.environ, {'SGLANG_AX_BACKLOG_RELIEF': '1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'COLD_CAP'):
                ax.backlog_config(2)
            with self.assertRaisesRegex(ValueError, 'COLD_CAP'):
                ax.backlog_config(0)
        env = {'SGLANG_AX_BACKLOG_RELIEF': '1', 'SGLANG_AX_BACKLOG_INTERVAL': '3'}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                ax.backlog_config(2)

    def test_invalid_deadline_config_refuses(self):
        with patch.dict(os.environ, {'SGLANG_AX_DEADLINE_TIERS': '1', 'SGLANG_AX_DEADLINE_PER_TOKEN_S': '0'},
                        clear=True):
            with self.assertRaises(ValueError):
                ax.deadline_config()


if __name__ == '__main__':
    unittest.main()


class Family(unittest.TestCase):
    """128: families among waiting cold requests from block hashes of their prompts."""
    CFG = None

    def setUp(self):
        self.cfg = ax.FamilyConfig(block=256, link_min=4096, rider_max=6144)

    @staticmethod
    def prompt(shared_blocks, tail_blocks, seed):
        shared = [7] * (shared_blocks * 256)
        return shared + [1000 + seed] * (tail_blocks * 256)

    def req(self, rid, ids, matched=0):
        return NS(rid=rid, origin_input_ids=ids, output_ids=[], num_matched_prefix_tokens=matched)

    def shared_fn(self, reqs):
        hashes = {r.rid: ax.block_hashes(r.origin_input_ids, 256) for r in reqs}
        return lambda a, b: ax.shared_prefix_blocks(hashes[a.rid], hashes[b.rid]) * 256

    def test_block_hashes_ignore_the_partial_tail(self):
        self.assertEqual(len(ax.block_hashes([1] * 1000, 256)), 3)
        self.assertEqual(ax.block_hashes([1] * 512, 256), ax.block_hashes([1] * 700, 256))
        self.assertEqual(ax.shared_prefix_blocks((1, 2, 3), (1, 2, 9)), 2)

    def test_leader_ranks_by_work_per_rider_and_riders_are_held(self):
        # 104/109: a 36k leader and three 35k siblings sharing 32k (each 3k of its own), a 14k and a 19k head
        leader = self.req('leader', self.prompt(128, 12, 1))          # 32768 + 3072 = 35840
        sibs = [self.req(f's{i}', self.prompt(128, 12, 10 + i)) for i in range(3)]
        others = [self.req('h14', self.prompt(0, 55, 2)), self.req('h19', self.prompt(0, 75, 3))]
        reqs = [leader, *sibs, *others]
        # leader = the member with the least remaining work: make it 36k by 1 extra block over the siblings
        leader.origin_input_ids = self.prompt(128, 12, 1)[:-256] + [7] * 256  # same length; tie broken by rid
        work, held, families = ax.family_plan(reqs, self.shared_fn(reqs), self.cfg)
        self.assertEqual(held, {'s0', 's1', 's2'})
        self.assertEqual(len(families), 1)
        lead = families[0][0]
        self.assertIn(lead, {'leader', 's0', 's1', 's2'})
        self.assertEqual(work[lead], 35840 // 4)
        # unrelated heads are not in any family
        self.assertNotIn('h14', held)

    def test_link_needs_an_uncached_shared_prefix(self):
        # two heads sharing only the 8k app prefix that both already have cached do not form a family
        a = self.req('a', self.prompt(32, 40, 1), matched=8192)
        b = self.req('b', self.prompt(32, 60, 2), matched=8192)
        work, held, families = ax.family_plan([a, b], self.shared_fn([a, b]), self.cfg)
        self.assertEqual((work, held, families), ({}, set(), []))
        # the same pair with nothing cached is linked (8192 shared > 4096)
        a.num_matched_prefix_tokens = b.num_matched_prefix_tokens = 0
        work, held, families = ax.family_plan([a, b], self.shared_fn([a, b]), self.cfg)
        self.assertEqual(len(families), 1)

    def test_big_tailed_member_neither_holds_nor_dilutes(self):
        # a sibling with a 50k tail beyond the shared prefix rides nothing: it is listed as other, not held, and
        # the leader's work is diluted only by the real riders
        leader = self.req('leader', self.prompt(128, 12, 1))
        rider = self.req('rider', self.prompt(128, 12, 2))
        big = self.req('big', self.prompt(128, 200, 3))
        reqs = [leader, rider, big]
        work, held, families = ax.family_plan(reqs, self.shared_fn(reqs), self.cfg)
        lead, riders, others = families[0]
        self.assertEqual(set(riders), {'leader', 'rider'} - {lead})
        self.assertEqual(others, ['big'])
        self.assertEqual(held, set(riders))
        self.assertEqual(work[lead], 35840 // 2)

    def test_tier_order_uses_the_override_only_for_ranking(self):
        cfg = ax.DeadlineConfig()
        small = req('small', 14000)
        lead = req('lead', 36000)
        order = ax.tier_order([lead, small], lambda r: 0.0, set(), 8192, cfg, {'lead': 9000})
        self.assertEqual([r.rid for r in order], ['lead', 'small'])
        order = ax.tier_order([lead, small], lambda r: 0.0, set(), 8192, cfg)
        self.assertEqual([r.rid for r in order], ['small', 'lead'])

    def test_config(self):
        with patch.dict(os.environ, {'SGLANG_AX_DEADLINE_FAMILY': '1', 'SGLANG_AX_FAMILY_LINK_MIN': '2048'}):
            self.assertEqual(ax.family_config().link_min, 2048)
        with patch.dict(os.environ, {'SGLANG_AX_DEADLINE_FAMILY': '0'}):
            self.assertIsNone(ax.family_config())
        with patch.dict(os.environ, {'SGLANG_AX_DEADLINE_FAMILY': '1', 'SGLANG_AX_FAMILY_BLOCK': '0'}):
            with self.assertRaises(ValueError):
                ax.family_config()

    def test_starvation_bound_frees_a_held_rider(self):
        cfg = ax.DeadlineConfig(max_wait_s=120)
        lead, rider = req('lead', 36000), req('rider', 35000)
        waited = {'lead': 0.0, 'rider': 121.0}
        order = ax.tier_order([lead, rider], lambda r: waited[r.rid], set(), 8192, cfg, {'lead': 9000},
                              family_held={'rider'})
        self.assertEqual([r.rid for r in order], ['rider', 'lead'])

    def test_lpm_holdback_stays_last_even_when_starved(self):
        # 124 unchanged: an LPM in-batch holdback is last whatever it waited (so 128 off equals 124)
        cfg = ax.DeadlineConfig(max_wait_s=120)
        lead, held = req('lead', 36000), req('held', 35000)
        waited = {'lead': 0.0, 'held': 121.0}
        order = ax.tier_order([held, lead], lambda r: waited[r.rid], {'held'}, 8192, cfg)
        self.assertEqual([r.rid for r in order], ['lead', 'held'])
