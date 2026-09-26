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


def member(rid, shared, own, family_token):
    """A chain start whose prompt is `shared` tokens common to its family, then `own` tokens of its own."""
    r = req(rid, 0)
    r.origin_input_ids = [family_token] * shared + [hash((rid, i)) % 1000 + 1000 for i in range(own)]
    return r


class FamilyOrder(unittest.TestCase):
    # Shapes from run 104 (N26 opening): family 62bf0fac had 4 chain starts of ~36k sharing ~31k.
    def family(self):
        lead = member('lead', 31000, 4000, 7)
        followers = [member(f'f{i}', 31000, 5000, 7) for i in range(3)]
        return lead, followers

    def test_followers_link_to_the_leader_whose_tokens_they_share(self):
        lead, followers = self.family()
        other = member('other', 31000, 4000, 8)  # same length, different tokens
        links = ax.link_families([other, lead] + followers, {f.rid: 31000 for f in followers})
        self.assertEqual(links, {f.rid: ('lead', 31000) for f in followers})

    def test_followers_link_only_within_their_cache_namespace(self):
        # LPM held f0 back because of leader B (same salt); A has the same tokens under another salt.
        a, b = member('A', 31000, 4000, 7), member('B', 31000, 4000, 7)
        f = member('f0', 31000, 5000, 7)
        a.cache_salt, b.cache_salt, f.cache_salt = 'x', 'y', 'y'
        self.assertEqual(ax.link_families([a, b, f], {'f0': 31000}), {'f0': ('B', 31000)})

    def test_family_work_counts_the_shared_prefix_once(self):
        lead, followers = self.family()
        links = ax.link_families([lead] + followers, {f.rid: 31000 for f in followers})
        # (35000 + 3 * (36000 - 31000)) / 4 requests
        self.assertEqual(ax.family_unit_work([lead] + followers, links), {'lead': 12500.0})

    def test_leader_of_a_cheap_family_goes_before_a_smaller_single_start(self):
        lead, followers = self.family()
        solo = req('solo', 20000)
        q = [solo, lead] + followers
        held = {f.rid for f in followers}
        unit = ax.family_unit_work(q, ax.link_families(q, {f.rid: 31000 for f in followers}))
        plain = ax.tier_order(q, lambda r: 0.0, held, CHUNK, CFG)
        family = ax.tier_order(q, lambda r: 0.0, held, CHUNK, CFG, unit)
        self.assertEqual([r.rid for r in plain], ['solo', 'lead', 'f0', 'f1', 'f2'])
        self.assertEqual([r.rid for r in family], ['lead', 'solo', 'f0', 'f1', 'f2'])

    def test_a_leader_that_cannot_make_it_stays_hopeless(self):
        # Whether the budget can still be met stays the leader's own: 28 s waited leaves no time for 35k.
        lead, followers = self.family()
        solo = req('solo', 20000)
        q = [solo, lead] + followers
        unit = ax.family_unit_work(q, ax.link_families(q, {f.rid: 31000 for f in followers}))
        waited = {'solo': 0.0, 'lead': 28.0}
        order = ax.tier_order(q, lambda r: waited.get(r.rid, 0.0), {f.rid for f in followers}, CHUNK, CFG, unit)
        self.assertEqual([r.rid for r in order][:2], ['solo', 'lead'])

    def test_links_are_reused_while_the_leader_waits_and_dropped_after(self):
        lead, followers = self.family()
        cache = {}
        shared = {f.rid: 31000 for f in followers}
        ax.link_families([lead] + followers, shared, cache)
        self.assertEqual(cache['f0'], ('lead', 31000))
        lead.origin_input_ids = []  # would no longer match: a reused link is not re-verified
        self.assertEqual(ax.link_families([lead] + followers, shared, cache)['f0'], ('lead', 31000))
        # once the leader has been admitted, the followers find no leader in the queue
        self.assertEqual(ax.link_families(followers[:1], {'f0': 31000}, cache), {})
        self.assertEqual(cache, {})
