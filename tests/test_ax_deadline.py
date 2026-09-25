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


def req(rid, prompt, matched=0, output=0):
    return NS(rid=rid, origin_input_ids=[0] * prompt, output_ids=[0] * output, num_matched_prefix_tokens=matched)


class Estimates(unittest.TestCase):
    def test_remaining_counts_output_like_123(self):
        self.assertEqual(ax.remaining_tokens(req('a', 10000, matched=4000, output=5)), 6005)

    def test_budget_by_visible_hit_ratio(self):
        self.assertEqual(ax.budget_s(req('cold', 30000, matched=1000), CFG), 30.0)
        self.assertEqual(ax.budget_s(req('warm', 30000, matched=29000), CFG), 5.0)

    def test_252k_cold_start_is_rescuable_only_if_started_at_once(self):
        # 1.27 * (0.13 + 252000 * 68e-6) = 21.9 s of service against a 30 s budget
        giant = req('giant', 252000)
        self.assertGreater(ax.slack_s(giant, 0.0, CFG), 0)
        self.assertLess(ax.slack_s(giant, 8.0, CFG), 0)


class TierOrder(unittest.TestCase):
    def test_rescuable_small_first_then_hopeless_then_held(self):
        waited = {'giant': 20.0, 'small': 1.0, 'mid': 1.0, 'held': 0.0, 'late_small': 29.0}
        q = [req('giant', 252000), req('held', 12000), req('mid', 71000), req('small', 12000),
             req('late_small', 12000)]
        order = ax.tier_order(q, lambda r: waited[r.rid], {'held'}, CFG)
        # late_small (waited 29 s) cannot make 30 s any more: it joins the hopeless tier by size
        self.assertEqual([r.rid for r in order], ['small', 'mid', 'late_small', 'giant', 'held'])

    def test_stable_for_equal_work(self):
        q = [req('a', 12000), req('b', 12000), req('c', 12000)]
        self.assertEqual([r.rid for r in ax.tier_order(q, lambda r: 0.0, set(), CFG)], ['a', 'b', 'c'])

    def test_cache_hit_shortens_remaining_as_in_123(self):
        q = [req('cold', 20000), req('hit', 60000, matched=59500)]
        self.assertEqual([r.rid for r in ax.tier_order(q, lambda r: 0.0, set(), CFG)], ['hit', 'cold'])


class Parking(unittest.TestCase):
    def test_long_continuation_yields_to_a_rescuable_waiter_that_fits(self):
        cont, head = req('cont', 195000, matched=60000), req('small', 12000)
        self.assertTrue(ax.should_park(cont, 10.0, head, 2.0, 16384, 0, CFG))

    def test_waiter_that_would_become_a_second_partial_does_not_park(self):
        cont, head = req('cont', 195000, matched=60000), req('mid', 34000)
        self.assertFalse(ax.should_park(cont, 10.0, head, 2.0, 16384, 0, CFG))

    def test_hopeless_waiter_does_not_park(self):
        cont, head = req('cont', 195000, matched=60000), req('small', 12000)
        self.assertFalse(ax.should_park(cont, 10.0, head, 29.5, 16384, 0, CFG))

    def test_short_continuation_that_can_still_make_it_is_not_delayed(self):
        cont, head = req('cont', 40000, matched=10000), req('small', 12000)
        self.assertFalse(ax.should_park(cont, 1.0, head, 2.0, 16384, 0, CFG))

    def test_hopeless_short_continuation_yields(self):
        cont, head = req('cont', 40000, matched=10000), req('small', 12000)
        self.assertTrue(ax.should_park(cont, 40.0, head, 2.0, 16384, 0, CFG))

    def test_round_limit_and_disable(self):
        cont, head = req('cont', 195000, matched=60000), req('small', 12000)
        self.assertFalse(ax.should_park(cont, 10.0, head, 2.0, 16384, CFG.park_max_rounds, CFG))
        off = ax.DeadlineConfig(park_max_rounds=0)
        self.assertFalse(ax.should_park(cont, 10.0, head, 2.0, 16384, 0, off))
        self.assertFalse(ax.should_park(cont, 10.0, None, 0.0, 16384, 0, CFG))


class Backlog(unittest.TestCase):
    def state(self, **kw):
        return ax.BacklogState(ax.BacklogConfig(**kw))

    def test_hysteresis(self):
        s = self.state()
        s.note_prefill(9200, 1.0)
        self.assertFalse(s.decide(9200 * 20))      # 20 s of backlog: below high
        self.assertTrue(s.decide(9200 * 31))       # above 30 s: relieve
        self.assertTrue(s.decide(9200 * 15))       # between low and high: stay
        self.assertFalse(s.decide(9200 * 9))       # below 10 s: back to normal

    def test_guard_trips_for_good(self):
        s = self.state(max_slow=2)
        s.note_prefill(9200, 1.0)
        self.assertTrue(s.decide(9200 * 40))
        s.note_finished(0.12)
        s.note_finished(0.05)
        s.note_finished(0.11)
        self.assertEqual((s.slow, s.finished), (2, 3))
        self.assertFalse(s.decide(9200 * 40))

    def test_no_rate_no_relief_and_smoothing(self):
        s = self.state(rate_weight=0.5)
        self.assertFalse(s.decide(10 ** 6))
        s.note_prefill(8000, 1.0)
        s.note_prefill(12000, 1.0)
        self.assertEqual(s.rate, 10000.0)
        s.note_prefill(0, 1.0)
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

    def test_relief_needs_a_fixed_interval_above_the_relaxed_one(self):
        with patch.dict(os.environ, {'SGLANG_AX_BACKLOG_RELIEF': '1'}, clear=True):
            with self.assertRaises(ValueError):
                ax.backlog_config(1)
            with self.assertRaises(ValueError):
                ax.backlog_config(0)

    def test_invalid_deadline_config_refuses(self):
        with patch.dict(os.environ, {'SGLANG_AX_DEADLINE_TIERS': '1', 'SGLANG_AX_DEADLINE_PER_TOKEN_S': '0'},
                        clear=True):
            with self.assertRaises(ValueError):
                ax.deadline_config()


if __name__ == '__main__':
    unittest.main()
