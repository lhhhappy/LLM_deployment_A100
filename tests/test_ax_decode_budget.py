#!/usr/bin/env python3
"""[ax] 133 planner (engine/sglang/srt/managers/ax_decode_budget.py) on CPU.

Run: python3 -m unittest discover -s tests -p test_ax_decode_budget.py
"""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ax_decode_budget', ROOT / 'engine/sglang/srt/managers/ax_decode_budget.py')
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)

CFG = db.DecodeBudgetConfig()  # gate 0.10 x 0.9, step 35 ms, max 64 rounds, spend 60% of the 5%


def dec(rid, first_out, produced, n, spent=False):
    return db.Decoder(rid, first_out, produced, n, spent)


class Slack(unittest.TestCase):
    def test_slack_is_the_deadline_minus_now_minus_pure_decode_time(self):
        # n=198 (public p50): deadline = t0 + 0.09*197 = 17.73 s; after 50 tokens at t0+2 s, 148 left * 35 ms = 5.18 s
        d = dec('a', 100.0, 50, 198)
        self.assertAlmostEqual(db.slack_s(d, 102.0, CFG), 17.73 - 2.0 - 5.18, places=6)

    def test_short_output_has_almost_no_slack(self):
        # n=16: deadline 1.35 s; 8 tokens left needs 0.28 s -> a 1.1 s prefill batch cannot fit after 0.3 s
        d = dec('s', 100.0, 8, 16)
        self.assertLess(db.slack_s(d, 100.3, CFG), 1.1)


class Plan(unittest.TestCase):
    def test_prefill_runs_at_once_when_every_decoder_can_absorb_it(self):
        ds = [dec('long', 100.0, 20, 554), dec('mid', 100.0, 10, 198)]
        self.assertEqual(db.plan(ds, 101.0, 1.1, None, 100, 0, CFG), (0, []))

    def test_tight_decoder_arms_decode_rounds_until_it_finishes(self):
        # 16-token output with 6 tokens left: slack < 1.1 s -> 6 decode rounds, no prefill now
        ds = [dec('short', 100.0, 10, 16), dec('long', 100.0, 20, 554)]
        self.assertEqual(db.plan(ds, 100.35, 1.1, None, 100, 0, CFG), (6, []))

    def test_rounds_are_capped(self):
        ds = [dec('tight', 100.0, 5, 400)]        # deadline 35.9 s but 395 tokens left need 13.8 s...
        now = 100.0 + 35.9 - 13.8 - 0.5           # ...so slack is 0.5 s < 1.1 s: protect until done, capped at 64
        self.assertEqual(db.plan(ds, now, 1.1, None, 100, 0, CFG), (64, []))

    def test_lost_decoders_are_not_protected(self):
        d = dec('lost', 100.0, 3, 16)             # deadline 1.35 s, 13 left = 0.455 s, now +1.2 s -> slack < 0
        self.assertEqual(db.plan([d], 101.2, 1.1, None, 100, 0, CFG), (0, []))
        self.assertEqual(db.plan([dec('spent', 100.0, 8, 16, spent=True)], 100.3, 1.1, None, 100, 0, CFG), (0, []))

    def test_allowance_is_spent_for_an_urgent_chain_start(self):
        # a rescuable cold head has 0.1 s of slack; protecting the short decoder would take 6 rounds (0.21 s)
        ds = [dec('short', 100.0, 10, 16)]
        rounds, spent = db.plan(ds, 100.35, 1.1, 0.1, started=1000, spent=0, cfg=CFG)  # allowance 30
        self.assertEqual((rounds, spent), (0, ['short']))

    def test_allowance_is_not_exceeded(self):
        ds = [dec('short', 100.0, 10, 16)]
        # 40 requests started -> allowance int(0.6*0.05*40) = 1; already spent 1 -> protect instead
        self.assertEqual(db.plan(ds, 100.35, 1.1, 0.1, started=40, spent=1, cfg=CFG), (6, []))

    def test_no_head_waiting_never_spends(self):
        ds = [dec('short', 100.0, 10, 16)]
        self.assertEqual(db.plan(ds, 100.35, 1.1, None, 1000, 0, CFG), (6, []))


class Config(unittest.TestCase):
    def test_off_by_default_and_validated(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SGLANG_AX_DECODE_BUDGET', None)
            self.assertIsNone(db.decode_budget_config())
        with patch.dict(os.environ, {'SGLANG_AX_DECODE_BUDGET': '1'}):
            self.assertEqual(db.decode_budget_config().max_rounds, 64)
        with patch.dict(os.environ, {'SGLANG_AX_DECODE_BUDGET': '1', 'SGLANG_AX_DECODE_BUDGET_MARGIN': '1.5'}):
            with self.assertRaises(ValueError):
                db.decode_budget_config()


if __name__ == '__main__':
    unittest.main()
