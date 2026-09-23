#!/usr/bin/env python3
"""Patch 122 (adaptive decode rounds after each prefill batch) on the real scheduler code with CPU fakes.

Build the tree first:
  python3 scripts/patch_stack.py apply build/p122/candidate 000-interface-compliance 101-role-boundary-split \
      106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 120-sched-protect-chain 122-adaptive-decode-rounds
then: python3 -m unittest discover -s tests -p test_adaptive_decode_rounds.py
"""
import math
import os
import unittest
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Req, make_scheduler, step

P122 = ROOT / 'build/p122/candidate/sglang'
ENV = {'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '2048', 'SGLANG_AX_SCHED_SHORT_TOKENS': '4096'}


def owed(tokens, target=0.085, fixed=0.11, per_token=6.5e-5, step_s=0.022, cap=64):
    return max(1, min(cap, math.ceil((fixed + per_token * tokens) / (target - step_s))))


class AdaptiveRounds(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, dict(ENV, SGLANG_AX_TPOT_TARGET='0.085'))
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_rounds_scale_with_prefill_size_while_decoding(self):
        s, _ = make_scheduler(P122, waiting=[Req('c', 20000)], running=[Req('r', 1)])
        modes = [step(s)['mode'] for _ in range(1 + owed(8192) + 1)]
        # an 8192-token chunk (alone: uncapped) owes 11 rounds to the running stream, then prefill resumes
        self.assertEqual(owed(8192), 11)
        self.assertEqual(modes, ['prefill'] + ['decode'] * owed(8192) + ['prefill'])

    def test_short_prefill_owes_few_rounds(self):
        s, _ = make_scheduler(P122, waiting=[Req('short', 512, cached=65536)], running=[Req('r', 1)])
        modes = [step(s)['mode'] for _ in range(1 + owed(512))]
        self.assertEqual(owed(512), 3)
        self.assertEqual(modes, ['prefill'] + ['decode'] * 3)

    def test_nothing_owed_without_decoders(self):
        s, _ = make_scheduler(P122, waiting=[Req('c', 20000)])
        self.assertEqual([step(s)['mode'] for _ in range(3)], ['prefill'] * 3)

    def test_rounds_are_capped(self):
        with patch.dict(os.environ, {'SGLANG_AX_DECODE_ROUNDS_MAX': '4'}):
            s, _ = make_scheduler(P122, waiting=[Req('c', 20000)], running=[Req('r', 1)])
            self.assertEqual([step(s)['mode'] for _ in range(6)], ['prefill'] + ['decode'] * 4 + ['prefill'])

    def test_off_by_default_matches_120(self):
        with patch.dict(os.environ, {'SGLANG_AX_TPOT_TARGET': '0'}):
            s, _ = make_scheduler(P122, waiting=[Req('c', 20000)], running=[Req('r', 1)])
            self.assertEqual([step(s)['mode'] for _ in range(4)], ['prefill', 'decode', 'prefill', 'decode'])


if __name__ == '__main__':
    unittest.main()
