#!/usr/bin/env python3
"""Patch 124 (short-hit reserve next to a partial) on the real scheduler code with CPU fakes.

Build the trees first (official A stack, with and without 124):
  A="000-interface-compliance 101-role-boundary-split 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer \
     111-sm80-fp8-moe-marlin 114-indexer-row-shard 120-sched-protect-chain 121-sched-cap-while-decoding"
  B="130-async-tokenize 140-kda-dual-snapshot 150-startup-warmup 160-nextn-sm80 170-glm-bcg-prefill"
  python3 scripts/patch_stack.py apply build/p122/baseA $A $B
  python3 scripts/patch_stack.py apply build/p124/candidate $A 124-short-hit-reserve $B
then: python3 -m unittest discover -s tests -p test_short_hit_reserve.py

Motivation (044r, official A at dev N14, scripts/analysis/intra_attrib.py): cached hits with 4097-8192 new tokens were
over their TTFT limit 26/79 times (wait p50 6.56 s, a partial in every wait window), hits with <= 4096 new tokens 0/274.
"""
import os
import random
import unittest
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Req, make_scheduler, step

BASE = ROOT / 'build/p122/baseA/sglang'
CAND = ROOT / 'build/p124/candidate/sglang'
A_ENV = {'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '4096', 'SGLANG_AX_SCHED_SHORT_TOKENS': '8192'}
ON = {'SGLANG_AX_SHORT_RESERVE': '1'}


def build(root, **kw):
    kw.setdefault('budget', 8192)
    kw.setdefault('interval', 2)
    return make_scheduler(root, **kw)[0]


def partial():
    cold = Req('cold', 30000)
    cold.prefix_indices = [0] * 8192  # 8192 tokens already prefilled
    return cold


def shares(trace):
    return {rid: end - start for rid, start, end in trace['reqs']}


class ShortHitReserve(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, A_ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_both(self, **kw):
        out = []
        for root in (BASE, CAND):
            s = build(root, chunk=partial(), **kw)
            out.append(step(s))
        return out

    def test_off_is_identical_to_official_a(self):
        for seed in range(3):
            traces = []
            for root in (BASE, CAND):
                rng = random.Random(seed)
                s = build(root, running=[Req('r0', 1, output=50)])
                trace = []
                for i in range(60):
                    arr = []
                    if rng.random() < 0.3:
                        arr.append(Req(f'c{i}', rng.randint(3000, 30000), output=rng.randint(5, 60)))
                    if rng.random() < 0.4:
                        arr.append(Req(f's{i}', rng.randint(100, 8000), cached=4096, output=rng.randint(5, 60)))
                    t = step(s, arr)
                    trace.append((t['mode'], t['reqs'], t['chunk']))
                traces.append(trace)
            self.assertEqual(traces[0], traces[1])

    def test_hit_bigger_than_leftover_joins_the_partial(self):
        with patch.dict(os.environ, ON):
            base, cand = self.run_both(waiting=[Req('hit', 7000, cached=60000)])
        self.assertEqual(shares(base), {'cold': 4096})          # official A: the hit waits for the whole partial
        self.assertEqual(shares(cand), {'cold': 1024, 'hit': 7000})  # 8192 - paged(7000)=1152 -> grid 256 -> 1024

    def test_small_hit_and_no_hit_are_unchanged(self):
        with patch.dict(os.environ, ON):
            base, cand = self.run_both(waiting=[Req('hit', 3000, cached=60000)])
            self.assertEqual(shares(base), shares(cand))
            self.assertEqual(shares(cand), {'cold': 4096, 'hit': 3000})
            base, cand = self.run_both(waiting=[Req('other', 30000)], running=[Req('r', 1, output=50)])
            self.assertEqual(shares(base), shares(cand))

    def test_waiting_lengths_do_not_depend_on_unbuilt_fill_ids(self):
        with patch.dict(os.environ, ON):
            hit = Req('hit', 7000, cached=60000)
            hit.full_untruncated_fill_ids = []  # real waiting requests: built only at admission
            s = build(CAND, chunk=partial(), waiting=[hit])
            # the reserve is computed from seqlen, so the partial still yields; (the fake's admission does not
            # rebuild fill ids like the real _refresh_fill_ids, so only the partial's share is checked here)
            self.assertEqual(shares(step(s))['cold'], 1024)

    def test_partial_keeps_progressing_under_many_hits(self):
        with patch.dict(os.environ, ON):
            hits = [Req(f'h{i}', 7000, cached=60000) for i in range(3)]
            s = build(CAND, chunk=partial(), waiting=hits)
            t = step(s)
            self.assertGreaterEqual(shares(t)['cold'], 256)  # at least one grid unit
            self.assertEqual(sum(shares(t).values()) <= 8192, True)


if __name__ == '__main__':
    unittest.main()
