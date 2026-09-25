#!/usr/bin/env python3
"""[ax] 126 (demand-sized cold cap) on the real scheduler code (get_next_batch_to_run, PrefillAdder).

Uses the CPU fakes of test_sched_protect_chain (pools, cache tree, forwards); the scheduler and adder code is
the working tree's. Budget 8192 as served; 120's cap 4096; short hits up to 8192 new tokens.
Run: python3 -m unittest discover -s tests -p test_demand_cap.py
"""
import os
import random
import unittest
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Req, make_scheduler, step, tree_dir

TREE = ROOT / 'engine/sglang'
BASE = tree_dir('37e90023')
ENV = {'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '4096', 'SGLANG_AX_SCHED_SHORT_TOKENS': '8192'}


def continuation(done, left):
    return Req('cont', left, cached=done)  # a cold request mid-prefill: `done` tokens computed, `left` to go


def hit(rid, new):
    return Req(rid, new, cached=65536)  # a mid-chain request: long device prefix hit, `new` uncached tokens


def first_prefill(env, waiting, root=TREE):
    with patch.dict(os.environ, env):
        s, _ = make_scheduler(root, chunk=continuation(60000, 100000), waiting=waiting, budget=8192)
        return step(s)['reqs']


class DemandCap(unittest.TestCase):
    ON = {**ENV, 'SGLANG_AX_SCHED_COLD_CAP_MAX': '6144'}

    def test_a_3k_hit_joins_beside_the_continuation(self):
        # The fixed 6144 cap of probe 078 leaves 2048: a 3000-token hit cannot join and waits the whole
        # continuation. 126 gives the continuation 8192 - 3008 (the hit's page-rounded need), rounded down to
        # the checkpoint grid (256 in these fakes): 5120, and the hit runs in the same batch.
        fixed = first_prefill({**ENV, 'SGLANG_AX_SCHED_COLD_CAP': '6144'}, [hit('h', 3000)])
        self.assertEqual(fixed, [('cont', 60000, 66144)])
        self.assertEqual(first_prefill(self.ON, [hit('h', 3000)]),
                         [('cont', 60000, 65120), ('h', 65536, 68536)])

    def test_no_waiting_hit_gives_the_continuation_the_maximum(self):
        self.assertEqual(first_prefill(self.ON, [Req('other_cold', 50000)]), [('cont', 60000, 66144)])

    def test_only_hits_that_fit_beside_the_floor_are_reserved(self):
        # Two 3000-token hits: only one fits beside 120's 4096, so only it is reserved; the continuation takes
        # 8192 - 3008 = 5120 (grid) and the first hit joins. Reserving both would pin the chunk at 4096 and
        # leave 1088 idle.
        reqs = first_prefill(self.ON, [hit('h1', 3000), hit('h2', 3000)])
        self.assertEqual(reqs, [('cont', 60000, 65120), ('h1', 65536, 68536)])

    def test_a_hit_that_can_never_join_does_not_shrink_the_chunk(self):
        # 5000 new tokens (5056 paged) do not fit beside 120's 4096 in an 8192 round, so the hit cannot join
        # whatever the cap; the continuation keeps the maximum instead of idling at the floor.
        self.assertEqual(first_prefill(self.ON, [hit('h', 5000)]), [('cont', 60000, 66144)])

    def test_a_hit_refused_for_slots_is_not_reserved_again(self):
        # One request slot: the reserved hit is refused (the continuation holds the slot); from the next round
        # on it is no longer reserved and the continuation runs at the maximum.
        with patch.dict(os.environ, self.ON):
            s, _ = make_scheduler(TREE, chunk=continuation(60000, 100000), waiting=[hit('h', 3000)],
                                  budget=8192, slots=1)
            chunks = [step(s)['reqs'] for _ in range(3)]
        self.assertEqual([c[0][2] - c[0][1] for c in chunks], [5120, 6144, 6144])
        self.assertTrue(all(len(c) == 1 for c in chunks))

    def test_first_chunk_of_a_new_cold_request_leaves_room_for_a_waiting_hit(self):
        with patch.dict(os.environ, self.ON):
            s, _ = make_scheduler(TREE, waiting=[Req('cold', 100000), hit('h', 3000)], budget=8192)
            reqs = step(s)['reqs']
        self.assertEqual(sorted(r[0] for r in reqs), ['cold', 'h'])
        self.assertEqual(next(r for r in reqs if r[0] == 'cold')[2], 5120)


class Refusals(unittest.TestCase):
    def check(self, env, pattern):
        with patch.dict(os.environ, env):
            s, _ = make_scheduler(TREE)
            with self.assertRaisesRegex(ValueError, pattern):
                s._ax_demand_cap_max()

    def test_maximum_below_120s_cap(self):
        self.check({**ENV, 'SGLANG_AX_SCHED_COLD_CAP_MAX': '2048'}, 'below')

    def test_with_122(self):
        self.check({**ENV, 'SGLANG_AX_SCHED_COLD_CAP_MAX': '6144', 'SGLANG_AX_PACE_TPOT': '0.085'}, '122')

    def test_without_protection(self):
        self.check({'SGLANG_AX_SCHED_PROTECT': '0', 'SGLANG_AX_SCHED_COLD_CAP_MAX': '6144'}, 'protection')


class OffEqualsBase(unittest.TestCase):
    def trace(self, root, seed):
        rng = random.Random(seed)
        with patch.dict(os.environ, ENV):
            s, _ = make_scheduler(root, waiting=[Req('w0', 30000)], budget=8192, interval=2)
            out = []
            for i in range(80):
                arrivals = []
                if rng.random() < .5:
                    arrivals = [rng.choice([Req(f'c{i}', rng.choice([12000, 60000])), hit(f'h{i}', rng.choice([500, 3000]))])]
                t = step(s, arrivals=arrivals)
                out.append((t['mode'], t['reqs'], t['chunk'], t['interval']))
        return out

    def test_decision_sequences_match_the_base(self):
        for seed in range(5):
            self.assertEqual(self.trace(TREE, seed), self.trace(BASE, seed))


if __name__ == '__main__':
    unittest.main()
