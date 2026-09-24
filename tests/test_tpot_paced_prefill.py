#!/usr/bin/env python3
"""Patch 122 (TPOT-paced prefill budget) on the real scheduler code with CPU fakes and a fake clock.

Build the trees first (official A stack, with and without 122):
  A="000-interface-compliance 101-role-boundary-split 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer \
     111-sm80-fp8-moe-marlin 114-indexer-row-shard 120-sched-protect-chain 121-sched-cap-while-decoding"
  B="130-async-tokenize 140-kda-dual-snapshot 150-startup-warmup 160-nextn-sm80 170-glm-bcg-prefill"
  python3 scripts/patch_stack.py apply build/p122/baseA $A $B
  python3 scripts/patch_stack.py apply build/p122/candidate $A 122-tpot-paced-prefill $B
then: python3 -m unittest discover -s tests -p test_tpot_paced_prefill.py

The fake clock charges C0 + C1 * new tokens per prefill batch and DECODE_S per decode round (optionally scaled, to model
cost-model error). The simulation checks the invariant the patch is built on: when the cost model holds, every request's
TPOT stays under the gate; with the patch off the trace is identical to the official A stack.
"""
import math
import os
import random
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Req, make_scheduler, step

BASE = ROOT / 'build/p122/baseA/sglang'
CAND = ROOT / 'build/p122/candidate/sglang'
# official A launch: chunk 8192 (default), interval 2, COLD_CAP 4096, SHORT_TOKENS 8192
A_ENV = {'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '4096', 'SGLANG_AX_SCHED_SHORT_TOKENS': '8192'}
PACE = {'SGLANG_AX_PACE_TPOT': '0.085'}
C0, C1, DECODE_S = 0.08, 6e-5, 0.036


class Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    perf_counter = monotonic
    time = monotonic


def build(root, clock, **kw):
    kw.setdefault('budget', 8192)
    kw.setdefault('interval', 2)
    s, ns = make_scheduler(root, **kw)
    ns['time'] = clock
    return s, ns


def new_tokens(trace):
    return sum(end - start for _, start, end in trace['reqs'])


def advance(clock, trace, scale=1.0):
    if trace['mode'] == 'prefill':
        clock.t += scale * (C0 + C1 * new_tokens(trace))
    elif trace['mode'] == 'decode':
        clock.t += scale * DECODE_S
    else:
        clock.t += 0.01


def simulate(root, env, seconds=240.0, seed=0, scale=1.0, accept=3.3, gap=1.0):
    """Open-loop Poisson arrivals, one per second on average: 20% cold 24k-40k, 80% short hits (200-3000 new tokens on a
    60k cached prefix). About 7.7k new tokens/s, close to the 8.0k/s measured in 044r (official A config, dev N14)."""
    rng = random.Random(seed)
    clock = Clock()
    with patch.dict(os.environ, env, clear=False):
        s, _ = build(root, clock)
        arrivals, t, k = [], clock.t, 0
        while t < clock.t + seconds:
            t += rng.expovariate(1 / gap)
            k += 1
            if rng.random() < 0.2:
                arrivals.append((t, Req(f'c{k}', rng.randint(24000, 40000), output=rng.randint(60, 400))))
            else:
                arrivals.append((t, Req(f's{k}', rng.randint(200, 3000), cached=60000, output=rng.randint(40, 300))))
        pending = sorted(arrivals, key=lambda a: a[0])
        seen, first, done, recv = {}, {}, {}, {}
        prefill_tokens = 0
        end = clock.t + seconds * 3
        while (pending or s.waiting_queue or s.chunked_req or not s.running_batch.is_empty()) and clock.t < end:
            now_arr = [r for ta, r in pending if ta <= clock.t]
            pending = [(ta, r) for ta, r in pending if ta > clock.t]
            for r in now_arr:
                recv[r.rid] = clock.t
                seen[r.rid] = r
            trace = step(s, now_arr)
            advance(clock, trace, scale)
            if trace['mode'] == 'decode' and accept > 1:
                # MTP: step() appended one token per decoder; add the accepted drafts (mean `accept` per step)
                for r in s.running_batch.reqs:
                    extra = int(accept - 1) + (rng.random() < (accept - int(accept)))
                    room = r.sampling_params.max_new_tokens - len(r.output_ids)
                    r.output_ids.extend([1] * max(0, min(extra, room)))
            if trace['mode'] == 'prefill':
                prefill_tokens += new_tokens(trace)
            for rid, r in seen.items():
                if r.output_ids and rid not in first:
                    first[rid] = clock.t
                if r.finished() and rid not in done:
                    done[rid] = clock.t
            if trace['mode'] == 'idle' and pending:
                clock.t = max(clock.t, pending[0][0])
        tpot = {rid: (done[rid] - first[rid]) / (seen[rid].sampling_params.max_new_tokens - 1)
                for rid in done if seen[rid].sampling_params.max_new_tokens > 1}
        ttft = {rid: first[rid] - recv[rid] for rid in first}
        return NS(tpot=tpot, ttft=ttft, n=len(seen), done=len(done), prefill_tokens=prefill_tokens, s=s)


class PacedPrefill(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, A_ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_off_is_identical_to_official_a(self):
        for seed in range(3):
            traces = []
            for root in (BASE, CAND):
                rng, clock = random.Random(seed), Clock()
                s, _ = build(root, clock, running=[Req('r0', 1, output=50)])
                trace = []
                for i in range(60):
                    arr = []
                    if rng.random() < 0.3:
                        arr.append(Req(f'c{i}', rng.randint(3000, 30000), output=rng.randint(5, 60)))
                    if rng.random() < 0.4:
                        arr.append(Req(f's{i}', rng.randint(100, 6000), cached=4096, output=rng.randint(5, 60)))
                    t = step(s, arr)
                    trace.append((t['mode'], t['reqs'], t['chunk']))
                    advance(clock, t)
                traces.append(trace)
            self.assertEqual(traces[0], traces[1])

    def test_no_decoders_uses_full_chunk_minus_short_hit_reserve(self):
        clock = Clock()
        with patch.dict(os.environ, PACE):
            s, _ = build(CAND, clock, waiting=[Req('cold', 30000), Req('cold2', 30000)])
            self.assertEqual(new_tokens(step(s)), 8192)  # official A caps this first chunk at 4096
            s, _ = build(CAND, clock, waiting=[Req('cold', 30000), Req('hit', 1000, cached=60000)])
            t = step(s)
            self.assertEqual(sorted(r[0] for r in t['reqs']), ['cold', 'hit'])
            self.assertEqual(dict((r[0], r[2] - r[1]) for r in t['reqs'])['cold'], 7168)  # 8192 - paged(1000)

    def test_budget_fits_a_fresh_decoders_allowed_deficit(self):
        clock = Clock()
        with patch.dict(os.environ, dict(PACE, SGLANG_AX_PACE_MIN_CHUNK='2048')):
            dec = Req('dec', 1, output=300)
            dec.output_ids = [1]
            s, _ = build(CAND, clock, waiting=[Req('cold', 30000)], running=[dec])
            t = step(s)
            # slack = deficit 0.5 s -> (0.5 - 0.08) / 6e-5 = 7000 tokens -> grid 256 -> 6912
            self.assertEqual(t['mode'], 'prefill')
            self.assertEqual(new_tokens(t), 6912)

    def test_decoder_behind_pace_gets_decode_rounds_until_it_recovers(self):
        clock = Clock()
        with patch.dict(os.environ, PACE):
            dec = Req('dec', 1, output=300)
            dec.output_ids = [1]
            s, _ = build(CAND, clock, waiting=[Req('cold', 30000)], running=[dec])
            dec._ax_pace_anchor = (clock.t - 1.0, 1)  # 1 s behind the 0.085 s/token pace, 0.5 s deficit allowed
            modes = []
            for _ in range(60):
                t = step(s)
                modes.append(t['mode'])
                advance(clock, t)
                if t['mode'] == 'prefill':
                    break
            # slack starts at -1 + 0.5; each decode round gains 0.085 - 0.036 s; prefill resumes once slack covers
            # C0 + 8192 * C1 (by default the minimum chunk is the full chunk)
            need = math.ceil((C0 + 8192 * C1 + 0.5) / (0.085 - DECODE_S))
            self.assertEqual(modes, ['decode'] * need + ['prefill'])

    def test_decode_rounds_are_bounded(self):
        clock = Clock()
        with patch.dict(os.environ, dict(PACE, SGLANG_AX_PACE_MAX_DECODE='2')):
            dec = Req('dec', 1, output=300)
            dec.output_ids = [1]
            s, _ = build(CAND, clock, waiting=[Req('cold', 30000)], running=[dec])
            dec._ax_pace_anchor = (clock.t - 100.0, 1)
            modes = [step(s)['mode'] for _ in range(3)]
            self.assertEqual(modes, ['decode', 'decode', 'prefill'])

    def test_waiting_lengths_do_not_depend_on_unbuilt_fill_ids(self):
        # real waiting requests have empty full_untruncated_fill_ids until admission (schedule_batch.py:974)
        clock = Clock()
        with patch.dict(os.environ, PACE):
            dec = Req('dec', 1, output=300)
            dec.output_ids = [1]
            cold = Req('cold', 30000)
            cold.full_untruncated_fill_ids = []
            s, _ = build(CAND, clock, waiting=[cold], running=[dec])
            # fresh decoder: slack 0.5 s < cost of a full 8192 chunk (0.57 s) -> decode, not a sliver of prefill
            self.assertEqual(step(s)['mode'], 'decode')
            hit = Req('hit', 1000, cached=60000)
            hit.full_untruncated_fill_ids = []
            s, _ = build(CAND, clock, waiting=[Req('cold', 30000), hit])
            t = step(s)
            self.assertEqual(dict((r[0], r[2] - r[1]) for r in t['reqs'])['cold'], 7168)  # reserve still seen

    def test_decoder_whose_first_token_is_unprocessed_still_counts(self):
        clock = Clock()
        with patch.dict(os.environ, PACE):
            lag = Req('lag', 1, output=300)  # in the running batch, output not yet appended (overlap)
            s, _ = build(CAND, clock, waiting=[Req('cold', 30000)], running=[lag])
            self.assertEqual(step(s)['mode'], 'decode')
            self.assertEqual(lag._ax_pace_anchor, (1000.0, 1))

    def test_guard_admits_a_full_chunk(self):
        clock = Clock()
        with patch.dict(os.environ, dict(PACE, SGLANG_AX_PACE_MAX_DECODE='1')):
            dec = Req('dec', 1, output=300)
            dec.output_ids = [1]
            s, _ = build(CAND, clock, waiting=[Req('cold', 30000)], running=[dec])
            dec._ax_pace_anchor = (clock.t - 100.0, 1)
            self.assertEqual(step(s)['mode'], 'decode')
            t = step(s)
            self.assertEqual((t['mode'], new_tokens(t)), ('prefill', 8192))

    def test_ranks_agree_on_one_clock(self):
        clock = Clock()
        calls = []

        class T:
            def __init__(self, v):
                self.v = list(v)

            def item(self):
                return self.v[0]

        def all_reduce(t, op=None, group=None):
            calls.append(op)
            t.v[0] = t.v[0] + 5.0  # another rank is 5 s ahead: the max must be used by everyone

        with patch.dict(os.environ, PACE):
            dec = Req('dec', 1, output=300)
            dec.output_ids = [1]
            s, ns = build(CAND, clock, waiting=[Req('cold', 30000)], running=[dec])
            s.ps.tp_size = 2
            s.tp_cpu_group = object()
            ns['torch'] = NS(tensor=lambda v, dtype=None: T(v), float64='f64',
                             distributed=NS(all_reduce=all_reduce, ReduceOp=NS(MAX='max')))
            step(s)
            self.assertEqual(calls, ['max'])
            self.assertEqual(dec._ax_pace_anchor[0], clock.t + 5.0)
            # no decoder -> no collective
            s2, ns2 = build(CAND, clock, waiting=[Req('cold', 30000)])
            s2.ps.tp_size = 2
            ns2['torch'] = ns['torch']
            step(s2)
            self.assertEqual(calls, ['max'])

    def test_simulated_tpot_stays_under_gate_when_cost_model_holds(self):
        base = simulate(BASE, {})
        cand = simulate(CAND, PACE)
        for r in (base, cand):
            self.assertEqual(r.done, r.n)
        worst = max(cand.tpot.values())
        self.assertLessEqual(worst, 0.10)
        p95 = lambda v: sorted(v)[int(0.95 * len(v))]
        print(f'\n  [sim, fake cost model] requests={cand.n}'
              f'\n  official A : tpot max {max(base.tpot.values()):.3f} p95 {p95(base.tpot.values()):.3f}'
              f' | ttft p95 {p95(base.ttft.values()):.2f}s max {max(base.ttft.values()):.2f}s'
              f'\n  A + 122    : tpot max {worst:.3f} p95 {p95(cand.tpot.values()):.3f}'
              f' | ttft p95 {p95(cand.ttft.values()):.2f}s max {max(cand.ttft.values()):.2f}s')

    def test_cost_model_error_degrades_gracefully(self):
        # real chunks 25% slower than the model: the pace loop sees the lost slack and backs off
        cand = simulate(CAND, PACE, scale=1.25)
        self.assertEqual(cand.done, cand.n)
        over = sum(v > 0.10 for v in cand.tpot.values())
        print(f'\n  [sim, cost x1.25] A + 122: tpot max {max(cand.tpot.values()):.3f}, over 0.10: {over}/{len(cand.tpot)}')
        self.assertLessEqual(over / len(cand.tpot), 0.05)


if __name__ == '__main__':
    unittest.main()
