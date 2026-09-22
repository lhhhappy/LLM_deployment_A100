#!/usr/bin/env python3
"""D2 simulator contracts, including differential admission vs production code."""
from dataclasses import replace
import random
import unittest

import sim_closed_loop as sim
from test_sim_closed_loop import workload, run, engine
from test_spf_scheduling import adder, load_source, policy, request


def req(rid, work, order=0, matched=0, arrived_tokens=0):
    return sim.Request(rid, 0, 0, 0, 0, 0, order, max(1, work), 0, work, 1,
                       matched=matched, arrival_processed_tokens=arrived_tokens)


class UpstreamSimulatorTests(unittest.TestCase):
    def test_reserves_before_slot_admission(self):
        e = engine(chunk_tokens=4096, page_tokens=64, scheduler='spf-upstream')
        c, s = req('c', 16384), req('s', 512)
        batch, partial = sim.select_prefill([s], c, 0, e, 1)
        self.assertEqual([(r.rid, n) for r, n in batch], [('c', 3584)])
        legacy, _ = sim.select_prefill([req('s', 512)], req('c', 16384), 0, replace(e, scheduler='spf'), 1)
        self.assertEqual([(r.rid, n) for r, n in legacy], [('c', 4096)])
        self.assertIs(partial, c)

    def test_production_differential_random_rounds(self):
        ns = load_source()
        p = policy(ns)
        rng = random.Random(40024)
        for _ in range(500):
            page = rng.choice((1, 64, 256))
            budget = page * rng.randint(1, 32)
            size = rng.randint(1, budget * 3) if rng.choice((False, True)) else 0
            lengths = [rng.randint(1, budget * 2) for _ in range(rng.randint(0, 8))]
            slots = rng.randint(1, 8)
            decode = rng.randint(0, slots - bool(size))
            a = adder(ns, budget, page)
            waiting = [request(str(i), n, arrived=i) for i, n in enumerate(lengths)]
            p._sort_by_shortest_prefill(waiting, set())
            active = None
            if size:
                active = request('continuation', size)
                a.chunked_req_limit = p.shortest_prefill_chunk_limit(active, waiting, budget, page)
                active = a.add_chunked_req(active)
            for r in waiting:
                if decode + len(a.can_run_list) >= slots:
                    break
                verdict = a.add_one_req(r, active is not None, None)
                if verdict.name != 'CONTINUE':
                    break
            expected = [(r.rid, r.extend_range.length) for r in a.can_run_list]
            simulated, partial = sim.select_prefill(
                [req(str(i), n, i) for i, n in enumerate(lengths)],
                req('continuation', size) if size else None, decode,
                engine(chunk_tokens=budget, page_tokens=page, scheduler='spf-upstream'), slots)
            self.assertEqual([(r.rid, n) for r, n in simulated], expected)
            self.assertEqual(getattr(partial, 'rid', None), getattr(active or a.new_chunked_req, 'rid', None))

    def test_hrrn_uses_token_age_rid_tie_and_does_not_reserve(self):
        e = engine(scheduler='hrrn')
        q = [req('new-short', 5, arrived_tokens=100), req('old-long', 50, arrived_tokens=0)]
        batch, _ = sim.select_prefill(q, None, 0, e, 1, 100)
        self.assertEqual(batch[0][0].rid, 'old-long')
        q = [req('z', 1), req('a', 50)]
        batch, _ = sim.select_prefill(q, None, 0, e, 1, 0)
        self.assertEqual(batch[0][0].rid, 'a')
        q = [req('s', 1)]
        batch, _ = sim.select_prefill(q, req('c', 300), 0, e, 2, 1000)
        self.assertEqual([(r.rid, n) for r, n in batch], [('c', 100)])

    def test_hrrn_lpm_zero_work_fallback_and_stable_ties(self):
        q = [req('old', 50), req('zero', 0)]
        batch, _ = sim.select_prefill(q, None, 0, engine(scheduler='hrrn'), 1, 10)
        self.assertEqual(batch[0][0].rid, 'zero')
        q = [req('first', 50, matched=500), req('short', 1, matched=50)]
        batch, _ = sim.select_prefill(q, None, 0, engine(scheduler='lpm'), 1)
        self.assertEqual(batch[0][0].rid, 'first')
        for name in ('hrrn', 'lpm'):
            q = [req(str(i), 50, i, matched=i*100) for i in range(129)]
            batch, _ = sim.select_prefill(q, None, 0, engine(scheduler=name), 1)
            self.assertEqual(batch[0][0].rid, '0')

    def test_sorting_matches_stock_hrrn_lpm_functions(self):
        ns = load_source()
        rng = random.Random(20)
        for name in ('hrrn', 'lpm'):
            p = policy(ns, name)
            for _ in range(50):
                rs = [request(str(i), rng.randrange(0, 100), cached=rng.randrange(0, 300)) for i in range(10)]
                for r in rs:
                    r.arrival_processed_tokens = rng.randrange(100)
                sq = [req(r.rid, len(r.origin_input_ids)-r.num_matched_prefix_tokens,
                          i, r.num_matched_prefix_tokens, r.arrival_processed_tokens) for i, r in enumerate(rs)]
                if name == 'hrrn':
                    p._sort_by_hrrn(rs, set(), 100)
                else:
                    p._sort_by_longest_prefix(rs, set())
                batch, _ = sim.select_prefill(sq, None, 0, engine(scheduler=name), 1, 100)
                self.assertEqual(batch[0][0].rid, rs[0].rid)

    def test_decode_coalescing_and_single_partial(self):
        rng = random.Random(31)
        w = workload([[(rng.randrange(1, 500), rng.randrange(2, 100), rng.randrange(0, 800))
                       for _ in range(3)] for _ in range(8)])
        for scheduler in ('spf-upstream', 'hrrn', 'lpm'):
            for interval in (0, 2):
                e = engine(scheduler=scheduler, prefill_decode_interval=interval)
                a, b = run(w, 4, e), run(w, 4, e, coalesce_decode=False)
                self.assertLessEqual(a['summary']['engine_stats']['peak_running'], 4)
                self.assertAlmostEqual(a['summary']['wall_s'], b['summary']['wall_s'], places=7)
                byid = {r['req_id']: r for r in b['requests']}
                for r in a['requests']:
                    self.assertAlmostEqual(r['ttft_s'], byid[r['req_id']]['ttft_s'], places=7)


if __name__ == '__main__':
    unittest.main()
