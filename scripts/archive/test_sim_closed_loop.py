#!/usr/bin/env python3
"""CPU-only regression tests for replay causality, forward scheduling and gates.

Run: python3 -B -m unittest discover -s scripts -p test_sim_closed_loop.py -v
No GPU, model, network, external service or writes to s1-dev/.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import sim_closed_loop as sim


def workload(sequences):
    """Each item is (uncached, output, gap_ms[, phase[, frozen_expected]])."""
    rows, chains = {}, []
    for c, seq in enumerate(sequences):
        ids = []
        for i, args in enumerate(seq):
            uncached, output, gap = args[:3]
            phase = args[3] if len(args) > 3 else "intra"
            expected = args[4] if len(args) > 4 else uncached
            rid = f"request-{c}-{i}"
            rows[rid] = dict(_req_id=rid, source_req_id=rid, chain_id=f"chain-{c}",
                _idx_in_chain=i, phase=phase, glm_tokens=max(10000, uncached, expected),
                uncached_expected=expected, max_output_i=output, replay_gap_ms=gap,
                pack="biomaster", view="canon", logical_call_id=rid,
                sampling_weight=1.0, source_completion_tokens=None)
            rows[rid]["test_uncached"] = uncached
            ids.append(rid)
        chains.append({"chain_id": f"chain-{c}", "req_ids": ids})
    gaps, stats = sim.build_gap_plan(chains, rows, 3600000)
    return sim.Workload(rows, chains, gaps, dict(mix="dev", label=sim.LABEL,
        gap_stats=stats, formal_reference={"chain_start": {"n": 808}, "intra": {"n": 9023},
                                         "turn_start": {"n": 65}}))


def profiles(w):
    return {rid: {"stock": r["test_uncached"], "role_conservative": r["test_uncached"],
                  "output_tokens": r["max_output_i"]} for rid, r in w.rows.items()}


def engine(**changes):
    defaults = dict(prefill_tps=100, chunk_tokens=100, page_tokens=1,
                    decode_ms=100, decode_batch_slope=0, forward_overhead_ms=0)
    return sim.Engine(**dict(defaults, **changes))


def run(w, n=1, e=None, **kwargs):
    return sim.simulate(w, profiles(w), n, e or engine(), keep_trace=True, **kwargs)


class ReplayTests(unittest.TestCase):
    def test_gap_rounding_exact_last_nonzero_remainder(self):
        w = workload([[(1, 1, 4), (1, 1, 4), (1, 1, 4), (1, 1, 0)]])
        gaps, stats = sim.build_gap_plan(w.chains, w.rows, 10)
        self.assertEqual(gaps["chain-0"], [3, 3, 4, 0])
        self.assertEqual(stats["n_chains_compressed"], 1)
        self.assertEqual(stats["effective_gap_sum_ms"], 10)

    def test_gap_below_cap_and_disabled_are_identical(self):
        w = workload([[(1, 1, 1), (1, 1, 7), (1, 1, 0)]])
        for cap in (0, 8, 10, 3600000):
            plan, _ = sim.build_gap_plan(w.chains, w.rows, cap)
            self.assertEqual(plan["chain-0"], [1, 7, 0])

    def test_serial_requests_and_chain_queue_use_finish_plus_gap(self):
        w = workload([[(100, 3, 500), (50, 2, 300)], [(100, 1, 200)]])
        result = run(w)
        a, b, c = result["requests"]
        self.assertAlmostEqual(a["sim_dispatch_s"], .5)
        self.assertAlmostEqual(a["sim_first_token_s"], 1.5)
        self.assertAlmostEqual(a["sim_finish_s"], 1.7)
        self.assertAlmostEqual(b["sim_dispatch_s"], 2.0)
        self.assertAlmostEqual(b["sim_finish_s"], 2.6)
        self.assertAlmostEqual(c["sim_dispatch_s"], 2.8)
        self.assertAlmostEqual(result["summary"]["wall_s"], 3.8)
        self.assertAlmostEqual(a["tpot_s"], .1)

    def test_workers_claim_chains_when_free_not_round_robin(self):
        w = workload([[(10, 20, 0)], [(10, 1, 0)], [(10, 1, 0)]])
        rs = {r["req_id"]: r for r in run(w, 2)["requests"]}
        self.assertEqual(rs["request-2-0"]["worker"], 1)
        self.assertAlmostEqual(rs["request-2-0"]["sim_dispatch_s"], rs["request-1-0"]["sim_finish_s"])

    def test_frontend_delay_is_in_ttft_and_feedback(self):
        w = workload([[(10, 1, 0), (10, 1, 0)]])
        a, b = run(w, e=engine(frontend_ms=200))["requests"]
        self.assertAlmostEqual(a["ttft_s"], .3)
        self.assertAlmostEqual(a["queue_time_s"], 0)
        self.assertAlmostEqual(b["sim_dispatch_s"], .3)
        self.assertAlmostEqual(b["sim_finish_s"], .6)

    def test_single_token_and_zero_uncached_still_need_forward(self):
        w = workload([[(0, 1, 0)]])
        result = run(w)
        rec = result["requests"][0]
        self.assertAlmostEqual(rec["ttft_s"], .01)
        self.assertIsNone(rec["tpot_s"])
        self.assertEqual(result["summary"]["engine_stats"]["prefill_work_tokens"], 1)
        self.assertEqual(rec["uncached_actual_model"], 0)

    def test_decode_batch_changes_when_short_request_finishes(self):
        w = workload([[(10, 3, 0)], [(10, 2, 0)]])
        rs = {r["req_id"]: r for r in run(w, 2, engine(decode_batch_slope=1))["requests"]}
        self.assertAlmostEqual(rs["request-1-0"]["tpot_s"], .2)
        self.assertAlmostEqual(rs["request-0-0"]["tpot_s"], .15)
        self.assertAlmostEqual(rs["request-0-0"]["sim_finish_s"], .5)

    def test_prefill_stalls_existing_decode_and_is_in_tpot(self):
        w = workload([[(10, 3, 0)], [(100, 1, 150)]])
        rs = {r["req_id"]: r for r in run(w, 2)["requests"]}
        self.assertAlmostEqual(rs["request-1-0"]["sim_exec_start_s"], .2)
        self.assertAlmostEqual(rs["request-0-0"]["tpot_s"], .6)

    def test_fixed_overhead_charged_once_per_shared_forward(self):
        w = workload([[(10, 2, 0)], [(10, 2, 0)]])
        rs = run(w, 2, engine(forward_overhead_ms=10))["requests"]
        for rec in rs:
            self.assertAlmostEqual(rec["ttft_s"], .21)
            self.assertAlmostEqual(rec["tpot_s"], .11)

    def test_prefill_length_penalty(self):
        e = engine(prefill_length_alpha=1, prefill_length_reference=1000)
        self.assertAlmostEqual(e.prefill_seconds(100, 2000), 2)
        self.assertAlmostEqual(e.prefill_seconds(100, 500), 1)

    def test_decode_curve_interpolates_and_clamps(self):
        e = engine(decode_curve=((1, 10), (5, 30)), forward_overhead_ms=2)
        self.assertAlmostEqual(e.decode_seconds(1), .012)
        self.assertAlmostEqual(e.decode_seconds(3), .022)
        self.assertAlmostEqual(e.decode_seconds(9), .032)

    def test_spf_short_waiter_bypasses_long_continuation(self):
        w = workload([[(300, 1, 0)], [(10, 1, 500)]])
        fcfs = {r["req_id"]: r for r in run(w, 2)["requests"]}
        spf = run(w, 2, engine(scheduler="spf"))
        short = next(r for r in spf["requests"] if r["req_id"] == "request-1-0")
        self.assertLess(short["ttft_s"], fcfs["request-1-0"]["ttft_s"])
        second = spf["trace"][1]
        self.assertEqual(second["tokens"], [10, 90])
        self.assertEqual(second["partial_after"], "request-0-0")

    def test_prefill_decode_interval_allows_decode_between_chunks(self):
        w = workload([[(10, 8, 0)], [(300, 1, 150)]])
        base = run(w, 2)
        delayed = run(w, 2, engine(prefill_decode_interval=1))
        pf = [i for i, x in enumerate(delayed["trace"]) if x["kind"] == "prefill"]
        self.assertTrue(all(delayed["trace"][i+1]["kind"] == "decode" for i in pf[1:-1]))
        long_base = next(r for r in base["requests"] if r["req_id"] == "request-1-0")
        long_delayed = next(r for r in delayed["requests"] if r["req_id"] == "request-1-0")
        self.assertGreater(long_delayed["ttft_s"], long_base["ttft_s"])

    def test_engine_running_cap_is_separate_from_N(self):
        w = workload([[(10, 3, 0)], [(10, 3, 0)], [(10, 3, 0)]])
        result = run(w, 3, engine(max_running=1))
        self.assertEqual(result["summary"]["engine_stats"]["peak_running"], 1)
        ordered = sorted(result["requests"], key=lambda r: r["sim_exec_start_s"])
        for a, b in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(b["sim_exec_start_s"], a["sim_finish_s"])

    def test_chunk_budget_conservation_and_closed_loop_concurrency(self):
        w = workload([[(281 + j, 7, 15), (43 + j, 4, 10)] for j in range(5)])
        for scheduler in ("fcfs", "spf"):
            result = run(w, 3, engine(scheduler=scheduler))
            prefill = [t for t in result["trace"] if t["kind"] == "prefill"]
            self.assertEqual(sum(sum(t["tokens"]) for t in prefill), sum(r["stock"] for r in profiles(w).values()))
            self.assertTrue(all(sum(t["tokens"]) <= 100 for t in prefill))
            self.assertLessEqual(result["summary"]["engine_stats"]["peak_running"], 3)
            rs = result["requests"]
            for worker in range(3):
                seq = sorted((r for r in rs if r["worker"] == worker), key=lambda r: r["sim_dispatch_s"])
                for a, b in zip(seq, seq[1:]):
                    self.assertGreaterEqual(b["sim_dispatch_s"], a["sim_finish_s"])

    def test_decode_coalescing_matches_literal_forward_events(self):
        rng = random.Random(31)
        w = workload([[(rng.randrange(1, 500), rng.randrange(2, 100), rng.randrange(0, 800))
                        for _ in range(3)] for _ in range(8)])
        for scheduler in ("fcfs", "spf"):
            for interval in (0, 2):
                e = engine(scheduler=scheduler, prefill_decode_interval=interval, decode_batch_slope=.07)
                a = run(w, 4, e)
                b = run(w, 4, e, coalesce_decode=False)
                self.assertAlmostEqual(a["summary"]["wall_s"], b["summary"]["wall_s"], places=7)
                byid = {r["req_id"]: r for r in b["requests"]}
                for r in a["requests"]:
                    for key in ("ttft_s", "tpot_s", "sim_dispatch_s", "sim_finish_s"):
                        self.assertAlmostEqual(r[key], byid[r["req_id"]][key], places=7)


class CacheAndScoringTests(unittest.TestCase):
    def test_frozen_fast_gate_does_not_follow_actual_cache_miss(self):
        w = workload([[(1, 2, 0), (9000, 2, 0, "intra", 4096), (10, 2, 0, "intra", 4097)]])
        rs = run(w)["requests"]
        self.assertTrue(sim.in_ttft_gate(rs[1], "fast_intra"))
        self.assertFalse(sim.in_ttft_gate(rs[2], "fast_intra"))
        self.assertTrue(sim.in_ttft_gate(rs[0], "chain_start"))

    def test_phase_gate_first_request_wins_over_turn_start(self):
        for field in ("idx_in_chain", "_idx_in_chain"):
            self.assertEqual(sim.phase_gate({field: 0, "phase": "turn_start"}), 30)
        self.assertEqual(sim.phase_gate({"idx_in_chain": 5, "phase": "context_reset"}), 30)
        self.assertEqual(sim.phase_gate({"idx_in_chain": 1, "phase": "turn_start"}), 15)

    def test_f24_frozen_proxy_and_clipping(self):
        w = workload([[(1, 2, 0), (3144, 2, 0), (14033, 2, 0)]])
        p, _ = sim.prepare_profiles(w)
        self.assertEqual(p["request-0-1"]["stock"], 7238)
        self.assertEqual(p["request-0-1"]["role_conservative"], 3144)
        self.assertEqual(p["request-0-2"]["stock"], 14033)  # clip to prompt, not 15744
        w.rows["request-0-2"]["glm_tokens"] = 20000
        p, _ = sim.prepare_profiles(w)
        self.assertEqual(p["request-0-2"]["stock"], 15744)

    def test_supplied_cache_counts_used_exactly_missing_and_bad_are_errors(self):
        w = workload([[(10, 2, 0)]])
        supplied = {"request-0-0": {"stock": 77, "role_conservative": 23}}
        p, _ = sim.prepare_profiles(w, supplied)
        self.assertEqual(p["request-0-0"]["stock"], 77)
        self.assertEqual(p["request-0-0"]["role_conservative"], 23)
        for bad in ({}, {"request-0-0": {"stock": 10001, "role_conservative": 2}},
                    {"request-0-0": {"stock": -1, "role_conservative": 2}}):
            with self.assertRaises(ValueError):
                sim.prepare_profiles(w, bad)

    def test_output_budget_source_and_measured_profiles(self):
        w = workload([[(10, 8, 0), (10, 5, 0)]])
        w.rows["request-0-0"]["source_completion_tokens"] = 4
        p, meta = sim.prepare_profiles(w, output_mode="source")
        self.assertEqual(p["request-0-0"]["output_tokens"], 4)
        self.assertEqual(p["request-0-1"]["output_tokens"], 5)
        self.assertEqual(meta["source_output_budget_fallbacks"], 1)
        measured = {rid: {"output_tokens": 2} for rid in w.rows}
        p, _ = sim.prepare_profiles(w, output_profiles=measured)
        self.assertTrue(all(r["output_tokens"] == 2 for r in p.values()))
        measured["request-0-0"]["output_tokens"] = 9
        with self.assertRaises(ValueError):
            sim.prepare_profiles(w, output_profiles=measured)

    def test_dev_gates_reused_exactly_and_tpot_is_separate(self):
        w = workload([[(1, 3, 0), (1, 3, 0), (1, 3, 0, "turn_start")]])
        result = run(w, e=engine(decode_ms=200))
        raw, s = result["requests"], result["summary"]
        official = sim.evaluate(raw, s["wall_s"], {"lane": "dev"})
        self.assertEqual(s["dev_gates"], official["gates"])
        self.assertTrue(s["dev_all_pass"])
        self.assertFalse(s["model_pass_dev_plus_tpot"])
        self.assertAlmostEqual(s["tpot_p95_s"], .2)

    def test_order_statistic_threshold_equality_and_missing_samples(self):
        w = workload([[(1, 2, 0), (1, 2, 0), (1, 2, 0, "turn_start")]])
        raw = run(w)["requests"]
        raw[0]["ttft_s"], raw[1]["ttft_s"], raw[2]["ttft_s"] = 30., 3., 15.
        report = sim.score_records(raw, 100)
        self.assertTrue(report["dev_all_pass"])
        raw[1]["ttft_s"] = math.nextafter(3., math.inf)
        self.assertFalse(sim.score_records(raw, 100)["dev_all_pass"])
        missing = sim.score_records(raw[:1], 100)
        self.assertFalse(missing["dev_gates"]["gated_phases_have_samples"])
        self.assertEqual(sim.q(list(range(20)), .95), 19)
        self.assertEqual(sim.q(list(range(21)), .95), 19)

    def test_ladder_reports_nonmonotonicity_and_first_binding_gate(self):
        runs = [{"N": n, "model_pass_dev_plus_tpot": ok, "failed_gates": [] if ok else ["tpot_p95<=0.10"]}
                for n, ok in ((2, True), (6, False), (10, True))]
        ladder = sim.ladder_summary(runs)
        self.assertEqual(ladder["max_passing_tested_N"], 10)
        self.assertEqual(ladder["contiguous_pass_through_N"], 2)
        self.assertEqual(ladder["first_failing_N"], 6)
        self.assertTrue(ladder["pass_after_fail"])
        self.assertEqual(ladder["first_binding_gates"], ["tpot_p95<=0.10"])

    def test_formal_mix_exact_counts_determinism_and_profile_identity(self):
        w = workload([[(10, 2, 0), (3, 2, 20), (40, 2, 10, "turn_start")]])
        a = sim.formal_mix(w)
        b = sim.formal_mix(w)
        self.assertEqual(a.chains, b.chains)
        self.assertEqual(len(a.rows), 9896)
        self.assertEqual(len(a.chains), 808)
        self.assertEqual(sim.composition(a.rows.values()), {"chain_start": 808, "intra": 9023, "turn_start": 65})
        self.assertIn("WHAT-IF ONLY", a.metadata["label"])
        self.assertTrue(all(r["source_req_id"] in w.rows for r in a.rows.values()))
        for ch in a.chains:
            self.assertEqual(a.rows[ch["req_ids"][0]]["_idx_in_chain"], 0)
        p, _ = sim.prepare_profiles(a, profiles(w))
        self.assertEqual(len(p), 9896)

    def test_small_formal_mix_replays_with_whatif_labels(self):
        w = workload([[(10, 2, 0), (3, 2, 20), (40, 2, 10, "turn_start")]])
        mixed = sim.formal_mix(w, counts={"chain_start": 3, "intra": 20, "turn_start": 2})
        p, _ = sim.prepare_profiles(mixed)
        result = sim.simulate(mixed, p, 2, engine(), "stock")
        self.assertEqual(len(result["requests"]), 25)
        self.assertTrue(all("WHAT-IF ONLY" in r["label"] for r in result["requests"]))


class InputIntegrationTests(unittest.TestCase):
    def test_real_cohort_order_gate_counts_and_cap(self):
        cohort_path = sim.HARNESS / "g0a/samples_v3/cohort_dev-combined-v1.json"
        w = sim.load_workload(sim.REPO / "s1-dev/data/dev-combined-v1", cohort_path)
        cohort = json.loads(cohort_path.read_text())
        self.assertEqual([(ch["chain_id"], ch["req_ids"]) for ch in w.chains],
                         [(ch["chain_id"], ch["req_ids"]) for ch in cohort["chains"]])
        self.assertEqual(w.metadata["composition"], {"chain_start": 314, "intra": 388, "turn_start": 20})
        self.assertEqual(w.metadata["n_requests"], 722)
        self.assertEqual(w.metadata["gap_stats"]["chain_gap_cap_ms"], 3600000)
        self.assertEqual(sum(sim.in_ttft_gate(r, "fast_intra") for r in w.rows.values()), 328)

    def test_invalid_engine_parameters(self):
        bad = ({"prefill_tps": 0}, {"decode_ms": -1}, {"chunk_tokens": 63},
               {"max_running": -1}, {"prefill_decode_interval": 1.2},
               {"forward_overhead_ms": float("nan")}, {"decode_curve": ((3, 10), (2, 20))})
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sim.Engine(**kwargs)

    def test_cli_writes_raw_and_sweep_without_harness_mutation(self):
        sources = sorted(p for p in sim.HARNESS.glob("*.py"))
        before = {str(p): sim.sha256(p) for p in sources}
        bytecode = {str(p): sim.sha256(p) for p in sim.HARNESS.rglob("*.pyc")}
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, "-B", str(sim.REPO / "scripts/sim_closed_loop.py"),
                "--levels", "2", "--policies", "stock", "--out-dir", temp],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((Path(temp) / "sweep.json").read_text())
            self.assertIn("MODEL OUTPUT", report["label"])
            raw_file = next(Path(temp).glob("*.requests.jsonl"))
            raw = [json.loads(line) for line in raw_file.read_text().splitlines()]
            self.assertEqual(len(raw), 722)
            self.assertTrue(all(r["ttft_source"] == "model" for r in raw))
        self.assertEqual(before, {str(p): sim.sha256(p) for p in sources})
        self.assertEqual(bytecode, {str(p): sim.sha256(p) for p in sim.HARNESS.rglob("*.pyc")})

    def test_cli_rejects_readonly_output_directory(self):
        result = subprocess.run([sys.executable, "-B", str(sim.REPO / "scripts/sim_closed_loop.py"),
            "--out-dir", str(sim.HARNESS / "forbidden-test")], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("read-only", result.stderr)
        self.assertFalse((sim.HARNESS / "forbidden-test").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
