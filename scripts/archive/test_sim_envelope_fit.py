#!/usr/bin/env python3
"""CPU tests of public-prior extraction, residuals, and envelope-fit CLI."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import sim_closed_loop as sim
import sim_envelope_fit as fit
from test_sim_closed_loop import engine, workload, profiles


def attempt(n=6, value=1., passed=True):
    stress = dict.fromkeys(fit.METRICS, value)
    stress.update(n=n, n_at_slo=18, passed=passed)
    return {"id": n, "authorId": "example", "scorecard": {
        "scorewheel_stress": stress, "scorewheel_raw_result": {"stress": dict(stress)}}}


def summary(n=6, value=1.):
    return {"N": n, "tpot_mean_s": value, "tpot_p95_s": value,
            "ttft_gate_detail": {g: {"p95": value} for g in (
                "fast_intra(uncached_expected<=4096,<=3s)", "overall_intra(<=5s)",
                "turn_start(<=15s)", "chain_start(<=30s)")},
            "model_pass_dev_plus_tpot": True, "failed_gates": [], "tpot_p95_le_0_10": True}


class EnvelopeTests(unittest.TestCase):
    def test_canonical_stress_once_filter_passing_n_not_n_at_slo(self):
        a, b = attempt(value=2.), attempt(value=4.)
        a["scorecard"]["scorewheel_raw_result"]["stress"]["tpot_mean"] = 999
        result = fit.public_envelope([a, b, attempt(passed=False), attempt(n=18)], (6,))
        row = result["levels"]["6"]
        self.assertEqual(row["n_passing_attempts"], 2)
        self.assertEqual(row["n_authors"], 1)
        self.assertEqual(row["median"]["tpot_mean"], 3.)
        self.assertIn("PUBLIC PRIOR INPUT", result["label"])

    def test_null_card_and_nested_fallback(self):
        a = attempt()
        a["scorecard"]["scorewheel_stress"] = None
        result = fit.public_envelope([{"scorecard": None}, a], (6,))
        self.assertEqual(result["levels"]["6"]["n_passing_attempts"], 1)

    def test_invalid_or_absent_prior_does_not_silently_fit(self):
        for value in (None, True, 0, -1, math.nan):
            a = attempt()
            a["scorecard"]["scorewheel_stress"]["tpot_mean"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit.public_envelope([a], (6,))
        for rows in ([], {}, [attempt(n=2)]):
            with self.assertRaises(ValueError):
                fit.public_envelope(rows, (6,))

    def test_real_public_counts_and_f35_medians(self):
        # all_att.json is refreshed (F40); pin the historical F35 cohort instead
        # of treating new public submissions as an extraction regression.
        saved = json.loads((sim.REPO / "evidence/T18_calibration/refined/envelope_fit.json").read_text())["prior"]
        ids = {i for level in saved["levels"].values() for i in level["attempt_ids"]}
        rows = json.loads((sim.REPO / "data/all_att.json").read_text())
        frozen = [row for row in rows if row.get("id") in ids]
        self.assertEqual(len(frozen), len(ids))
        prior = fit.public_envelope(frozen)
        self.assertEqual([prior["levels"][str(n)]["n_passing_attempts"] for n in fit.ANCHORS],
                         [44, 26, 33, 7])
        self.assertAlmostEqual(prior["levels"]["6"]["median"]["fast_intra_p95"], 2.9094258546829224)
        self.assertAlmostEqual(prior["levels"]["18"]["median"]["chain_start_p95"], 46.287394762039185)

    def test_residual_sign_log_symmetry_and_band_equality(self):
        prior = fit.public_envelope([attempt()], (6,))
        low = fit.residuals([summary(value=.5)], prior)
        high = fit.residuals([summary(value=2)], prior)
        self.assertAlmostEqual(low["full"]["log_rmse"], math.log(2))
        self.assertAlmostEqual(low["full"]["log_rmse"], high["full"]["log_rmse"])
        self.assertEqual(low["cells"][0]["residual_s"], -.5)
        self.assertTrue(high["full"]["all_in_band"])
        self.assertEqual(high["core"]["total"], 4)
        outside = fit.residuals([summary(value=2.01)], prior)
        self.assertFalse(outside["full"]["all_in_band"])

    def test_missing_fit_anchor_and_unevaluable_metric_rejected(self):
        prior = fit.public_envelope([attempt()], (6,))
        with self.assertRaises(ValueError):
            fit.residuals([summary(n=10)], prior)
        for value in (None, 0, math.inf):
            with self.assertRaises(ValueError):
                fit.residuals([summary(value=value)], prior)
        for factor in (0, 1, math.nan):
            with self.assertRaises(ValueError):
                fit.residuals([summary()], prior, factor)

    def test_ttft_first_requires_earlier_pass_and_no_cobinding_tpot(self):
        runs = [summary(n=n) for n in (6, 10, 14, 18, 22)]
        self.assertFalse(fit.ttft_first_near_ceiling(runs))
        runs[-1].update(model_pass_dev_plus_tpot=False, failed_gates=["fast_intra(<=3s)"])
        self.assertTrue(fit.ttft_first_near_ceiling(runs))
        bad = copy.deepcopy(runs)
        bad[-1]["failed_gates"].append("tpot_p95<=0.10")
        self.assertFalse(fit.ttft_first_near_ceiling(bad))
        bad = copy.deepcopy(runs)
        bad[0].update(model_pass_dev_plus_tpot=False, failed_gates=["chain_start(<=30s)"])
        self.assertFalse(fit.ttft_first_near_ceiling(bad))
        runs[-1]["tpot_p95_le_0_10"] = False
        self.assertFalse(fit.ttft_first_near_ceiling(runs))

    def test_candidates_are_deterministic_and_supplied_values_preserved(self):
        base = sim.Engine(decode_speedup=2)
        supplied = [{"prefill_tps": 23456, "chunk_tokens": 2048}]
        a = fit.candidate_engines(base, 5, 71, supplied)
        self.assertEqual(a, fit.candidate_engines(base, 5, 71, supplied))
        self.assertEqual(a[0].prefill_tps, 23456)
        self.assertTrue(all(e.decode_speedup == 2 for e in a))
        for count, extra in ((-1, []), (0, []), (1, {}), (0, [{"chunk_tokens": 17}]),
                             (0, [{"unknown_parameter": 1}]), (0, [None])):
            with self.assertRaises(ValueError):
                fit.candidate_engines(base, count, 71, extra)

    def test_decode_speedup_amortizes_compute_only_including_curve(self):
        e = engine(decode_ms=40, forward_overhead_ms=2, decode_speedup=2)
        self.assertAlmostEqual(e.decode_seconds(1), .022)
        self.assertAlmostEqual(replace(e, decode_curve=((1, 40), (3, 80))).decode_seconds(2), .032)
        self.assertEqual(e.prefill_seconds(100, 2000), replace(e, decode_speedup=1).prefill_seconds(100, 2000))
        for value in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                replace(e, decode_speedup=value)

    def test_decode_base_speedup_degeneracy_preserves_whole_replay(self):
        w = workload([[(200, 12, 0), (17, 5, 30)], [(50, 8, 10)]])
        a = engine(decode_ms=20, decode_speedup=1, decode_batch_slope=.07)
        b = replace(a, decode_ms=60, decode_speedup=3)
        x = sim.simulate(w, profiles(w), 2, a)["requests"]
        y = sim.simulate(w, profiles(w), 2, b)["requests"]
        for left, right in zip(x, y):
            for key in ("sim_dispatch_s", "ttft_s", "tpot_s", "sim_finish_s"):
                self.assertAlmostEqual(left[key], right[key])

    def test_fit_cli_actual_formal_mix_and_readonly_input_hashes(self):
        sources = list(sim.HARNESS.glob("*.py")) + list(sim.HARNESS.rglob("*.pyc"))
        before = {str(p): sim.sha256(p) for p in sources}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidates = root / "candidates.json"
            candidates.write_text('[{"prefill_tps": 25000, "decode_ms": 5}]')
            cmd = [sys.executable, "-B", str(sim.REPO / "scripts/sim_closed_loop.py"),
                   "--envelope-fit", str(sim.REPO / "data/all_att.json"), "--mix", "formal-mix",
                   "--fit-samples", "0", "--fit-candidates", str(candidates),
                   "--fit-compare-top", "0", "--out-dir", str(root / "output")]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((root / "output/envelope_fit.json").read_text())
            self.assertIn("MODEL OUTPUT", data["label"])
            self.assertEqual(data["workload"]["n_requests"], 9896)
            self.assertEqual(data["counts"]["candidates"], 1)
            self.assertEqual(data["candidates"][0]["fit"]["full"]["total"], 24)
            self.assertEqual(len(data["candidates"][0]["runs"]), 5)
            result = subprocess.run(cmd + ["--fit-scheduler", "spf"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((root / "output/envelope_fit.json").read_text())
            self.assertEqual(data["fit_scheduler"], "spf")
            self.assertEqual(data["candidates"][0]["engine"]["scheduler"], "spf")
            result = subprocess.run(cmd + ["--levels", "6,10"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("must include", result.stderr)
            result = subprocess.run(cmd + ["--mix", "dev"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("requires --mix formal-mix", result.stderr)
        self.assertEqual(before, {str(p): sim.sha256(p) for p in sources})


if __name__ == "__main__":
    unittest.main(verbosity=2)
