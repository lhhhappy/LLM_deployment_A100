#!/usr/bin/env python3
"""Meaningful CPU regressions for T16; HTTP stubs, no engine/GPU/Trisol."""
from contextlib import contextmanager, redirect_stdout
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import make_case_sets as cases
import plan_8gpu_session as planner
import preflight_8gpu as preflight
import replay_chains as replay
import score_formal as score
from test_ladder_search import record, records, run_metadata


class WeightTests(unittest.TestCase):
    def test_equal_weight_quantile_matches_dev(self):
        scorer = score.load_harness()
        for n in (1, 2, 19, 20, 40, 100):
            for weight in (.1, 2.5):
                rows = [{"value": i, "sampling_weight": weight} for i in range(n)]
                self.assertEqual(score.weighted_distribution(rows, "value")["p95"],
                                 scorer.q(list(range(n)), .95))

    def test_fractional_weights_change_diagnostic_not_gates(self):
        rows = records()+[record(index=5, ttft=9, sampling_weight=.01)]
        rows[2]["ttft_s"] = .001  # At least one within the official budget.
        a = score.score_records(rows, run_metadata())
        rows[-1]["sampling_weight"] = 100
        b = score.score_records(rows, run_metadata())
        self.assertEqual(a["dev"]["gates"], b["dev"]["gates"])
        self.assertEqual(a["estimated"], b["estimated"])
        g = next(k for k in a["ttft_estimated"] if k.startswith("fast_intra"))
        self.assertLess(a["ttft_estimated"][g]["weighted"]["p95"], b["ttft_estimated"][g]["weighted"]["p95"])
        self.assertEqual(a["ttft_estimated"][g]["rate_ci_lower"], b["ttft_estimated"][g]["rate_ci_lower"])
        self.assertNotEqual(a["dev"]["budget_attainment"]["weighted"], b["dev"]["budget_attainment"]["weighted"])

    def test_limit_equality_zero_and_missing_weights(self):
        rows = [{"x": 3, "sampling_weight": .3}, {"x": 4, "sampling_weight": .1},
                {"x": 100, "sampling_weight": 0}, {"x": 200}]
        d = score.weighted_distribution(rows, "x", 3)
        self.assertAlmostEqual(d["exceed_rate"], .25)
        self.assertEqual(d["p95"], 4)
        self.assertEqual(d["n_missing_weight"], 1)
        self.assertIsNone(score.weighted_distribution([{"x": 1}], "x")["p95"])

    def test_index_join_overrides_raw_without_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"requests.jsonl"
            path.write_text(json.dumps({"pack": "p", "view": "v", "logical_call_id": "c", "sampling_weight": 7})+"\n")
            original = [{"req_id": "p:v:c", "sampling_weight": 1}, {"req_id": "other"}]
            joined, info = score.attach_weights(original, path)
            self.assertEqual(joined[0]["sampling_weight"], 7)
            self.assertEqual(info, {"from_requests": 1, "from_raw": 0, "missing": 1, "raw_conflicts": 1})
            self.assertEqual(original[0]["sampling_weight"], 1)

    def test_invalid_weights_rejected(self):
        for weight in (-1, float("nan"), float("inf"), True, "2"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                score.attach_weights([{"sampling_weight": weight}])

    def test_formal_budget_authority_matches_public_scorer(self):
        rows = records()
        rows[0]["sampling_weight"] = 100
        harness = score.load_harness()
        formal = harness.evaluate(rows, 20, {"lane": "formal"})
        report = score.score_records(rows, run_metadata(), harness)
        self.assertEqual(report["dev"]["budget_attainment"]["weighted"], formal["budget_attainment"]["weighted"])
        self.assertEqual(formal["budget_attainment"]["authoritative_for_lane"], "weighted")
        self.assertFalse(formal["budget_attainment"]["is_slo_gate"])
        self.assertEqual(report["dev"]["gates"], formal["gates"])


class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.metadata, cls.groups = cases.load_data()

    def test_all_shipped_cases_are_real_contiguous_prefixes(self):
        for name in ("formal_like", "reminder_heavy", "strict_append", "cold_heavy", "smoke"):
            root = cases.REPO/"cases"
            case = cases.validate_case(json.loads((root/(name+".json")).read_text()), self.groups)
            ids = [rid for chain in case for rid in chain["req_ids"]]
            self.assertEqual(len(ids), len(set(ids)))
            prediction = [json.loads(line) for line in (root/(name+".predictions.jsonl")).read_text().splitlines()]
            self.assertEqual(ids, [r["req_id"] for r in prediction])
            for row in prediction:
                for policy in ("stock", "role_conservative"):
                    self.assertEqual(row["policy_details"][policy]["cached_tokens"]+row[policy], row["glm_tokens"])
            manifest = json.loads((root/(name+".manifest.json")).read_text())
            self.assertEqual(manifest["composition"]["n_requests"], len(ids))
            self.assertTrue(all(c["reason"] for c in manifest["chains"]))

    def test_case_purposes_and_honest_formal_limit(self):
        def manifest(name):
            return json.loads((cases.REPO/"cases"/(name+".manifest.json")).read_text())
        formal = manifest("formal_like")
        self.assertFalse(formal["availability_limits"]["formal_mix_attainable"])
        self.assertLess(formal["availability_limits"]["max_intra_share_any_available_prefix"],
                        formal["formal_target_gate_shares"]["intra"])
        self.assertGreater(formal["composition"]["gates"]["intra"]["share"], .80)
        self.assertGreater(formal["composition"]["gates"]["turn_start"]["n"], 0)
        for name, field in (("reminder_heavy", "reminder_intra_edges"), ("strict_append", "strict_intra_edges")):
            for c in manifest(name)["chains"]:
                self.assertGreaterEqual(c[field]/(c["selected_requests"]-1), .8)
        self.assertGreater(manifest("cold_heavy")["composition"]["gates"]["chain_start"]["share"], .5)
        self.assertEqual(len(manifest("smoke")["chains"]), 3)

    def test_bad_prefixes_and_duplicates_rejected(self):
        cid, rows = next((c, r) for c, r in self.groups.items() if len(r) >= 3)
        for ids in ([rows[1]["_req_id"]], [rows[0]["_req_id"], rows[2]["_req_id"]], []):
            with self.assertRaises(ValueError):
                cases.validate_case([{"chain_id": cid, "req_ids": ids}], self.groups)
        good = {"chain_id": cid, "req_ids": [rows[0]["_req_id"]]}
        with self.assertRaises(ValueError):
            cases.validate_case([good, good], self.groups)

    def test_frozen_strict_append_keeps_end_and_no_future_oracle(self):
        rows = [{"glm_tokens": 200, "glm_lcp_with_prev": 0},
                {"glm_tokens": 260, "glm_lcp_with_prev": 200}]
        for policy in ("stock", "role_conservative"):
            got = cases.predict_frozen(rows, policy, tail=64)
            self.assertEqual(got[1]["cached_tokens"], 192)
            changed = [rows[0], {"glm_tokens": 900, "glm_lcp_with_prev": 128}]
            self.assertEqual(got[0], cases.predict_frozen(changed, policy, tail=64)[0])

    def test_case_replay_honors_order_prefix_and_all_chains(self):
        # Avoid renderer/dependencies: exercise real main selection with CPU fakes.
        import s1_common, s1_loadgen
        groups = {f"c{i}": [{"_req_id": f"r{i}-{j}", "phase": "intra"} for j in range(3)] for i in range(4)}
        case = [{"chain_id": f"c{i}", "req_ids": [f"r{i}-0"], "prefix_len": 1} for i in (3, 1, 2, 0)]
        class Tokenizer:
            def convert_tokens_to_ids(self, _): return [9, 8]
            def encode(self, text, **_): return [1]*128+[9]+[2]*80
        class Renderer:
            tokenizer = Tokenizer()
            def __init__(self, *_): pass
            def render(self, body): return "fake"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/"case.json"; path.write_text(json.dumps(case))
            argv = ["replay", "--dev-root", str(cases.REPO/"s1-dev"), "--case-file", str(path),
                    "--output", str(root/"run"), "--plan-only"]
            with patch.object(sys, "argv", argv), patch.object(s1_common, "load_index", return_value=({}, {}, groups)), \
                 patch.object(s1_common, "Renderer", Renderer), \
                 patch.object(s1_common, "materialize_bodies", return_value={r: {} for c in case for r in c["req_ids"]}), \
                 patch.object(s1_loadgen, "call_engine", side_effect=AssertionError("no HTTP in plan")), \
                 redirect_stdout(io.StringIO()):
                replay.main()
            selected = json.loads((root/"run/manifest.json").read_text())["selection"]
            self.assertEqual([c["chain_id"] for c in selected], [c["chain_id"] for c in case])
            self.assertEqual([c["req_ids"] for c in selected], [c["req_ids"] for c in case])


class PlannerTests(unittest.TestCase):
    def test_gap_floor_and_reference_calibration(self):
        rows, _, _ = cases.load_data()
        from s1_loadgen import build_gap_plan
        cohort = json.loads(cases.COHORT.read_text())
        gaps, _ = build_gap_plan(cohort["chains"], rows, 3600000)
        scale = planner.calibrate(cohort["chains"], rows, gaps, 6, 2100)
        self.assertAlmostEqual(planner.makespan(cohort["chains"], rows, gaps, 6, scale), 2100)
        self.assertTrue(all(sum(g) <= 3600000 for g in gaps.values()))
        with self.assertRaises(ValueError):
            planner.calibrate(cohort["chains"], rows, gaps, 6, 1)

    def test_fifo_tail_is_not_total_divided_by_n(self):
        chains = [{"chain_id": "a", "req_ids": ["a"]}, {"chain_id": "b", "req_ids": ["b"]}]
        rows = {"a": {"max_output_i": 1, "uncached_expected": 0}, "b": {"max_output_i": 1, "uncached_expected": 0}}
        self.assertEqual(planner.makespan(chains, rows, {"a": [100000], "b": [0]}, 2, 1), 101)

    def test_budget_fit_including_two_startups_and_reserve(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"plan.json"
            proc = subprocess.run([sys.executable, "-B", str(cases.REPO/"scripts/plan_8gpu_session.py"),
                                   "--budget-minutes", "240", "--out", str(path)], capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            plan = json.loads(path.read_text())
            self.assertLessEqual(plan["selected_with_reserve_minutes"], 240)
            self.assertEqual(sum(r["startup_minutes"] for r in plan["requested_runs"]), 42)
            self.assertEqual(plan["selected_runs"], 6)
            self.assertFalse(plan["requested_runs"][-1]["selected"])

    def test_invalid_rungs(self):
        for spec in ("stock:3", "D1:0", "unknown:6", ""):
            with self.assertRaises(ValueError): planner.parse_runs(spec)


@contextmanager
def endpoint_stub(fault=None):
    state = {"cached": False, "requests": [], "generates": 0}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def send_json(self, body):
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(body).encode())
        def do_GET(self):
            self.send_json({"data": [{"id": "default"}]})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body))
            if self.path.startswith("/flush_cache"):
                if fault != "flush_no_clear": state["cached"] = False
                self.send_json({"success": "true" if fault == "flush_string" else True})
            elif self.path == "/generate":
                n = body["sampling_params"]["max_new_tokens"]
                cached = 128 if state["cached"] and fault != "no_repeat_hit" else 0
                state["cached"] = True; state["generates"] += 1
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for tokens in (1, n if fault != "short_output" else n-1):
                    meta = {"completion_tokens": tokens, "prompt_tokens": 256, "cached_tokens": cached,
                            "request_received_ts": 100, "prefill_finished_time": 101}
                    if fault == "missing_counter": del meta["cached_tokens"]
                    if fault == "missing_timing": del meta["prefill_finished_time"]
                    if fault == "negative_ttft": meta["prefill_finished_time"] = 99
                    self.wfile.write(("data: "+json.dumps({"text": "secret-server-output", "meta_info": meta})+"\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_json({"model": "default", "choices": [{"message": {
                    "content": "" if fault == "empty_chat" else "secret-server-output"}}]})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield preflight.Client(f"http://127.0.0.1:{server.server_port}", api_key="secret-key"), state
    finally:
        server.shutdown(); server.server_close(); thread.join()


class PreflightTests(unittest.TestCase):
    def test_complete_checklist_with_stub_and_no_secrets(self):
        report = []
        with endpoint_stub() as (client, state):
            preflight.run_http_checks(client, report)
        self.assertEqual(len(report), 8)
        generated = [body for path, body in state["requests"] if path == "/generate"]
        self.assertEqual(len(generated), 3)
        self.assertEqual(len({b["text"] for b in generated}), 1)
        self.assertTrue(all(b["sampling_params"]["ignore_eos"] is True for b in generated))
        self.assertNotIn("secret", json.dumps(report))
        self.assertEqual([r["cached_tokens"] for r in report if "cached_tokens" in r], [0, 128, 0])

    def test_real_failures_stop_the_checklist(self):
        for fault in ("flush_string", "flush_no_clear", "no_repeat_hit", "short_output",
                      "missing_counter", "missing_timing", "negative_ttft", "empty_chat"):
            with self.subTest(fault=fault), endpoint_stub(fault) as (client, _):
                with self.assertRaises(preflight.CheckFailed): preflight.run_http_checks(client, [])

    def test_dry_run_has_no_writes_no_auth_echo(self):
        import os
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"report.json"
            result = subprocess.run(["bash", str(cases.REPO/"scripts/preflight_8gpu.sh"), "--dry-run", "--out", str(path)],
                                    env={**os.environ, "S1_API_KEY": "sensitive-key"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn("ignore_eos", result.stdout)
            self.assertNotIn("sensitive-key", result.stdout+result.stderr)
            self.assertFalse(path.exists())

    def test_url_credentials_rejected_without_echo(self):
        with self.assertRaises(preflight.CheckFailed) as error:
            preflight.Client("http://user:secret@example.com")
        self.assertNotIn("secret", str(error.exception))

    def test_harness_unittest_fallback_and_no_bytecode(self):
        report, commands = [], []
        def run(command, **kwargs):
            commands.append(command)
            self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertTrue(kwargs["capture_output"])
            return subprocess.CompletedProcess(command, 0)
        with patch.object(preflight.subprocess, "run", side_effect=run), \
             patch.object(preflight.importlib.util, "find_spec", return_value=None):
            preflight.harness_checks(cases.REPO/"s1-dev", report)
        self.assertTrue(any("unittest" in command for command in commands))
        self.assertFalse(any("pip" in command for command in commands))

    def test_harness_installs_missing_requirements_and_uses_pytest_no_cache(self):
        commands, statuses = [], iter((1, 0, 0, 0))
        def run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, next(statuses))
        with patch.object(preflight.subprocess, "run", side_effect=run), \
             patch.object(preflight.importlib.util, "find_spec", return_value=object()):
            report = []
            preflight.harness_checks(cases.REPO/"s1-dev", report)
        self.assertTrue(report[0]["pip_installed"])
        self.assertIn("pip", commands[1])
        self.assertTrue(commands[1][-1].endswith("harness/requirements.txt"))
        self.assertIn("pytest", commands[-1])
        self.assertIn("no:cacheprovider", commands[-1])


if __name__ == "__main__":
    unittest.main()
