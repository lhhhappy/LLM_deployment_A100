#!/usr/bin/env python3
"""README — CPU-only synthetic regression tests for the T12 tools.

Run: python3 -B -m unittest discover -s scripts -p test_ladder_search.py -v
Uses temporary raw/run JSON files, an inert fake run_dev subprocess and loopback
HTTP stubs. No inference engine, GPU, dataset replay, Trisol or submission.
Tests the real dev scorer read-only; fixtures never touch s1-dev/.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from . import ladder_search as ladder, score_formal as scoring
except ImportError:
    import ladder_search as ladder
    import score_formal as scoring


def record(phase="intra", index=1, ttft=1.0, **changes):
    result = dict(req_id=f"{phase}-{index}", chain_id="chain-a", phase=phase,
                  idx_in_chain=index, error=None, error_class=None, ttft_s=ttft,
                  ttft_source="client_proxy", prompt_tokens=10000, glm_tokens=10000,
                  cached_tokens=9000, uncached_expected=1000, output_tokens=11,
                  tpot_s=.02, wall_s=ttft + .2, sampling_weight=1)
    result.update(changes)
    return result


def records():
    return [record(index=0, ttft=10), record("turn_start", ttft=3),
            record(), record(index=2, uncached_expected=5000, cached_tokens=5000, ttft=4)]


def run_metadata(n=10):
    return {"config": {"N": n, "set": "synthetic", "instance_id": "test-measure"}, "wall_s": 20}


def synthetic_files(directory, rows=None, n=10, name="synthetic"):
    raw, run = directory / f"raw_{name}.jsonl", directory / f"run_{name}.json"
    raw.write_text("".join(json.dumps(r) + "\n" for r in (records() if rows is None else rows)))
    run.write_text(json.dumps(run_metadata(n)))
    return raw, run


@contextmanager
def http_stub(responses):
    """Responses: (HTTP status, raw body, optional delay before body)."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            calls.append({"path": self.path, "authorization": self.headers.get("Authorization")})
            item = responses[min(len(calls) - 1, len(responses) - 1)]
            self.send_response(item[0])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if len(item) == 3:
                time.sleep(item[2])
            try:
                self.wfile.write(item[1])
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class SearchTests(unittest.TestCase):
    def trace(self, outcomes, mode="official-climb", hint=18, max_n=None):
        search = ladder.Search(mode, hint, max_n)
        seen = []
        for passed in outcomes:
            seen.append(search.next_n)
            search.observe(passed)
        return seen, search

    def test_all_official_spec_examples(self):
        examples = [([False, False, False], [10, 6, 2], None),
                    ([False, False, True], [10, 6, 2], 2),
                    ([False, True], [10, 6], 6),
                    ([True, False], [10, 14], 10),
                    ([True, True, False], [10, 14, 18], 14),
                    ([True, True, True, False], [10, 14, 18, 22], 18),
                    ([True] * 6 + [False], [10, 14, 18, 22, 26, 30, 34], 30)]
        for outcomes, expected, critical in examples:
            with self.subTest(expected=expected):
                seen, search = self.trace(outcomes)
                self.assertEqual(seen, expected)
                self.assertIsNone(search.next_n)
                self.assertTrue(search.result()["complete"])
                self.assertEqual(search.result()["critical_n"], critical)

    def test_fast_bracketing_and_bisection_exhaustive(self):
        for hint in (2, 6, 18, 62, 126):
            for critical_index in range(-1, 65):
                critical = None if critical_index == -1 else 2 + 4 * critical_index
                search = ladder.Search("fast", hint)
                seen = []
                while search.next_n is not None:
                    n = search.next_n
                    self.assertNotIn(n, seen)
                    seen.append(n)
                    search.observe(critical is not None and n <= critical)
                    self.assertLess(len(seen), 20)
                self.assertEqual(seen[0], hint)
                self.assertEqual(search.result()["critical_n"], critical)
                self.assertTrue(search.result()["complete"])

    def test_fast_saves_runs_for_distant_capacity(self):
        search = ladder.Search("fast", hint=10)
        count = 0
        while search.next_n is not None:
            count += 1
            search.observe(search.next_n <= 402)
        self.assertEqual(search.result()["critical_n"], 402)
        self.assertLess(count, 18)  # official traversal would take 100 levels

    def test_ceiling_is_incomplete_not_capacity(self):
        _, search = self.trace([True, True, True], max_n=18)
        self.assertEqual(search.reason, "max_n_reached")
        self.assertFalse(search.result()["complete"])
        self.assertIsNone(search.result()["critical_n"])
        self.assertEqual(search.result()["largest_observed_pass"], 18)

    def test_invalid_rungs_and_start_above_ceiling(self):
        for n in (0, 1, 3, 9, 11, 24):
            with self.assertRaises(ValueError):
                ladder.rung(n)
        with self.assertRaises(ValueError):
            ladder.Search(max_n=6)


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = scoring.load_harness()

    def score(self, rows):
        with tempfile.TemporaryDirectory() as temporary:
            raw, run = synthetic_files(Path(temporary), rows)
            return scoring.score_files(raw, run)

    def test_dev_report_unchanged_and_tpot(self):
        rows = records()
        rows[-1]["tpot_s"] = .06
        result = self.score(rows)
        cfg = dict(run_metadata()["config"], lane="dev", strict_ttft_basis=False,
                   steady_start_min=10, steady_len_min=60)
        self.assertEqual(result["dev"], self.scorer.evaluate(rows, 20, cfg))
        self.assertEqual(len(result["dev"]["gates"]), 10)
        self.assertTrue(result["dev"]["ALL_PASS"])
        self.assertAlmostEqual(result["tpot"]["tpot_mean"], .03)
        self.assertEqual(result["tpot"]["tpot_p95"], .06)
        self.assertTrue(result["estimated"]["passed"])
        self.assertNotIn("formal PASS", json.dumps(result))

    def test_statistical_allowance_preserves_failing_dev_verdict(self):
        rows = records()[:2] + [record(ttft=3.5 if i < 2 else 1) for i in range(20)]
        result = self.score(rows)
        fast = result["ttft_estimated"][self.scorer.TTFT_GATE_SPECS[0][0]]
        self.assertFalse(result["dev"]["ALL_PASS"])
        self.assertEqual(fast["over_limit"], 2)
        self.assertEqual(fast["p95"], 3.5)
        self.assertFalse(fast["pass_point"])
        self.assertLess(fast["rate_ci_lower"], .05)
        self.assertTrue(fast["pass_estimated"])
        self.assertTrue(result["estimated"]["passed"])
        self.assertEqual(fast["method"], scoring.METHOD)

    def test_statistically_supported_exceedance_fails(self):
        rows = records()[:2] + [record(ttft=3.5 if i < 4 else 1) for i in range(20)]
        result = self.score(rows)
        fast = next(iter(result["ttft_estimated"].values()))
        self.assertGreater(fast["rate_ci_lower"], .05)
        self.assertFalse(result["estimated"]["passed"])

    def test_exact_limit_is_not_exceedance(self):
        rows = [record(index=0, ttft=30), record("turn_start", ttft=15),
                record(ttft=3), record(ttft=5, uncached_expected=5000)]
        result = self.score(rows)
        self.assertTrue(result["dev"]["ALL_PASS"])
        self.assertTrue(all(d["over_limit"] == 0 for d in result["ttft_estimated"].values()))

    def test_tpot_gate_has_no_statistical_allowance(self):
        for value, passed in ((.1, True), (.100001, False)):
            rows = records()
            rows[-1]["tpot_s"] = value
            result = self.score(rows)
            self.assertTrue(result["dev"]["ALL_PASS"])
            self.assertEqual(result["tpot"]["passed"], passed)
            self.assertEqual(result["estimated"]["passed"], passed)

    def test_missing_tpot_blocks_but_single_token_is_counted(self):
        rows = records()
        rows[0]["tpot_s"] = None
        result = self.score(rows)
        self.assertFalse(result["tpot"]["passed"])
        self.assertEqual(result["tpot"]["n_missing_or_invalid"], 1)
        rows[0]["output_tokens"] = 1
        result = self.score(rows)
        self.assertTrue(result["tpot"]["passed"])
        self.assertEqual(result["tpot"]["n_single_token_undefined"], 1)
        for r in rows:
            r.update(output_tokens=1, tpot_s=None)
        result = self.score(rows)
        self.assertFalse(result["tpot"]["passed"])
        self.assertIsNone(result["tpot"]["tpot_mean"])

    def test_empty_buckets_and_entire_run_fail(self):
        for rows in ([], [record()]):
            result = self.score(rows)
            self.assertFalse(result["dev"]["gates"]["gated_phases_have_samples"])
            self.assertFalse(result["estimated"]["passed"])
            for gate in result["ttft_estimated"].values():
                if gate["n"] == 0:
                    self.assertIsNone(gate["rate_ci_lower"])
                    self.assertFalse(gate["pass_estimated"])

    def test_error_and_coverage_gates_are_not_filtered_away(self):
        for error_class, gate in (("HARNESS_DATA", "harness_data==0"),
                                  ("HARNESS_RENDER", "harness_render==0"),
                                  ("ENGINE", "engine_error<1%"), ("INFRA", "infra_error<1%")):
            rows = records() + [record(error="synthetic failure", error_class=error_class)]
            result = self.score(rows)
            self.assertFalse(result["estimated"]["gates"][gate])
            self.assertFalse(result["estimated"]["passed"])
        rows = records()
        rows[0]["cached_tokens"] = None
        result = self.score(rows)
        self.assertFalse(result["estimated"]["gates"]["coverage=100%"])

    def test_frozen_bucket_membership_and_cache_reconciliation(self):
        rows = records()
        rows[2]["cached_tokens"] = 1000  # actual miss 9000; frozen miss stays 1000 (fast)
        result = self.score(rows)
        fast_name = self.scorer.TTFT_GATE_SPECS[0][0]
        self.assertEqual(result["ttft_estimated"][fast_name]["n"], 1)
        cache = result["cache_reconciliation"]["by_gate"][fast_name]
        self.assertEqual(cache["extra_uncached_tokens"]["p95"], 8000)
        self.assertEqual(cache["cached_minus_frozen_expected"]["p95"], -8000)
        self.assertEqual(cache["frozen_cached_expected"]["p95"], 9000)
        self.assertEqual(cache["extra_uncached_histogram"][">4096"], 1)

    def test_binomial_analytic_endpoints_and_small_exact_sums(self):
        self.assertIsNone(scoring.binomial_lower(0, 0))
        self.assertEqual(scoring.binomial_lower(0, 30), 0)
        self.assertEqual(scoring.binomial_lower(1, 1), .05)
        for n in (2, 10, 100, 1000):
            self.assertAlmostEqual(scoring.binomial_lower(n, n), .05 ** (1/n), places=12)
            self.assertAlmostEqual(scoring.binomial_lower(1, n), 1 - .95 ** (1/n), places=12)
        for n in range(1, 20):
            for k in range(n + 1):
                # Independent direct polynomial sum for small n.
                tail = sum(math.comb(n, j) * .05**j * .95**(n-j) for j in range(k, n+1))
                self.assertAlmostEqual(scoring.binomial_tail(n, k, .05), tail, places=12)
                if 0 < k < n:
                    lower = scoring.binomial_lower(k, n)
                    self.assertAlmostEqual(sum(math.comb(n, j) * lower**j * (1-lower)**(n-j)
                                               for j in range(k, n+1)), .05, places=12)
        self.assertEqual(scoring.allowed_over(20), 3)
        self.assertEqual(scoring.allowed_over(1), 1)

    def test_directory_selection_uses_summary_not_latest_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw, run = synthetic_files(directory, name="measure")
            synthetic_files(directory, name="preflight")
            with self.assertRaises(ValueError):
                scoring.resolve_inputs(None, None, directory)
            (directory / "summary.json").write_text(json.dumps({"raw": str(raw), "run": str(run)}))
            self.assertEqual(scoring.resolve_inputs(None, None, directory), (raw, run))
            output = directory / "estimated.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(scoring.main(["--run-dir", str(directory), "--out", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["dev"]["n_attempted"], 4)

    def test_malformed_raw_is_execution_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw, run = synthetic_files(Path(temporary))
            for invalid in ('{"ttft_s": NaN}\n', '[]\n', 'not json\n'):
                raw.write_text(invalid)
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(scoring.main(["--raw", str(raw), "--run", str(run)]), 2)


class FlushTests(unittest.TestCase):
    def test_http_and_json_must_both_succeed(self):
        responses = [(200, b'{"success":true}'), (200, b'{"success":false}'),
                     (200, b'{"success":"true"}'), (200, b'{"success":1}'),
                     (200, b'Cache flushed.'), (204, b''), (200, b'[]'),
                     (500, b'{"success":true}'), (302, b'{"success":true}')]
        with http_stub(responses) as (base, calls):
            results = [ladder.verified_flush(base, "test-only-secret", 2) for _ in responses]
        self.assertEqual([r["success"] for r in results], [True] + [False] * 8)
        self.assertTrue(all(c["path"].split("?")[0] == "/flush_cache" for c in calls))
        self.assertTrue(all(c["authorization"] == "Bearer test-only-secret" for c in calls))
        self.assertNotIn("test-only-secret", json.dumps(results))

    def test_network_error_is_not_slo_failure(self):
        with patch("urllib.request.OpenerDirector.open", side_effect=urllib.error.URLError("secret")):
            result = ladder.verified_flush("http://example.invalid", "secret", 1)
        self.assertFalse(result["success"])
        self.assertIsNone(result["http_status"])
        self.assertNotIn("secret", json.dumps(result))

    def test_url_normalization_and_secret_rejection(self):
        self.assertEqual(ladder.engine_url("http://localhost:8000/v1/"), "http://localhost:8000")
        for url in ("", "file:///tmp/test", "http://u:secret@host", "http://host/?token=secret", "http://host/#secret"):
            with self.assertRaises(ValueError):
                ladder.engine_url(url)


class DriverTests(unittest.TestCase):
    def make_runner(self, directory):
        runner = directory / "fake_run_dev.py"
        runner.write_text(textwrap.dedent('''\
            import argparse, json, os, sys, urllib.request
            from pathlib import Path
            sys.path.insert(0, SCRIPTS_DIR)
            import score_formal as scoring
            parser = argparse.ArgumentParser()
            parser.add_argument('--n', type=int)
            parser.add_argument('--out', type=Path)
            args, other = parser.parse_known_args()
            (args.out / 'fake_started').write_text('started')
            if os.environ.get('T12_FAKE_CRASH'):
                sys.exit(9)
            # Deliberately mirrors run_dev ignoring HTTP failure; guard must kill us.
            try:
                urllib.request.urlopen(urllib.request.Request(os.environ['S1_FLUSH_URL'], method='POST'), timeout=30).read()
            except OSError:
                pass
            (args.out / 'measurement_started').write_text('started')
            rows = FIXTURE_ROWS
            if args.n > int(os.environ.get('T12_FAKE_CAPACITY', '10')):
                rows[2]['ttft_s'] = 4
            raw = args.out / 'raw_synthetic.jsonl'
            run = args.out / 'run_synthetic.json'
            raw.write_text(''.join(json.dumps(r) + '\\n' for r in rows))
            run.write_text(json.dumps({'wall_s':20, 'config':{'N':args.n, 'set':'synthetic'}}))
            report = args.out / 'report_synthetic.json'
            report.write_text(json.dumps(scoring.score_files(raw, run)['dev']))
            (args.out / 'summary.json').write_text(json.dumps({'raw':str(raw), 'run':str(run), 'report':str(report)}))
            ''').replace("SCRIPTS_DIR", repr(str(Path(ladder.__file__).parent)))
                            .replace("FIXTURE_ROWS", repr(records())))
        return runner

    def run_driver(self, out, base, *extra):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return ladder.main(["--out", str(out), "--base-url", base, *extra])

    def test_dry_run_no_network_process_or_output_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "never-created"
            stream = io.StringIO()
            with patch.object(ladder, "verified_flush") as flush, patch.object(ladder.subprocess, "Popen") as popen:
                with redirect_stdout(stream):
                    rc = ladder.main(["--out", str(out), "--base-url", "http://example.invalid/v1", "--dry-run"])
                self.assertEqual(rc, 0)
                flush.assert_not_called()
                popen.assert_not_called()
            self.assertFalse(out.exists())
            commands = [line for line in stream.getvalue().splitlines() if "run_dev.py " in line]
            self.assertEqual(len(commands), 2)
            self.assertIn("--n 10", commands[0])
            self.assertNotIn("--skip-warmup", commands[0])
            self.assertIn("--n 14", commands[1])
            self.assertIn("--skip-warmup", commands[1])

    def test_failed_initial_flush_aborts_without_starting_runner(self):
        with tempfile.TemporaryDirectory() as temporary, http_stub([(200, b'{"success":false}')]) as (base, _):
            out = Path(temporary) / "session"
            with patch.object(ladder.subprocess, "Popen") as popen:
                self.assertEqual(self.run_driver(out, base), 2)
                popen.assert_not_called()
            ledger = json.loads((out / "ledger.json").read_text())
            self.assertEqual(ledger["status"], "aborted")
            self.assertIsNone(ledger["levels"][0]["passed"])

    def test_subprocess_ledger_warmup_and_dev_verdict(self):
        with tempfile.TemporaryDirectory() as temporary, http_stub([(200, b'{"success":true}')]) as (base, calls):
            directory = Path(temporary)
            runner = self.make_runner(directory)
            out = directory / "session"
            with patch.object(ladder, "RUNNER", runner), patch.dict(os.environ, {"API_KEY": "test-only-secret", "S1_SKIP_WARMUP": "1"}):
                self.assertEqual(self.run_driver(out, base), 0)
            ledger = json.loads((out / "ledger.json").read_text())
            self.assertEqual(ledger["search"]["critical_n"], 10)
            self.assertEqual([level["N"] for level in ledger["levels"]], [10, 14])
            self.assertEqual([level["passed"] for level in ledger["levels"]], [True, False])
            self.assertEqual(len(calls), 4)
            self.assertNotIn("test-only-secret", (out / "ledger.json").read_text())
            for level in ledger["levels"]:
                self.assertEqual(len(level["dev_gates"]), 10)
                self.assertEqual(len(level["estimated_gates"]), 11)
                self.assertEqual([f["stage"] for f in level["flushes"]],
                                 ["before_level", "after_preflight_warmup"])
                self.assertTrue(all(Path(p).exists() for p in level["paths"].values()))
            self.assertNotIn("--skip-warmup", ledger["levels"][0]["command"])
            self.assertIn("--skip-warmup", ledger["levels"][1]["command"])

    def test_failed_internal_flush_prevents_measurement(self):
        responses = [(200, b'{"success":true}'), (200, b'{"success":false}')]
        with tempfile.TemporaryDirectory() as temporary, http_stub(responses) as (base, calls):
            directory = Path(temporary)
            with patch.object(ladder, "RUNNER", self.make_runner(directory)):
                out = directory / "session"
                self.assertEqual(self.run_driver(out, base), 2)
            level = out / "level_001_N10"
            self.assertTrue((level / "fake_started").exists())
            self.assertFalse((level / "measurement_started").exists())
            ledger = json.loads((out / "ledger.json").read_text())
            self.assertIsNone(ledger["levels"][0]["passed"])
            self.assertEqual(ledger["error"], "measurement_flush_failed_or_missing")
            self.assertEqual(len(calls), 2)

    def test_internal_flush_deadline_kills_ignoring_runner(self):
        responses = [(200, b'{"success":true}'), (200, b'{"success":true}', .4)]
        with tempfile.TemporaryDirectory() as temporary, http_stub(responses) as (base, _):
            directory = Path(temporary)
            with patch.object(ladder, "RUNNER", self.make_runner(directory)):
                out = directory / "session"
                self.assertEqual(self.run_driver(out, base, "--flush-timeout", ".1"), 2)
            self.assertFalse((out / "level_001_N10/measurement_started").exists())

    def test_unexpected_guard_error_also_prevents_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initial = {"at": ladder.now(), "http_status": 200, "success": True}
            with patch.object(ladder, "RUNNER", self.make_runner(directory)), \
                    patch.object(ladder, "verified_flush", side_effect=[initial, RuntimeError("synthetic")]):
                out = directory / "session"
                self.assertEqual(self.run_driver(out, "http://example.invalid"), 2)
            self.assertFalse((out / "level_001_N10/measurement_started").exists())

    def test_real_unmodified_run_dev_with_synthetic_loadgen(self):
        """Exercise the exact subprocess API while replacing only loadgen in /tmp."""
        with tempfile.TemporaryDirectory() as temporary, http_stub([(200, b'{"success":true}')]) as (base, calls):
            directory = Path(temporary)
            harness = directory / "synthetic_harness"
            harness.mkdir()
            for name in ("s1_score.py", "s1_common.py"):
                (harness / name).write_bytes((scoring.DEFAULT_HARNESS / name).read_bytes())
            (harness / "s1_loadgen.py").write_text(textwrap.dedent('''\
                import argparse, json, os
                from pathlib import Path
                p = argparse.ArgumentParser()
                p.add_argument('--n', type=int)
                p.add_argument('--out-dir', type=Path)
                p.add_argument('--instance-id')
                p.add_argument('--warmup', action='store_true')
                args, _ = p.parse_known_args()
                out = args.out_dir
                phase = args.instance_id.rsplit('-', 1)[-1]
                with (out / 'synthetic_phases.jsonl').open('a') as f:
                    f.write(json.dumps({'phase':phase}) + '\\n')
                if args.warmup:
                    raise SystemExit(0)
                rows = FIXTURE_ROWS
                if args.n > 10:
                    rows[2]['ttft_s'] = 4
                raw, run = out / ('raw_' + phase + '.jsonl'), out / ('run_' + phase + '.json')
                raw.write_text(''.join(json.dumps(r) + '\\n' for r in rows))
                run.write_text(json.dumps({'wall_s':20, 'config':{'N':args.n, 'set':'synthetic', 'instance_id':args.instance_id}}))
                stamp = 1000 if phase == 'preflight' else 2000
                os.utime(raw, (stamp, stamp))
                os.utime(run, (stamp, stamp))
                ''').replace("FIXTURE_ROWS", repr(records())))
            out = directory / "session"
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", S1_SKIP_WARMUP="1", API_KEY="test-only-secret")
            result = subprocess.run([sys.executable, "-B", str(Path(ladder.__file__).resolve()),
                                     "--base-url", base, "--out", str(out), "--harness-dir", str(harness)],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            ledger = json.loads((out / "ledger.json").read_text())
            self.assertEqual(ledger["search"]["critical_n"], 10)
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(c["authorization"] == "Bearer test-only-secret" for c in calls))
            for index, level in enumerate(ledger["levels"]):
                self.assertEqual(level["command"][1], str(ladder.RUNNER))
                phases = [json.loads(line)["phase"] for line in
                          (Path(level["paths"]["directory"]) / "synthetic_phases.jsonl").read_text().splitlines()]
                self.assertEqual(phases, ["preflight", "warmup", "measure"] if index == 0 else ["preflight", "measure"])
                self.assertTrue(level["paths"]["raw"].endswith("raw_measure.jsonl"))

    def test_execution_crash_and_budget_stop_are_not_slo_fails(self):
        with tempfile.TemporaryDirectory() as temporary, http_stub([(200, b'{"success":true}')]) as (base, _):
            directory = Path(temporary)
            with patch.object(ladder, "RUNNER", self.make_runner(directory)):
                out = directory / "budget"
                self.assertEqual(self.run_driver(out, base, "--max-levels", "1"), 3)
                ledger = json.loads((out / "ledger.json").read_text())
                self.assertEqual(ledger["status"], "incomplete")
                self.assertIsNone(ledger["search"]["critical_n"])
                self.assertEqual(ledger["search"]["largest_observed_pass"], 10)
                out = directory / "crash"
                with patch.dict(os.environ, {"T12_FAKE_CRASH": "1"}):
                    self.assertEqual(self.run_driver(out, base), 2)
                ledger = json.loads((out / "ledger.json").read_text())
                self.assertEqual(ledger["status"], "aborted")
                self.assertIsNone(ledger["levels"][0]["passed"])

    def test_gate_policy_keeps_point_and_estimated_results_separate(self):
        report = {"dev": {"ALL_PASS": False}, "tpot": {"passed": True},
                  "estimated": {"passed": True}}
        self.assertFalse(ladder.policy_pass(report, "dev"))
        self.assertFalse(ladder.policy_pass(report, "dev+tpot"))
        self.assertTrue(ladder.policy_pass(report, "estimated"))


if __name__ == "__main__":
    unittest.main()
