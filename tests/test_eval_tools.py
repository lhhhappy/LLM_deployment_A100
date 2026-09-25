"""CPU regressions for invalid measurements, file selection and score stability."""
import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKED = load("checked", "scripts/pod/verify/run_dev_checked.py")
LEVEL = load("verdict", "scripts/pod/verify/level_verdict.py")
FETCH = load("fetch", "scripts/analysis/fetch_level.py")
SCORE = load("score", "scripts/score_formal.py")


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def fixture(out, n=22):
    out.mkdir(parents=True, exist_ok=True)
    raw, run = out / f"raw_N{n}_measure.jsonl", out / f"run_N{n}_measure.json"
    write(raw, {"req_id": "measured"})
    write(run, {"config": {"N": n, "instance_id": "dev-1790200000-measure"}, "raw_file": raw.name})
    write(out / "summary.json", {"n": n, "raw": f"/pod/{raw.name}", "run": f"/pod/{run.name}", "score_rc": 0})
    return raw, run


class FlushTest(unittest.TestCase):
    def test_success_receipt_tracks_original_runner_selected_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"S1_HARNESS_DIR": str(ROOT / "s1-dev/harness")}):
            out = Path(directory)
            original = load("original_runner_success", "s1-dev/run_dev.py")
            calls = []
            def fake_run(command, log, env):
                calls.append(command)
                if "--instance-id" in command:
                    write(out / "raw_measure.jsonl", {"req_id": "fixture"})
                    write(out / "run_measure.json", {"config": {"N": 22}})
                return 0
            original._run = fake_run
            spec = Mock(); spec.loader.exec_module = Mock()
            response = Mock(status=200)
            response.read.return_value = b'{"success":true}'
            response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
            argv = ["--out", str(out), "--n", "22", "--base-url", "http://test.invalid",
                    "--root", str(ROOT / "s1-dev/data/dev-combined-v1"),
                    "--cohort", str(ROOT / "s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json"),
                    "--skip-warmup", "--skip-preflight"]
            with patch.object(CHECKED.importlib.util, "spec_from_file_location", return_value=spec), \
                 patch.object(CHECKED.importlib.util, "module_from_spec", return_value=original), \
                 patch.object(CHECKED.urllib.request, "urlopen", return_value=response), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(CHECKED.run(ROOT / "s1-dev/run_dev.py", argv), 0)
            receipt = json.loads((out / "flush_evidence.json").read_text())
            self.assertTrue(receipt["flush_success"])
            self.assertEqual(receipt["runner_rc"], 0)
            self.assertEqual(receipt["raw"], "raw_measure.jsonl")
            self.assertEqual(receipt["run"], "run_measure.json")
            self.assertEqual(len(calls), 2)  # original measured replay and original scorer

    def test_original_runner_never_measures_after_flush_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"S1_HARNESS_DIR": str(ROOT / "s1-dev/harness")}):
            out = Path(directory)
            original = load("original_runner", "s1-dev/run_dev.py")
            original._run = Mock(return_value=0)
            spec = Mock()
            spec.loader.exec_module = Mock()
            argv = ["--out", str(out), "--n", "22", "--base-url", "http://test.invalid",
                    "--root", str(ROOT / "s1-dev/data/dev-combined-v1"),
                    "--cohort", str(ROOT / "s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json"),
                    "--skip-warmup", "--skip-preflight"]
            with patch.object(CHECKED.importlib.util, "spec_from_file_location", return_value=spec), \
                 patch.object(CHECKED.importlib.util, "module_from_spec", return_value=original), \
                 patch.object(CHECKED.urllib.request, "urlopen", side_effect=OSError("failed")), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                CHECKED.run(ROOT / "s1-dev/run_dev.py", argv)
            self.assertEqual(caught.exception.code, 2)
            original._run.assert_not_called()
            receipt = json.loads((out / "flush_evidence.json").read_text())
            self.assertFalse(receipt["flush_success"])
            self.assertEqual(receipt["runner_rc"], 2)

    def test_http_success_does_not_override_explicit_rejection(self):
        for payload, status, succeeds in ((b'{"success":true}', 200, True),
                                          (b'{"success":true}', 201, True),
                                          (b'{}', 200, False), (b'', 204, False),
                                          (b'Cache flushed.', 200, False),
                                          (b'{"success":1}', 200, False),
                                          (b'{"success":"true"}', 200, False),
                                          (b'{"success":false}', 200, False),
                                          (b'{"flushed":false}', 200, False),
                                          (b'false', 200, False), (b'{}', 503, False)):
            with self.subTest(payload=payload, status=status), tempfile.TemporaryDirectory() as directory:
                response = Mock(status=status)
                response.read.return_value = payload
                response.__enter__ = Mock(return_value=response)
                response.__exit__ = Mock(return_value=False)
                evidence = {}
                with patch.object(CHECKED.urllib.request, "urlopen", return_value=response), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if succeeds:
                        self.assertTrue(CHECKED.strict_flush("http://test.invalid", evidence, Path(directory) / "receipt"))
                    else:
                        with self.assertRaises(SystemExit):
                            CHECKED.strict_flush("http://test.invalid", evidence, Path(directory) / "receipt")
                self.assertEqual(evidence["flush_success"], succeeds)

    def test_current_flush_required_and_receipt_bound_to_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            raw, run = fixture(out)
            metadata = json.loads(run.read_text())
            rows = [{"client_dispatch_at_s": 1790200020}]
            (out / "run_dev.log").write_text("flushed KV via checked runner\n")
            stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("[%Y-%m-%d %H:%M:%S TP0] Cache flushed successfully!\n")
            for timestamp in (1790199990, 1790200021):
                (out / "server.log").write_text(stamp(timestamp))
                with self.assertRaisesRegex(ValueError, "no successful server flush"):
                    LEVEL.check_flush(out, raw, run, metadata, rows, 22)
            (out / "server.log").write_text(stamp(1790200010))
            self.assertTrue(LEVEL.check_flush(out, raw, run, metadata, rows, 22)["verified"])
            receipt = {"flush_success": True, "runner_rc": 0, "n": 22, "raw": raw.name, "run": run.name,
                       "runner_started_s": 1790200000, "flush_started_s": 1790200010,
                       "flush_finished_s": 1790200011}
            write(out / "flush_evidence.json", receipt)
            self.assertTrue(LEVEL.check_flush(out, raw, run, metadata, rows, 22)["verified"])
            for change in ({"n": 26}, {"raw": "old_raw.jsonl"}, {"flush_success": False},
                           {"runner_rc": 1}, {"flush_finished_s": 1790200021}):
                write(out / "flush_evidence.json", dict(receipt, **change))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    LEVEL.check_flush(out, raw, run, metadata, rows, 22)
            (out / "flush_evidence.json").unlink()
            (out / "run_dev.log").write_text("flush failed: connection error\n")
            with self.assertRaisesRegex(ValueError, "flush missing/failed"):
                LEVEL.check_flush(out, raw, run, metadata, rows, 22)

    def test_invalid_exit_status_removes_stale_score(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            fixture(out)
            (out / "rundev_exit_code").write_text("not a status")
            (out / "score_formal.json").write_text('{"passed":true}')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(LEVEL.evaluate(out, 22, ROOT / "s1-dev/harness", ROOT / "s1-dev/data/dev-combined-v1"), 2)
            self.assertFalse((out / "score_formal.json").exists())
            self.assertEqual(json.loads((out / "level_verdict.json").read_text())["status"], "INVALID")

    def test_vllm_flush_binds_response_to_server_and_client_window(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            raw, run = fixture(out)
            metadata = json.loads(run.read_text())
            rows = [{"client_dispatch_at_s": 1790200020}]
            (out / "run_dev.log").write_text("flushed KV via checked runner\n")
            payload = {
                "success": True, "engine": "vllm", "engine_version": "frozen-test",
                "flush_started_ts": 1790200010.2, "flush_finished_ts": 1790200010.8,
                "reset_connector": True, "kv_connector": None,
                "unfinished_requests_at_start": 0,
            }
            receipt = {
                "flush_success": True, "runner_rc": 0, "n": 22,
                "raw": raw.name, "run": run.name, "runner_started_s": 1790200000,
                "flush_started_s": 1790200010, "flush_finished_s": 1790200011,
                "flush_response": payload,
            }
            event = lambda p: "INFO [ax] flush_cache " + json.dumps(p) + "\n"
            write(out / "flush_evidence.json", receipt)
            (out / "server.log").write_text(event(payload))
            result = LEVEL.check_flush(out, raw, run, metadata, rows, 22)
            self.assertTrue(result["verified"])
            self.assertEqual(result["engine"], "vllm")
            self.assertEqual(result["matched_server_flush_s"], payload["flush_finished_ts"])

            # HTTP success alone, unrelated/malformed receipts, or familiar
            # SGLang wording must not substitute for this exact service event.
            for log in ("flush_cache: prefix cache reset\n", "INFO [ax] flush_cache {bad\n",
                        event(dict(payload, engine_version="other-engine")),
                        "[2026-09-23 21:46:50 TP0] Cache flushed successfully!\n"):
                (out / "server.log").write_text(log)
                with self.subTest(log=log), self.assertRaisesRegex(ValueError, "no matching vLLM"):
                    LEVEL.check_flush(out, raw, run, metadata, rows, 22)

            changes = (
                {"kv_connector": "OffloadingConnector"}, {"success": False},
                {"reset_connector": False}, {"unfinished_requests_at_start": 1},
                {"unfinished_requests_at_start": False}, {"engine_version": ""},
                {"flush_started_ts": 1790200009}, {"flush_finished_ts": 1790200021},
                {"flush_started_ts": 1790200010.9}, {"flush_finished_ts": float("nan")},
            )
            for change in changes:
                altered = dict(payload, **change)
                write(out / "flush_evidence.json", dict(receipt, flush_response=altered))
                (out / "server.log").write_text(event(altered))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    LEVEL.check_flush(out, raw, run, metadata, rows, 22)
            for key in ("kv_connector", "reset_connector", "unfinished_requests_at_start",
                        "engine_version", "flush_started_ts", "flush_finished_ts"):
                altered = {k: v for k, v in payload.items() if k != key}
                write(out / "flush_evidence.json", dict(receipt, flush_response=altered))
                (out / "server.log").write_text(event(altered))
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    LEVEL.check_flush(out, raw, run, metadata, rows, 22)


class FetchTest(unittest.TestCase):
    def test_full_archive_recheck_and_missing_flush_log(self):
        source = ROOT / "evidence/L042"
        if not (source / "summary.json").is_file():
            self.skipTest("historical L042 evidence not installed")
        summary = json.loads((source / "summary.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for with_log in (True, False):
                archive = root / "level.tgz"
                with tarfile.open(archive, "w:gz") as tar:
                    for name in ("summary.json", Path(summary["raw"]).name, Path(summary["run"]).name, "server.log"):
                        tar.add(source / name, arcname=name)
                    if with_log:
                        # Test fixture only: historical fetch omitted this client log.
                        data = b"flushed KV via fixture\n"
                        member = tarfile.TarInfo("run_dev.log"); member.size = len(data)
                        tar.addfile(member, io.BytesIO(data))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = FETCH.main(["fixture-042", "22", "--archive", str(archive), "--evidence-root", str(root / "out")])
                self.assertEqual(status, 1 if with_log else 2)
                out = FETCH.destination("fixture-042", 22, root / "out")
                verdict = json.loads((out / "level_verdict.json").read_text())
                self.assertEqual(verdict["status"], "VALID" if with_log else "INVALID")
                self.assertEqual((out / "badcases.csv").exists(), with_log)

    def test_pack_selects_requested_level_and_dereferences_live_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for n in (22, 26):
                fixture(root / f"N{n}", n)
                (root / f"N{n}" / "raw_000_preflight.jsonl").write_text("WRONG\n")
                (root / f"N{n}" / "run_dev.log").write_text(f"N{n} log\n")
            (root / "live.log").write_text("live server log\n")
            (root / "server.log").symlink_to(root / "live.log")
            archive = root / "archive.tgz"
            proc = subprocess.run([sys.executable, "-B", "-c", FETCH.PACK_CODE, str(root), "26", str(archive)],
                                  capture_output=True, text=True, check=True)
            meta = json.loads(FETCH.marked(proc.stdout, "FETCH_META "))
            self.assertEqual(meta["size"], archive.stat().st_size)
            with tarfile.open(archive) as tar:
                self.assertIn("raw_N26_measure.jsonl", tar.getnames())
                self.assertNotIn("raw_N22_measure.jsonl", tar.getnames())
                self.assertNotIn("raw_000_preflight.jsonl", tar.getnames())
                self.assertTrue(tar.getmember("server.log").isfile())
            staging = root / "staging"; staging.mkdir()
            FETCH.extract(archive, staging)
            out = FETCH.destination("run-variant", 26, root / "evidence")
            out.mkdir(parents=True)
            (out / "flush_evidence.json").write_text("stale receipt")
            self.assertEqual(FETCH.publish(staging, out, 26), "raw_N26_measure.jsonl")
            self.assertFalse((out / "flush_evidence.json").exists())
            with self.assertRaisesRegex(ValueError, "N mismatch"):
                LEVEL.selected_files(out, 22)
            self.assertNotEqual(FETCH.destination("037-c", 22, root), FETCH.destination("037-d", 22, root))
            self.assertNotEqual(FETCH.destination("037-c", 22, root), FETCH.destination("037-c", 26, root))

    def test_archive_rejects_traversal_and_links(self):
        for name, type_ in (("../oops", tarfile.REGTYPE), ("/oops", tarfile.REGTYPE), ("server.log", tarfile.SYMTYPE)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); archive = root / "bad.tgz"
                with tarfile.open(archive, "w:gz") as tar:
                    member = tarfile.TarInfo(name); member.type = type_; tar.addfile(member)
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    FETCH.extract(archive, root)

    def test_analysis_preserves_fail_and_stops_on_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); raw, _ = fixture(out)
            for verdict, analysis, expected in ((0, 0, 0), (1, 0, 1), (1, 4, 2), (2, 0, 2)):
                with self.subTest(verdict=verdict, analysis=analysis), \
                     patch.object(FETCH, "logged", side_effect=[verdict, analysis]) as logged, \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(FETCH.analyze(out, 22), expected)
                    self.assertEqual(logged.call_count, 1 if verdict == 2 else 2)
                    if verdict != 2:
                        self.assertIn(str(raw), logged.call_args_list[1].args[0])

    def test_transfer_rejects_truncation_and_hash_mismatch(self):
        for sha, data in (("0" * 64, "YWJj"), ("0" * 64, "YQ==")):
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(FETCH, "remote", side_effect=[
                     'FETCH_META ' + json.dumps({"size": 3, "sha256": sha}), "FETCH_DATA " + data]):
                with self.assertRaises(ValueError):
                    FETCH.download("test", 22, Path(directory) / "archive")

    def test_custom_dataset_and_harness_reach_both_analysis_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            fixture(out)
            data_root, harness_dir = out / "longchain", out / "harness"
            with patch.object(FETCH, "logged", side_effect=[1, 0]) as logged:
                self.assertEqual(FETCH.analyze(out, 22, data_root=data_root, harness_dir=harness_dir), 1)
            verdict = logged.call_args_list[0].args[0]
            badcase = logged.call_args_list[1].args[0]
            self.assertEqual(verdict[verdict.index("--data-root") + 1], str(data_root))
            self.assertEqual(verdict[verdict.index("--harness-dir") + 1], str(harness_dir))
            self.assertEqual(badcase[badcase.index("--harness-dir") + 1], str(harness_dir))


class ScoringTest(unittest.TestCase):
    def test_successful_replay_requires_frozen_work_not_raw_claimed_budget(self):
        index = {"r": {"glm_tokens": 100, "max_output_i": 8}}
        row = {"req_id": "r", "prompt_tokens": 100, "output_tokens": 8,
               "cached_tokens": 64, "max_output_i": 8}
        self.assertEqual(SCORE.validate_replay_tokens([row], index)["successful_checked"], 1)
        for change, field in (({"output_tokens": 7, "max_output_i": 7}, "output_tokens"),
                              ({"output_tokens": 9}, "output_tokens"),
                              ({"output_tokens": 8.0}, "output_tokens"),
                              ({"prompt_tokens": 99}, "prompt_tokens"),
                              ({"cached_tokens": 101}, "cached_tokens"),
                              ({"cached_tokens": -1}, "cached_tokens"),
                              ({"cached_tokens": None}, "cached_tokens"),
                              ({"cached_tokens": True}, "cached_tokens")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, field):
                SCORE.validate_replay_tokens([dict(row, **change)], index)

    def test_failed_requests_preserve_harness_error_rate_semantics(self):
        rows = [{"req_id": "failed", "error": "ENGINE:EOF", "output_tokens": 3},
                {"req_id": "infra", "error_class": "infra_error"},
                {"req_id": "ok", "prompt_tokens": 100, "output_tokens": 512, "cached_tokens": 0}]
        receipt = SCORE.validate_replay_tokens(rows, {"ok": {"glm_tokens": 100}})
        self.assertEqual(receipt["successful_checked"], 1)
        self.assertEqual(receipt["error_rows_skipped"], 2)

    def test_file_entrypoint_rejects_early_successful_eof_on_complete_real_cohort(self):
        source = ROOT / "evidence/L042"
        if not (source / "summary.json").is_file():
            self.skipTest("historical L042 evidence not installed")
        summary = json.loads((source / "summary.json").read_text())
        rows = [json.loads(line) for line in (source / Path(summary["raw"]).name).read_text().splitlines() if line.strip()]
        rows[0]["output_tokens"] -= 1  # Same full cohort, but less actual decode work.
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "replay token contract mismatch: output_tokens"):
                SCORE.score_files(raw, source / Path(summary["run"]).name)

    def test_interval_boundary_is_diagnostic_not_a_tpot_waiver(self):
        for n, expected in ((20, (3, 2, 3)), (314, (22, 22, 23)), (328, (23, 22, 24)), (388, (27, 26, 27))):
            self.assertEqual((SCORE.allowed_over(n), SCORE.alternative_allowed(n, "wilson"),
                              SCORE.alternative_allowed(n, "wald")), expected)
        sensitivity = SCORE.interval_sensitivity(23, 314)
        self.assertTrue(sensitivity["method_sensitive"])
        self.assertFalse(sensitivity["methods"]["clopper_pearson"]["pass_estimated"])
        self.assertTrue(sensitivity["methods"]["wald"]["pass_estimated"])


class ShellTest(unittest.TestCase):
    def test_ladder_preserves_failures_and_invalid_on_descent(self):
        body = (ROOT / "scripts/pod/jobs/dev_ladder_template.sh").read_text().split("lvl=0\n", 1)[1]
        for up, down, statuses, expected in (("22 26", "", {22: 0, 26: 1}, 1),
                                            ("22", "18", {22: 1, 18: 2}, 2),
                                            ("22", "18 14", {22: 1, 18: 1, 14: 0}, 0),
                                            ("22", "18 14", {22: 1, 18: 1, 14: 1}, 1)):
            function = 'run_level() { case "$1" in ' + ' '.join(f'{n}) return {rc};;' for n, rc in statuses.items()) + ' esac; }\n'
            script = function + f"UP={shlex.quote(up)}; DOWN={shlex.quote(down)}; lvl=0\n" + body
            proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(proc.returncode, expected, proc.stdout + proc.stderr)

    def test_reused_engine_gets_live_log_without_starting_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run").mkdir()
            (root / "live.log").write_text("first\n")
            (root / "engine_log_path").write_text(str(root / "live.log") + "\n")
            script = '''source "$LIB"
curl() { return 0; }
start_engine() { echo UNEXPECTED_ENGINE_START; return 99; }
envkey=$(env | grep -E '^(SGLANG_AX_|SGLANG_ARENA_|NCCL_|SGLANG_MAMBA|SGLANG_OPT_)' | sort | tr '\\n' ' ')
printf '%s\\n' "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa |  | $envkey" > "$AX/engine.sig"
ensure_engine aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
'''
            env = {**os.environ, "AX": str(root), "RUN_DIR": str(root / "run"), "LIB": str(ROOT / "scripts/pod/lib.sh")}
            proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertNotIn("UNEXPECTED", proc.stdout)
            (root / "live.log").write_text("first\nsecond\n")
            self.assertIn("second", (root / "run/server.log").read_text())
            (root / "engine_log_path").unlink()
            proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertNotIn("UNEXPECTED", proc.stdout)


if __name__ == "__main__":
    unittest.main()
