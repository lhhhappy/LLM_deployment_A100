import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.analysis import longchain_replay as replay


class ReplayGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "data"
        self.root.mkdir()
        (self.root / "manifest.json").write_text(json.dumps({"generator": "test", "set": "test-set"}))
        self.args = argparse.Namespace(root=self.root, out=self.base / "new-run", tok_dir=self.base / "tok",
                                       self_check=True, n=6, base_url="", api_key="", model="model", skip_warmup=False)

    def valid(self):
        return {"status": "VALID", "errors": [], "counts": {
            "expected_ids": 2, "serving_request_ids": 2, "matched_bodies": 2, "rendered_prompts": 2}}

    def test_no_harness_invocation_when_body_missing_or_render_incomplete(self):
        for key in ("matched_bodies", "rendered_prompts"):
            report = self.valid()
            report["counts"][key] = 1
            self.args.out = self.base / key
            with patch.object(replay, "check_dataset", return_value=report), patch.object(replay.subprocess, "run") as run:
                self.assertEqual(replay.execute(self.args), 2)
                run.assert_not_called()

    def test_structural_only_or_invalid_is_rejected(self):
        for status in ("STRUCTURAL_OK", "INVALID"):
            report = self.valid()
            report["status"] = status
            self.args.out = self.base / status
            with patch.object(replay, "check_dataset", return_value=report), patch.object(replay.subprocess, "run") as run:
                self.assertEqual(replay.execute(self.args), 2)
                run.assert_not_called()

    def test_nonempty_errors_rejected_even_when_status_claims_valid(self):
        report = self.valid()
        report["errors"] = ["hash mismatch"]
        with patch.object(replay, "check_dataset", return_value=report), patch.object(replay.subprocess, "run") as run:
            self.assertEqual(replay.execute(self.args), 2)
            run.assert_not_called()

    def test_existing_output_and_output_inside_dataset_rejected(self):
        for out in (self.base, self.root / "new-out", replay.HARNESS / "must-not-create"):
            self.args.out = out
            with patch.object(replay, "check_dataset") as check, patch.object(replay.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    replay.execute(self.args)
                check.assert_not_called()
                run.assert_not_called()

    def test_cpu_selfcheck_bypasses_original_cache_and_no_engine_or_scorer(self):
        with patch.object(replay, "check_dataset", return_value=self.valid()) as check, \
             patch.object(replay.subprocess, "run", return_value=argparse.Namespace(returncode=0)) as run:
            self.assertEqual(replay.execute(self.args), 0)
            check.assert_called_once_with(str(self.root), str(replay.HARNESS), str(self.args.tok_dir), str(self.root / "cohort.json"))
            command = run.call_args.args[0]
            self.assertIn("--self-check", command)
            self.assertIn("--no-body-cache", command)
            self.assertNotIn("--base-url", command)
            self.assertEqual(run.call_count, 1)

    def test_measured_path_reuses_checked_runner_and_scores_current_dataset(self):
        self.args.self_check = False
        self.args.base_url = "http://localhost:1234/v1"
        self.args.api_key = "secret-not-in-logs"
        self.args.skip_warmup = True
        with patch.object(replay, "check_dataset", return_value=self.valid()), \
             patch.dict(replay.os.environ, {"S1_SKIP_PREFLIGHT": "1", "S1_SKIP_WARMUP": "1", "S1_FLUSH_URL": "wrong-service"}), \
             patch.object(replay.subprocess, "run", return_value=argparse.Namespace(returncode=0)) as run:
            self.assertEqual(replay.execute(self.args), 0)
            command, score = [x.args[0] for x in run.call_args_list]
            self.assertIn(str(replay.CHECKED_RUNNER), command)
            self.assertIn(str(replay.RUNNER), command)
            self.assertIn("--skip-warmup", command)
            self.assertNotIn("--skip-preflight", command)
            self.assertNotIn(self.args.api_key, command)
            env = run.call_args_list[0].kwargs["env"]
            self.assertEqual(env["S1_HARNESS_DIR"], str(replay.HARNESS))
            self.assertEqual(env["S1_SKIP_PREFLIGHT"], "0")
            self.assertEqual(env["S1_SKIP_WARMUP"], "0")
            self.assertNotIn("S1_FLUSH_URL", env)
            self.assertEqual(env["S1_API_KEY"], self.args.api_key)
            self.assertEqual(score[score.index("--requests") + 1], str(self.root / "requests.jsonl"))
        for path in self.args.out.iterdir():
            self.assertNotIn(self.args.api_key, path.read_text())

    def test_runner_failure_never_scores_partial_measurement(self):
        self.args.self_check = False
        self.args.base_url = "http://localhost:1234"
        with patch.object(replay, "check_dataset", return_value=self.valid()), \
             patch.object(replay.subprocess, "run", return_value=argparse.Namespace(returncode=2)) as run:
            self.assertEqual(replay.execute(self.args), 2)
            self.assertEqual(run.call_count, 1)

    def test_manifest_change_during_validation_rejected(self):
        def mutate(*_):
            (self.root / "manifest.json").write_text('{"changed":true}')
            return self.valid()
        with patch.object(replay, "check_dataset", side_effect=mutate), patch.object(replay.subprocess, "run") as run:
            self.assertEqual(replay.execute(self.args), 2)
            run.assert_not_called()

    def test_actual_checker_rejects_missing_body_before_original_selfcheck(self):
        # Missing expected request body must not reach the original false-PASS branch.
        row = {"pack": "p", "view": "canon", "logical_call_id": "q", "chain_id": "c", "in_serving_load": True,
               "glm_tokens": 3, "max_output_i": 2, "replay_gap_ms": 0}
        (self.root / "requests.jsonl").write_text(json.dumps(row) + "\n")
        (self.root / "chains.jsonl").write_text(json.dumps({"view": "canon", "chain_id": "c", "n_requests": 1}) + "\n")
        (self.root / "cohort.json").write_text(json.dumps({"chains": [{"chain_id": "c", "req_ids": ["p:canon:q"]}]}))
        with patch.object(replay.subprocess, "run") as run:
            self.assertEqual(replay.execute(self.args), 2)
            run.assert_not_called()
        report = json.loads((self.args.out / "dataset-validation.json").read_text())
        self.assertTrue(any("has no body" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
