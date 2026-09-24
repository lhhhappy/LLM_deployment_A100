"""Offline checks for the repeated baseline diagnostic entry point."""

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from urllib.parse import urlsplit
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pod/verify/numcheck_baseline_diag.py"


class BaselineDiagnosticTest(unittest.TestCase):
    def execute(self, fail=None):
        calls = []
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "diag"

            def urlopen(req, timeout):
                path = urlsplit(req.full_url).path.rsplit("/", 1)[-1]
                body = json.loads(req.data)
                calls.append((path, body))
                if path == "flush_cache":
                    self.assertEqual(urlsplit(req.full_url).query, "timeout=30")
                    if fail == "flush":
                        raise OSError("flush failed")
                    return io.BytesIO(b"{}" if fail == "flush_ack" else b'{"success":true}')
                if fail == "request":
                    raise OSError("generate failed")
                n = body["sampling_params"]["max_new_tokens"]
                if fail == "short":
                    n -= 1
                result = {"meta_info": {
                    "cached_tokens": 0,
                    "output_token_logprobs": [[-0.1, 42] for _ in range(n)],
                    "output_top_logprobs": [[[-0.1, i, None] for i in range(10)] for _ in range(n)],
                }}
                return io.BytesIO(json.dumps(result).encode())

            with (mock.patch.object(sys, "argv", [str(SCRIPT), str(out)]),
                  mock.patch.dict(os.environ, {"VOCAB_MAX": "150000"}),
                  mock.patch("urllib.request.urlopen", side_effect=urlopen),
                  contextlib.redirect_stdout(stdout)):
                if fail:
                    with self.assertRaises((OSError, RuntimeError)):
                        runpy.run_path(str(SCRIPT), run_name="__main__")
                else:
                    runpy.run_path(str(SCRIPT), run_name="__main__")
            summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else None
            failure = json.loads((out / "failure.json").read_text()) if (out / "failure.json").exists() else None
            lines = (out / "responses.jsonl").read_text().splitlines() if (out / "responses.jsonl").exists() else []
            return calls, stdout.getvalue(), summary, [json.loads(line) for line in lines], failure

    def test_default_cases_are_fixed_and_complete(self):
        calls, stdout, summary, records, failure = self.execute()
        self.assertEqual(len(records), 16)
        self.assertEqual(sum(path == "flush_cache" for path, _ in calls), 16)
        self.assertEqual(len({r["prompt_sha256"] for r in records if r["case"] == "cold_37"}), 1)
        self.assertEqual(len({r["prompt_sha256"] for r in records if r["case"] == "text_short"}), 1)
        for name in ("cold_37", "cold_256", "cold_1024", "text_short"):
            prompts = [r["request"].get("input_ids", r["request"].get("text"))
                       for r in records if r["case"] == name]
            self.assertTrue(all(prompt == prompts[0] for prompt in prompts), name)
        self.assertEqual([(r["case"], r["max_new_tokens"]) for r in records[:3]],
                         [("cold_37", 1)] * 3)
        self.assertEqual(summary["requests"], 16)
        self.assertIsNone(failure)
        self.assertIn("DIAGNOSTIC_COMPLETE", stdout)

    def test_flush_failure_stops_without_summary(self):
        calls, stdout, summary, records, failure = self.execute("flush")
        self.assertEqual([path for path, _ in calls], ["flush_cache"])
        self.assertEqual(records, [])
        self.assertIsNone(summary)
        self.assertIsNone(failure)
        self.assertNotIn("DIAGNOSTIC_COMPLETE", stdout)

    def test_generate_failure_stops_without_summary(self):
        _, stdout, summary, records, failure = self.execute("request")
        self.assertEqual(records, [])
        self.assertIsNone(summary)
        self.assertIsNone(failure)
        self.assertNotIn("DIAGNOSTIC_COMPLETE", stdout)

    def test_missing_flush_ack_stops_before_generate(self):
        calls, _, summary, records, _ = self.execute("flush_ack")
        self.assertEqual([path for path, _ in calls], ["flush_cache"])
        self.assertEqual(records, [])
        self.assertIsNone(summary)

    def test_short_response_stops_without_summary(self):
        _, stdout, summary, records, failure = self.execute("short")
        self.assertEqual(records, [])
        self.assertIsNone(summary)
        self.assertEqual(failure["case"], "cold_37")
        self.assertIn("truncated diagnostic output", failure["error"])
        self.assertIn("output_token_logprobs", failure["raw_response"])
        self.assertNotIn("DIAGNOSTIC_COMPLETE", stdout)

    def test_divergence_summary_excludes_changed_history(self):
        summarize = runpy.run_path(str(SCRIPT))["summarize"]
        def record(repeat, tokens, logprobs):
            return {"case": "sample", "max_new_tokens": 3, "repeat": repeat,
                    "response": {"meta_info": {
                        "output_token_logprobs": [[lp, token] for lp, token in zip(logprobs, tokens)],
                        "output_top_logprobs": [[[0, token, None]] for token in tokens],
                    }}}
        rows = summarize([record(1, [1, 2, 3], [-1, -2, -3]),
                          record(2, [1, 9, 3], [-1.1, -2.2, -30]),
                          record(3, [8, 2, 3], [-10, -2, -3])])
        first, second = rows[0]["comparisons_to_first"]
        self.assertEqual(first["first_token_divergence"], 1)
        self.assertAlmostEqual(first["max_abs_logprob_delta_before_divergence"], 0.1)
        self.assertEqual(first["max_abs_logprob_delta"], 27)
        self.assertIsNone(second["max_abs_logprob_delta_before_divergence"])


if __name__ == "__main__":
    unittest.main()
