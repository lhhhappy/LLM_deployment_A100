"""CPU regressions for the numeric probe's failure propagation and output."""

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock


PROBE = Path(__file__).resolve().parents[1] / "scripts/pod/verify/numcheck.py"


class Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()


class NumcheckProbeTest(unittest.TestCase):
    def run_probe(self, failure=None):
        requests = []
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "numcheck.json"

            def urlopen(request, timeout):
                path = request.full_url.rsplit("/", 1)[-1]
                requests.append(path)
                if path == "flush_cache":
                    if failure == "flush":
                        raise OSError("mock flush failure")
                    return Response({"success": True})

                self.assertEqual(path, "generate")
                body = json.loads(request.data)
                length = len(body.get("input_ids", []))
                if failure == "mix" and length == 700:
                    raise OSError("mock mixed request failure")
                count = body["sampling_params"]["max_new_tokens"]
                if failure == "short" and length == 37:
                    count -= 1
                return Response({"meta_info": {
                    "output_token_logprobs": [[-0.25, 42] for _ in range(count)],
                    "cached_tokens": 0,
                    "prompt_tokens": length,
                }})

            with (
                mock.patch.object(sys, "argv", [str(PROBE), directory, "37"]),
                mock.patch.dict(os.environ, {"NUMCHECK_PREFIX_LEN": "64"}),
                mock.patch("urllib.request.urlopen", side_effect=urlopen),
                mock.patch("time.sleep"),
                contextlib.redirect_stdout(output),
            ):
                if failure is None:
                    runpy.run_path(str(PROBE), run_name="__main__")
                else:
                    with self.assertRaisesRegex(
                        (RuntimeError, OSError),
                        {"flush": "could not flush cache",
                         "mix": "mixed request failure",
                         "short": "truncated probe output"}[failure],
                    ):
                        runpy.run_path(str(PROBE), run_name="__main__")

            data = json.loads(result_file.read_text()) if result_file.exists() else None
            return requests, output.getvalue(), data

    def test_complete_output(self):
        requests, stdout, data = self.run_probe()
        self.assertEqual(requests.count("flush_cache"), 2)
        self.assertEqual(len(data), 8)
        self.assertEqual(set(data), {
            "cold_37", "hit_64+333", "hit_64+512", "mix_a_700", "mix_b_1500",
            "text_short", "text_mid", "text_long",
        })
        self.assertTrue(all(len(case["tokens"]) == len(case["logprobs"]) == 48
                            for case in data.values()))
        self.assertIn("NUMCHECK_DONE 8", stdout)

    def test_flush_retry_exhaustion_fails(self):
        requests, stdout, data = self.run_probe("flush")
        self.assertEqual(requests, ["flush_cache"] * 30)
        self.assertNotIn("NUMCHECK_DONE", stdout)
        self.assertIsNone(data)

    def test_mixed_request_failure_propagates(self):
        requests, stdout, data = self.run_probe("mix")
        self.assertIn("generate", requests)
        self.assertNotIn("NUMCHECK_DONE", stdout)
        self.assertIsNone(data)

    def test_short_output_fails(self):
        requests, stdout, data = self.run_probe("short")
        self.assertIn("generate", requests)
        self.assertNotIn("NUMCHECK_DONE", stdout)
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
