#!/usr/bin/env python3
"""CPU/unit checks for E1's token-only predictor and strict flush parsing."""
import io
import unittest
from unittest.mock import patch

from replay_chains import flush_cache, lcp, predict_policy


class ReplayTests(unittest.TestCase):
    def test_lcp(self):
        self.assertEqual(lcp([1, 2], [1, 3]), 1)
        self.assertEqual(lcp([1], [1, 2]), 1)
        self.assertEqual(lcp([], []), 0)

    def test_role_tail_diverges(self):
        a = [1] * 128 + [9] + [2] * 71
        b = a[:128] + [8] + [3] * 71
        stock = predict_policy([a, b], {9}, chunk_size=8192)
        role = predict_policy([a, b], {9}, "role_conservative")
        self.assertEqual(stock[1]["cached_tokens"], 0)
        self.assertEqual(stock[1]["branch_boundary"], 128)
        self.assertEqual(role[1]["cached_tokens"], 128)
        self.assertEqual(role[0]["checkpoints_added"], [128, 192])

    def test_chunk_boundary_is_strict(self):
        p = list(range(256))
        stock = predict_policy([p, p + [900]], set(), chunk_size=128)
        self.assertEqual(stock[0]["checkpoints_added"], [128, 256])
        self.assertEqual(stock[1]["cached_tokens"], 256)

    def test_path_identity(self):
        a, b = [1] * 200, [2] * 200
        stock = predict_policy([a, b], set())
        self.assertEqual(stock[1]["cached_tokens"], 0)

    def test_branch_conflict(self):
        a = list(range(1000, 1200))
        b = a[:128] + [3] * 64 + [9] + [4] * 63
        conservative = predict_policy([a, b], {9}, "role_conservative")
        all_points = predict_policy([a, b], {9}, "role_all")
        self.assertEqual(conservative[1]["checkpoints_added"], [128])
        self.assertEqual(all_points[1]["checkpoints_added"], [192, 256, 128])

    def test_invalid_alignment(self):
        with self.assertRaises(ValueError):
            predict_policy([], set(), chunk_size=100)

    def test_flush(self):
        for body, require_json, expected in [
            (b"Cache flushed.\nDetails", False, True),
            (b"Cache flushed.\nDetails", True, False),
            (b'{"success":true}', True, True),
            (b'{"success":false}', False, False),
            (b'{"success":"true"}', False, False),
            (b"something else", False, False),
        ]:
            with self.subTest(body=body, require_json=require_json):
                response = io.BytesIO(body)
                response.status = 200
                with patch("urllib.request.urlopen", return_value=response):
                    if expected:
                        self.assertTrue(flush_cache("http://localhost", require_json)["success"])
                    else:
                        with self.assertRaises(RuntimeError):
                            flush_cache("http://localhost", require_json)


if __name__ == "__main__":
    unittest.main()
