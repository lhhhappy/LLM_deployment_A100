#!/usr/bin/env python3
"""CPU checks for E2 joins, harness quantile, pair selection and strict tolerance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from compare_e2 import compare, quantile, rows


class JoinTests(unittest.TestCase):
    def test_quantile_matches_harness_at_integer_rank(self):
        self.assertEqual(quantile(list(range(20))), 19)
        self.assertIsNone(quantile([]))

    def test_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rows.jsonl'
            path.write_text('{"req_id":"x"}\n' * 2)
            with self.assertRaises(AssertionError):
                rows(path)

    def test_join_and_mismatched_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row = dict(req_id='r1', chain_id='c', idx_in_chain=1, phase='intra',
                       uncached_expected=1, prompt_sha256='h', prompt_tokens=1000,
                       rendered_prompt_tokens=1000, output_tokens=4,
                       predictions={'stock': {'cached_tokens': 100},
                                    'role_conservative': {'cached_tokens': 800}})
            for v, cached in [('stock', 100), ('off', 100), ('on', 800)]:
                target = root / (v + '_test')
                target.mkdir()
                (target / 'requests.jsonl').write_text(json.dumps(dict(row, cached_tokens=cached)) + '\n')
            result = compare(root, 'test')['summary']
            self.assertEqual(result['off_equals_stock'], 1)
            self.assertEqual(result['on_better'], 1)
            self.assertEqual(result['fast_uncached_p95']['on'], 200)
            (root / 'on_test/requests.jsonl').write_text(json.dumps(dict(row, cached_tokens=800, prompt_sha256='bad')))
            with self.assertRaises(AssertionError):
                compare(root, 'test')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'remote E1 env supplies torch')
class NumericTests(unittest.TestCase):
    def test_zero_noise_does_not_hide_warm_drift(self):
        import torch
        from e2_raw_logits import compare_tensors
        cold = torch.zeros((1, 8))
        result = compare_tensors(cold, cold, cold + 0.001)
        self.assertEqual(result['tolerance_2x_cold_noise'], 0)
        self.assertFalse(result['logits_within_declared_tolerance'])
        self.assertEqual(result['different_vocab_entries'], 8)

    def test_select_distinct_improved_chains(self):
        from e2_raw_logits import select_pairs
        data = [{'chain_id': c, 'on_cached': x, 'stock_cached': 1}
                for c, x in [('z', 0), ('a', 2), ('a', 3), ('b', 2), ('c', 2)]]
        self.assertEqual([r['chain_id'] for r in select_pairs({'reminder_heavy': {'requests': data}})],
                         ['a', 'b', 'c'])
        with self.assertRaises(RuntimeError):
            select_pairs({'reminder_heavy': {'requests': data[:3]}})


if __name__ == '__main__':
    unittest.main()
