#!/usr/bin/env python3
"""CPU checks for T29 diagnostic accounting, not model correctness."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from e2_completion_checks import merge_cases, summarize_rounds, pool_verdict


class CompletionTools(unittest.TestCase):
    def test_pool_pass_requires_complete_workload(self):
        baseline = {'kv_free': 42, 'mamba_free': 7, 'request_free': 2}
        final = {'success': True, 'after': baseline}
        self.assertEqual(pool_verdict(False, baseline, baseline, final), 'blocked')
        self.assertEqual(pool_verdict(True, baseline, baseline, final), 'pass')
        self.assertEqual(pool_verdict(True, baseline, baseline,
            {'success': True, 'after': {**baseline, 'mamba_free': 6}}), 'fail')

    def test_union_keeps_longest_original_prefix(self):
        a = {'chain_id': 'a', 'prefix_len': 1, 'req_ids': ['a0']}
        long_a = {'chain_id': 'a', 'prefix_len': 2, 'req_ids': ['a0', 'a1']}
        b = {'chain_id': 'b', 'prefix_len': 1, 'req_ids': ['b0']}
        self.assertEqual(merge_cases([a, b], [long_a]), [long_a, b])
        self.assertEqual(a['req_ids'], ['a0'])

    def test_conflicting_prefix_rejected(self):
        with self.assertRaises(Exception):
            merge_cases([{'chain_id': 'a', 'req_ids': ['x']}],
                        [{'chain_id': 'a', 'req_ids': ['y']}])

    def test_counter_denominators_not_hidden(self):
        row = {'kind': 'round', 'role_delta': {'taken': 1}, 'attempts': 3,
               'partial_count': 1, 'batch_rids': ['a', 'b'],
               'commits': [{'kind': 'new', 'partial': True}, {'kind': 'continuation', 'partial': False}]}
        summary = summarize_rounds([row])
        self.assertEqual(summary['role_stats_sum'], 1)
        self.assertEqual(summary['new_commits'], 1)
        self.assertEqual(summary['all_commits'], 2)
        self.assertEqual(summary['new_admission_attempts'], 3)

    def test_round_tap_returns_original_result(self):
        path = Path(__file__).with_name('e2_completion_trace.py')
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prefill')
        receipts = []
        ns = {'ACTIVE': None, 'ROUND': 0, 'policy': NS(ROLE_BOUNDARY_STATS={}),
              'emit': lambda kind, **fields: receipts.append({'kind': kind, **fields})}
        result = NS(batch_to_run=NS(reqs=[NS(rid='a'), NS(rid='b')]))
        def original(self):
            ns['ACTIVE']['commits'].extend([
                {'rid': 'a', 'kind': 'continuation', 'partial': False},
                {'rid': 'b', 'kind': 'new', 'partial': True}])
            return result
        ns['_prefill'] = original
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        self.assertIs(ns['prefill'](None), result)
        self.assertIsNone(ns['ACTIVE'])
        self.assertEqual(receipts[0]['partial_count'], 1)
        self.assertEqual(receipts[0]['batch_rids'], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
