#!/usr/bin/env python3
"""D1-07 CPU tests of the ACTUAL patched method, AST-isolated from torch/CUDA.

No translated implementation: compile _PrefillAdmission and the method from
build/d1/b, plus the actual checkpoint-grid helper from read-only runtime_context.
Fake only request/cache/adder fields and the configured model grid (64).
Patch-to-working-copy parity is asserted so a stale working copy cannot pass.
Run: python3 -B -m unittest discover -s scripts -p test_role_boundary_split.py -v
"""
import ast
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'build/d1/b/python/sglang/srt/managers/schedule_policy.py'
PATCH = REPO / 'patches/001-role-boundary-mamba-ckpt.patch'


def load_method():
    tree = ast.parse(SOURCE.read_text())
    admission = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_PrefillAdmission')
    adder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PrefillAdder')
    method = next(n for n in adder.body if isinstance(n, ast.FunctionDef) and n.name == '_maybe_role_boundary_split')
    runtime = ast.parse((REPO / 'src/sglang/python/sglang/srt/runtime_context.py').read_text())
    grid = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == 'mamba_checkpoint_grid')
    # Avoid importing annotations or sglang dependencies, preserving the method body.
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             admission, grid, method], type_ignores=[])
    namespace = {'dataclass': dataclass, 'math': math, 'mamba_cache_chunk_size': lambda: 64,
                 '_role_boundary_token_ids': lambda: frozenset({99}),
                 'ROLE_BOUNDARY_STATS': Counter(), '_ROLE_BOUNDARY_SCAN_WINDOW': 32768}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    return namespace, method


class RoleBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.method_ast = load_method()

    def setUp(self):
        self.ns['ROLE_BOUNDARY_STATS'].clear()
        self.ns['_ROLE_BOUNDARY_SCAN_WINDOW'] = 32768
        self.ns['_role_boundary_token_ids'] = lambda: frozenset({99})
        self.adder = SimpleNamespace(dllm_config=None, new_chunked_req=None, rem_chunk_tokens=1024,
            tree_cache=SimpleNamespace(page_size=64, supports_mamba=lambda: True), _arena_truncation_align=None)
        self.req = SimpleNamespace(full_untruncated_fill_ids=[1] * 512, mamba_branching_seqlen=None,
                                   sampling_params=SimpleNamespace(max_new_tokens=240))
        self.req.full_untruncated_fill_ids[350] = 99
        self.admission = self.ns['_PrefillAdmission'](128, 384, 240, False)

    def invoke(self, active=False):
        return self.ns['_maybe_role_boundary_split'](self.adder, self.req, self.admission, active)

    def skipped(self, key, active=False):
        self.assertIsNone(self.invoke(active))
        self.assertEqual(self.ns['ROLE_BOUNDARY_STATS'], Counter({key: 1}))
        self.assertEqual(self.req.sampling_params.max_new_tokens, 240)

    def test_patch_matches_working_method(self):
        added = '\n'.join(line[1:] for line in PATCH.read_text().splitlines()
                          if line.startswith('+') and not line.startswith('+++'))
        start = added.index('    def _maybe_role_boundary_split(')
        patch_method = ast.parse('class PrefillAdder:\n' + added[start:]).body[0].body[0]
        self.assertEqual(ast.dump(patch_method), ast.dump(self.method_ast))

    def test_no_boundary(self):
        self.req.full_untruncated_fill_ids[350] = 1
        self.skipped('skipped_no_boundary')

    def test_short_prompt_less_than_two_grids(self):
        for size, boundary in ((63, 30), (127, 90)):
            with self.subTest(size=size):
                self.ns['ROLE_BOUNDARY_STATS'].clear()
                self.req.full_untruncated_fill_ids = [1] * size
                self.req.full_untruncated_fill_ids[boundary] = 99
                self.admission = self.ns['_PrefillAdmission'](0, size, 240, False)
                self.skipped('skipped_short')

    def test_unaligned_prefix(self):
        self.admission = self.ns['_PrefillAdmission'](129, 383, 240, False)
        self.skipped('skipped_unaligned_prefix')

    def test_branch_conflict(self):
        self.req.mamba_branching_seqlen = 192
        self.skipped('skipped_branch_conflict')

    def test_active_chunk(self):
        self.skipped('skipped_active_chunk', active=True)

    def test_new_chunked_req(self):
        self.adder.new_chunked_req = object()
        self.skipped('skipped_active_chunk')

    def test_no_chunk_budget(self):
        self.adder.rem_chunk_tokens = None
        self.skipped('skipped_no_chunked_prefill')

    def test_already_chunked_and_dllm(self):
        self.admission = self.ns['_PrefillAdmission'](128, 384, 0, True)
        self.skipped('skipped_already_chunked')
        self.ns['ROLE_BOUNDARY_STATS'].clear()
        self.admission = self.ns['_PrefillAdmission'](128, 384, 240, False)
        self.adder.dllm_config = object()
        self.skipped('skipped_already_chunked')

    def test_normal_split_and_sampling_budget_unchanged(self):
        result = self.invoke()
        self.assertEqual((result.prefix_len, result.extend_len, result.max_new_tokens, result.is_chunked),
                         (128, 192, 0, True))  # floor((350-128)/64)*64
        self.assertEqual(self.req.sampling_params.max_new_tokens, 240)
        self.assertEqual(self.admission.extend_len, 384)
        self.assertEqual(self.ns['ROLE_BOUNDARY_STATS'], Counter(taken=1))

    def test_scan_window_excludes_old_boundary_and_includes_lower_edge(self):
        self.ns['_ROLE_BOUNDARY_SCAN_WINDOW'] = 128  # lower bound = 384
        self.skipped('skipped_no_boundary')
        self.ns['ROLE_BOUNDARY_STATS'].clear()
        self.req.full_untruncated_fill_ids[384] = 99
        self.assertEqual(self.invoke().extend_len, 256)
        self.assertEqual(self.ns['ROLE_BOUNDARY_STATS'], Counter(taken=1))

    def test_scan_uses_last_boundary_not_first(self):
        self.req.full_untruncated_fill_ids[420] = 99
        self.assertEqual(self.invoke().extend_len, 256)

    def test_boundary_at_stock_end_checkpoint_skipped(self):
        self.req.full_untruncated_fill_ids = [1] * 530
        self.req.full_untruncated_fill_ids[525] = 99
        self.admission = self.ns['_PrefillAdmission'](128, 402, 240, False)
        self.skipped('skipped_short')

    def test_grid_and_truncation_lcm(self):
        self.adder.tree_cache.page_size = 128
        self.adder._arena_truncation_align = 192
        self.admission = self.ns['_PrefillAdmission'](0, 1024, 240, False)
        self.req.full_untruncated_fill_ids = [1] * 1024
        self.req.full_untruncated_fill_ids[700] = 99
        self.assertEqual(self.invoke().extend_len, 384)

    def test_disabled_and_non_mamba_are_noops(self):
        self.ns['_role_boundary_token_ids'] = lambda: frozenset()
        self.assertIsNone(self.invoke())
        self.ns['_role_boundary_token_ids'] = lambda: frozenset({99})
        self.adder.tree_cache.supports_mamba = lambda: False
        self.assertIsNone(self.invoke())
        self.assertEqual(self.ns['ROLE_BOUNDARY_STATS'], Counter())


if __name__ == '__main__':
    unittest.main()
