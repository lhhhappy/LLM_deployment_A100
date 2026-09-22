#!/usr/bin/env python3
"""D2 CPU tests: compile actual production classes, fake only runtime dependencies.

No torch/CUDA import. Patch chain parity, real selection/commit/budget methods,
and stock FCFS differential tests. Run with python3 -B -m unittest discover
-s scripts -p test_spf_scheduling.py -v.
"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from typing import Union
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
REL = Path('python/sglang/srt/managers/schedule_policy.py')
SOURCE = ROOT / 'build/d2/b' / REL


def load_source(path=SOURCE):
    """Preserve class/method ASTs; replace imports/hardware/runtime globals only."""
    names = {'CacheAwarePolicy', 'CacheAgnosticPolicy', 'SchedulePolicy',
             'AddReqResult', '_PrefillAdmission', 'PrefillAdder', '_ceil_div',
             'estimate_prefill_extend_tile_metrics'}
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, 'name', '') in names]
    mod = ModuleType('spf_test_' + str(len(sys.modules)))
    sys.modules[mod.__name__] = mod
    ns = mod.__dict__
    real_import = os.environ.get('D2_REAL_IMPORT') == '1'
    if real_import:
        # GPU-box CPU mode: import the complete module against its real pinned
        # dependencies before substituting the same runtime/cache fakes.
        exec(compile(path.read_text(), str(path), 'exec'), ns)
    ns.update(Enum=Enum, auto=auto, dataclass=dataclass, Union=Union, math=math,
              contextmanager=contextmanager, Counter=Counter, random=random,
              logger=logging.getLogger('spf-test'), _use_exact_chunk_fill=lambda: False,
              is_swa_req_ring=lambda _: False, _IS_HIP=False, PREFILL_TILE_BUDGET=0,
              PREFILL_TILE_BUDGET_MODE='compact', CLIP_MAX_NEW_TOKENS=4096,
              _role_boundary_token_ids=lambda: frozenset(), ROLE_BOUNDARY_STATS=Counter(),
              _ROLE_BOUNDARY_SCAN_WINDOW=32768, mamba_checkpoint_grid=lambda p: math.lcm(p, 64),
              InitLoadBackParams=lambda **kw: NS(**kw), torch=NS(cat=lambda xs: sum(xs, [])),
              get_schedule=lambda: NS(schedule_policy='shortest-prefill-first'),
              get_disagg=lambda: NS(disaggregation_mode='null'),
              RadixCache=NS(create_simulated=lambda: None),
              split_cached_prefix_by_tier=lambda **kw: (kw['prefix_len'], 0, 0))
    for name in ('SWATokenToKVPoolAllocator', 'DeepSeekV4HiSparseTokenToKVPoolAllocator',
                 'PureSWATokenToKVPoolAllocator', 'UnifiedMambaTokenToKVPoolAllocator',
                 'UnifiedMambaSWATokenToKVPoolAllocator'):
        ns[name] = type(name, (), {})
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    if not real_import:
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns


def request(rid, work, cached=0, host=0, arrived=0, boundary=None):
    ids = [1] * (work + cached)
    if boundary is not None:
        ids[boundary] = 99
    req = NS(rid=rid, origin_input_ids=ids[:], output_ids=[],
             full_untruncated_fill_ids=ids, prefix_indices=[0] * cached,
             num_matched_prefix_tokens=cached + host, host_hit_length=host,
             swa_host_hit_length=0, storage_hit_length=0, storage_hit_start=None,
             host_hit_is_storage=False, host_loaded_length=0, mamba_branching_seqlen=None,
             sampling_params=NS(ignore_eos=False, max_new_tokens=4),
             time_stats=NS(wait_queue_entry_time=arrived), retracted_stain=False,
             kv=NS(holds_mamba=False, cache_protected_len=cached),
             last_node=object(), best_match_node=None, cache_request_handle=None,
             materialized_host_hit_len=lambda: 0, fulfilled_storage_hit_len=lambda _: 0,
             needs_host_load_back=lambda: bool(host), priority=0)
    def set_range(start, end):
        req.extend_range = NS(start=start, end=end, length=end-start)
        req.extend_input_len = end-start
    req.set_extend_range = MagicMock(side_effect=set_range)
    return req


def adder(ns, budget=4096, page=64, available=1000000, mamba=False):
    tree = MagicMock()
    tree.disable = False
    tree.page_size = page
    tree.supports_mamba.return_value = mamba
    tree.is_tree_cache.return_value = False
    tree.supports_fast_match_prefix.return_value = False
    for name in ('evictable_size', 'full_evictable_size', 'mamba_evictable_size'):
        getattr(tree, name).return_value = 0
    allocator = MagicMock()
    allocator.available_size.return_value = available
    obj = ns['PrefillAdder'](page, tree, allocator, NS(reqs=[]), 1.0, 65536, budget)
    return obj


def policy(ns, name='shortest-prefill-first'):
    cache = NS(disable=False, supports_fast_match_prefix=lambda: False)
    return ns['SchedulePolicy'](name, cache, False, False, False)


class SPFTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_source()
        self.a = adder(self.ns)
        self.p = policy(self.ns)

    def add(self, req, active=False, align=None):
        return self.a.add_one_req(req, active, align)

    def test_order_by_uncached_output_ties_and_deprioritized(self):
        cached = request('cached', 16, cached=4096)
        short = request('short', 32, arrived=2)
        old = request('old', 32, arrived=1)
        duplicate = request('duplicate', 1)
        resumed = request('resumed', 1)
        resumed.output_ids = [1] * 64
        q = [short, old, resumed, duplicate, cached]
        self.p._compute_prefix_matches = MagicMock(return_value={'duplicate'})
        self.p.calc_priority(q)
        self.assertEqual([r.rid for r in q], ['cached', 'old', 'short', 'resumed', 'duplicate'])
        self.assertEqual(self.p._shortest_prefill_work(request('empty', 0)), 1)

    def test_short_waiters_share_continuation_budget(self):
        c = request('c', 16384)
        wait = [request('a', 512), request('b', 1024)]
        self.a.chunked_req_limit = self.p.shortest_prefill_chunk_limit(c, wait, 4096, 64)
        self.assertIs(self.a.add_chunked_req(c), c)
        for req in wait:
            self.add(req, active=True)
        self.assertEqual(c.extend_range.length, 2560)
        self.assertEqual(self.a.can_run_list, [c, *wait])
        self.assertEqual(self.a.rem_chunk_tokens, 0)
        self.assertIsNone(self.a.new_chunked_req)

    def test_minimum_page_no_reservation_and_scan_break(self):
        c = request('c', 10000)
        for size in (1, 63, 64, 65, 3968, 4032, 4033):
            limit = self.p.shortest_prefill_chunk_limit(c, [request('s', size)], 4096, 64)
            if size == 4033:
                self.assertIsNone(limit)
            else:
                self.assertGreaterEqual(limit, 64)
                self.assertEqual(limit % 64, 0)
        self.assertIsNone(self.p.shortest_prefill_chunk_limit(c, [], 4096, 64))
        self.assertIsNone(self.p.shortest_prefill_chunk_limit(c, [request('s', 1)], 64, 64))
        self.assertIsNone(self.p.shortest_prefill_chunk_limit(c, [request('equal', 10000), request('s', 1)], 4096, 64))

    def test_dsa_continuation_alignment(self):
        c = request('c', 10000)
        limit = self.p.shortest_prefill_chunk_limit(c, [request('s', 65)], 4096, 64, 512)
        self.assertEqual(limit, 3584)
        self.a.chunked_req_limit = limit
        self.a.add_chunked_req(c)
        self.assertEqual(c.extend_range.length % 512, 0)
        self.assertEqual(self.p.shortest_prefill_chunk_limit(c, [request('s', 3584)], 4096, 64, 512), 512)

    def test_reject_second_partial_before_delayer_or_load(self):
        self.a.rem_chunk_tokens = 512
        req = request('r', 1024)
        self.a.prefill_delayer_single_pass = MagicMock()
        self.assertEqual(self.add(req, True).name, 'OTHER')
        req.set_extend_range.assert_not_called()
        self.a.prefill_delayer_single_pass.negotiate_should_allow_prefill.assert_not_called()
        self.a.tree_cache.init_load_back.assert_not_called()
        self.assertEqual(self.a.can_run_list, [])

    def test_host_miss_reselect_checks_existing_and_new_partial(self):
        for existing in (True, False):
            self.a = adder(self.ns, budget=512)
            if not existing:
                self.a.new_chunked_req = request('already-created', 1000)
            req = request('r', 1024, host=768)
            self.a.tree_cache.init_load_back.return_value = ([], req.last_node)
            self.assertEqual(self.add(req, existing).name, 'OTHER')
            req.set_extend_range.assert_not_called()
            self.a.tree_cache.init_load_back.assert_called_once()
            self.assertEqual(self.a.can_run_list, [])
            self.assertEqual(self.a.rem_chunk_tokens, 512)

    def test_host_hit_commit_and_miss_without_partial(self):
        for hit in (0, 768):
            self.a = adder(self.ns, budget=512)
            req = request('r', 1024, host=768)
            self.a.tree_cache.init_load_back.return_value = ([0] * hit, req.last_node)
            self.add(req)
            self.assertEqual(req.extend_range.length, 256 if hit else 512)
            self.assertEqual(self.a.new_chunked_req is req, hit == 0)

    def test_d1_shared_guard_skip_split_during_continuation(self):
        self.ns['_role_boundary_token_ids'] = lambda: {99}
        self.a = adder(self.ns, mamba=True)
        req = request('r', 512, boundary=350)
        self.add(req, active=True)
        self.assertEqual(req.extend_range.length, 512)
        self.assertIsNone(self.a.new_chunked_req)
        self.assertEqual(self.ns['ROLE_BOUNDARY_STATS']['skipped_active_chunk'], 1)

    def test_d1_split_then_full_and_no_second_partial(self):
        self.ns['_role_boundary_token_ids'] = lambda: {99}
        self.a = adder(self.ns, budget=1024, mamba=True)
        first = request('first', 512, boundary=350)
        self.add(first)
        self.assertEqual(first.extend_range.length, 320)
        short = request('short', 256, boundary=130)
        self.add(short)
        self.assertEqual(short.extend_range.length, 256)
        long = request('long', 2048)
        self.assertEqual(self.add(long).name, 'OTHER')
        self.assertEqual(self.a.can_run_list, [first, short])
        self.assertIs(self.a.new_chunked_req, first)
        self.assertEqual(first.sampling_params.max_new_tokens, 4)

    def test_d1_on_fcfs_uses_same_host_miss_guard(self):
        self.ns['get_schedule'] = lambda: NS(schedule_policy='fcfs')
        self.ns['_role_boundary_token_ids'] = lambda: {99}
        self.a = adder(self.ns, budget=1024, mamba=True)
        self.add(request('first', 512, boundary=350))
        r = request('miss', 1024, host=768)
        self.a.tree_cache.init_load_back.return_value = ([], r.last_node)
        self.assertEqual(self.add(r).name, 'OTHER')
        r.set_extend_range.assert_not_called()

    def test_memory_mamba_and_page_debits(self):
        self.a = adder(self.ns, budget=512, available=500)
        self.assertEqual(self.add(request('oom', 512)).name, 'NO_TOKEN')
        self.a = adder(self.ns, budget=512, mamba=True)
        self.a._mamba_slot_cost = 128
        self.a.rem_mamba_slots = 2
        req = request('r', 65)
        self.add(req)
        self.assertEqual(self.a.rem_chunk_tokens, 384)
        self.assertEqual(self.a.cur_rem_token_offset, 128 + 64 + 128)
        self.assertEqual(self.a.rem_total_token_offset, 128 + 64 + 128 + 4)
        self.assertEqual(self.a.rem_mamba_slots, 1)

    def test_tile_gate_and_dsa_candidate_alignment(self):
        self.a = adder(self.ns, budget=1000)
        req = request('dsa', 2000)
        self.add(req, align=256)
        self.assertEqual(req.extend_range.length, 768)
        self.a = adder(self.ns, budget=512)
        self.ns['_IS_HIP'] = True
        self.ns['PREFILL_TILE_BUDGET'] = 2
        self.add(request('first', 64))
        req = request('tile-reject', 128)
        self.assertEqual(self.add(req).name, 'OTHER')
        req.set_extend_range.assert_not_called()

    def test_disabled_cache_rejected_and_fcfs_not_reordered(self):
        with self.assertRaises(ValueError):
            self.ns['SchedulePolicy']('shortest-prefill-first', NS(disable=True), False, False, False)
        p = policy(self.ns, 'fcfs')
        q = [request('long', 10000), request('short', 1)]
        p.calc_priority(q)
        self.assertEqual([r.rid for r in q], ['long', 'short'])
        self.assertIsNone(p.shortest_prefill_chunk_limit(q[0], q[1:], 4096, 64))

    def test_fcfs_differential_against_clean_stock(self):
        stock = load_source(ROOT / 'src/sglang' / REL)
        for ns in (stock, self.ns):
            ns['get_schedule'] = lambda: NS(schedule_policy='fcfs')
        def snapshot(ns, budget, lengths, host_miss, continuation):
            a = adder(ns, budget=budget)
            verdicts = []
            if continuation:
                a.add_chunked_req(request('continuation', continuation))
            for i, size in enumerate(lengths):
                r = request(str(i), size, host=min(size//2, 256) if host_miss else 0)
                a.tree_cache.init_load_back.return_value = ([], r.last_node)
                verdicts.append(a.add_one_req(r, False, 64).name)
                if verdicts[-1] != 'CONTINUE':
                    break
            return repr((verdicts, [(r.rid, r.extend_range.start, r.extend_range.end) for r in a.can_run_list],
                         getattr(a.new_chunked_req, 'rid', None), a.rem_chunk_tokens,
                         a.rem_total_token_offset, a.cur_rem_token_offset, a.rem_input_tokens))
        for budget in (None, 64, 512, 4096):
            for lengths in ([1, 63, 64], [256, 1024], [5000], [65, 511, 3000]):
                for host in (False, True):
                    for cont in ((0, 128) if budget else (0,)):
                        with self.subTest(budget=budget, lengths=lengths, host=host, cont=cont):
                            self.assertEqual(snapshot(stock, budget, lengths, host, cont), snapshot(self.ns, budget, lengths, host, cont))

    def test_patch_chain_matches_all_production_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            for path in (ROOT / 'build/d2/a').rglob('*.py'):
                rel = path.relative_to(ROOT / 'build/d2/a')
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / 'src/sglang' / rel, target)
            for name in ('001-role-boundary-mamba-ckpt.patch', '002-spf-scheduling.patch'):
                result = subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(ROOT / 'patches' / name)], cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for expected in (ROOT / 'build/d2/b').rglob('*.py'):
                self.assertEqual((dest / expected.relative_to(ROOT / 'build/d2/b')).read_bytes(), expected.read_bytes())

    def test_preserved_budget_and_commit_methods_are_ast_identical(self):
        def methods(path):
            cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'PrefillAdder')
            return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        base, port = methods(ROOT / 'src/sglang' / REL), methods(SOURCE)
        for name in ('_commit_prefill_admission', '_update_prefill_budget', '_mamba_gap_budget_for_req',
                     '_swa_admission_gate', '_check_prefill_tile_budget', 'rem_total_tokens', 'cur_rem_tokens'):
            self.assertEqual(base[name], port[name], name)


if __name__ == '__main__':
    unittest.main()
