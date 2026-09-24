#!/usr/bin/env python3
"""T41: actual baseline/candidate scheduler + PrefillAdder ASTs on CPU fakes.

No CUDA/torch import; model forwards, cache/pools and ScheduleBatch are faked.
The decision methods, 101 split, resource accounting and LPM sort are production
code. Trees come from the engine git history (scripts/engine/tree.py, cached under build/engine/trees):
the commit before 120, the 120 commit, and HEAD (all candidates, default off) for the HiCache tier tests.
Run: python3 -m unittest discover -s tests -p test_sched_protect_chain.py
"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from enum import Enum, auto
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace as NS
import unittest
from typing import Union
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/engine'))
from tree import tree_dir  # noqa: E402

BASE = tree_dir('before:120')
CANDIDATE = tree_dir('mech:120')
EVIDENCE = ROOT / 'evidence/T41'


class Mode:
    def __init__(self, mode):
        self.mode = mode

    def is_extend(self):
        return self.mode == 'prefill'


class Batch:
    def __init__(self, reqs, batch_is_full=False, chunked_req=None):
        self.reqs = list(reqs)
        self.batch_is_full = batch_is_full
        self.chunked_req = chunked_req
        self.forward_mode = Mode('prefill')
        self.is_prefill_only = False
        self.return_logprob = False
        self.input_embeds = None
        self.decoding_reqs = None

    @classmethod
    def init_new(cls, reqs, *args, chunked_req=None):
        return cls(reqs, chunked_req=chunked_req)

    @property
    def extend_num_tokens(self):
        # mirrors ScheduleBatch.extend_num_tokens: new tokens computed by this extend batch
        if not self.forward_mode.is_extend():
            return None
        return sum(r.extend_range.length for r in self.reqs if getattr(r, 'extend_range', None))

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, chunked_req_to_exclude=()):
        self.reqs = [r for r in self.reqs if not r.finished() and r not in chunked_req_to_exclude]

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)

    def prepare_for_extend(self):
        self.forward_mode = Mode('prefill')


class Req:
    def __init__(self, rid, work, cached=0, host=0, output=240, boundary=None):
        self.rid = rid
        self.origin_input_ids = [1] * (cached + work)
        if boundary is not None:
            self.origin_input_ids[boundary] = 99
        self.output_ids = []
        self.full_untruncated_fill_ids = self.origin_input_ids[:]
        self.prefix_indices = [0] * cached
        self.num_matched_prefix_tokens = cached
        self.host_hit_length = host
        self.swa_host_hit_length = 0
        self.storage_hit_length = 0
        self.mamba_branching_seqlen = None
        self.retracted_stain = False
        self.sampling_params = NS(max_new_tokens=output, ignore_eos=True)
        self.time_stats = NS(set_forward_entry_time=lambda: None)
        self.kv = NS(holds_mamba=False, mamba_pool_idx=None,
                     mamba_cow_src_index=None, mamba_needs_clear=False)
        self.last_node = object()
        self.best_match_node = None
        self.beam_group = None
        self.session = None
        self.inflight_middle_chunks = 0
        self.set_extend_range(cached, cached + work)

    def set_extend_range(self, start, end):
        self.extend_range = NS(start=start, end=end, length=end-start)
        self.extend_input_len = end-start

    def needs_host_load_back(self):
        return self.host_hit_length > 0

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def finished(self):
        return len(self.output_ids) >= self.sampling_params.max_new_tokens

    def init_next_round_input(self, tree_cache=None):
        self.set_extend_range(len(self.prefix_indices), len(self.full_untruncated_fill_ids))
        if tree_cache is not None and getattr(tree_cache, 'allocate_on_match', False):
            self.kv.holds_mamba = True
            self.kv.mamba_pool_idx = MagicMock()
            self.kv.mamba_cow_src_index = 7
            self.kv.mamba_needs_clear = True


def compile_nodes(path, names, ns):
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, 'name', '') in names]
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), ns)


def load_source(root=CANDIDATE):
    ns = dict(Union=Union, Enum=Enum, auto=auto, math=math, os=os, random=random, time=time,
              lru_cache=lru_cache, contextmanager=contextmanager, Counter=Counter,
              logger=logging.getLogger('p120'), _IS_HIP=False, PREFILL_TILE_BUDGET=0,
              PREFILL_TILE_BUDGET_MODE='compact', CLIP_MAX_NEW_TOKENS=4096,
              IGNORE_EOS_RESERVE_TOKENS=1, _ROLE_BOUNDARY_SCAN_WINDOW=32768,
              ROLE_BOUNDARY_STATS=Counter(), mamba_checkpoint_grid=lambda p: math.lcm(p, 256),
              is_dsa_prefill_cp_in_seq_split=lambda: False,
              is_prefill_context_parallel_enabled=lambda: False,
              get_memory=lambda: NS(enable_flexkv=False),
              get_parallel=lambda: NS(pp_max_micro_batch_size=64),
              get_schedule=lambda: NS(prefill_max_requests=None),
              get_disagg=lambda: NS(disaggregation_mode='null'),
              InitLoadBackParams=lambda **kw: NS(**kw), torch=NS(cat=lambda xs: sum(xs, [])),
              ScheduleBatch=Batch, DisaggregationMode=NS(NULL='null', PREFILL='prefill'),
              NextBatchPlan=lambda **kw: NS(**kw), TEST_RETRACT=False,
              scheduler_nvtx_method=lambda _: (lambda f: f),
              PrefillStats=NS(from_adder=lambda *a, **kw: None),
              set_time_batch=lambda *a: None, set_schedule_time_batch=lambda *a: None,
              split_cached_prefix_by_tier=lambda **kw: (kw['prefix_len'], 0, 0))
    for name in ('SWATokenToKVPoolAllocator', 'DeepSeekV4HiSparseTokenToKVPoolAllocator',
                 'PureSWATokenToKVPoolAllocator', 'UnifiedMambaTokenToKVPoolAllocator',
                 'UnifiedMambaSWATokenToKVPoolAllocator'):
        ns[name] = type(name, (), {})
    compile_nodes(root / 'srt/managers/schedule_policy.py',
                  {'AddReqResult', 'PrefillAdder', '_role_boundary_token_ids',
                   '_ax_sched_protect_config', '_ax_srpt_aging', 'SchedulePolicy', 'CacheAwarePolicy',
                   'CacheAgnosticPolicy', '_ceil_div', 'estimate_prefill_extend_tile_metrics'}, ns)
    tree = ast.parse((root / 'srt/managers/scheduler.py').read_text())
    source_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    names = {'get_next_batch_to_run', 'get_new_batch_prefill', '_get_new_batch_prefill_raw',
             '_arm_prefill_decode_interval', '_should_defer_prefill',
             '_ax_sched_protect_enabled', '_ax_sched_protect_blocker', '_ax_mechanism_report',
             '_ax_sched_protect_limits', '_ax_should_decode',
             'get_num_allocatable_reqs', '_ax_pace', '_ax_pace_now', '_ax_pace_slack',
             '_ax_pace_should_decode', '_ax_pace_limits', '_ax_short_reserve_limits'}
    cls = ast.ClassDef(name='Scheduler', bases=[], keywords=[], decorator_list=[],
                      body=[n for n in source_cls.body if getattr(n, 'name', '') in names])
    ns.setdefault('math', math)
    ns.setdefault('os', os)
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(root / 'srt/managers/scheduler.py'), 'exec'), ns)
    # Only this dependency import is inside a production method.
    runtime = ModuleType('sglang.srt.runtime_context')
    runtime.mamba_checkpoint_grid = ns['mamba_checkpoint_grid']
    sys.modules['sglang.srt.runtime_context'] = runtime
    return ns


def make_scheduler(root=CANDIDATE, waiting=(), chunk=None, running=(), budget=8192,
                   interval=0, role=False, slots=64, available=10000000):
    ns = load_source(root)
    ns['_role_boundary_token_ids'] = lambda: frozenset({99}) if role else frozenset()
    s = ns['Scheduler'].__new__(ns['Scheduler'])
    for name in ('enable_fpm', 'require_mlp_sync', 'enable_hisparse', 'enable_lora',
                 'enable_hierarchical_cache', 'enable_hicache_storage', 'is_hybrid_swa',
                 'enable_priority_preemption', 'enable_priority_scheduling',
                 'enable_dynamic_chunking', 'is_mixed_chunk', 'enable_overlap'):
        setattr(s, name, False)
    s.ps = NS(pp_size=1, tp_size=1, tp_rank=0)
    s.dllm_config = None
    s.disaggregation_mode = 'null'
    s.chunked_prefill_size = budget
    s.prefill_decode_interval = interval
    s._prefill_decode_interval_remaining = 0
    s.chunked_req = chunk
    if chunk is not None:
        # A parked continuation has no newly computed KV to stash yet.
        chunk.set_extend_range(len(chunk.prefix_indices), len(chunk.prefix_indices))
    s.waiting_queue = list(waiting)
    s.running_batch = Batch(running)
    s.last_batch = None
    s.page_size = 64
    s.truncation_align_size = 4
    s.min_free_slots_delayer = None
    s.prefill_delayer = None
    s.max_prefill_tokens = 16384
    s.max_prefill_bs = 64
    s.max_running_requests = 64
    s.priority_scheduling_preemption_threshold = 10
    s.new_token_ratio_tracker = NS(current=1.0)
    s.tree_cache = MagicMock()
    s.tree_cache.disable = False
    s.tree_cache.allocate_on_match = False
    s.tree_cache.page_size = 64
    s.tree_cache.supports_mamba.return_value = True
    s.tree_cache.is_tree_cache.return_value = False
    for n in ('evictable_size', 'full_evictable_size', 'mamba_evictable_size'):
        getattr(s.tree_cache, n).return_value = 0
    s.token_to_kv_pool_allocator = MagicMock()
    s.token_to_kv_pool_allocator.available_size.return_value = available
    s.req_to_token_pool = NS(available_size=lambda: slots, mamba_allocator=None)
    s.tree_cache.req_to_token_pool = NS(mamba_allocator=NS(free=MagicMock()))
    s.beam_coordinator = NS(pending_member_rows=lambda _: 0)
    s.tp_worker = NS(model_runner=NS(attn_backend=NS(), prefill_aware_swa=False))
    s.model_config = NS()
    s.spec_algorithm = NS(is_none=lambda: True)
    s.grammar_manager = NS(has_waiting_grammars=lambda: False)
    s.policy = NS(calc_priority=lambda q, _: ns['SchedulePolicy']._sort_by_longest_prefix(q, set()))
    s.load_inquirer = NS(_get_num_pending_tokens=lambda **kw: 0)
    s.ngram_embedding_manager = NS(prepare_for_forward=lambda r, **kw: r)
    s.dp_attn_adapter = NS(maybe_prepare_mlp_sync_batch=lambda r, **kw: r,
                           maybe_convert_decode_to_extend=lambda r: r)
    s.process_pending_chunked_abort = lambda: None
    s._abort_on_waiting_timeout = lambda: None
    s._abort_on_running_timeout = lambda _: None
    def stash(req):
        req.prefix_indices = [0] * req.extend_range.end
    s.stash_chunked_request = stash
    def decode(batch):
        batch.filter_batch()
        batch.forward_mode = Mode('decode')
        return batch
    s.update_running_batch = decode
    s.adders = []
    real_adder = ns['PrefillAdder']
    def factory(*a, **kw):
        adder = real_adder(*a, **kw)
        adder.verdicts = []
        original = adder.add_one_req
        def add(req, *a, **kw):
            res = original(req, *a, **kw)
            adder.verdicts.append((req.rid, res.name))
            return res
        adder.add_one_req = add
        s.adders.append(adder)
        return adder
    ns['PrefillAdder'] = factory
    return s, ns


def step(s, arrivals=()):
    s.waiting_queue.extend(arrivals)
    plan = s.get_next_batch_to_run(s.running_batch, s.last_batch)
    batch = plan.batch_to_run
    s.running_batch = plan.running_batch
    trace = dict(mode=batch.forward_mode.mode if batch else 'idle',
                 reqs=[(r.rid, r.extend_range.start, r.extend_range.end) for r in batch.reqs] if batch else [],
                 waiting=[r.rid for r in s.waiting_queue],
                 chunk=s.chunked_req.rid if s.chunked_req else None,
                 full=s.running_batch.batch_is_full,
                 interval=s._prefill_decode_interval_remaining)
    if batch:
        for r in batch.reqs:
            if not batch.forward_mode.is_extend() or r is not s.chunked_req:
                r.output_ids.append(1)
    if s.adders:
        a = s.adders[-1]
        trace['budget'] = [a.rem_chunk_tokens, a.rem_input_tokens,
                           a.rem_total_token_offset, a.cur_rem_token_offset]
        trace['verdicts'] = a.verdicts[:]
    s.last_batch = batch
    return trace


class ProtectTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'SGLANG_AX_SCHED_PROTECT': '1',
                              'SGLANG_AX_SCHED_COLD_CAP': '2048',
                              'SGLANG_AX_SCHED_SHORT_TOKENS': '4096'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_cold_and_short_hits_interleave_actual_scheduler(self):
        cold = Req('cold', 100000)
        s, _ = make_scheduler(waiting=[cold], running=[Req('running', 1)])
        trace = [step(s)]
        trace.append(step(s, [Req('short', 512, cached=65536)]))
        trace.append(step(s))
        trace.append(step(s))
        self.assertEqual([r['mode'] for r in trace], ['prefill', 'decode', 'prefill', 'decode'])
        self.assertEqual([r[0] for r in trace[2]['reqs']], ['cold', 'short'])
        # 120 caps a cold chunk only while other requests wait: alone it takes the full 8192 budget,
        # once the short hit is waiting the continuation is capped to 2048 and the short joins the batch.
        self.assertEqual(trace[0]['reqs'][0][2], 8192)
        self.assertEqual(trace[2]['reqs'][0][2], 8192 + 2048)
        self.assertIn('short', [r[0] for r in trace[3]['reqs']])
        (EVIDENCE / 'interleave.json').write_text(json.dumps(trace, indent=2) + '\n')

    def test_only_cold_no_idle_or_decode_gap(self):
        cold = Req('cold', 100000)
        s, _ = make_scheduler(waiting=[cold])
        count = 0
        while not cold.output_ids:
            t = step(s)
            self.assertEqual(t['mode'], 'prefill')
            self.assertLessEqual(cold.extend_range.length, 8192)   # alone: uncapped, full chunk budget
            count += 1
            self.assertLess(count, 60)
        self.assertEqual(count, math.ceil(100000 / 8192))
        self.assertEqual(cold.extend_range.end, len(cold.origin_input_ids))

    def test_only_decode_unchanged(self):
        traces = []
        for root in (BASE, CANDIDATE):
            s, _ = make_scheduler(root, running=[Req('r', 1)])
            traces.append([step(s) for _ in range(10)])
        self.assertEqual(traces[0], traces[1])
        self.assertTrue(all(t['mode'] == 'decode' for t in traces[1]))

    def test_lpm_order_short_hits_after_skipped_long(self):
        c = Req('cold', 20000)
        long = Req('long', 10000, cached=100000)
        low = Req('low', 512, cached=1000)
        high = Req('high', 512, cached=50000)
        s, _ = make_scheduler(waiting=[low, long, high], chunk=c)
        t = step(s)
        self.assertEqual([r[0] for r in t['reqs']], ['cold', 'high', 'low'])
        self.assertEqual(t['waiting'], ['long'])
        self.assertFalse(t['full'])

    def test_short_requires_device_hit_and_complete_fit(self):
        c = Req('cold', 20000)
        s, _ = make_scheduler(chunk=c, waiting=[Req('host', 100, cached=10000, host=100),
                     Req('large', 4097, cached=9000), Req('fits', 4096, cached=8000),
                     Req('too_big_for_rest', 4096, cached=7000), Req('tiny', 64, cached=6000),
                     Req('miss', 1)])
        t = step(s)
        self.assertEqual([r[0] for r in t['reqs']], ['cold', 'fits', 'tiny'])
        self.assertIsNone(s.adders[-1].new_chunked_req)
        s.tree_cache.init_load_back.assert_not_called()
        self.assertGreaterEqual(t['budget'][0], 0)

    def test_skipped_mamba_cow_is_freed_before_next_waiter(self):
        long = Req('long', 10000, cached=10000)
        s, _ = make_scheduler(chunk=Req('c', 20000), waiting=[long, Req('s', 64, cached=9000)])
        s.tree_cache.allocate_on_match = True
        step(s)
        self.assertIsNone(long.kv.mamba_pool_idx)
        self.assertIsNone(long.kv.mamba_cow_src_index)
        self.assertFalse(long.kv.mamba_needs_clear)
        s.tree_cache.req_to_token_pool.mamba_allocator.free.assert_called_once()

    def test_session_mamba_slot_is_not_freed(self):
        long = Req('session', 10000, cached=10000)
        long.session = object()
        s, _ = make_scheduler(chunk=Req('c', 20000), waiting=[long])
        s.tree_cache.allocate_on_match = True
        step(s)
        self.assertIsNotNone(long.kv.mamba_pool_idx)
        s.tree_cache.req_to_token_pool.mamba_allocator.free.assert_not_called()

    def test_reserve_request_slot_for_continuation(self):
        s, _ = make_scheduler(chunk=Req('c', 20000), waiting=[Req('s', 64, cached=1000)], slots=1)
        t = step(s)
        self.assertEqual([r[0] for r in t['reqs']], ['c'])
        self.assertEqual(t['waiting'], ['s'])

    def test_kv_no_token_still_refuses_waiter(self):
        s, _ = make_scheduler(chunk=Req('c', 20000), waiting=[Req('s', 4096, cached=1000)], available=5000)
        t = step(s)
        self.assertEqual([r[0] for r in t['reqs']], ['c'])
        self.assertEqual(t['verdicts'], [('s', 'NO_TOKEN')])
        self.assertTrue(t['full'])

    def test_budget_input_and_page_cost_preserved(self):
        s, _ = make_scheduler(chunk=Req('c', 20000), waiting=[Req('s', 65, cached=1024)])
        s.max_prefill_tokens = 2304
        t = step(s)
        a = s.adders[-1]
        self.assertEqual(a.rem_input_tokens, 128)
        self.assertEqual(a.rem_chunk_tokens, 8192 - 2048 - 128)
        self.assertEqual(a.cur_rem_token_offset, 2048 + 64 + 128 + 64)
        self.assertEqual(a.rem_total_token_offset, a.cur_rem_token_offset + 240)

    def test_d1_admission_and_tail_splits_preserved(self):
        for continuation in (False, True):
            r = Req('role', 1536, boundary=1100)
            s, ns = make_scheduler(waiting=[] if continuation else [r], chunk=r if continuation else None, role=True)
            t = step(s)
            self.assertEqual(r.extend_range.end, 1024)
            self.assertIs(s.chunked_req, r)
            self.assertEqual(ns['ROLE_BOUNDARY_STATS'][('tail_' if continuation else 'admit_') + 'taken'], 1)
            self.assertEqual(r.sampling_params.max_new_tokens, 240)
            self.assertEqual(step(s)['reqs'][0][2], 1536)

    def test_d1_active_chunk_prevents_short_second_partial(self):
        short = Req('s', 1536, cached=4096, boundary=5196)
        s, _ = make_scheduler(chunk=Req('c', 20000), waiting=[short], role=True)
        step(s)
        self.assertEqual(short.extend_range.length, 1536)
        self.assertIsNone(s.adders[-1].new_chunked_req)

    def test_d1_branch_conflict_stays_native(self):
        r = Req('r', 1536, boundary=1100)
        r.mamba_branching_seqlen = 512
        s, ns = make_scheduler(waiting=[r], role=True)
        step(s)
        self.assertIsNone(s.chunked_req)
        self.assertEqual(r.extend_range.length, 1536)
        self.assertEqual(ns['ROLE_BOUNDARY_STATS']['admit_skip_branch_conflict'], 1)

    def test_config_alignment_and_minimum_progress(self):
        for cap, expected in [('1', 256), ('2100', 2048), ('99999', 8192)]:
            with patch.dict(os.environ, {'SGLANG_AX_SCHED_COLD_CAP': cap}):
                # a second waiting request makes the cap apply at admission
                s, _ = make_scheduler(waiting=[Req('c', 20000), Req('w', 20000)])
                self.assertEqual(s._ax_sched_protect_limits(8192), (expected, 4096, 256))
                step(s)
                self.assertEqual(s.chunked_req.extend_range.length, expected)
        s, _ = make_scheduler(waiting=[Req('c', 20000)])
        s.truncation_align_size = 1024
        self.assertEqual(s._ax_sched_protect_limits(8192)[2], 1024)
        s.max_prefill_tokens = 128
        self.assertIsNone(s._ax_sched_protect_limits(8192))

    def test_explicit_decode_interval_is_not_doubled(self):
        s, _ = make_scheduler(waiting=[Req('c', 20000)], running=[Req('r', 1)], interval=2)
        self.assertEqual([step(s)['mode'] for _ in range(7)],
                         ['prefill', 'decode', 'decode', 'prefill', 'decode', 'decode', 'prefill'])

    def test_finished_decode_batch_does_not_starve_cold(self):
        # 20000 tokens need three 8192-budget prefill rounds; a finished decoder must not insert decode turns
        s, _ = make_scheduler(waiting=[Req('c', 20000)], running=[Req('done', 1, output=0)])
        modes = [step(s)['mode'] for _ in range(3)]
        self.assertEqual(modes, ['prefill'] * 3)

    def test_unsupported_modes_bypass_protection(self):
        for attr, value in [('is_mixed_chunk', True), ('require_mlp_sync', True),
                            ('enable_lora', True), ('enable_hierarchical_cache', True),
                            ('enable_hisparse', True), ('is_hybrid_swa', True),
                            ('enable_priority_preemption', True), ('dllm_config', object()),
                            ('disaggregation_mode', 'prefill'), ('prefill_delayer', object()),
                            ('chunked_prefill_size', None)]:
            s, _ = make_scheduler()
            setattr(s, attr, value)
            self.assertFalse(s._ax_sched_protect_enabled(), attr)
        s, ns = make_scheduler()
        s.ps.pp_size = 2
        self.assertFalse(s._ax_sched_protect_enabled())
        s.ps.pp_size = 1
        s.tree_cache.disable = True
        self.assertFalse(s._ax_sched_protect_enabled())
        s.tree_cache.disable = False
        ns['is_prefill_context_parallel_enabled'] = lambda: True
        self.assertFalse(s._ax_sched_protect_enabled())

    def test_config_off_ignores_invalid_knobs(self):
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_PROTECT': '0', 'SGLANG_AX_SCHED_COLD_CAP': 'invalid'}):
            s, ns = make_scheduler()
            self.assertIsNone(ns['_ax_sched_protect_config']())
            self.assertFalse(s._ax_sched_protect_enabled())
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_COLD_CAP': '0'}):
            _, ns = make_scheduler()
            with self.assertRaises(ValueError):
                ns['_ax_sched_protect_config']()

    def test_off_byte_identical_decision_sequences(self):
        receipts = []
        raw_pairs = []
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_PROTECT': '0'}):
            for seed in range(24):
                pair = []
                for root in (BASE, CANDIDATE):
                    rng = random.Random(seed)
                    initial = [Req('cold', rng.choice([10000, 20000]), boundary=8500)]
                    s, _ = make_scheduler(root, waiting=initial, running=[Req('running', 1)],
                                          role=seed % 2 == 0, interval=seed % 3,
                                          budget=[1024, 4096, 8192][seed % 3])
                    trace = []
                    for tick in range(40):
                        arrivals = [Req(f'a{tick}', rng.choice([64, 512, 3000, 9000]),
                                        cached=rng.choice([0, 4096, 16384]), output=8)] if tick % 3 == 0 else []
                        try:
                            trace.append(step(s, arrivals))
                        except AssertionError as exc:
                            site = traceback.extract_tb(exc.__traceback__)[-1]
                            trace.append({'baseline_assertion': str(exc), 'tick': tick,
                                          'method': site.name, 'statement': site.line})
                            break
                    pair.append(json.dumps(trace, sort_keys=True, separators=(',', ':')).encode())
                self.assertEqual(pair[0], pair[1], f'seed={seed}')
                raw_pairs.append({'seed': seed, 'baseline': json.loads(pair[0]), 'off': json.loads(pair[1])})
                receipts.append(dict(seed=seed, rounds=len(trace), baseline_assertion='baseline_assertion' in trace[-1], baseline_sha256=hashlib.sha256(pair[0]).hexdigest(),
                                     off_sha256=hashlib.sha256(pair[1]).hexdigest()))
        (EVIDENCE / 'off_parity.json').write_text(json.dumps(receipts, indent=2) + '\n')
        (EVIDENCE / 'off_decision_traces.json').write_text(json.dumps(raw_pairs, separators=(',', ':')) + '\n')

    def test_starvation_bound_with_endless_short_arrivals(self):
        rows = []
        for length, role, cap in [(100000, False, 2048), (100000, True, 2048),
                                  (32769, True, 256), (10000, True, 4096)]:
            with patch.dict(os.environ, {'SGLANG_AX_SCHED_COLD_CAP': str(cap)}):
                c = Req('cold', length, boundary=length-400 if role else None)
                s, _ = make_scheduler(waiting=[c], running=[Req('r', 1, output=10000)], role=role)
                prefill = 0
                rounds = 0
                while not c.output_ids:
                    arrivals = [Req(f's{rounds}', 512, cached=65536, output=2)] if rounds else []
                    t = step(s, arrivals)
                    prefill += t['mode'] == 'prefill'
                    rounds += 1
                    self.assertLessEqual(rounds, 2 * (math.ceil(length / cap) + 1))
                    if t['mode'] == 'prefill':
                        self.assertIn('cold', [r[0] for r in t['reqs']])
                self.assertLessEqual(prefill, math.ceil(length / cap) + int(role))
                rows.append(dict(length=length, cap=cap, role=role, prefill=prefill, rounds=rounds,
                                 bound=2 * (math.ceil(length / cap) + int(role))))
        (EVIDENCE / 'starvation_bound.json').write_text(json.dumps(rows, indent=2) + '\n')

    def test_baseline_second_partial_reproducer_protected(self):
        def run(root):
            # The final chunk role-splits, leaving budget. Native 101 only
            # guards adder.new_chunked_req, not the existing continuation.
            c = Req('c', 1536, boundary=1100)
            s, _ = make_scheduler(root, chunk=c, waiting=[Req('long', 10000, cached=4096)], role=True)
            return step(s)
        # The one-partial guard is part of 101 (T57 folded the former 105 into it), so neither tree crashes
        # even with 120's protection switched off.
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_PROTECT': '0'}):
            for root in (BASE, CANDIDATE):
                run(root)
        t = run(CANDIDATE)
        self.assertEqual(t['chunk'], 'c')
        self.assertEqual(t['waiting'], ['long'])
        self.assertEqual([r[0] for r in t['reqs']], ['c'])

    def test_protection_is_default_on(self):
        with patch.dict(os.environ):
            os.environ.pop('SGLANG_AX_SCHED_PROTECT', None)
            s, ns = make_scheduler()
            self.assertEqual(ns['_ax_sched_protect_config'](), (2048, 4096))
            self.assertTrue(s._ax_sched_protect_enabled())

    def test_small_chunk_and_changed_threshold(self):
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_SHORT_TOKENS': '512'}):
            s, _ = make_scheduler(chunk=Req('c', 10000), budget=4096,
                                   waiting=[Req('long', 513, cached=10000), Req('s', 512, cached=9000)])
            self.assertEqual([r[0] for r in step(s)['reqs']], ['c', 's'])
        s, _ = make_scheduler(waiting=[Req('c', 10000)], budget=1024)
        self.assertEqual(step(s)['reqs'][0][2], 1024)

    def test_new_cold_cannot_overwrite_101_partial_in_same_batch(self):
        first = Req('role', 1536, cached=4096, boundary=5196)
        s, _ = make_scheduler(waiting=[first, Req('cold', 20000)], role=True)
        t = step(s)
        self.assertEqual(t['chunk'], 'role')
        self.assertEqual(t['waiting'], ['cold'])
        self.assertEqual(first.extend_range.length, 1024)

    def test_continuation_survives_zero_request_slots(self):
        s, _ = make_scheduler(chunk=Req('c', 10000), slots=0)
        self.assertEqual(step(s)['reqs'][0][0], 'c')

    def test_on_randomized_admission_never_two_partials_or_overbudget(self):
        for seed in range(16):
            rng = random.Random(seed)
            s, _ = make_scheduler(waiting=[Req('cold', 20000)], running=[Req('r', 1)], role=bool(seed % 2))
            for tick in range(30):
                req = Req(f'q{tick}', rng.choice([64, 1536, 4096, 9000]),
                          cached=rng.choice([0, 4096, 16000]))
                t = step(s, [req] if tick else [])
                if t['mode'] == 'prefill':
                    a = s.adders[-1]
                    partials = [r for r in a.can_run_list if r.extend_range.end < len(r.full_untruncated_fill_ids)]
                    self.assertLessEqual(len(partials), 1)
                    self.assertGreaterEqual(a.rem_input_tokens, 0)
                    self.assertGreaterEqual(a.rem_chunk_tokens, 0)

    def test_original_lpm_role_budget_and_interfaces_unchanged(self):
        def classes(path):
            return {n.name: {m.name: ast.dump(m) for m in n.body if isinstance(m, ast.FunctionDef)}
                    for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}
        base = classes(BASE / 'srt/managers/schedule_policy.py')
        new = classes(CANDIDATE / 'srt/managers/schedule_policy.py')
        self.assertEqual(base['SchedulePolicy'], new['SchedulePolicy'])
        for method in ('_role_split_len', '_update_prefill_budget', '_mamba_gap_budget_for_req',
                       '_req_inc_lock_ref', '_lock_node', 'rem_total_tokens', 'cur_rem_tokens'):
            self.assertEqual(base['PrefillAdder'][method], new['PrefillAdder'][method], method)
        for rel in ('srt/entrypoints/http_server.py', 'srt/managers/tokenizer_manager.py',
                    'srt/managers/scheduler_components/flush_wrapper.py',
                    'srt/managers/schedule_batch.py'):
            self.assertEqual((BASE / rel).read_bytes(), (CANDIDATE / rel).read_bytes(), rel)


# HEAD: official A + all default-off candidates (incl. 180) + the mechanism report.
TREE_180 = tree_dir('HEAD')


class HiCacheTierTests(unittest.TestCase):
    """180 keeps 120/121's contracts with the L1/L2 host tier; only L3 storage bypasses them."""

    def setUp(self):
        self.env = patch.dict(os.environ, {'SGLANG_AX_SCHED_PROTECT': '1',
                              'SGLANG_AX_SCHED_COLD_CAP': '2048',
                              'SGLANG_AX_SCHED_SHORT_TOKENS': '4096'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_trace(self, hicache):
        s, _ = make_scheduler(TREE_180, waiting=[Req('cold', 100000)], running=[Req('running', 1)])
        s.enable_hierarchical_cache = hicache
        self.assertTrue(s._ax_sched_protect_enabled())
        trace = [step(s)]
        trace.append(step(s, [Req('short', 512, cached=65536)]))
        trace += [step(s) for _ in range(4)]
        return trace

    def test_host_tier_keeps_protection_storage_bypasses(self):
        s, _ = make_scheduler(TREE_180)
        s.enable_hierarchical_cache = True
        self.assertTrue(s._ax_sched_protect_enabled())
        s.enable_hicache_storage = True
        self.assertFalse(s._ax_sched_protect_enabled())

    def test_device_hit_decisions_identical_with_and_without_host_tier(self):
        off, on = self.run_trace(False), self.run_trace(True)
        self.assertEqual(off, on)
        # Protection is active: the cold chunk stays capped and the short hit joins it.
        self.assertIn(['cold', 'short'], [[r[0] for r in t['reqs']][:2] for t in on])
        self.assertTrue(all(t['reqs'][0][2] - p['reqs'][0][2] <= 2048
                            for p, t in zip(on, on[1:]) if t['mode'] == 'prefill' and p['mode'] == 'prefill'
                            and t['reqs'][0][0] == p['reqs'][0][0] == 'cold'))

    def test_host_hit_does_not_join_active_chunk(self):
        s, _ = make_scheduler(TREE_180, chunk=Req('cold', 20000),
                              waiting=[Req('host', 100, cached=10000, host=100), Req('fits', 64, cached=8000)])
        s.enable_hierarchical_cache = True
        t = step(s)
        self.assertEqual([r[0] for r in t['reqs']], ['cold', 'fits'])
        self.assertEqual(t['waiting'], ['host'])
        s.tree_cache.init_load_back.assert_not_called()

    def test_blocker_names_the_reason(self):
        s, _ = make_scheduler(TREE_180)
        self.assertIsNone(s._ax_sched_protect_blocker())
        s.enable_hierarchical_cache = True
        self.assertIsNone(s._ax_sched_protect_blocker())
        s.enable_hicache_storage = True
        self.assertEqual(s._ax_sched_protect_blocker(), 'hicache_storage')
        s.enable_hicache_storage = False
        s.is_mixed_chunk = True
        self.assertEqual(s._ax_sched_protect_blocker(), 'mixed_chunk')

    def test_mechanism_report_official_a_env(self):
        s, ns = make_scheduler(TREE_180)
        s.schedule_policy = 'lpm'
        policy = ModuleType('sglang.srt.managers.schedule_policy')
        policy._role_boundary_token_ids = ns['_role_boundary_token_ids']
        policy._ax_srpt_aging = ns['_ax_srpt_aging']
        mods = patch.dict(sys.modules, {'sglang.srt.managers.schedule_policy': policy})
        mods.start()
        self.addCleanup(mods.stop)
        ns['get_spec'] = lambda: NS(speculative_algorithm='NEXTN')
        ns['get_parallel'] = lambda: NS(dcp_size=1)
        env = {'SGLANG_AX_KDA_DUAL_SNAPSHOT': '0', 'SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS': ''}
        with patch.dict(os.environ, env):
            for k in ('SGLANG_AX_PACE_TPOT', 'SGLANG_AX_SRPT_AGING'):
                os.environ.pop(k, None)
            rep = s._ax_mechanism_report()
        head = rep.split(' | ')[0].split()
        self.assertEqual(head, ['101=off:role_ids_unset', '120=on', '122=off:SGLANG_AX_PACE_TPOT_unset',
                                '123=off:SGLANG_AX_SRPT_AGING_unset', '140=off', '180=off:no_hierarchical_cache'])
        s.enable_hierarchical_cache = True
        with patch.dict(os.environ, dict(env, SGLANG_AX_PACE_TPOT='0.085')):
            rep = s._ax_mechanism_report()
        self.assertIn('120=on 122=on', rep)
        self.assertIn('180=on', rep)
        self.assertIn('spec=NEXTN dcp=1', rep)

if __name__ == '__main__':
    unittest.main()
