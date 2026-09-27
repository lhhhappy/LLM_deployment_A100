"""128g uses the real policy, RadixKey and simulated RadixCache on CPU.

Only tensor storage / event sinks / physical cache contents are faked. Neither
the in-batch trie nor hold decisions are copied into these tests. The scheduler
integration test also executes 124/128p and the native PrefillAdder.
"""
import dataclasses
import math
import os
import sys
import time
import unittest
from array import array
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple
from unittest.mock import patch

from test_sched_protect_chain import compile_nodes, load_source, make_scheduler, step
from test_prefix_producer import ENV, request


ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "engine/sglang"
SWITCH = "SGLANG_AX_LPM_REUSE_GUARD"


class Tensor(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Tensor(value) if isinstance(key, slice) else value

    def clone(self):
        return Tensor(self)


class CpuBaseCache:
    def is_chunk_cache(self):
        return False

    def root_node_handle(self, **kwargs):
        return self.root_node


def load_policy(ns=None, state_chunk=64):
    ns = load_source(TREE) if ns is None else ns
    ns.update(dataclasses=dataclasses, NamedTuple=NamedTuple, array=array,
              defaultdict=defaultdict, sys=sys, time=time,
              BasePrefixCache=CpuBaseCache,
              KVCacheEventRecorder=lambda **kw: NS(
                  record_all_cleared=lambda: None, record_store=lambda n: None),
              get_eviction_strategy=lambda _: None,
              split_node_hash_value=lambda value, n, page: (None, None),
              torch=NS(empty=lambda n, **kw: Tensor([0] * (n[0] if isinstance(n, tuple) else n)),
                       cat=lambda chunks: Tensor(x for chunk in chunks for x in chunk),
                       tensor=lambda xs, **kw: Tensor(xs), device=str,
                       bool="bool", int64="int64"),
              IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD=32,
              IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD=32,
              envs=NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False)),
              mamba_cache_chunk_size=lambda: state_chunk)
    compile_nodes(TREE / 'srt/runtime_context.py', {'mamba_checkpoint_grid'}, ns)
    for file, names in (
        ('cache_init_params.py', {'CacheInitParams'}),
        ('base_prefix_cache.py', {'MatchPrefixParams', 'InsertParams', 'InsertResult',
                                 'MatchResult', 'zero_match_result'}),
        ('radix_cache.py', {'RadixKey', 'TreeNode', 'RadixCache'}),
    ):
        compile_nodes(TREE / 'srt/mem_cache' / file, names, ns)
    compile_nodes(TREE / 'srt/managers/schedule_policy.py', {'match_prefix_for_req'}, ns)
    return ns


def cache_for(ns, page=64, mamba=True, extra_buffer=True):
    # Real radix algorithms supply physical full-KV matches. No state is claimed
    # beyond them; all reproduction requests begin with an empty physical tree.
    cache = ns['RadixCache'].create_simulated(page_size=page)
    cache.supports_mamba = lambda: mamba
    cache.enable_mamba_extra_buffer = extra_buffer
    cache.swa_reprefill_tail_tokens = lambda: 0
    cache.supports_fast_match_prefix = lambda: False
    cache.is_chunk_cache = lambda: False
    cache.root_node_handle = lambda **kw: cache.root_node
    return cache


def policy_for(ns, cache=None, name='lpm'):
    return ns['SchedulePolicy'](name, cache or cache_for(ns), False, False, False)


def pair(shared, total=20000):
    return [request('large', shared=shared, tail=60000-shared),
            request('small', shared=shared, tail=total-shared)]


class LpmReuseGuard(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {**ENV, SWITCH: '1'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.ns = load_policy()

    def test_19k_head_native_hold_and_guarded_release(self):
        for enabled in ('0', '1'):
            with self.subTest(enabled=enabled), patch.dict(os.environ, {SWITCH: enabled}):
                reqs = pair(58, 19335)
                target = reqs[1]
                p = policy_for(self.ns)
                p.calc_priority(reqs)
                self.assertEqual(p.ax_held, {'small'} if enabled == '0' else set())
                self.assertEqual(len(target.prefix_indices), 0)
                if enabled == '1':
                    self.assertEqual(target._ax_lpm_hold_check,
                        dict(shared=58, grid=64, reuse_upper=0, gain_upper=0,
                             decision='release_zero_gain'))
                    self.assertNotIn('small', p.ax_prefix_held_by)
                else:
                    self.assertEqual(p.ax_prefix_held_depth['small'], 58)
                    self.assertFalse(hasattr(target, '_ax_lpm_hold_check'))

    def test_real_scheduler_serves_19k_before_unrelated_50k(self):
        for enabled, expected in (('0', 'large'), ('1', 'small')):
            with self.subTest(enabled=enabled), patch.dict(os.environ, {SWITCH: enabled}):
                reqs = pair(58, 19335)
                s, ns = make_scheduler(TREE, waiting=reqs)
                load_policy(ns)
                s.policy = policy_for(ns)
                s._ax_rank0_decide = lambda f: f()
                s._ax_prefix_consensus = lambda value: True
                result = step(s)
                self.assertEqual(result['reqs'][0][0], expected)
                if enabled == '1':
                    row = next(r for r in s._ax_prefix_tracker.rows if r['rid'] == 'small')
                    self.assertFalse(row['held'])
                    self.assertFalse(row['effective_held'])
                    self.assertEqual(row['lpm_hold']['decision'], 'release_zero_gain')

    def test_rotating_shallow_owners_cannot_restore_the_hold(self):
        target = request('small', shared=58, tail=19335-58)
        p = policy_for(self.ns)
        for index, shared in enumerate([50, 58] * 20):
            owner = request('o'+str(index), shared=shared, tail=50000)
            p.calc_priority([owner, target])
            self.assertNotIn(target.rid, p.ax_held)
            self.assertEqual(target._ax_lpm_hold_check['shared'], shared)

    def test_released_representative_retains_its_real_deep_sibling(self):
        unrelated = request('u', shared=58, tail=50000)
        producer = request('p', shared=4096, tail=1000)
        rider = request('r', shared=4096, tail=500)
        p = policy_for(self.ns)
        p.calc_priority([unrelated, producer, rider])
        self.assertEqual(p.ax_held, {'r'})
        self.assertEqual(p.ax_prefix_held_by['r'], 'p')
        self.assertEqual(p.ax_prefix_held_depth['r'], 4096)
        self.assertEqual(rider._ax_lpm_hold_check['decision'], 'keep')

    def test_actual_page_and_mamba_grid_boundaries(self):
        for page, chunk in ((64, 64), (128, 64), (256, 64), (64, 256), (192, 256)):
            ns = load_policy(state_chunk=chunk)
            grid = math.lcm(page, chunk)
            for shared in (32, 50, 58, grid-1, grid, grid+1, 4096):
                with self.subTest(page=page, chunk=chunk, shared=shared):
                    p = policy_for(ns, cache_for(ns, page=page))
                    reqs = pair(shared)
                    p.calc_priority(reqs)
                    self.assertEqual(p.ax_lpm_reuse_grid, grid)
                    self.assertEqual('small' in p.ax_held, shared >= grid)

    def test_no_buffer_cache_does_not_acquire_extra_buffer_grid(self):
        p = policy_for(self.ns, cache_for(self.ns, page=1, extra_buffer=False))
        p.calc_priority(pair(58))
        self.assertEqual(p.ax_lpm_reuse_grid, 1)
        self.assertIn('small', p.ax_held)

    def test_logits_and_logprob_limits_can_remove_all_gain(self):
        p = policy_for(self.ns)
        owner = request('p', shared=64, tail=1000)
        target = request('s', shared=64, tail=0)
        p.calc_priority([owner, target])
        self.assertNotIn('s', p.ax_held)  # last token must be recomputed
        target = request('s', shared=4096, tail=1000)
        target._compute_max_prefix_len = lambda n: 32
        owner = request('p', shared=4096, tail=2000)
        p.calc_priority([owner, target])
        self.assertNotIn('s', p.ax_held)

    def test_swa_tail_limit_can_remove_all_gain(self):
        cache = cache_for(self.ns)
        cache.swa_reprefill_tail_tokens = lambda: 19970
        p = policy_for(self.ns, cache)
        p.calc_priority(pair(4096))
        self.assertNotIn('small', p.ax_held)

    def test_existing_device_prefix_is_not_new_gain(self):
        cache = cache_for(self.ns, page=32, mamba=False)
        cache.insert(self.ns['InsertParams'](key=self.ns['RadixKey']([7] * 32)))
        reqs = pair(50)
        target = reqs[1]
        p = policy_for(self.ns, cache)
        p.calc_priority(reqs)
        self.assertEqual(len(target.prefix_indices), 32)
        self.assertEqual(target._ax_lpm_hold_check['reuse_upper'], 32)
        self.assertNotIn('small', p.ax_held)

    def test_cache_domains_are_kept_separate(self):
        for field in ('cache_salt', 'extra_key'):
            reqs = pair(4096)
            setattr(reqs[1], field, 'separate')
            p = policy_for(self.ns)
            p.calc_priority(reqs)
            self.assertFalse(p.ax_held)

    def test_next_round_and_fcfs_fallback_clear_diagnostic_state(self):
        reqs = pair(58)
        target = reqs[1]
        p = policy_for(self.ns)
        p.calc_priority(reqs)
        self.assertIsNotNone(target._ax_lpm_hold_check)
        p.calc_priority([target])
        self.assertIsNone(target._ax_lpm_hold_check)
        p.calc_priority(reqs)
        p.calc_priority([target] + [request(str(i), shared=0, tail=2) for i in range(128)])
        self.assertFalse(p.ax_held)
        self.assertIsNone(target._ax_lpm_hold_check)

    def test_force_miss_never_constructs_hold(self):
        self.ns['envs'].SGLANG_RADIX_FORCE_MISS.get = lambda: True
        p = policy_for(self.ns)
        p.calc_priority(pair(4096))
        self.assertFalse(p.ax_held)

    def test_guard_can_run_without_128p(self):
        self.ns['ax_prefix_readiness'].ENABLED = False
        p = policy_for(self.ns)
        p.calc_priority(pair(58))
        self.assertFalse(p.ax_held)
        p.calc_priority(pair(4096))
        self.assertEqual(p.ax_held, {'small'})

    def test_unsupported_configuration_refuses_on_but_not_off(self):
        for mode in ('fcfs', 'disabled', 'bigram', 'in_batch_disabled'):
            with self.subTest(mode=mode):
                cache = cache_for(self.ns)
                cache.disable = mode == 'disabled'
                cache.is_eagle = mode == 'bigram'
                name = 'fcfs' if mode == 'fcfs' else 'lpm'
                self.ns['IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD'] = -1 if mode == 'in_batch_disabled' else 32
                with self.assertRaises(ValueError):
                    policy_for(self.ns, cache, name)
                with patch.dict(os.environ, {SWITCH: '0'}):
                    self.assertIsNone(policy_for(self.ns, cache, name).ax_lpm_reuse_grid)


if __name__ == '__main__':
    unittest.main()
