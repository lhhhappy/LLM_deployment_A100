"""Real scheduler/adder control flow, with explicit cache publication and pools.

The fake cache publishes KV/KDA checkpoints only after a simulated forward.
It deliberately permits private producer KV beyond the last reusable state.
No torch import; the native Mamba validator and COW method are also exercised.
"""
import ast
import copy
import json
import os
import random
import sys
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from test_sched_protect_chain import ROOT, Req, compile_nodes, load_source, make_scheduler, step, tree_dir

TREE = ROOT / 'engine/sglang'
ENV = {'SGLANG_AX_PREFIX_PRODUCER': '1', 'SGLANG_AX_DEADLINE_TIERS': '1',
       'SGLANG_AX_DEADLINE_FREEZE_CLASS': '1', 'SGLANG_AX_DEADLINE_FAMILY': '0',
       'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '8192',
       'SGLANG_AX_SCHED_SHORT_TOKENS': '4096', 'SGLANG_AX_PREFIX_STALL_S': '2',
       'SGLANG_AX_PREFIX_MAX_HOLD_S': '8', 'SGLANG_AX_PREFIX_TRACE_S': '120',
       'SGLANG_AX_PREFIX_TRACE_ROUNDS': '2048', 'SGLANG_AX_PARK_MAX_ROUNDS': '0'}


class KV:
    mamba_pool_idx = None
    mamba_cow_src_index = None
    mamba_needs_clear = False

    @property
    def holds_mamba(self):
        return self.mamba_pool_idx is not None


def request(rid, shared=16384, tail=1024, tag=7, cached=0, waited=0):
    req = Req(rid, shared + tail - cached, cached=cached)
    req.origin_input_ids = [tag] * shared + [100 + ord(rid[0])] * tail
    req.full_untruncated_fill_ids = req.origin_input_ids[:]
    req.time_stats.scheduler_recv_time = time.perf_counter() - waited
    req.time_stats.prefill_finished_time = 0.0
    req.extra_key = req.cache_salt = None
    req.mamba_host_hit_length = 0
    req.retraction_count = 0
    req.kv = KV()
    req._compute_max_prefix_len = lambda n: max(0, n - 1)
    return req


class PublishedCache:
    def __init__(self, s, ns):
        self.s, self.ns = s, ns
        self.checkpoints = []
        self.full = 0
        self.host = self.mamba_host = 0
        self.admission_hook = None

    def publish(self, req, depth):
        self.checkpoints.append((req.origin_input_ids[:depth], req.extra_key, req.cache_salt))

    def match(self, req):
        cap = req._compute_max_prefix_len(req.seqlen)
        prefix = max([0, *(len(tokens) for tokens, key, salt in self.checkpoints
            if len(tokens) <= cap and (key, salt) == (req.extra_key, req.cache_salt)
            and req.origin_input_ids[:len(tokens)] == tokens)])
        req.prefix_indices = [0] * prefix
        req.num_matched_prefix_tokens = prefix + self.host
        req.host_hit_length, req.mamba_host_hit_length = self.host, self.mamba_host
        mod = self.ns.get('ax_prefix_readiness')
        if mod:
            mod.capture(req, NS(device_indices=req.prefix_indices,
                full_kv_hit_length=max(prefix, self.full), host_hit_length=self.host,
                mamba_host_hit_length=self.mamba_host, swa_host_hit_length=0), cap)

    def bind(self, req):
        def init(cache=None):
            if cache is not None:
                self.match(req)
                if self.admission_hook:
                    self.admission_hook(req)
            req.set_extend_range(len(req.prefix_indices), len(req.full_untruncated_fill_ids))
        req.init_next_round_input = init


def scheduler(waiting=(), chunk=None, root=TREE, held_by=None, **kwargs):
    s, ns = make_scheduler(root, waiting=waiting, chunk=chunk, **kwargs)
    s._ax_rank0_decide = lambda f: f()
    s._ax_prefix_consensus = lambda value: True
    c = PublishedCache(s, ns)
    for r in [*waiting, *([chunk] if chunk else [])]:
        c.bind(r)
    held_by = held_by or {}
    def priority(reqs, _):
        for r in reqs:
            c.match(r)
        s.policy.ax_held = {rid for rid in held_by if any(r.rid == rid and not r.prefix_indices for r in reqs)}
        s.policy.ax_prefix_held_by = held_by
        ns['SchedulePolicy']._sort_by_longest_prefix(reqs, s.policy.ax_held)
    s.policy.calc_priority = priority
    def stash(req):
        end = req.extend_range.end
        req.prefix_indices = [0] * end
        c.publish(req, end)
    s.stash_chunked_request = stash
    return s, ns, c


class ProducerAdmission(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def establish(self, **kwargs):
        producer = request('p', tail=24576, cached=8192)
        rider = request('r', tail=2048)
        s, ns, cache = scheduler([rider], producer, **kwargs)
        cache.publish(producer, 8192)
        self.assertEqual(step(s)['reqs'][0], ('p', 8192, 16384))
        self.assertEqual(s._ax_prefix_tracker.dependencies['r'].state, 'WAIT_PREFIX')
        return s, ns, cache, producer, rider

    def test_producer_can_replace_native_held_representative(self):
        small = request('a', tail=512)
        rep = request('z', tail=2048)
        ordinary = request('b', shared=0, tail=14000)
        s, _, _ = scheduler([rep, ordinary, small], held_by={'a': 'z'})
        trace = step(s)
        self.assertEqual(trace['reqs'][0][0], 'a')
        self.assertEqual(s._ax_prefix_tracker.dependencies['z'].producer, 'a')
        self.assertNotIn('a', s._ax_prefix_tracker.effective_held)

    def test_unexplained_native_hold_is_preserved(self):
        a, z = request('a'), request('z')
        s, _, _ = scheduler([a, z], held_by={'a': 'external', 'z': 'external'})
        step(s)
        self.assertEqual(s._ax_prefix_tracker.effective_held, {'a', 'z'})
        self.assertFalse(s._ax_prefix_tracker.dependencies)

    def test_deep_producer_can_replace_unrelated_shallow_system_prefix_owner(self):
        a, z = request('a', tail=512), request('z')
        unrelated = request('u', tag=9, tail=65536)
        for r in (a, z, unrelated):
            r.origin_input_ids[:256] = [1] * 256
            r.full_untruncated_fill_ids = r.origin_input_ids[:]
        s, _, _ = scheduler([unrelated, z, a], held_by={'a': 'u', 'z': 'u'})
        s.policy.ax_prefix_held_depth = {'a': 256, 'z': 256}
        self.assertEqual(step(s)['reqs'][0][0], 'a')
        self.assertNotIn('a', s._ax_prefix_tracker.effective_held)

    def test_ready_short_tail_runs_before_partial_consumes_full_budget(self):
        s, _, _, producer, rider = self.establish()
        trace = step(s)
        self.assertEqual(trace['reqs'], [('r', 16384, 18432), ('p', 16384, 22528)])
        self.assertEqual(trace['chunk'], 'p')
        self.assertLessEqual(sum(end - start for _, start, end in trace['reqs']), 8192)
        self.assertEqual(s._ax_prefix_tracker.stats['dependent_admitted'], 1)

    def test_kv_without_kda_and_host_only_are_not_ready(self):
        for host, state in ((0, 0), (0, 1), (8192, 0)):
            with self.subTest(host=host, state=state):
                s, ns, cache, producer, rider = self.establish()
                cache.full = 16384
                cache.host, cache.mamba_host = host, state
                s.stash_chunked_request = lambda r: setattr(r, 'prefix_indices', [0] * r.extend_range.end)
                trace = step(s)
                self.assertNotIn('r', [rid for rid, _, _ in trace['reqs']])
                self.assertFalse(s._ax_prefix_tracker.ready('r'))

    def test_rejected_cow_restores_full_continuation_budget_same_pass(self):
        s, ns, cache, producer, rider = self.establish()
        def fail(req):
            if req.rid == 'r':
                raise ns['ax_prefix_readiness'].ReservationUnavailable('mamba_cow_slots')
        cache.admission_hook = fail
        trace = step(s)
        self.assertEqual(trace['reqs'], [('p', 16384, 24576)])
        self.assertEqual(s._ax_prefix_tracker.stats['admission_mamba_cow_slots'], 1)
        self.assertFalse(rider._ax_prefix_reserving)

    def test_126_does_not_keep_a_failed_ready_seat_or_blacklist_its_retry(self):
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_COLD_CAP': '2048',
                                    'SGLANG_AX_SCHED_COLD_CAP_MAX': '8192'}):
            s, ns, cache, _, rider = self.establish()
            def fail(req):
                raise ns['ax_prefix_readiness'].ReservationUnavailable('mamba_cow_slots')
            cache.admission_hook = fail
            self.assertEqual(step(s)['reqs'], [('p', 16384, 24576)])
            cache.admission_hook = None
            self.assertEqual(step(s)['reqs'][0][0], 'r')
            self.assertNotIn('r', s._ax_reserve_state[2])

    def test_lock_time_capacity_rejection_resumes_continuation(self):
        s, _, cache, producer, rider = self.establish()
        capacity = [10000000]
        s.token_to_kv_pool_allocator.available_size.side_effect = lambda: capacity[0]
        s.tree_cache.inc_lock_ref.side_effect = lambda node: capacity.__setitem__(0, 100)
        s.tree_cache.dec_lock_ref.side_effect = lambda *args: capacity.__setitem__(0, 10000000)
        self.assertEqual(step(s)['reqs'], [('p', 16384, 24576)])
        self.assertEqual(s._ax_prefix_tracker.stats['admission_kv_budget'], 1)

    def test_authoritative_adder_rejection_frees_new_cow_and_restores_budget(self):
        s, ns, cache, producer, rider = self.establish()
        factory = ns['PrefillAdder']
        def make(*args, **kwargs):
            adder = factory(*args, **kwargs)
            native = adder.add_one_req
            def add(req, *args, **kwargs):
                if not getattr(req, '_ax_prefix_reserving', False):
                    return native(req, *args, **kwargs)
                s.token_to_kv_pool_allocator.available_size.return_value = 1
                try:
                    return native(req, *args, **kwargs)
                finally:
                    s.token_to_kv_pool_allocator.available_size.return_value = 10000000
            adder.add_one_req = add
            return adder
        ns['PrefillAdder'] = make
        cache.admission_hook = lambda req: setattr(req.kv, 'mamba_pool_idx', MagicMock())
        self.assertEqual(step(s)['reqs'], [('p', 16384, 24576)])
        self.assertIsNone(rider.kv.mamba_pool_idx)
        self.assertEqual(s._ax_prefix_tracker.stats['admission_adder_no_token'], 1)

    def test_rematch_regression_frees_fresh_state_and_retries_next_round(self):
        s, ns, cache, producer, rider = self.establish()
        def regress(req):
            if req.rid == 'r':
                req.kv.mamba_pool_idx = MagicMock()
                req.kv.mamba_cow_src_index = 7
                req.prefix_indices = [0] * 8192
                req._ax_prefix_match = ns['ax_prefix_readiness'].PrefixReadiness(8192, 16384, 0, 0, 0)
        cache.admission_hook = regress
        trace = step(s)
        self.assertEqual(trace['reqs'], [('p', 16384, 24576)])
        self.assertIsNone(rider.kv.mamba_pool_idx)
        self.assertIsNone(rider.kv.mamba_cow_src_index)
        s.tree_cache.req_to_token_pool.mamba_allocator.free.assert_called_once()
        cache.admission_hook = None
        trace = step(s)
        self.assertIn('r', [rid for rid, _, _ in trace['reqs']])

    def test_preserves_preexisting_mamba_on_rejection(self):
        s, ns, cache, producer, rider = self.establish()
        original = rider.kv.mamba_pool_idx = MagicMock()
        decisions = iter((False, True))
        s._ax_prefix_consensus = lambda _: next(decisions)
        trace = step(s)
        self.assertIs(rider.kv.mamba_pool_idx, original)
        s.tree_cache.req_to_token_pool.mamba_allocator.free.assert_not_called()
        self.assertEqual(trace['reqs'], [('p', 16384, 24576)])

    def test_one_row_keeps_the_producer_progressing(self):
        s, _, _, producer, _ = self.establish(slots=1)
        trace = step(s)
        self.assertEqual(trace['reqs'], [('p', 16384, 24576)])
        self.assertEqual(s._ax_prefix_tracker.stats['admission_preview_request_slots'], 1)

    def test_dependency_survives_producer_departure_if_checkpoint_is_real(self):
        s, _, cache, producer, rider = self.establish()
        cache.publish(producer, 16384)
        s.chunked_req = None
        s.last_batch = None
        self.assertEqual(step(s)['reqs'], [('r', 16384, 18432)])

    def test_cancellation_without_checkpoint_releases_dependency(self):
        s, _, _, _, rider = self.establish()
        s.chunked_req = s.last_batch = None
        trace = step(s)
        self.assertEqual(s._ax_prefix_tracker.dependencies['r'].reason, 'producer_gone')
        self.assertEqual(trace['reqs'][0][0], 'r')

    def test_flush_and_reused_rid_do_not_keep_old_dependencies(self):
        s, _, _, producer, _ = self.establish()
        epoch = s._ax_prefix_tracker.epoch
        s._ax_flush_admission_state()
        self.assertEqual(s._ax_prefix_tracker.epoch, epoch + 1)
        self.assertFalse(s._ax_prefix_tracker.dependencies)
        self.assertFalse(s._ax_prefix_tracker.pairs)
        self.assertEqual(s._ax_prefix_applied_sequence, 0)
        # Reused producer RID with unrelated tokens is a different generation.
        s, _, cache, producer, _ = self.establish()
        replacement = request('p', tag=99, tail=24576, cached=8192)
        cache.bind(replacement)
        s.chunked_req, s.last_batch = replacement, None
        step(s)
        self.assertEqual(s._ax_prefix_tracker.dependencies['r'].reason, 'producer_gone')

    def test_retraction_keeps_deadline_but_invalidates_producer_generation(self):
        s, _, _, producer, _ = self.establish()
        due = s._ax_prefix_tracker.records['p'].due
        producer.retraction_count += 1
        producer.prefix_indices = []
        producer.set_extend_range(0, 0)
        s.last_batch = None
        step(s)
        self.assertEqual(s._ax_prefix_tracker.records['p'].due, due)
        self.assertEqual(s._ax_prefix_tracker.dependencies['r'].reason, 'producer_gone')

    def test_deadline_does_not_shrink_after_match_growth(self):
        with patch.dict(os.environ, {'SGLANG_AX_DEADLINE_FREEZE_CLASS': '0'}):
            s, _, _, _, rider = self.establish()
            due = s._ax_prefix_tracker.records['r'].due
            step(s)
            self.assertEqual(s._ax_prefix_tracker.records['r'].due, due)
            self.assertTrue(s._ax_prefix_tracker.records['r'].cold)

    def test_cache_domain_prevents_cross_salt_family(self):
        a, z = request('a'), request('z')
        a.cache_salt, z.cache_salt = 'tenant-a', 'tenant-z'
        s, _, _ = scheduler([a, z])
        step(s)
        self.assertFalse(s._ax_prefix_tracker.dependencies)

    def test_two_ranks_apply_same_plan_with_different_local_clocks(self):
        payloads, traces = [], []
        for rank in (0, 1):
            with patch('time.perf_counter', return_value=1000 + rank * 100):
                producer, rider = request('p', cached=8192, tail=24576), request('r', tail=2048)
                s, _, cache = scheduler([rider], producer)
                cache.publish(producer, 8192)
                cursor = 0
                def decide(callback):
                    nonlocal cursor
                    if rank == 0:
                        payloads.append(copy.deepcopy(callback()))
                    result = payloads[cursor]
                    cursor += 1
                    return result
                s._ax_rank0_decide = decide
                traces.append([step(s), step(s)])
        self.assertEqual(*traces)
        self.assertEqual([r[0] for r in traces[0][-1]['reqs']], ['r', 'p'])

    def test_parking_uses_original_deadline_and_failed_seat_resumes_owner(self):
        for fail in (False, True):
            with self.subTest(fail=fail), patch.dict(os.environ, {
                'SGLANG_AX_DEADLINE_FREEZE_CLASS': '0', 'SGLANG_AX_PARK_MAX_ROUNDS': '8',
                'SGLANG_AX_SCHED_SHORT_TOKENS': '8192'}), patch('time.perf_counter', return_value=100):
                p = request('p', cached=8192, tail=131072, waited=20)
                r = request('r', tail=8000, waited=20)
                s, ns, cache = scheduler([r], p)
                cache.publish(p, 8192)
                step(s)
                if fail:
                    def reject(req):
                        raise ns['ax_prefix_readiness'].ReservationUnavailable('mamba_cow_slots')
                    cache.admission_hook = reject
                trace = step(s)
                self.assertEqual(trace['reqs'][0][0], 'p' if fail else 'r')
                self.assertEqual(p._ax_parked_rounds, 0 if fail else 1)
                self.assertEqual(trace['chunk'], 'p')

    def test_s1_relief_and_shallower_checkpoint_ready(self):
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_COLD_CAP': '6144',
            'SGLANG_AX_SCHED_SHORT_TOKENS': '8192', 'SGLANG_AX_BACKLOG_RELIEF': '1',
            'SGLANG_AX_BACKLOG_COLD_CAP': '8192', 'SGLANG_AX_BACKLOG_INTERVAL': '0',
            'SGLANG_AX_BACKLOG_HIGH_S': '0.1', 'SGLANG_AX_BACKLOG_LOW_S': '0.01',
            'SGLANG_AX_BACKLOG_MAX_SLOW': '80'}):
            p, r = request('p', shared=32768, tail=16384), request('r', shared=32768, tail=3072)
            # Force this request to be the already-active producer, as happens
            # when its sibling arrives after the first chunk.
            s, _, cache = scheduler([r], p, interval=2)
            s._ax_admission_cfgs()
            s._ax_backlog.rate = 10000  # the CPU fixture has no real forward timing observer
            first = step(s)
            self.assertEqual(first['reqs'][0][2], 6144)
            self.assertTrue(s._ax_backlog_relieved)
            admitted = None
            for _ in range(10):
                trace = step(s)
                matches = [x for x in trace['reqs'] if x[0] == 'r']
                if matches:
                    admitted = matches[0]
                    break
            self.assertIsNotNone(admitted)
            self.assertLessEqual(admitted[2] - admitted[1], 8192)
            self.assertLess(admitted[1], s._ax_prefix_tracker.dependencies['r'].target)

    def test_equal_group_work_prefers_shorter_producer_not_random_rid(self):
        long, short = request('a', tail=3072), request('z', tail=512)
        s, _, _ = scheduler([long, short], held_by={'z': 'a'})
        self.assertEqual(step(s)['reqs'][0][0], 'z')

    def test_stall_timeout_releases_held_rider_without_resetting_deadline(self):
        with patch('time.perf_counter', return_value=100):
            s, _, _, producer, _ = self.establish()
            due = s._ax_prefix_tracker.records['r'].due
        producer.set_extend_range(8192, 8192)
        s.last_batch = None
        with patch('time.perf_counter', return_value=103):
            step(s)
        self.assertEqual(s._ax_prefix_tracker.dependencies['r'].reason, 'producer_stalled')
        self.assertNotIn('r', s._ax_prefix_tracker.wait_prefix)
        self.assertEqual(s._ax_prefix_tracker.records['r'].due, due)

    def test_hash_collision_does_not_create_a_prefix(self):
        a, b = request('a', tag=7), request('b', tag=8)
        s, _, _ = scheduler([a, b])
        with patch('builtins.hash', return_value=1):
            step(s)
        self.assertFalse(s._ax_prefix_tracker.dependencies)

    def test_shared_prefix_does_not_turn_nested_families_into_a_union(self):
        a, b, c = request('a', tail=1024), request('b', tail=1024), request('c', tail=1024)
        c.origin_input_ids[8192:] = [89] * (len(c.origin_input_ids) - 8192)
        c.full_untruncated_fill_ids = c.origin_input_ids[:]
        s, _, _ = scheduler([a, b, c])
        step(s)
        self.assertIn('b', s._ax_prefix_tracker.dependencies)
        self.assertNotIn('c', s._ax_prefix_tracker.dependencies)

    def test_no_second_partial_or_token_overflow_over_many_rounds(self):
        rng = random.Random(127)
        reqs = [request(chr(97 + i), shared=16384 if i < 8 else 8192,
                        tail=rng.randrange(128, 4000), tag=7 if i < 8 else i) for i in range(16)]
        s, _, cache = scheduler(reqs)
        for _ in range(100):
            trace = step(s)
            if trace['mode'] == 'prefill':
                self.assertLessEqual(sum(b-a for _, a, b in trace['reqs']), 8192)
                partials = [r for r in s.last_batch.reqs if r.extend_range.end < r.seqlen - len(r.output_ids)]
                self.assertLessEqual(len(partials), 1)
            if not s.waiting_queue and s.chunked_req is None:
                break
        self.assertFalse(s.waiting_queue)
        self.assertIsNone(s.chunked_req)

    def test_every_planned_ready_has_result_and_trace_has_window_boundary(self):
        with patch.dict(os.environ, {'SGLANG_AX_PREFIX_TRACE_ROUNDS': '1'}):
            with self.assertLogs('p120', level='INFO') as capture:
                s, _, _, _, _ = self.establish()
                step(s)
            lines = capture.output
            self.assertTrue(any('[ax-prefix-decision]' in l for l in lines))
            self.assertTrue(any('[ax-prefix-trace-end]' in l for l in lines))

    def test_native_finish_timestamp_is_logged_even_without_another_prefill_plan(self):
        s, ns, _, _, rider = self.establish()
        step(s)
        rider.time_stats.prefill_finished_time = time.perf_counter()
        tracker = s._ax_prefix_tracker
        with self.assertLogs('p120', level='INFO') as records:
            tracker.note_prefill_finished(rider, ns['logger'])
            tracker.note_prefill_finished(rider, ns['logger'])
        self.assertEqual(len(records.output), 1)
        self.assertEqual(tracker.stats['prefill_finished'], 1)

    def test_scheduler_logs_finish_from_its_separate_result_processor(self):
        s, ns, _, _, rider = self.establish()
        step(s)
        path = TREE / 'srt/managers/scheduler.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
        method = next(n for n in cls.body if getattr(n, 'name', '') == 'process_batch_result')
        mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
        ns['flush_trace_batch'] = lambda reqs: None
        exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), ns)
        s.publish_load_snapshot = lambda **kw: None
        s.load_inquirer.get_loads = lambda: None
        s.load_publisher = NS(publish_load_stat=lambda *a, **kw: None)
        # Production uses a separate frozen SchedulerBatchResultProcessor;
        # it cannot see scheduler-owned tracker state via getattr(self, ...).
        s.batch_result_processor = NS(process_batch_result_prefill=lambda b, r: setattr(
            rider.time_stats, 'prefill_finished_time', time.perf_counter()))
        for name in ('_record_step_counters', '_maybe_clear_mm_inputs', 'maybe_send_health_check_signal'):
            setattr(s, name, lambda *a: None)
        s.metrics_reporter = NS(log_batch_result_stats=lambda *a: None, update_device_timer=lambda: None)
        batch = NS(reqs=[rider], chunked_req=None, is_dllm=lambda: False,
                   forward_mode=NS(is_decode=lambda: False, is_extend=lambda: True))
        with self.assertLogs('p120', level='INFO') as logs:
            ns['process_batch_result'](s, batch, None)
        self.assertTrue(any('[ax-prefix-finish]' in line for line in logs.output))

    def test_adjacent_lcp_acceleration_matches_exact_pairs(self):
        rng = random.Random(2026)
        reqs = []
        for i in range(32):
            r = request(str(i), shared=4096, tail=64)
            split = rng.randrange(1, 4096)
            r.origin_input_ids[split:] = [rng.randrange(2, 10) for _ in range(r.seqlen - split)]
            r.cache_salt = str(i % 3)
            reqs.append(r)
        s, _, _ = scheduler(reqs)
        step(s)
        tracker = s._ax_prefix_tracker
        for i, a in enumerate(reqs):
            for b in reqs[i + 1:]:
                expected = 0
                if a.cache_salt == b.cache_salt:
                    for x, y in zip(a.origin_input_ids, b.origin_input_ids):
                        if x != y:
                            break
                        expected += 1
                self.assertEqual(tracker._shared(a, b), expected)

    def test_default_off_matches_741_scheduling_traces(self):
        results = []
        with patch.dict(os.environ, {'SGLANG_AX_PREFIX_PRODUCER': '0',
                                    'SGLANG_AX_PREFIX_MAX_READY': 'malformed_ignored_when_off'}):
            for root in (tree_dir('741f3eda'), TREE):
                with patch('time.perf_counter', return_value=500):
                    reqs = [request('a'), request('b', shared=0, tail=28000), request('c', tail=2048)]
                    s, _, _ = scheduler(reqs, root=root)
                    results.append([step(s) for _ in range(25)])
        self.assertEqual(*results)

    def test_legacy_family_and_unsupported_modes_refuse(self):
        for knobs, attr in (({'SGLANG_AX_DEADLINE_FAMILY': '1'}, None),
                            ({'SGLANG_AX_DEADLINE_TIERS': '0'}, None),
                            ({}, 'enable_priority_scheduling')):
            with self.subTest(knobs=knobs, attr=attr), patch.dict(os.environ, knobs):
                s, _, _ = scheduler([request('a')])
                if attr:
                    setattr(s, attr, True)
                with self.assertRaises(ValueError):
                    s._ax_admission_cfgs()


class NativeCacheContracts(unittest.TestCase):
    def native_mamba(self, ns):
        path = TREE / 'srt/mem_cache/unified_cache/components/mamba_component.py'
        tree = ast.parse(path.read_text())
        source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MambaComponent')
        cls = ast.ClassDef(name='Mamba', bases=[], keywords=[], decorator_list=[], body=[
            n for n in source.body if getattr(n, 'name', '') in (
                'create_match_validator', 'finalize_match_result_in_cache')])
        mod = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
        ns['EvictParams'] = lambda **kw: NS(**kw)
        exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), ns)
        return ns['Mamba']()

    def test_native_validator_rejects_full_kv_tombstone_and_host_only_on_device(self):
        ns = load_source(TREE)
        mamba = self.native_mamba(ns)
        mamba.component_type = 'KDA'
        node = NS(component_data={'KDA': NS(value=None, host_value=None)})
        device, either = mamba.create_match_validator(True), mamba.create_match_validator(False)
        self.assertFalse(device(node))
        self.assertFalse(either(node))
        node.component_data['KDA'].host_value = 17
        self.assertFalse(device(node))
        self.assertTrue(either(node))
        node.component_data['KDA'].value = 23
        self.assertTrue(device(node))

    def test_native_cow_failure_is_recoverable_only_inside_ready_reservation(self):
        ns = load_source(TREE)
        mamba = self.native_mamba(ns)
        mamba.component_type = 'KDA'
        mamba.tree_core = NS(get_component_device_value=lambda *a: 7)
        mamba.cache = MagicMock()
        for reserving in (False, True):
            req = request('r')
            req._ax_prefix_reserving = reserving
            mamba.cache.req_to_token_pool.mamba_allocator.alloc.side_effect = [None, None]
            exception = ns['ax_prefix_readiness'].ReservationUnavailable if reserving else AssertionError
            with patch.dict(sys.modules, {'sglang.srt.mem_cache.ax_prefix_readiness': ns['ax_prefix_readiness']}):
                with self.assertRaises(exception):
                    mamba.finalize_match_result_in_cache(NS(cow_mamba=True, req=req), NS(best_match_node='node'))
            self.assertIsNone(req.kv.mamba_pool_idx)
            self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertEqual(mamba.cache.inc_lock_ref.call_count, 2)
        self.assertEqual(mamba.cache.dec_lock_ref.call_count, 2)

    def test_policy_uses_native_consumer_limit_before_matching(self):
        with patch.dict(os.environ, ENV):
            ns = load_source(TREE)
        ns['MatchPrefixParams'] = lambda **kw: NS(**kw)
        ns['RadixKey'] = lambda **kw: NS(**kw)
        ns['envs'] = NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False))
        compile_nodes(TREE / 'srt/managers/schedule_policy.py', {'match_prefix_for_req'}, ns)
        req = request('r', shared=16384, tail=0)
        cache = NS(swa_reprefill_tail_tokens=lambda: 0)
        observed = []
        def match(params):
            observed.append(params.key.limit)
            # 16384 is cached; a split at limit=16383 has no state. Native
            # validation returns the previous real checkpoint, not 16383.
            return NS(device_indices=[0] * 8192, last_device_node='checkpoint', last_host_node='checkpoint',
                      best_match_node='checkpoint', host_hit_length=0, swa_host_hit_length=0,
                      mamba_host_hit_length=0, full_kv_hit_length=16320,
                      mamba_branching_seqlen=16320, cache_protected_len=8192)
        cache.match_prefix = match
        ns['match_prefix_for_req'](cache, req, include_req=True)
        self.assertEqual(observed, [16383])
        self.assertEqual(req._ax_prefix_match.device, 8192)
        self.assertEqual(req._ax_prefix_match.reason(req.seqlen, 4096), 'checkpoint_tail_long')


if __name__ == '__main__':
    unittest.main()
