"""Opt-in 120 fallback: real scheduler/adder, CPU fake pools/cache/forwards."""
import os
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

from test_sched_protect_chain import ROOT, Req, make_scheduler, step
from test_admission_trace import module as collector

CURRENT = ROOT / 'engine/sglang'


class KvAdmissionScanTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '4096',
            'SGLANG_AX_SCHED_SHORT_TOKENS': '8192', 'SGLANG_AX_SCHED_KV_SCAN': '4',
            'SGLANG_AX_PACE_TPOT': '0', 'SGLANG_AX_SRPT_AGING': '0',
            'SGLANG_AX_ADMISSION_TRACE': '0',
        })
        env.start()
        self.addCleanup(env.stop)

    def build(self, tail=(), **kw):
        kwargs = dict(waiting=[Req('head', 6000, cached=16384), *tail],
                      running=[Req('decoder', 1, output=100)], available=5000)
        kwargs.update(kw)
        s, ns = make_scheduler(CURRENT, **kwargs)
        s.enable_hierarchical_cache = True
        return s, ns

    def test_rejected_head_now_admits_feasible_tail(self):
        traces = []
        for limit in ('0', '4'):
            with patch.dict(os.environ, {'SGLANG_AX_SCHED_KV_SCAN': limit}):
                s, _ = self.build([Req('tail', 256, cached=4096)])
                traces.append(step(s))
        self.assertEqual(traces[0]['mode'], 'decode')
        self.assertEqual(traces[0]['waiting'], ['head', 'tail'])
        self.assertEqual(traces[1]['mode'], 'prefill')
        self.assertEqual([r[0] for r in traces[1]['reqs']], ['tail'])
        self.assertEqual(traces[1]['waiting'], ['head'])
        self.assertEqual(traces[1]['verdicts'], [('head', 'NO_TOKEN'), ('tail', 'CONTINUE')])
        self.assertTrue(traces[1]['full'])  # No change to latch-reset policy.
        self.assertGreater(s.adders[-1].rem_total_tokens, 0)

    def test_limit_counts_inspected_positions_including_ineligible(self):
        for limit, admitted in ((1, False), (2, True)):
            with patch.dict(os.environ, {'SGLANG_AX_SCHED_KV_SCAN': str(limit)}):
                host = Req('host', 4160, cached=8192, host=4096)
                s, _ = self.build([host, Req('tail', 256, cached=4096)])
                t = step(s)
                self.assertEqual('tail' not in t['waiting'], admitted)
                self.assertIn('host', t['waiting'])
                s.tree_cache.init_load_back.assert_not_called()

    def test_ownership_exclusions_do_not_match_or_free_session_slots(self):
        session = Req('session', 256, cached=12000)
        session.session = object()
        beam = Req('beam', 256, cached=10000)
        beam.beam_group = NS(beam_width=2)
        for req in (session, beam):
            req.init_next_round_input = MagicMock(side_effect=AssertionError('must not match'))
        s, _ = self.build([session, beam, Req('tail', 256, cached=4096)])
        self.assertEqual(step(s)['waiting'], ['head', 'session', 'beam'])

    def test_all_failed_matches_release_cow_and_fresh_state_once(self):
        host = Req('host', 4160, cached=12000, host=4096)
        too_big = Req('big', 6000, cached=10000)
        tail = Req('tail', 256, cached=4096)
        s, _ = self.build([host, too_big, tail])
        rejected = s.waiting_queue[:-1]
        s.tree_cache.allocate_on_match = True
        group = MagicMock()
        s.req_to_token_pool.mamba_allocator = group
        step(s)
        self.assertEqual(s.tree_cache.req_to_token_pool.mamba_allocator.free.call_count, 3)
        for req in rejected:
            self.assertIsNone(req.kv.mamba_pool_idx)
            self.assertIsNone(req.kv.mamba_cow_src_index)
            self.assertFalse(req.kv.mamba_needs_clear)
        self.assertIsNotNone(tail.kv.mamba_pool_idx)
        group.alloc_group_begin.assert_called_once_with(4)
        group.alloc_group_end.assert_called_once_with()

    def test_request_slots_stop_fallback_without_matching_another_request(self):
        tail = [Req(f't{i}', 256, cached=4096-i*64) for i in range(3)]
        tail[1].init_next_round_input = MagicMock(side_effect=AssertionError('no slots'))
        s, _ = self.build(tail, slots=1)
        self.assertEqual([r[0] for r in step(s)['reqs']], ['t0'])

    def test_future_output_reservation_is_not_relaxed(self):
        costly = Req('short_but_long_decode', 256, cached=8192, output=4096)
        s, _ = self.build([costly, Req('tail', 256, cached=4096)], available=4000)
        t = step(s)
        self.assertEqual([r[0] for r in t['reqs']], ['tail'])
        self.assertIn(('short_but_long_decode', 'NO_TOKEN'), t['verdicts'])

    def test_marginal_kv_budget_includes_page_rounding(self):
        # Legacy gate: 257 + 240 + 64 < 580; actual charge is 320+240+64.
        s, _ = self.build([Req('tail', 257, cached=4096)], available=680)
        self.assertEqual(step(s)['mode'], 'decode')
        self.assertEqual(s.adders[-1].can_run_list, [])

    def test_retracted_output_uses_full_native_charge(self):
        tail = Req('tail', 256, cached=4096, output=4096)
        tail.output_ids = [1] * 4000
        tail.retracted_stain = True
        s, _ = self.build([tail], available=1000)
        self.assertEqual(step(s)['mode'], 'decode')
        self.assertEqual(s.adders[-1].can_run_list, [])

    def test_shared_state_shortage_is_not_funded_by_full_kv(self):
        s, ns = self.build([Req('tail', 256, cached=4096)])
        pool = ns['UnifiedMambaTokenToKVPoolAllocator']()
        pool.available_size = lambda: 5000
        pool.mamba_slot_full_token_cost = lambda: 1000
        pool.mamba_allocator = NS(schedulable_available_size=lambda: 0)
        s.token_to_kv_pool_allocator = pool
        self.assertEqual(step(s)['mode'], 'decode')
        self.assertEqual(s.adders[-1].rem_mamba_slots, 0)

    def test_shared_state_cost_is_charged_for_admitted_tail(self):
        s, ns = self.build([Req('tail', 256, cached=4096)])
        pool = ns['UnifiedMambaTokenToKVPoolAllocator']()
        pool.available_size = lambda: 5000
        pool.mamba_slot_full_token_cost = lambda: 1000
        pool.mamba_allocator = NS(schedulable_available_size=lambda: 1)
        s.token_to_kv_pool_allocator = pool
        self.assertEqual([r[0] for r in step(s)['reqs']], ['tail'])
        self.assertEqual(s.adders[-1].rem_mamba_slots, 0)
        self.assertEqual(s.adders[-1].rem_total_token_offset, 100 + 256 + 240 + 64 + 1000)

    def test_recheck_under_prefix_lock_and_release_on_rejection(self):
        tail = Req('tail', 256, cached=4096)
        s, _ = self.build([tail])
        # The head fails before locking. Locking the tail removes evictable KV.
        s.tree_cache.inc_lock_ref.side_effect = lambda *a, **kw: setattr(
            s.token_to_kv_pool_allocator.available_size, 'return_value', 500)
        s.tree_cache.allocate_on_match = True
        t = step(s)
        self.assertEqual(t['mode'], 'decode')
        self.assertEqual(t['verdicts'], [('head', 'NO_TOKEN'), ('tail', 'NO_TOKEN')])
        s.tree_cache.dec_lock_ref.assert_called_once()
        self.assertIsNone(tail.kv.mamba_pool_idx)

    def test_active_partial_remains_unique_and_has_budget_first(self):
        s, _ = self.build([Req('tail', 256, cached=4096)], chunk=Req('partial', 30000),
                          available=9000)
        t = step(s)
        self.assertEqual([r[0] for r in t['reqs']], ['partial', 'tail'])
        self.assertEqual(t['chunk'], 'partial')
        self.assertEqual(s.adders[-1].new_chunked_req, None)
        self.assertGreaterEqual(s.adders[-1].rem_chunk_tokens, 0)

    def test_complete_only_check_uses_page_rounding_and_input_budget(self):
        for budget in (256, 512):
            tail = Req('tail', 257, cached=4096)
            s, _ = self.build([tail], budget=budget)
            s.max_prefill_tokens = 256
            t = step(s)
            self.assertEqual(t['mode'], 'decode')
            self.assertIn('tail', t['waiting'])
            self.assertIsNone(s.chunked_req)

    def test_successful_head_does_not_trigger_fallback_on_budget_stop(self):
        s, ns = self.build([Req('tail', 256, cached=4096)], available=100000)
        # Exercise the native post-admission return contract explicitly: a
        # budget result alone cannot tell whether the candidate was admitted.
        factory = ns['PrefillAdder']
        def with_no_token_result(*a, **kw):
            adder = factory(*a, **kw)
            adder.budget_state = lambda: ns['AddReqResult'].NO_TOKEN
            return adder
        ns['PrefillAdder'] = with_no_token_result
        s._ax_scan_after_kv_rejection = MagicMock(side_effect=AssertionError('head was added'))
        self.assertEqual([r[0] for r in step(s)['reqs']], ['head'])

    def test_successful_fallback_stops_on_budget_result(self):
        s, ns = self.build([Req('first', 256, cached=8192), Req('next', 256, cached=4096)])
        factory = ns['PrefillAdder']
        def with_no_token_result(*a, **kw):
            adder = factory(*a, **kw)
            adder.budget_state = lambda: ns['AddReqResult'].NO_TOKEN
            return adder
        ns['PrefillAdder'] = with_no_token_result
        self.assertEqual([r[0] for r in step(s)['reqs']], ['first'])

    def test_visited_tail_not_reported_as_unscanned(self):
        with patch.dict(os.environ, {'SGLANG_AX_SCHED_KV_SCAN': '1'}):
            visited = Req('visited', 256, cached=8192)
            unseen = Req('unseen', 256, cached=4096)
            s, _ = self.build([visited, unseen])
            s._ax_admission_collector = collector.AxAdmissionTrace(MagicMock())
            step(s)
            self.assertIsNone(visited._ax_admission_trace_state)
            self.assertEqual(unseen._ax_admission_trace_state['decisions'], {'unscanned_after_kv_scan': 1})

    def test_invalid_or_unsupported_configuration_refused(self):
        for limit in ('-1', '17', 'bogus'):
            with patch.dict(os.environ, {'SGLANG_AX_SCHED_KV_SCAN': limit}):
                s, _ = self.build()
                with self.assertRaises(ValueError):
                    s._ax_kv_scan_limit()
        for attr, value in (('is_mixed_chunk', True), ('enable_hicache_storage', True),
                            ('enable_priority_preemption', True), ('require_mlp_sync', True)):
            s, _ = self.build()
            setattr(s, attr, value)
            with self.assertRaises(ValueError):
                s._ax_kv_scan_limit()
        s, _ = self.build(role=True)
        with self.assertRaisesRegex(ValueError, 'role_boundary'):
            s._ax_kv_scan_limit()

    def test_random_fallback_retains_resource_and_single_partial_invariants(self):
        for seed in range(40):
            rng = random.Random(seed)
            tails = [Req(f't{i}', rng.choice([1, 256, 257, 1024, 6000, 9000]),
                         cached=(12-i)*256, output=rng.choice([1, 240, 4096])) for i in range(8)]
            s, _ = self.build(tails, available=rng.randint(800, 5000),
                              budget=rng.choice([256, 1024, 8192]), slots=rng.randint(1, 4))
            step(s)
            adder = s.adders[-1]
            self.assertIsNone(adder.new_chunked_req)
            self.assertGreaterEqual(adder.rem_chunk_tokens, 0)
            self.assertGreaterEqual(adder.rem_input_tokens, 0)
            if adder.can_run_list:
                self.assertGreater(adder.rem_total_tokens, 0)
            self.assertLessEqual(len(adder.can_run_list), s.req_to_token_pool.available_size())
            for req in adder.can_run_list:
                self.assertEqual(req.extend_range.end, len(req.full_untruncated_fill_ids))


if __name__ == '__main__':
    unittest.main()
