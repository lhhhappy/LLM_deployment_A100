"""124m production scheduler/adder/result callbacks with CPU pools and a clock.

These tests exercise ownership and callback ordering; they do not validate GPU
KV/KDA contents. The TP8 boundary/abort probe remains required before rollout.
"""
import ast
import copy
import logging
import os
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from test_sched_protect_chain import Batch, Req, make_scheduler

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "engine/sglang"
ENV = {
    "SGLANG_AX_SCHED_PROTECT": "1",
    "SGLANG_AX_SCHED_COLD_CAP": "16384",
    "SGLANG_AX_SCHED_SHORT_TOKENS": "2048",
    "SGLANG_AX_DEADLINE_TIERS": "1",
    "SGLANG_AX_DEADLINE_MAX_WAIT_S": "600",
    "SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S": "120",
    "SGLANG_AX_DEADLINE_LOAD": "1.05",
    "SGLANG_AX_MULTI_ROUND_PARK": "1",
}


def methods(path, cls_name, names, ns):
    source = ast.parse(path.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    chosen = [n for n in cls.body if getattr(n, "name", "") in names]
    for n in chosen:
        n.decorator_list = []
    mod = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
                          *chosen], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), ns)
    return {n.name: ns[n.name] for n in chosen}


def request(rid, work, *, computed=0, age=0, matched=0):
    r = Req(rid, work, cached=computed)
    r.num_matched_prefix_tokens = matched
    r.time_stats.scheduler_recv_time = 1000.0 - age
    r.time_stats.prefill_finished_time = 0.0
    r.time_stats.set_prefill_finished_time = lambda: setattr(r.time_stats, "prefill_finished_time", 1000.0)
    r.time_stats.set_last_chunked_prefill_finish_time = lambda: None
    r.time_stats.trace_ctx = MagicMock()
    r.time_stats.set_completion_time = lambda: None
    r.kv.req_pool_idx = 1 if computed else None
    r.kv.cache_protected_len = computed // 256 * 256
    r.kv.kv_allocated_len = computed
    r.kv.holds_kv = bool(computed)
    r.is_retracted = False
    r.return_sampling_mask = False
    r.grammar = None
    r.update_finish_state = lambda: None
    r.to_finish = None
    return r


class MultiRoundTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.clock = patch("time.perf_counter", return_value=1000.0).start()
        self.addCleanup(patch.stopall)

    def scheduler(self, a=None, b=None, **kw):
        a = a or request("A", 200000, computed=16384, age=20)
        b = b or request("B", 35000, age=2)
        s, ns = make_scheduler(TREE, chunk=a, waiting=[b], budget=16384, **kw)
        s.server_args = NS(enable_unified_memory=False)
        s.forward_stream = NS(synchronize=MagicMock())
        s._ax_rank0_decide = lambda fn: fn()
        s._ax_prefix_consensus = lambda value: True
        s._ax_admission_cfgs()
        return s, ns, a, b

    def processor(self):
        ns = dict(get_memory=lambda: NS(enable_hisparse=False),
                  release_kv_cache=MagicMock(), maybe_cache_unfinished_req=MagicMock())
        method = methods(TREE / "srt/managers/scheduler_components/batch_result_processor.py",
                         "SchedulerBatchResultProcessor", {"process_batch_result_prefill"}, ns)
        p = MagicMock()
        p.is_generation = True
        p.snapshot_auxiliary_output_starts.return_value = None
        p.consume = method["process_batch_result_prefill"].__get__(p)
        return p, ns

    def callback(self, p, batch):
        batch.return_hidden_states = batch.return_logprob = False
        batch.decoding_reqs = None
        batch.prefill_stats = None
        result = NS(copy_done=None, auxiliary_host_output=None, routed_experts_output=None,
                    indexer_topk_output=None, logits_output=NS(),
                    next_token_ids=NS(tolist=lambda: [1] * len(batch.reqs)),
                    extend_input_len_per_req=None, extend_logprob_start_len_per_req=None,
                    can_run_cuda_graph=False)
        p.consume(batch, result)

    def schedule(self, s):
        plan = s.get_next_batch_to_run(s.running_batch, s.last_batch)
        s.running_batch = plan.running_batch
        batch = plan.batch_to_run
        s.last_batch = batch
        snapshot = copy.copy(batch) if batch else None
        if snapshot:
            snapshot.reqs = list(batch.reqs)
        return snapshot

    def test_rescue_three_chunks_then_resume_exact_live_prefix(self):
        for computed in (255, 256, 257, 16383, 16384, 16385):
            with self.subTest(computed=computed):
                a = request("A", 200000, computed=computed, age=20)
                s, _, a, b = self.scheduler(a=a)
                p, _ = self.processor()
                receive_a = a.time_stats.scheduler_recv_time
                batches = []
                for _ in range(3):
                    batch = self.schedule(s)
                    batches.append([r.rid for r in batch.reqs])
                    self.callback(p, batch)
                    s._ax_multi_park.observe(s)
                    self.assertEqual(len(a.prefix_indices), computed)
                self.assertEqual(batches, [["B"], ["B"], ["B"]])
                self.assertEqual(len(b.output_ids), 1)
                batch = self.schedule(s)
                self.assertIs(s.chunked_req, a)
                if a not in batch.reqs:  # preserve 120's decode turn
                    batch = self.schedule(s)
                self.assertIn(a, batch.reqs)
                self.assertEqual(a.extend_range.start, computed)
                self.assertEqual(a.kv.req_pool_idx, 1)
                self.assertEqual(a.time_stats.scheduler_recv_time, receive_a)
                self.assertFalse(s._ax_multi_park.parked)
                self.assertEqual(a.inflight_middle_chunks, 1)

    def test_overlap_late_a_callback_does_not_complete_or_free_it(self):
        s, _, a, b = self.scheduler()
        p, ns = self.processor()
        a.inflight_middle_chunks = 1
        old = Batch([a])
        first = self.schedule(s)
        self.assertIs(s.chunked_req, b)
        self.callback(p, old)
        self.assertEqual(a.inflight_middle_chunks, 0)
        self.assertEqual(a.output_ids, [])
        ns["release_kv_cache"].assert_not_called()
        p.output_streamer.stream_output.assert_called_with([a], False, a)
        self.callback(p, first)
        second = self.schedule(s)
        third = self.schedule(s)  # two in-flight middle callbacks on B
        self.callback(p, second)
        self.assertIsNone(s.chunked_req)  # B final chunk scheduled, not received
        self.assertEqual(s._ax_multi_park.reqs(), (a,))
        self.schedule(s)  # no third partial may be admitted at this boundary
        self.assertEqual(s._ax_multi_park.reqs(), (a,))
        self.callback(p, third)
        self.assertEqual(b.inflight_middle_chunks, 0)
        self.assertEqual(b.output_ids, [1])
        self.schedule(s)
        self.assertIs(s.chunked_req, a)

    def test_partially_cached_rescuer_keeps_its_admission_match(self):
        b = request("B", 35000, computed=8192, matched=8192, age=2)
        s, _, _, _ = self.scheduler(b=b)
        batch = self.schedule(s)
        self.assertEqual(batch.reqs, [b])
        self.assertEqual(b.extend_range.start, 8192)

    def test_native_rejection_restores_owner_same_pass(self):
        s, ns, a, b = self.scheduler()
        real_factory = ns["PrefillAdder"]
        def factory(*args, **kwargs):
            adder = real_factory(*args, **kwargs)
            adder.add_one_req = lambda *a, **kw: ns["AddReqResult"].NO_TOKEN
            return adder
        ns["PrefillAdder"] = factory
        batch = self.schedule(s)
        self.assertEqual(batch.reqs, [a])
        self.assertIs(s.chunked_req, a)
        self.assertEqual(s.waiting_queue, [b])
        self.assertFalse(s._ax_multi_park.parked)

    def test_plan_counters_do_not_classify_held_requests(self):
        s, ns, a, b = self.scheduler()
        state = s._ax_multi_park
        adder = NS(rem_total_tokens=1000000,
                   _request_total_tokens=lambda req, work: work)
        # plan() must not read a deadline class for requests excluded by hold:
        # freeze_class could otherwise change a later admission decision.
        self.assertFalse(hasattr(b, "_ax_deadline_cold"))
        s.policy.ax_held = [b.rid]
        with patch.object(ns["ax_multiround_park"], "logger") as log:
            state.plan(s, adder, s.running_batch, 16384, None)
            self.assertEqual(state.plan_stats.rejections["b_held"], 1)
            self.assertEqual(state.plan_stats.outcomes["none"], 1)
            self.assertFalse(hasattr(b, "_ax_deadline_cold"))
            self.clock.return_value = 1030.0
            state.plan(s, adder, s.running_batch, 16384, None)
            self.assertEqual(log.info.call_count, 1)
            self.assertEqual(log.info.call_args.args[0], "[ax-124m-plan] %s")
        self.assertIs(s.chunked_req, a)
        self.assertEqual(s.waiting_queue, [b])
        self.assertFalse(state.parked)

    def test_observed_rescue_decision_matches_unobserved(self):
        s, ns, a, b = self.scheduler()
        module = ns["ax_multiround_park"]
        cfg = s._ax_admission_cfgs()[0]
        for work, waited in ((35000, 2), (1200, 2), (35000, 40)):
            with self.subTest(work=work, waited=waited):
                b = request("B", work, age=waited)
                plain_a, plain_b = copy.deepcopy(a), copy.deepcopy(b)
                stats = module.PlanStats()
                plain = module.rescue_decision(plain_a, plain_b, 1000, 16384, cfg)
                observed = module.rescue_decision(a, b, 1000, 16384, cfg, stats)
                self.assertEqual(plain, observed)
                self.assertEqual(getattr(plain_b, "_ax_deadline_cold", None),
                                 getattr(b, "_ax_deadline_cold", None))

    def test_150k_owner_with_30s_cold_budget_is_not_dead_at_35k_arrival(self):
        # ezn9 raw, server t_recv difference: B arrived 8.000929 s after A.
        # Give A all its original work (a pessimistic bound); real progress
        # only improves its slack. This is a CPU counterfactual, not a replay.
        a = request("A", 150249, matched=14848, age=8.000929355621338)
        b = request("B", 35296, matched=0, age=0)
        s, ns, _, _ = self.scheduler(a=a, b=b)
        stats = ns["ax_multiround_park"].PlanStats()
        decision = ns["ax_multiround_park"].rescue_decision(
            a, b, 1000, 16384, s._ax_admission_cfgs()[0], stats)
        self.assertIsNone(decision)
        self.assertEqual(stats.rejections, {"owner_rescuable": 1})
        self.assertGreater(stats.examples["owner_rescuable"]["a_slack"], 9)

    def test_lease_and_absolute_age_restore_without_nested_third_partial(self):
        for reason in ("lease", "age"):
            with self.subTest(reason=reason):
                s, _, a, b = self.scheduler()
                self.schedule(s)
                c = request("C", 24000)
                s.waiting_queue.append(c)
                if reason == "lease":
                    self.clock.return_value = s._ax_multi_park.lease_end + 0.01
                else:
                    a.time_stats.scheduler_recv_time = 430.0
                    self.clock.return_value = 1015.0
                batch = self.schedule(s)
                self.assertIn(a, batch.reqs)
                self.assertEqual(s._ax_multi_park.reqs(), (b,))
                self.assertTrue(s._ax_multi_park.protect)
                self.assertIn(c, s.waiting_queue)
                self.clock.return_value = 1100.0
                self.schedule(s)
                self.assertIs(s.chunked_req, a)
                self.clock.return_value = 1000.0

    def test_last_chunk_or_rescuable_owner_is_not_preempted(self):
        for left, waited in ((12000, 40), (35000, 0)):
            s, _, a, _ = self.scheduler(a=request("A", left, computed=16384, age=waited))
            batch = self.schedule(s)
            self.assertIn(a, batch.reqs)
            self.assertFalse(s._ax_multi_park.parked)

    def test_pending_and_uncached_token_counts_include_paused_tail(self):
        s, _, a, b = self.scheduler()
        self.schedule(s)
        ns = dict(DisaggregationMode=NS(DECODE="decode"))
        funcs = methods(TREE / "srt/managers/scheduler_components/load_inquirer.py",
                        "SchedulerLoadInquirer", {"_get_num_pending_tokens", "get_num_waiting_uncached_tokens"}, ns)
        q = NS(get_waiting_queue=lambda: [], get_chunked_req=lambda: b,
               get_parked_reqs=lambda: (a,), disaggregation_mode="null",
               waiting_queue_prefix_matched=lambda: True, get_recent_cache_hit_rate=lambda: 0)
        self.assertEqual(funcs["_get_num_pending_tokens"](q, chunk_deduct=16384), 218616)
        self.assertEqual(funcs["get_num_waiting_uncached_tokens"](q), 235000)

    def test_busy_memory_accounting_deduplicates_parked_last_batch(self):
        s, _, a, _ = self.scheduler()
        self.schedule(s)
        a.kv.kv_allocated_len = 16449
        a.kv.cache_protected_len = 16384
        ns = dict(ceil_align=lambda n, p: ((n + p - 1) // p) * p)
        fn = methods(TREE / "srt/managers/scheduler_components/invariant_checker.py",
                     "SchedulerInvariantChecker", {"_get_total_uncached_sizes"}, ns)["_get_total_uncached_sizes"]
        for last in (Batch([a]), Batch([])):
            obj = NS(get_last_batch=lambda: last, get_running_batch=lambda: Batch([]),
                     get_parked_reqs=lambda: (a,), page_size=64, is_hybrid_swa=False)
            self.assertEqual(fn(obj), (128, 0))

    def lifecycle(self, s, ns):
        ns.update(logging=logging, prepare_abort=lambda r, msg: setattr(r.sampling_params, "max_new_tokens", 0),
                  _make_abort_req=lambda r: r.rid, AbortReq=lambda **kw: NS(abort_all=False, **kw),
                  FINISH_ABORT=lambda: "abort", release_kv_cache=MagicMock())
        ns["DisaggregationMode"].DECODE = "decode"
        names = {"abort_request", "collect_inflight_reqs", "process_pending_chunked_abort",
                 "_abort_partial_req", "is_fully_idle", "_pp_microbatches_drained",
                 "flush_cache", "pause_generation"}
        for name, fn in methods(TREE / "srt/managers/scheduler.py", "Scheduler", names, ns).items():
            setattr(s, name, fn.__get__(s))
        s._pending_chunked_abort_req = None
        s.mm_receiver = None
        s.dllm_manager = NS(any_staging_reqs=lambda: False)
        s.grammar_manager.grammar_queue = []
        s.grammar_manager.abort_requests = lambda _: None
        s.grammar_manager.clear = MagicMock()
        s.ipc_channels = NS(send_to_tokenizer=MagicMock())
        s._engine_paused = False
        s.req_to_token_pool.clear = MagicMock()
        s.req_to_token_pool.reset_aux_cache_allocator = MagicMock()
        s.metrics_reporter = MagicMock()
        s.kv_events_publisher = MagicMock()
        s.draft_worker = None
        s.hisparse_coordinator = None
        s._add_request_to_queue = s.waiting_queue.append
        s.beam_coordinator.retire_group = lambda _: None
        ns["retract_all"] = MagicMock()

    def test_abort_each_owner_and_abort_all_release_once_then_flush(self):
        for target in ("A", "B", "all"):
            with self.subTest(target=target):
                s, ns, a, b = self.scheduler()
                self.schedule(s)
                self.lifecycle(s, ns)
                self.assertFalse(s.flush_cache(empty_cache=False))
                s.tree_cache.reset.assert_not_called()
                s.abort_request(NS(rid=target, abort_all=target == "all"))
                s.process_pending_chunked_abort()
                released = [call.args[0] for call in ns["release_kv_cache"].call_args_list]
                self.assertEqual(set(released), {a, b} if target == "all" else {a if target == "A" else b})
                s.process_pending_chunked_abort()
                self.assertEqual(len(ns["release_kv_cache"].call_args_list), len(released))
                if target == "B":
                    self.schedule(s)
                    self.assertIs(s.chunked_req, a)
                if target == "all":
                    # The already-launched middle chunk must still drain its
                    # counter; it must not free the request for a second time.
                    p, pns = self.processor()
                    self.callback(p, Batch([b]))
                    self.assertEqual(b.inflight_middle_chunks, 0)
                    self.assertEqual(b.output_ids, [])
                    pns["release_kv_cache"].assert_not_called()
                    s.last_batch = None
                    s.running_batch.reqs = []
                    self.assertTrue(s.flush_cache(empty_cache=False))
                    s.req_to_token_pool.clear.assert_called_once()
                    s.tree_cache.reset.assert_called_once()

    def test_retract_pause_returns_both_requests_without_resetting_receive_age(self):
        s, ns, a, b = self.scheduler()
        self.schedule(s)
        self.lifecycle(s, ns)
        stamps = [r.time_stats.scheduler_recv_time for r in (a, b)]
        s.pause_generation(NS(mode="in_place"))
        self.assertEqual(s._ax_multi_park.reqs(), (a,))
        s.pause_generation(NS(mode="retract"))
        reqs = ns["retract_all"].call_args.kwargs["reqs"]
        self.assertEqual(set(reqs), {a, b})
        self.assertEqual(len(reqs), 2)
        self.assertEqual(set(s.waiting_queue), {a, b})
        self.assertIsNone(s.chunked_req)
        self.assertFalse(s._ax_multi_park.parked)
        self.assertEqual([r.time_stats.scheduler_recv_time for r in (a, b)], stamps)

    def test_rank0_controls_rescue_with_follower_clock_skew(self):
        payloads = []
        states = []
        for rank in (0, 1):
            s, _, a, b = self.scheduler()
            if rank:
                # Follower would judge the opposite if it evaluated deadlines.
                a.time_stats.scheduler_recv_time = 999.0
                b.time_stats.scheduler_recv_time = 900.0
            cursor = iter(payloads)
            def decide(fn):
                if rank == 0:
                    result = fn()
                    payloads.append(result)
                    return result
                return next(cursor)
            s._ax_rank0_decide = decide
            batch = self.schedule(s)
            states.append(([r.rid for r in batch.reqs], s.chunked_req.rid,
                           [r.rid for r in s._ax_multi_park.reqs()]))
            if rank:
                self.assertEqual(list(cursor), [])
        self.assertEqual(states, [(["B"], "B", ["A"])] * 2)

    def test_off_does_not_create_second_partial(self):
        with patch.dict(os.environ, {"SGLANG_AX_MULTI_ROUND_PARK": "0"}):
            s, _, a, b = self.scheduler()
            batch = self.schedule(s)
            self.assertEqual(batch.reqs, [a])
            self.assertEqual(s.waiting_queue, [b])
            self.assertFalse(hasattr(s, "_ax_multi_park"))

    def test_128p_tracks_parked_owner_and_does_not_reserve_away_rescue(self):
        with patch.dict(os.environ, {"SGLANG_AX_PREFIX_PRODUCER": "1",
                                     "SGLANG_AX_DEADLINE_FREEZE_CLASS": "1"}):
            s, _, a, b = self.scheduler()
            p, _ = self.processor()
            first = self.schedule(s)
            self.assertEqual(first.reqs, [b])
            self.callback(p, first)
            second = self.schedule(s)
            self.assertEqual(second.reqs, [b])
            self.assertIn(a.rid, s._ax_prefix_tracker.records)
            self.assertEqual(s._ax_multi_park.reqs(), (a,))

    def test_insufficient_kv_and_slot_capacity_do_not_park(self):
        for kwargs in ({"available": 25000}, {"slots": 0}):
            s, _, a, b = self.scheduler(**kwargs)
            batch = self.schedule(s)
            self.assertIn(a, batch.reqs)
            self.assertIs(s.chunked_req, a)
            self.assertEqual(s.waiting_queue, [b])
            self.assertFalse(s._ax_multi_park.parked)

    def test_rescuer_final_tail_cannot_role_split_third_partial(self):
        s, _, a, b = self.scheduler(role=True)
        p, _ = self.processor()
        for _ in range(2):
            self.callback(p, self.schedule(s))
        c = request("C", 1024, computed=32768, matched=32768)
        c.origin_input_ids[32768 + 512] = 99
        c.full_untruncated_fill_ids = c.origin_input_ids[:]
        s.waiting_queue.append(c)
        final = self.schedule(s)
        self.assertEqual(final.reqs, [b, c])
        self.assertIsNone(s.chunked_req)
        self.assertEqual(c.extend_range.length, 1024)
        self.callback(p, final)
        self.schedule(s)
        self.assertIs(s.chunked_req, a)
        self.assertEqual(c.output_ids, [1])


if __name__ == "__main__":
    unittest.main()
