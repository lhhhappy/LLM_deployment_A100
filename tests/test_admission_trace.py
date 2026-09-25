"""Real scheduler CPU regressions: telemetry must preserve admission and budgets."""
import importlib.util
import json
import os
import random
import sys
import unittest
from unittest.mock import MagicMock, patch

from test_sched_protect_chain import ROOT, Req, make_scheduler, step, tree_dir

SPEC = importlib.util.spec_from_file_location(
    "sglang.srt.managers.ax_admission_trace",
    ROOT / "engine/sglang/srt/managers/ax_admission_trace.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
CURRENT = ROOT / "engine/sglang"
FROZEN = tree_dir("759a6ebb8e31723519ad5daf438e26e24b32501a")


class AdmissionTraceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "SGLANG_AX_SCHED_PROTECT": "1", "SGLANG_AX_SCHED_COLD_CAP": "2048",
            "SGLANG_AX_SCHED_SHORT_TOKENS": "4096", "SGLANG_AX_PACE_TPOT": "0",
            "SGLANG_AX_ADMISSION_TRACE": "0",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.mods = patch.dict(sys.modules, {SPEC.name: module})
        self.mods.start()
        self.addCleanup(self.mods.stop)

    def traced(self, **kwargs):
        s, ns = make_scheduler(CURRENT, **kwargs)
        s._ax_admission_collector = module.AxAdmissionTrace(MagicMock())
        return s, ns

    def reasons(self, req):
        return req._ax_admission_trace_state["decisions"]

    def test_off_and_on_preserve_frozen_scheduling_and_pool_budgets(self):
        for seed in (0, 1, 2):
            traces = []
            for tree, enabled in ((FROZEN, False), (CURRENT, False), (CURRENT, True)):
                rng = random.Random(seed)
                s, _ = make_scheduler(tree, running=[Req("decoder", 1, output=20)])
                if enabled:
                    s._ax_admission_collector = module.AxAdmissionTrace(MagicMock())
                trace = []
                for i in range(80):
                    arrivals = []
                    if rng.random() < .3:
                        arrivals.append(Req(f"cold{i}", rng.randint(1000, 30000)))
                    if rng.random() < .4:
                        arrivals.append(Req(f"hit{i}", rng.randint(1, 6000), cached=32768))
                    trace.append(step(s, arrivals))
                traces.append(trace)
            self.assertEqual(traces[0], traces[1])
            self.assertEqual(traces[0], traces[2])

    def test_distinguish_partial_host_cold_tail_and_leftover_budget(self):
        host = Req("host", 64, cached=80000, host=64)
        long = Req("long", 5000, cached=70000)
        fits = Req("fits", 4096, cached=60000)
        budget = Req("budget", 3000, cached=50000)
        cold = Req("cold", 64)
        s, _ = self.traced(chunk=Req("partial", 20000), waiting=[host, long, fits, budget, cold])
        trace = step(s)
        self.assertEqual([x[0] for x in trace["reqs"]], ["partial", "fits"])
        for req, reason in ((host, "partial_host_restore"), (long, "partial_long_tail"),
                            (budget, "partial_token_budget"), (cold, "partial_no_device_prefix")):
            self.assertEqual(self.reasons(req), {reason: 1})
        self.assertIsNone(fits._ax_admission_trace_state)
        s.tree_cache.init_load_back.assert_not_called()

    def test_paced_prefill_trace_preserves_budget_and_decode_decisions(self):
        from test_tpot_paced_prefill import Clock, advance

        traces = []
        with patch.dict(os.environ, {"SGLANG_AX_PACE_TPOT": "0.085"}):
            for tree, enabled in ((FROZEN, False), (CURRENT, True)):
                clock = Clock()
                s, ns = make_scheduler(tree, running=[Req("dec", 1, output=150)], interval=2)
                ns["time"] = clock
                if enabled:
                    s._ax_admission_collector = module.AxAdmissionTrace(MagicMock(), clock=clock.monotonic)
                trace = []
                for i in range(80):
                    arrivals = [Req(f"cold{i}", 15000)] if i % 20 == 0 else []
                    if i % 7 == 0:
                        arrivals.append(Req(f"hit{i}", 1024, cached=32768))
                    t = step(s, arrivals)
                    trace.append(t)
                    advance(clock, t)
                traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_kv_rejection_does_not_mislabel_unvisited_tail(self):
        head, tail = Req("head", 4096, cached=1000), Req("tail", 64, cached=512)
        s, _ = self.traced(chunk=Req("partial", 20000), waiting=[head, tail], available=5000)
        step(s)
        self.assertEqual(self.reasons(head), {"kv_budget": 1})
        self.assertEqual(self.reasons(tail), {"unscanned_after_no_token": 1})
        self.assertTrue(s.running_batch.batch_is_full)

    def test_cadence_slots_and_latched_full_are_distinct(self):
        waiting = Req("wait", 64, cached=1024)
        s, _ = self.traced(waiting=[waiting], running=[Req("dec", 1)], interval=2)
        s._prefill_decode_interval_remaining = 1
        step(s)
        self.assertEqual(self.reasons(waiting), {"decode_cadence": 1})
        s.req_to_token_pool.available_size = lambda: 0
        step(s)
        self.assertEqual(self.reasons(waiting)["request_slots"], 1)
        step(s)
        self.assertEqual(self.reasons(waiting)["batch_full_latched"], 1)

    def test_rank_zero_only_and_off_does_not_attach_request_state(self):
        req = Req("new", 64, cached=1024)
        s, _ = make_scheduler(CURRENT, waiting=[req])
        step(s)
        self.assertFalse(hasattr(req, "_ax_admission_trace_state"))
        with patch.dict(os.environ, {"SGLANG_AX_ADMISSION_TRACE": "1"}):
            s, _ = make_scheduler(CURRENT)
            s.ps.tp_rank = 1
            self.assertIsNone(s._ax_admission_trace())
            s, _ = make_scheduler(CURRENT)
            self.assertIsInstance(s._ax_admission_trace(), module.AxAdmissionTrace)

    def test_snapshot_bounded_and_admission_resets_retracted_request(self):
        logger = MagicMock()
        now = [10.0]
        trace = module.AxAdmissionTrace(logger, clock=lambda: now[0])
        reqs = [Req(str(i), 64) for i in range(40)]
        trace.record_many(reqs, "decode_cadence")
        trace.snapshot(reqs)
        msg = json.loads(logger.info.call_args.args[1])
        self.assertEqual(msg["queue_size"], 40)
        self.assertEqual(len(msg["sample"]), 32)
        self.assertTrue(msg["truncated"])
        trace.snapshot(reqs)
        self.assertEqual(logger.info.call_count, 1)
        now[0] = 40.0
        trace.snapshot(reqs)
        self.assertEqual(logger.info.call_count, 2)
        trace.admitted(reqs[0])
        self.assertIsNone(reqs[0]._ax_admission_trace_state)
        trace.begin_attempt(reqs[0])
        self.assertEqual(self.reasons(reqs[0]), {})
        self.assertFalse(any(isinstance(v, dict) for v in vars(trace).values()))

    def test_byte_budget_stops_logging_and_observation(self):
        logger = MagicMock()
        trace = module.AxAdmissionTrace(logger, max_bytes=2048)
        for i in range(100):
            req = Req("请求" + str(i), 64)
            trace.record(req, "decode_cadence")
            trace.admitted(req)
        self.assertTrue(trace.exhausted)
        self.assertLessEqual(trace.bytes_emitted, 2048)
        payloads = [json.loads(call.args[1]) for call in logger.info.call_args_list]
        self.assertEqual(sum(p["event"] == "budget_exhausted" for p in payloads), 1)
        self.assertEqual(payloads[-1]["event"], "budget_exhausted")
        calls = logger.info.call_count
        unseen = Req("unseen", 64)
        trace.record_many([unseen], "decode_cadence")
        trace.snapshot([unseen])
        self.assertEqual(logger.info.call_count, calls)
        self.assertFalse(hasattr(unseen, "_ax_admission_trace_state"))


if __name__ == "__main__":
    unittest.main()
