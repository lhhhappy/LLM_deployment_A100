"""131/132 and the chain-max warm seat: production scheduler ASTs, CPU pools.

Run with PYTHONPATH=tests python3 -B -m unittest test_chain_risk_scheduler.
No model forward is simulated; distributed transport has a separate Gloo probe.
"""
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Batch, Mode, Req, make_scheduler, step
from test_prefix_producer import request, scheduler

TREE = ROOT / 'engine/sglang'
ENV = {
    'SGLANG_AX_SCHED_PROTECT': '1', 'SGLANG_AX_SCHED_COLD_CAP': '16384',
    'SGLANG_AX_SCHED_SHORT_TOKENS': '2048', 'SGLANG_AX_DEADLINE_TIERS': '1',
    'SGLANG_AX_DEADLINE_FREEZE_CLASS': '1', 'SGLANG_AX_DEADLINE_LOAD': '1.05',
    'SGLANG_AX_CHAIN_RISK_INTERVAL': '1', 'SGLANG_AX_DEADLINE_CHAIN_FIRST': '1',
    'SGLANG_AX_PARK_MAX_ROUNDS': '0',
}


class ChainRiskScheduler(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('SGLANG_')}
        self.env = patch.dict(os.environ, {**env, **ENV}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def pair(self, waited=5.0, left=200000, current=16384, chunk=16384):
        s, ns = make_scheduler(TREE, budget=16384, interval=2)
        s._ax_rank0_decide = lambda f: f()
        ns['time'] = NS(perf_counter=lambda: 100.0)
        r = Req('head', left + current)
        r.set_extend_range(0, current)
        r.time_stats.scheduler_recv_time = 100.0 - waited
        b = Batch([r])
        b._ax_chain_risk_chunk = chunk
        return s, ns, b

    def test_default_model_uses_plan_and_rejects_negative_override(self):
        s, ns, _ = self.pair()
        self.assertEqual(s._ax_chain_risk_cfg().chunk, 0)
        with patch.dict(os.environ, {'SGLANG_AX_CHAIN_RISK_CHUNK': '-1'}):
            with self.assertRaisesRegex(ValueError, 'invalid chain-risk'):
                ns['ax_deadline'].chain_risk_config(2)

    def test_automatic_chunk_tracks_16k_not_fixed_8k(self):
        s, _, b = self.pair(waited=3.5)
        self.assertIsNone(s._ax_chain_risk_interval(b))
        b._ax_chain_risk_chunk = 8192
        self.assertEqual(s._ax_chain_risk_interval(b), 1)

    def test_explicit_chunk_override_is_preserved(self):
        with patch.dict(os.environ, {'SGLANG_AX_CHAIN_RISK_CHUNK': '8192'}):
            s, _, b = self.pair(waited=3.5)
            self.assertEqual(s._ax_chain_risk_interval(b), 1)

    def test_unexecuted_batch_is_charged_once_including_riders(self):
        s, ns, b = self.pair(current=8192, waited=5.0)
        ax = ns['ax_deadline']
        dl, _ = s._ax_admission_cfgs()
        # Projection without the current forward is <22; including it is >22.
        self.assertIsNone(ax.chain_risk_interval(b.reqs[0], 200000, 5.0,
                          dl, s._ax_chain_risk_cfg(), chunk=16384))
        self.assertEqual(s._ax_chain_risk_interval(b), 1)
        original = ax.chain_risk_interval
        with patch.object(ax, 'chain_risk_interval', wraps=original) as call:
            rider = Req('rider', 512, cached=65536)
            b.reqs.append(rider)
            s._ax_chain_risk_interval(b)
            costs = [c.kwargs['inflight_s'] for c in call.call_args_list]
        self.assertEqual(costs, [ax.service_s(8704, 8704, dl)] * 2)

    def test_all_ranks_follow_root_even_with_clock_and_frozen_class_skew(self):
        for root_wait in (3.5, 5.0, 40.0):
            with self.subTest(root_wait=root_wait):
                root, _, rb = self.pair(waited=root_wait)
                peer, _, pb = self.pair(waited=24.0)
                # Peer would choose a different class, and must never evaluate it.
                pb.reqs[0].num_matched_prefix_tokens = 180000
                chosen = []
                root._ax_rank0_decide = lambda f: chosen.append(f()) or chosen[-1]
                peer._ax_rank0_decide = lambda f: chosen[-1]
                with patch.object(peer, '_ax_chain_risk_interval', side_effect=AssertionError('peer evaluated risk')):
                    root._arm_prefill_decode_interval(rb)
                    peer._arm_prefill_decode_interval(pb)
                self.assertEqual(chosen, [1 if root_wait == 5.0 else 2])
                self.assertEqual([root._should_defer_prefill() for _ in range(3)],
                                 [peer._should_defer_prefill() for _ in range(3)])

    def test_backlog_lower_interval_wins_and_is_broadcast(self):
        s, ns, b = self.pair()
        dl, _ = s._ax_admission_cfgs()
        s._ax_admission_cfg = (dl, NS(relaxed_interval=0))
        s._ax_backlog_relieved = True
        decisions = []
        s._ax_rank0_decide = lambda f: decisions.append(f()) or decisions[-1]
        s._arm_prefill_decode_interval(b)
        self.assertEqual(decisions, [0])
        self.assertFalse(s._should_defer_prefill())

    def test_off_decode_idle_do_not_add_collectives(self):
        for mode in ('off', 'decode', 'idle', 'interval_zero'):
            with self.subTest(mode=mode):
                s, _, b = self.pair()
                if mode == 'off':
                    s._ax_chain_risk_cfg_ = None
                elif mode == 'decode':
                    b.forward_mode = Mode('decode')
                elif mode == 'idle':
                    b = None
                else:
                    s.prefill_decode_interval = 0
                s._ax_rank0_decide = lambda f: self.fail('unexpected collective')
                s._arm_prefill_decode_interval(b)
                self.assertEqual(s._prefill_decode_interval_remaining, 2 if mode == 'off' else 0)

    def test_132_without_124_fails_loudly(self):
        with patch.dict(os.environ, {'SGLANG_AX_DEADLINE_TIERS': '0'}):
            s, _, _ = self.pair()
            with self.assertRaisesRegex(ValueError, '132 needs 124'):
                s._ax_admission_cfgs()

    def test_runtime_receipt_reports_effective_risk_and_chain_first(self):
        for on in (False, True):
            with self.subTest(on=on):
                with patch.dict(os.environ, {'SGLANG_AX_DEADLINE_CHAIN_FIRST': str(int(on))}):
                    s, ns, _ = self.pair()
                    if not on:
                        s._ax_chain_risk_cfg_ = None
                    ns['get_spec'] = lambda: NS(speculative_algorithm=None)
                    ns['get_parallel'] = lambda: NS(dcp_size=1)
                    policy = ModuleType('sglang.srt.managers.schedule_policy')
                    policy._ax_srpt_aging = lambda: None
                    policy._role_boundary_token_ids = lambda: ()
                    local = ModuleType('sglang.srt.layers.dcp.local_extend')
                    local.local_extend_mechanism_tokens = lambda _: 'dcp_local=off'
                    with patch.dict(sys.modules, {policy.__name__: policy, local.__name__: local}):
                        tokens = dict(p.split('=', 1) for p in s._ax_mechanism_report().split() if '=' in p)
                    self.assertEqual(tokens['132'].split(':')[0], 'on' if on else 'off')
                    self.assertEqual(tokens['131_sync'], 'rank0' if on else 'off')
                    self.assertEqual(tokens['131_chunk'], 'auto' if on else 'off')

    def reserve_run(self, hits=(512,), demand=True, relieved=False):
        env = {'SGLANG_AX_PREFIX_PRODUCER': '1', 'SGLANG_AX_PREFIX_TRACE_S': '0',
               'SGLANG_AX_SCHED_COLD_CAP': '12288' if demand else '16384',
               'SGLANG_AX_SCHED_COLD_CAP_MAX': '16384' if demand else '0',
               'SGLANG_AX_BACKLOG_RELIEF': '1', 'SGLANG_AX_BACKLOG_COLD_CAP': '16384',
               'SGLANG_AX_BACKLOG_INTERVAL': '0'}
        with patch.dict(os.environ, env):
            cold = request('c', shared=8192, tail=100000, cached=8192)
            cold._ax_deadline_cold = True
            warm = [request(f'w{i}', shared=16384, tail=n, cached=16384, tag=9+i)
                    for i, n in enumerate(hits)]
            s, _, cache = scheduler(warm, cold, budget=16384, interval=2)
            cache.publish(cold, 8192)
            for r in warm:
                cache.publish(r, 16384)
            s._ax_admission_cfgs()
            s._ax_backlog_relieved = relieved
            trace = step(s)
            return s, trace

    def test_ordinary_warm_hit_gets_actual_tokens_and_risk_uses_final_plan(self):
        for relieved in (False, True):
            with self.subTest(relieved=relieved):
                off, before = self.reserve_run(demand=False, relieved=relieved)
                on, after = self.reserve_run(demand=True, relieved=relieved)
                self.assertEqual([r[0] for r in before['reqs']], ['c'])
                self.assertEqual([r[0] for r in after['reqs']], ['c', 'w0'])
                self.assertEqual(after['reqs'][0][2] - after['reqs'][0][1], 15872)
                self.assertEqual(on.last_batch._ax_chain_risk_chunk, 15872)
                self.assertLessEqual(sum(b-a for _, a, b in after['reqs']), 16384)
                self.assertEqual(off.last_batch._ax_chain_risk_chunk, 16384)

    def test_no_eligible_hit_keeps_full_cold_chunk(self):
        for hits in ((), (4096,)):
            with self.subTest(hits=hits):
                s, trace = self.reserve_run(hits=hits)
                self.assertEqual(trace['reqs'][0][2] - trace['reqs'][0][1], 16384)
                self.assertEqual(s.last_batch._ax_chain_risk_chunk, 16384)

    def test_two_2k_hits_fit_beside_12k_floor_without_second_partial(self):
        s, trace = self.reserve_run(hits=(2048, 2048))
        self.assertEqual(trace['reqs'], [('c', 8192, 20480), ('w0', 16384, 18432), ('w1', 16384, 18432)])
        self.assertEqual(trace['chunk'], 'c')
        self.assertEqual(s.last_batch._ax_chain_risk_chunk, 12288)


if __name__ == '__main__':
    unittest.main()
