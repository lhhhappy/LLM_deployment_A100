#!/usr/bin/env python3
"""Patch 123 (shortest remaining prefill first with aging) on the real scheduler code with CPU fakes.

The tree is the current working engine source, including the TP order repair.
Run: python3 -m unittest discover -s tests -p test_srpt_admission.py
"""
import os
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_sched_protect_chain import ROOT, Req, load_source, make_scheduler, step, tree_dir

P123 = ROOT / 'engine/sglang'


def req(rid, work, cached=0, waited=0.0):
    r = Req(rid, work, cached=cached)
    r.time_stats.wait_queue_entry_time = time.perf_counter() - waited
    return r


class SrptAdmission(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING': '2000', 'SGLANG_AX_SCHED_PROTECT': '1',
                                           'SGLANG_AX_SCHED_COLD_CAP': '2048', 'SGLANG_AX_SCHED_SHORT_TOKENS': '4096'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.ns = load_source(P123)
        self.sort = self.ns['SchedulePolicy']._ax_sort_by_remaining_work

    def order(self, q, held=()):
        self.sort(q, set(held))
        return [r.rid for r in q]

    def test_hit_then_small_cold_then_big_cold(self):
        q = [req('big', 100000), req('small', 3000), req('hit', 500, cached=60000)]
        self.assertEqual(self.order(q), ['hit', 'small', 'big'])

    def test_aging_lets_a_long_waiting_request_through(self):
        # 2000 tokens/s: after 40 s a 100k request still ranks behind a fresh 3k one, after 60 s it goes first
        self.assertEqual(self.order([req('small', 3000), req('big', 100000, waited=40)]), ['small', 'big'])
        self.assertEqual(self.order([req('small', 3000), req('big', 100000, waited=60)]), ['big', 'small'])

    def test_in_batch_prefix_sharing_holdbacks_stay_last(self):
        self.assertEqual(self.order([req('a', 1000), req('b', 2000)], held={'a'}), ['b', 'a'])

    def test_off_when_unset(self):
        with patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING': '0'}):
            self.assertIsNone(self.ns['_ax_srpt_aging']())

    def test_scheduler_admits_the_small_cold_request_first(self):
        s, ns = make_scheduler(P123, waiting=[req('big', 100000), req('small', 3000)])
        s.policy = NS(calc_priority=lambda q, _: ns['SchedulePolicy']._ax_sort_by_remaining_work(q, set()))
        t = step(s)
        self.assertEqual(t['mode'], 'prefill')
        self.assertEqual(t['reqs'][0][0], 'small')


if __name__ == '__main__':
    unittest.main()


class SharedOrdering(unittest.TestCase):
    def test_rank_local_arrival_near_tie_uses_leader_order(self):
        ns = load_source(P123)
        ns['time'] = NS(perf_counter=lambda: 100.0)
        queues = []
        for entered in [99.993, 99.994]:
            a, b = req('long',1001), req('short',1000)
            a.num_matched_prefix_tokens = b.num_matched_prefix_tokens = 0
            a.time_stats.wait_queue_entry_time = 99.990
            b.time_stats.wait_queue_entry_time = entered
            queues.append([a,b])
        with patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING':'300'}):
            # Reproduce the actual old divergence, even with identical now.
            for q in queues: ns['SchedulePolicy']._ax_sort_by_remaining_work(q,set())
            self.assertEqual([[r.rid for r in q] for q in queues], [['short','long'],['long','short']])
            payload = []
            for rank, q in enumerate(queues):
                p = ns['SchedulePolicy'].__new__(ns['SchedulePolicy'])
                p.policy = ns['CacheAwarePolicy'].LPM
                p._determine_active_policy = lambda _: p.policy
                p._compute_prefix_matches = lambda *_: set()
                def decide(compute):
                    if rank == 0: payload.append(compute())
                    return payload[0]
                p.rank0_decide = decide
                p.calc_priority(q)
            self.assertEqual([[r.rid for r in q] for q in queues], [['short','long'],['short','long']])
            # Off never calls the collective callback.
            with patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING':'0'}):
                p.rank0_decide = lambda _: self.fail('off path broadcast')
                p.calc_priority(queues[1])

    def test_broadcast_uses_group_source_and_leader_only_callback(self):
        import importlib.util, sys
        path=P123/'srt/managers/ax_rank0_decision.py'
        spec=importlib.util.spec_from_file_location('decision',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        payload=[]; group=object(); state={'rank':0,'calls':0}
        def broadcast(value,src,group):
            self.assertEqual(src,8)
            if state['rank']==0: payload.append(value[0])
            else: value[0]=payload[0]
        dist=NS(get_world_size=lambda _:2,get_rank=lambda _:state['rank'],
                get_global_rank=lambda g,r:8,broadcast_object_list=broadcast)
        with patch.dict(sys.modules,{'torch':NS(distributed=dist),'torch.distributed':dist}):
            self.assertEqual(module.rank0_decide(group,lambda: ['b','a']),['b','a'])
            state['rank']=1
            self.assertEqual(module.rank0_decide(group,lambda:self.fail('follower computed')),['b','a'])
