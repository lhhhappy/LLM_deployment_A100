#!/usr/bin/env python3
"""T25 CPU tests: exact statistics, real draft policy/adder, no oracle scheduling.

Run PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p
 test_slo_scheduling.py -v. Live KDA/cache/TP timing validation remains separate.
"""
import ast
import copy
from decimal import Decimal, localcontext
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, MagicMock

import sim_closed_loop as sim
import score_formal as score
import test_spf_scheduling as d2
from test_sim_closed_loop import workload, profiles
from check_slo_calibration import estimated_summary

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'build/d3/b'
HELPER=BASE/'python/sglang/srt/managers/arena_slo.py'
spec=importlib.util.spec_from_file_location('slo_draft',HELPER)
slo=importlib.util.module_from_spec(spec)
spec.loader.exec_module(slo)


def prod():
    ns=d2.load_source(BASE/d2.REL)
    ns.update(SloPolicy=slo.SloPolicy,SLO_POLICIES=slo.SLO_POLICIES)
    return ns


def recv(sid='s',namespace='n',arrival=0):
    return NS(arena_slo_session=sid,arena_slo_namespace=namespace,arena_slo_arrival=arrival)


def req(rid,work,arrival=0,cold=False):
    r=d2.request(rid,work,arrived=arrival)
    r._arena_slo_chain_start=cold
    r._arena_slo_arrival=arrival
    return r


class StatisticsTests(unittest.TestCase):
    def test_count_frontiers_against_independent_decimal_cdf(self):
        for n,expected in ((1,1),(20,3),(65,6),(100,9),(314,22),(388,27),(808,51),(9023,485)):
            with localcontext() as ctx:
                ctx.prec=60
                p=Decimal('0.05');mass=(1-p)**n;cdf=Decimal(0)
                allowed=0
                for k in range(n+1):
                    if 1-cdf >= p: allowed=k
                    else: break
                    cdf+=mass
                    mass=mass*Decimal(n-k)/Decimal(k+1)*p/(1-p)
            self.assertEqual(allowed,expected)
            self.assertEqual(score.allowed_over(n),expected)
            self.assertLessEqual(score.binomial_lower(expected,n),.05)
            if expected<n: self.assertGreater(score.binomial_lower(expected+1,n),.05)
        self.assertIsNone(score.allowed_over(0))
        with self.assertRaises(ValueError): score.allowed_over(-1)

    def test_61_seconds_and_arbitrary_magnitude_can_pass(self):
        for seconds in (61,1e9):
            vals=[20.0]*758+[seconds]*50
            self.assertEqual(sim.q(vals,.95),seconds)
            self.assertLess(score.binomial_lower(50,808),.05)
        self.assertGreater(score.binomial_lower(52,808),.05)

    def test_estimate_keeps_non_ttft_gate_and_strict_report(self):
        w=workload([[(10,2,0,'chain_start'),(10,2,0,'intra'),(10,2,0,'turn_start')]])
        result=sim.simulate(w,profiles(w),1,sim.Engine())
        s=result['summary'];before=copy.deepcopy(s)
        s['failed_gates'].append('tpot_p95<=0.10')
        estimated=estimated_summary(s,result['requests'])
        self.assertIn('tpot_p95<=0.10',estimated['failed_gates'])
        self.assertFalse(estimated['model_pass_dev_plus_tpot'])
        self.assertEqual(s['ttft_gate_detail'],before['ttft_gate_detail'])


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ, {k:v for k,v in os.environ.items() if not k.startswith('SGLANG_ARENA_SLO_')},clear=True)
        self.env.start();self.addCleanup(self.env.stop)
        self.ns=prod();self.p=d2.policy(self.ns,'arena-edf')

    def test_header_fields_are_separate_and_timestamp_is_server_owned(self):
        obj=NS(session_id='native',received_time=99,arena_slo_arrival=-123)
        raw=NS(headers={'x-s1-session-id':'s','x-s1-cache-namespace':'n','x-s1-phase':'turn_start'},scope={'arena_recv_perf':5.0})
        slo.apply_slo_headers(obj,raw)
        self.assertEqual((obj.arena_slo_session,obj.arena_slo_namespace,obj.arena_slo_arrival),('s','n',5.0))
        self.assertEqual((obj.session_id,obj.received_time),('native',99))
        with patch.object(slo.time,'perf_counter',return_value=17):
            slo.apply_slo_headers(obj,NS(headers={},scope={}))
        self.assertIsNone(obj.arena_slo_session)
        self.assertEqual(obj.arena_slo_arrival,17)

    def test_real_passthrough_ast_fields_and_unconditional_http_call(self):
        io=ast.parse((BASE/'python/sglang/srt/managers/io_struct.py').read_text())
        fields=('arena_slo_session','arena_slo_namespace','arena_slo_arrival')
        for clsname in ('GenerateReqInput','TokenizedGenerateReqInput'):
            cls=next(n for n in io.body if isinstance(n,ast.ClassDef) and n.name==clsname)
            names={n.target.id for n in cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
            self.assertTrue(set(fields)<=names)
        for filename,constructor,owner in (('io_struct.py','GenerateReqInput','self'),('tokenizer_manager.py','TokenizedGenerateReqInput','obj')):
            tree=ast.parse((BASE/'python/sglang/srt/managers'/filename).read_text())
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==constructor]
            call=next(c for c in calls if any(k.arg==fields[0] for k in c.keywords))
            for f in fields:
                expression=next(k.value for k in call.keywords if k.arg==f)
                self.assertEqual(ast.unparse(expression),owner+'.'+f)
        http=ast.parse((BASE/'python/sglang/srt/entrypoints/http_server.py').read_text())
        handler=next(n for n in http.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='generate_request')
        self.assertIn('apply_slo_headers(obj, request)',[ast.unparse(n) for n in handler.body])

    def test_first_session_namespace_missing_flush_and_bounded_state(self):
        for sid,namespace,cold in [('s','n',True),('s','n',False),('s','n2',True),(None,'n',False)]:
            r=d2.request('r',50)
            self.p.register_slo_request(r,recv(sid,namespace))
            self.assertEqual(r._arena_slo_chain_start,cold)
        self.p.reset_slo_sessions()
        self.assertEqual(len(self.p._arena_slo.sessions),0)
        self.p._arena_slo.session_cap=2
        for sid in ('a','b','c'):
            r=d2.request(sid,1);self.p.register_slo_request(r,recv(sid))
        self.assertEqual(list(self.p._arena_slo.sessions),[('n','b'),('n','c')])
        before=dict(r.__dict__);self.p.register_slo_request(r,recv('changed',arrival=500))
        self.assertEqual(r.__dict__,before)
        tree=ast.parse((BASE/'python/sglang/srt/managers/scheduler.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='flush_cache')
        ns={'logging':MagicMock(),'logger':MagicMock(),'current_platform':MagicMock()}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'actual_flush','exec'),ns)
        scheduler=MagicMock();scheduler.policy=self.p;scheduler.is_fully_idle.return_value=False
        self.assertFalse(ns['flush_cache'](scheduler))
        self.assertEqual(len(self.p._arena_slo.sessions),2)
        scheduler.tree_cache.reset.assert_not_called()
        scheduler.is_fully_idle.return_value=True
        self.assertTrue(ns['flush_cache'](scheduler))
        self.assertEqual(len(self.p._arena_slo.sessions),0)
        scheduler.tree_cache.reset.assert_called_once()
        scheduler.req_to_token_pool.clear.assert_called_once()

    def test_freeze_gate_threshold_and_no_future_or_eval_labels(self):
        for size,limit in ((4096,3),(4097,5)):
            r=req('r',size)
            r.phase='chain_start';r.uncached_expected=1
            self.p._arena_slo.prepare(r)
            self.assertEqual(r._arena_slo_limit,limit)
            r.num_matched_prefix_tokens=size
            self.assertEqual(self.p._arena_slo.key(r,10)[0],limit-10)
        cold=req('c',1,cold=True);self.p._arena_slo.prepare(cold)
        self.assertEqual(cold._arena_slo_limit,30)

    def test_edf_age_overdue_and_weighted_order(self):
        old=req('cold',60000,arrival=0,cold=True)
        intra=req('intra',128,arrival=40)
        q=[intra,old];self.p._compute_prefix_matches=MagicMock(return_value={'cold'})
        self.p.calc_priority(q)
        self.assertEqual([r.rid for r in q],['cold','intra'])
        weighted=d2.policy(self.ns,'arena-edf-chain-weighted')
        q=[req('cold',60000,cold=True),req('intra',128,arrival=40)]
        weighted._compute_prefix_matches=MagicMock(return_value=set());weighted.calc_priority(q)
        self.assertEqual([r.rid for r in q],['intra','cold'])
        self.assertEqual(q[1]._arena_slo_limit,30)
        self.assertEqual(q[1]._arena_slo_deadline,30)
        self.assertEqual(q[1]._arena_slo_scheduling_deadline,60)

    def test_least_slack_uses_service_and_can_favor_long_cold(self):
        p=slo.SloPolicy('arena-least-slack')
        cold=req('cold',250000,cold=True);fast=req('fast',100)
        p.sort([cold,fast])
        self.assertLess(p.key(cold,0),p.key(fast,0))
        with patch.dict(os.environ,{'SGLANG_ARENA_SLO_CHAIN_WEIGHT':'nan'}):
            with self.assertRaises(ValueError): slo.SloPolicy('arena-edf')

    def test_reserve_more_urgent_full_waiter_and_align(self):
        active=req('cold',16000,cold=True);wait=[req('fast',512)]
        self.p._arena_slo.sort(wait)
        self.assertEqual(self.p.shortest_prefill_chunk_limit(active,wait,4096,64),3584)
        a=d2.adder(self.ns)
        a.chunked_req_limit=self.p.shortest_prefill_chunk_limit(active,wait,4096,64)
        self.assertIs(a.add_chunked_req(active),active)
        a.add_one_req(wait[0],True,None)
        self.assertEqual(a.can_run_list,[active,*wait])
        self.assertEqual(a.rem_chunk_tokens,0)
        self.assertIsNone(a.new_chunked_req)
        self.assertEqual(self.p.shortest_prefill_chunk_limit(active,[req('r',65)],4096,64,512),3584)
        self.assertIsNone(self.p.shortest_prefill_chunk_limit(active,[req('r',4096)],4096,64))
        self.assertIsNone(self.p.shortest_prefill_chunk_limit(active,wait,64,64))
        urgent=req('urgent',20000,arrival=-100,cold=True)
        self.assertIsNone(self.p.shortest_prefill_chunk_limit(urgent,wait,4096,64))

    def test_shared_partial_guard_host_miss_and_d1_reuse(self):
        # Run original W7 adversarial cases against the actual 003 classes and
        # each newly selectable scheduler; the only mock is runtime dependencies.
        methods=('test_reject_second_partial_before_delayer_or_load',
                 'test_host_miss_reselect_checks_existing_and_new_partial',
                 'test_d1_shared_guard_skip_split_during_continuation',
                 'test_d1_split_then_full_and_no_second_partial',
                 'test_memory_mamba_and_page_debits','test_tile_gate_and_dsa_candidate_alignment')
        for policy in slo.SLO_POLICIES:
            for name in methods:
                case=d2.SPFTests(name);case.ns=prod()
                case.ns['get_schedule']=lambda: NS(schedule_policy=policy)
                case.a=d2.adder(case.ns);case.p=d2.policy(case.ns,policy)
                with self.subTest(policy=policy,case=name): getattr(case,name)()

    def test_legacy_disabled_path_and_budget_ast_unchanged(self):
        case=d2.SPFTests();case.ns=self.ns
        case.test_fcfs_differential_against_clean_stock()
        p=d2.policy(self.ns,'fcfs');q=[req('a',50000),req('b',1)]
        p.calc_priority(q);self.assertEqual([r.rid for r in q],['a','b'])
        self.assertIsNone(p.shortest_prefill_chunk_limit(q[0],q[1:],4096,64))
        for mode in slo.SLO_POLICIES:
            with self.assertRaises(ValueError):
                self.ns['SchedulePolicy'](mode,NS(disable=True),False,False,False)
        def methods(path):
            cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PrefillAdder')
            return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        old,new=methods(d2.SOURCE),methods(BASE/d2.REL)
        for name in old:
            if name != '_select_prefill_admission': self.assertEqual(old[name],new[name],name)

    def test_clean_patch_chain_and_d0_composition(self):
        for with_d0 in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                dest=Path(tmp)
                files={p.relative_to(ROOT/'build/d3/a') for p in (ROOT/'build/d3/a').rglob('*.py')}
                for patch_name in (('000-interface-compliance.patch',) if with_d0 else ()):
                    for line in (ROOT/'patches'/patch_name).read_text().splitlines():
                        if line.startswith('+++ b/'): files.add(Path(line.split()[1][2:]))
                for rel in files:
                    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(ROOT/'src/sglang'/rel,target)
                names=(['000-interface-compliance.patch'] if with_d0 else [])+[
                    '001-role-boundary-mamba-ckpt.patch','002-spf-scheduling.patch','003-slo-aware-scheduling.patch']
                for name in names:
                    result=subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(ROOT/'patches'/name)],cwd=dest,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                for expected in BASE.rglob('*.py'):
                    rel=expected.relative_to(BASE)
                    compile((dest/rel).read_text(),str(rel),'exec')
                    if with_d0 and rel.name in ('http_server.py','tokenizer_manager.py'): continue
                    self.assertEqual((dest/rel).read_bytes(),expected.read_bytes())


class SimulatorTests(unittest.TestCase):
    def test_label_changes_do_not_change_scheduling_trace(self):
        w=workload([[(10000,3,0),(100,3,0)],[(5000,3,0),(4200,3,0)]])
        other=copy.deepcopy(w)
        for row in other.rows.values(): row['phase']='turn_start';row['uncached_expected']=1
        for mode in sim.SLO_SCHEDULERS:
            e=sim.Engine(scheduler=mode,chunk_tokens=1024)
            a=sim.simulate(w,profiles(w),2,e,keep_trace=True)
            b=sim.simulate(other,profiles(other),2,e,keep_trace=True)
            self.assertEqual(a['trace'],b['trace'])
            self.assertEqual(sum(r['output_tokens'] for r in a['requests']),12)
            self.assertEqual(len(a['requests']),4)

    def test_random_reservation_matches_actual_production_and_one_partial(self):
        rng=random.Random(25)
        for mode in sim.SLO_SCHEDULERS:
            engine=sim.Engine(scheduler=mode,chunk_tokens=4096,prefill_length_alpha=1.2)
            policy=slo.SloPolicy('arena-'+mode)
            for _ in range(150):
                rs=[];ps=[]
                for i in range(rng.randrange(2,9)):
                    work=rng.randrange(1,20000);arrival=rng.randrange(0,20);cold=bool(rng.randrange(2))
                    r=sim.Request(str(i),0,0,0 if cold else 1,arrival,arrival,i,work,1,work,2)
                    cached=rng.randrange(0,150000)
                    r.matched=cached
                    p=req(str(i),work,arrival,cold)
                    p.origin_input_ids=[1]*(work+cached)
                    p.full_untruncated_fill_ids=p.origin_input_ids[:]
                    p.prefix_indices=[0]*cached
                    p.num_matched_prefix_tokens=cached
                    rs.append(r);ps.append(p)
                c,waiting=rs[0],rs[1:];pc,pw=ps[0],ps[1:]
                policy.sort(pw)
                self.assertEqual([r.rid for r in sorted(rs[1:],key=lambda r:sim.slo_priority(r,engine,50))],
                                 [r.rid for r in pw])
                cap=policy.chunk_limit(pc,pw,4096,64,64)
                initial=c.remaining
                batch,active=sim.select_prefill(waiting,c,0,engine,16,now=50)
                self.assertEqual(batch[0][1],min(initial,cap or 4096))
                self.assertLessEqual(sum(r.remaining>0 for r,_ in batch),1)
                self.assertLessEqual(sum(math.ceil(t/64)*64 for _,t in batch),4096)

    def test_weight_one_is_edf_and_old_policies_retain_trace(self):
        w=workload([[(20000,3,0),(200,3,0)],[(10000,3,0),(6000,3,0)]])
        a=sim.simulate(w,profiles(w),2,sim.Engine(scheduler='edf'),keep_trace=True)
        b=sim.simulate(w,profiles(w),2,sim.Engine(scheduler='edf-chain-weighted',chain_start_weight=1),keep_trace=True)
        self.assertEqual(a['trace'],b['trace'])
        spec=importlib.util.spec_from_file_location('before',ROOT/'build/d3/sim_closed_loop.before.py')
        import sys
        before=importlib.util.module_from_spec(spec);sys.modules['before']=before;spec.loader.exec_module(before)
        for mode in ('fcfs','spf','spf-upstream','hrrn','lpm'):
            a=sim.simulate(w,profiles(w),2,sim.Engine(scheduler=mode),keep_trace=True)
            b=before.simulate(w,profiles(w),2,before.Engine(scheduler=mode),keep_trace=True)
            a['summary']['engine'].pop('chain_start_weight')
            self.assertEqual(a,b)
        with self.assertRaises(ValueError):sim.Engine(chain_start_weight=.5)
        with tempfile.TemporaryDirectory() as output:
            cli=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/sim_closed_loop.py'),
                                '--levels','2','--schedulers',','.join(sim.SLO_SCHEDULERS),
                                '--chain-start-weight','1.5','--policies','stock',
                                '--summary-only','--out-dir',output],capture_output=True,text=True)
            self.assertEqual(cli.returncode,0,cli.stdout+cli.stderr)
            manifest=json.loads((Path(output)/'sweep.json').read_text())
            self.assertEqual(len(manifest['ladders']),3)
            self.assertTrue(all(e['engine']['chain_start_weight']==1.5 for e in manifest['ladders']))

if __name__=='__main__':unittest.main()
