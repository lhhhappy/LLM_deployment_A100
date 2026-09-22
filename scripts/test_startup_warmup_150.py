#!/usr/bin/env python3
"""CPU tests executing the delivered 150 module and flush wrapper with IPC/pool mocks."""
import asyncio
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'build/p150/candidate/sglang'


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, SOURCE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


io = ModuleType('sglang.srt.managers.io_struct')
io.GenerateReqInput = io.FlushCacheReqInput = io.FlushCacheReqOutput = NS
sys.modules[io.__name__] = io
ipc = ModuleType('sglang.srt.managers.scheduler_components.ipc_channels')
ipc.SchedulerIpcChannels = NS
sys.modules[ipc.__name__] = ipc
warm = load('warm150', 'srt/entrypoints/ax_shapes.py')
wrapper = load('flush150', 'srt/managers/scheduler_components/flush_wrapper.py')


class Manager:
    def __init__(self, **args):
        self.server_args = NS(**args)
        self.events = []
        self.cache = []
        self.abort = []
        self.requests = []
        self.fail = None
        self.flush_fail = False
        self.dirty = False
        self.no_hit = False
        self.metrics_cached = 0

    def abort_request(self, rid):
        self.abort.append(rid)

    async def flush_cache(self, **kw):
        self.events.append(('flush', kw))
        if not self.dirty:
            self.cache.clear()
        return NS(success=not self.flush_fail, message='mock')

    async def generate_request(self, req, request):
        self.events.append(('generate', req.rid))
        self.requests.append(req)
        if self.fail == 'error':
            raise ValueError('request failed')
        if self.fail == 'timeout':
            await asyncio.sleep(10)
        ids = req.input_ids if isinstance(req.input_ids[0], list) else [req.input_ids]
        outputs = []
        for seq in ids:
            best = 0
            for old in self.cache:
                n = 0
                for a, b in zip(seq, old):
                    if a != b:
                        break
                    n += 1
                best = max(best, n // 64 * 64)
            if self.no_hit:
                best = 0
            if req.log_metrics:
                self.metrics_cached += best
            outputs.append({'meta_info': {'cached_tokens': best, 'prompt_tokens': len(seq),
                'completion_tokens': (req.sampling_params[len(outputs)] if isinstance(req.sampling_params,list) else req.sampling_params)['max_new_tokens'], 'finish_reason': {
                'type': 'abort' if self.fail == 'abort' else 'length'}}})
            self.cache.append(seq)
        if self.fail == 'empty':
            return
        yield outputs if isinstance(req.input_ids[0], list) else outputs[0]
        # Full drain is required, even after a nonstream final response.
        self.events.append(('drained', req.rid))


class WarmTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequence_and_cleanup(self):
        m = Manager()
        await warm.run('null', m)
        plan = list(warm.shape_plan())
        self.assertEqual(len(m.requests), len(plan) + 1)
        self.assertEqual(len([e for e in m.events if e[0] == 'flush']), 3)
        self.assertEqual(len([e for e in m.events if e[0] == 'drained']), len(m.requests))
        self.assertEqual(m.cache, [])
        self.assertEqual(m.metrics_cached, 0)
        self.assertEqual(m.events[-1][0], 'flush')
        for req in m.requests:
            self.assertFalse(req.log_metrics)
            self.assertFalse(req.stream)
            self.assertFalse(hasattr(req, 'text'))
            params = req.sampling_params if isinstance(req.sampling_params, list) else [req.sampling_params]
            self.assertTrue(all(p['temperature'] == 0 and p['max_new_tokens'] <= 12 for p in params))
        for _, kw in (e for e in m.events if e[0] == 'flush'):
            self.assertTrue(kw['verify_empty'])
        # The exact formerly cached prompt was sent again after the flush.
        self.assertEqual(m.requests[0].input_ids, m.requests[-1].input_ids)

    async def test_mtp_skip(self):
        m = Manager(speculative_algorithm='NEXTN')
        await warm.run('null', m)
        self.assertEqual(m.events, [])

    async def test_unsupported(self):
        for args in ({'enable_hierarchical_cache': True}, {'dp_size': 2}, {'pp_size': 2},
                     {'tokenizer_worker_num': 2}, {'enable_dp_attention': True}):
            m = Manager(**args)
            with self.assertRaises(ValueError):
                await warm.run('null', m)
            self.assertFalse(m.events)
        with self.assertRaises(ValueError):
            await warm.run('prefill', Manager())

    async def test_request_failure_aborts_and_flushes(self):
        for failure in ('error', 'abort', 'empty'):
            m = Manager(); m.fail = failure
            with self.assertRaises((ValueError, RuntimeError)):
                await warm.run('null', m)
            self.assertTrue(m.abort)
            self.assertEqual(m.events[-1][0], 'flush')

    async def test_timeout_aborts_and_flushes(self):
        m = Manager(); m.fail = 'timeout'
        old = warm.REQUEST_TIMEOUT; warm.REQUEST_TIMEOUT = 0.001
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await warm.run('null', m)
        finally:
            warm.REQUEST_TIMEOUT = old
        self.assertTrue(m.abort)
        self.assertEqual(m.events[-1][0], 'flush')

    async def test_cancellation_cleanup(self):
        m = Manager(); m.fail = 'timeout'
        task = asyncio.create_task(warm.run('null', m))
        while not m.requests:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(m.abort)
        self.assertEqual(m.events[-1][0], 'flush')

    async def test_flush_failure_is_fatal(self):
        m = Manager(); m.flush_fail = True
        with self.assertRaisesRegex(RuntimeError, 'verified flush failed'):
            await warm.run('null', m)
        self.assertFalse(m.requests)

    async def test_lingering_cache_is_fatal(self):
        m = Manager(); m.dirty = True
        with self.assertRaisesRegex(RuntimeError, 'cache survived'):
            await warm.run('null', m)
        self.assertEqual(m.events[-1][0], 'flush')

    async def test_missing_prefix_hit_is_fatal(self):
        m = Manager(); m.no_hit = True
        with self.assertRaisesRegex(RuntimeError, 'prefix reuse'):
            await warm.run('null', m)

    def test_plan(self):
        p = list(warm.shape_plan())
        self.assertEqual([len(ids) for label, ids, _, _ in p if label.startswith('decode-')], list(range(6, 33)))
        chain = [ids[0] for label, ids, _, _ in p if label.startswith('chain-')]
        self.assertEqual([len(x) for x in chain], [600, 1600, 8600, 20600])
        for a, b in zip(chain, chain[1:]):
            self.assertEqual(a, b[:len(a)])
        self.assertEqual([len(b)-len(a) for a,b in zip(chain,chain[1:])], [1000,7000,12000])
        self.assertTrue(any(len(ids[0]) == 20000 for _,ids,_,_ in p))
        self.assertEqual({t for _,ids,_,_ in p for row in ids for t in row if t > 150000}, {154827,154829})

    async def test_exporter_obeys_metrics_flag(self):
        tree = ast.parse((SOURCE/'srt/observability/request_metrics_exporter.py').read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RequestMetricsExporterManager')
        node = next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='write_record')
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<exporter>', 'exec'), ns)
        seen = []
        async def write(obj, out): seen.append(out)
        m = NS(_exporters=[NS(write_record=write)])
        await ns['write_record'](m, NS(log_metrics=False), {'cached_tokens':512})
        self.assertEqual(seen, [])
        await ns['write_record'](m, NS(log_metrics=True), {'cached_tokens':512})
        self.assertEqual(seen, [{'cached_tokens':512}])


class FlushTests(unittest.TestCase):
    def setUp(self):
        self.calls = []; self.idle = True; self.verify_error = False
        def flush():
            self.calls.append('flush'); return self.idle
        def verify(success):
            self.calls.append('verify')
            if self.verify_error:
                raise RuntimeError('leak')
        self.sent = []
        self.obj = wrapper.SchedulerFlushWrapper(flush_cache=flush, is_fully_idle=lambda: self.idle,
            verify_empty=verify, ipc_channels=NS(send_to_tokenizer=NS(send_output=lambda *x: self.sent.append(x))))

    def test_immediate(self):
        out = self.obj.handle(NS(timeout_s=0, verify_empty=True))
        self.assertTrue(out.success); self.assertEqual(self.calls, ['flush', 'verify'])

    def test_default_flush_unchanged(self):
        self.assertTrue(self.obj.handle(NS(timeout_s=0, verify_empty=False)).success)
        self.assertEqual(self.calls, ['flush'])

    def test_busy_failure_still_enters_verifier(self):
        self.idle = False
        self.assertFalse(self.obj.handle(NS(timeout_s=0, verify_empty=True)).success)
        self.assertEqual(self.calls, ['flush', 'verify'])

    def test_deferred(self):
        self.idle = False
        self.assertIsNone(self.obj.handle(NS(timeout_s=60, verify_empty=True)))
        self.obj.check_pending(); self.assertEqual(self.calls, [])
        self.idle = True; self.obj.check_pending()
        self.assertTrue(self.sent[0][0].success)
        self.assertEqual(self.calls, ['flush', 'verify'])

    def test_verifier_failure_reply(self):
        self.verify_error = True
        out = self.obj.handle(NS(timeout_s=0, verify_empty=True))
        self.assertFalse(out.success); self.assertEqual(out.message, 'leak')

    def test_deferred_failure_and_timeout(self):
        self.idle = False; self.verify_error = True
        self.obj.handle(NS(timeout_s=60, verify_empty=True))
        self.idle = True; self.obj.check_pending()
        self.assertFalse(self.sent[0][0].success)
        self.idle = False
        self.obj.handle(NS(timeout_s=60, verify_empty=True))
        req, _ = self.obj._pending; self.obj._pending = (req, 0)
        self.obj.check_pending(); self.assertFalse(self.sent[-1][0].success)

    def test_actual_empty_assertions(self):
        req = NS(size=8, available_size=lambda: 8, mamba_pool=NS(size=16),
                 mamba_allocator=NS(available_size=lambda: 16), mamba_ckpt_pool=None)
        tree = NS(total_size=lambda: (0,0), **{x: lambda: 0 for x in (
            'full_evictable_size','full_protected_size','mamba_evictable_size','mamba_protected_size')})
        s = NS(is_fully_idle=lambda: True, req_to_token_pool=req,
               token_to_kv_pool_allocator=NS(size=1000,page_size=64,available_size=lambda: 960), tree_cache=tree)
        warm.assert_empty(s)
        for obj, attr, value in [(req,'available_size',lambda: 7),
                 (req.mamba_allocator,'available_size',lambda: 15),
                 (s.token_to_kv_pool_allocator,'available_size',lambda: 896),
                 (tree,'total_size',lambda: (64,1)), (tree,'full_protected_size',lambda: 64)]:
            old = getattr(obj,attr); setattr(obj,attr,value)
            with self.assertRaises(RuntimeError): warm.assert_empty(s)
            setattr(obj,attr,old)

    def test_real_scheduler_flush_restores_pools(self):
        import logging
        tree = ast.parse((SOURCE/'srt/managers/scheduler.py').read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
        node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='flush_cache')
        ns = {'logging':logging, 'logger':logging.getLogger('test'), 'current_platform':NS(empty_cache=lambda:None)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<scheduler>', 'exec'),ns)
        class Pool:
            def __init__(self,size): self.size=size; self.free=0; self.page_size=64
            def clear(self): self.free=self.size
            def available_size(self): return self.free
            def reset_aux_cache_allocator(self): pass
        req=Pool(8); req.mamba_pool=Pool(16); req.mamba_allocator=req.mamba_pool
        def req_clear(): req.free=req.size; req.mamba_pool.clear()
        req.clear=req_clear
        cache=NS(tokens=64)
        cache.reset=lambda:setattr(cache,'tokens',0)
        cache.total_size=lambda:(cache.tokens,cache.tokens//64)
        for key in ('full_evictable_size','full_protected_size','mamba_evictable_size','mamba_protected_size'):
            setattr(cache,key,lambda:cache.tokens)
        s=NS(is_fully_idle=lambda:True,tree_cache=cache,req_to_token_pool=req,
             token_to_kv_pool_allocator=Pool(1024),grammar_manager=NS(clear=lambda:None),
             metrics_reporter=NS(reset_metrics=lambda:None,is_stats_logging_rank=False),draft_worker=None)
        self.assertTrue(ns['flush_cache'](s)); warm.assert_empty(s)
        req.free=0; cache.tokens=64; s.is_fully_idle=lambda:False
        s.waiting_queue=[1]; s.running_batch=NS(reqs=[])
        self.assertFalse(ns['flush_cache'](s)); self.assertEqual(req.free,0); self.assertEqual(cache.tokens,64)

    def test_tp_peer_failure_propagates(self):
        from unittest.mock import patch
        class Tensor:
            def __init__(self, values): self.value=values[0]
            def item(self): return self.value
        dist=ModuleType('torch.distributed'); dist.ReduceOp=NS(MAX='MAX')
        seen=[]; remote=[False]
        def reduce(tensor, op, group):
            seen.append((tensor.value,op,group))
            tensor.value=max(tensor.value, int(remote[0]))
        dist.all_reduce=reduce
        torch=ModuleType('torch'); torch.distributed=dist; torch.int32='int32'
        torch.tensor=lambda values,**kw:Tensor(values)
        with patch.dict(sys.modules, {'torch':torch,'torch.distributed':dist}):
            with patch.object(warm,'assert_empty') as check:
                s=NS(tp_cpu_group='actual-tp-cpu-group')
                warm.verify_empty(s,True)
                remote[0]=True
                with self.assertRaisesRegex(RuntimeError,'another TP rank'): warm.verify_empty(s,True)
                remote[0]=False; check.side_effect=RuntimeError('local leak')
                with self.assertRaisesRegex(RuntimeError,'local leak'): warm.verify_empty(s,True)
                with self.assertRaisesRegex(RuntimeError,'refused'): warm.verify_empty(s,False)
        self.assertEqual(len(seen),4)
        self.assertEqual([x[0] for x in seen],[0,0,1,1])
        self.assertTrue(all(x[2]=='actual-tp-cpu-group' for x in seen))

    def test_control_aggregates_all_workers(self):
        # Execute actual patched tokenizer mixin method, no fake aggregation.
        mod = ast.parse((SOURCE/'srt/managers/tokenizer_control_mixin.py').read_text())
        node = next(n for n in ast.walk(mod) if isinstance(n,ast.AsyncFunctionDef) and n.name=='flush_cache')
        node.returns = None
        for arg in node.args.args: arg.annotation = None
        ns = {'FlushCacheReqInput':NS,'FlushCacheReqOutput':NS}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<flush>', 'exec'),ns)
        for replies, expected in [([],False),([NS(success=True)],True),
                    ([NS(success=True),NS(success=False,message='leak')],False)]:
            async def communicator(req):
                self.assertTrue(req.verify_empty)
                return replies
            manager = NS(auto_create_handle_loop=lambda: None, flush_cache_communicator=communicator,mm_processor=None)
            out = asyncio.run(ns['flush_cache'](manager, verify_empty=True))
            self.assertEqual(out.success, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
