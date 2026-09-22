#!/usr/bin/env python3
"""T42 CPU tests: patch real base copies, execute extracted production methods.

No GPU imports/server/harness changes. Requires transformers/tokenizers for --real.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import enum
import gzip
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ['000-interface-compliance', '101-d1v12-on-base',
         '110-sm80-dsa-indexer', '111-sm80-fp8-moe-marlin']
METHODS = {'_detect_input_format', '_prepare_tokenizer_input',
           '_extract_tokenizer_results', '_tokenize_texts', '_tokenize_texts_sync'}


def extracted(path, names, cls=None, namespace=None):
    tree = ast.parse(path.read_text())
    if cls:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    nodes = [n for n in tree.body if getattr(n, 'name', None) in names]
    for n in nodes:
        n.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    ns = {'Enum': enum.Enum, 'logger': logging.getLogger('T42'), **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns


def load_file(path):
    spec = importlib.util.spec_from_file_location('t42_async', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare(work, evidence):
    base, candidate = work/'baseline', work/'candidate'
    shutil.copytree(ROOT/'build/base_exact/sglang', base)
    with (evidence/'patch_apply.log').open('w') as log:
        for name in CHAIN:
            subprocess.run(['patch', '-p3', '--fuzz=0', '--batch', '-i', str(ROOT/'patches'/f'{name}.patch')], cwd=base, stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.copytree(base, candidate)
        subprocess.run(['patch', '-p3', '--fuzz=0', '--batch', '-i', str(ROOT/'patches/130-async-tokenize.patch')], cwd=candidate, stdout=log, stderr=subprocess.STDOUT, check=True)
    helper = load_file(candidate/'srt/managers/async_text_tokenizer.py').AsyncTextTokenizer
    types = []
    for tree in (base, candidate):
        path = tree/'srt/managers/tokenizer_manager.py'
        ns = extracted(path, {'InputFormat'})
        ns = extracted(path, METHODS, 'TokenizerManager', ns)
        types.append(type('Manager', (), {k: v for k, v in ns.items() if k in METHODS}))
    return base, candidate, helper, types


def manager(typ, tok, helper=None):
    obj = typ()
    obj.tokenizer = tok
    obj.async_dynamic_batch_tokenizer = None
    obj.async_text_tokenizer = helper
    obj.model_config = NS(is_embedding_gemma=False)
    return obj


class FakeTokenizer:
    is_fast = True
    eos_token_id = 99
    def __call__(self, texts, **kwargs):
        ids = [[ord(c) for c in (''.join(t) if isinstance(t, list) else t)] for t in texts]
        return {'input_ids': ids, 'token_type_ids': [[0]*len(x) for x in ids]}
    def encode(self, text):
        return [ord(c) for c in text]


class Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = HELPER()
        self.stock = manager(TYPES[0], FakeTokenizer())
        self.new = manager(TYPES[1], FakeTokenizer(), self.worker)
        self.off = manager(TYPES[1], FakeTokenizer())
    async def asyncTearDown(self):
        self.worker.close()

    async def test_shapes_fast_and_disabled(self):
        for text, cross in [('abc', False), (['abc','def'], False),
                            ([['query','doc']], True), ([['q','d'],['a','b']], True), ('query', True)]:
            expected = await self.stock._tokenize_texts(text, cross)
            self.assertEqual(expected, await self.new._tokenize_texts(text, cross))
            self.assertEqual(expected, await self.off._tokenize_texts(text, cross))

    async def test_slow_encode_branch(self):
        for obj in (self.stock, self.new, self.off):
            obj.tokenizer.is_fast = False
        expected = await self.stock._tokenize_texts(['abc','def'])
        self.assertEqual(expected, await self.new._tokenize_texts(['abc','def']))
        self.assertEqual(expected, await self.off._tokenize_texts(['abc','def']))

    async def test_embedding_eos(self):
        for obj in (self.stock, self.new):
            obj.model_config.is_embedding_gemma = True
        self.assertEqual(await self.stock._tokenize_texts(['abc','c']), await self.new._tokenize_texts(['abc','c']))

    async def test_invalid_inputs(self):
        for obj in (self.stock, self.new, self.off):
            for text in ('', [], None):
                with self.assertRaises(ValueError):
                    await obj._tokenize_texts(text)
            obj.tokenizer = None
            with self.assertRaises(ValueError):
                await obj._tokenize_texts('abc')

    async def test_dynamic_precedence(self):
        async def encode(text, **kw):
            return {'input_ids':[7], 'token_type_ids':[8]}
        for obj in (self.stock, self.new):
            obj.async_dynamic_batch_tokenizer = NS(encode=encode)
        self.assertEqual(await self.stock._tokenize_texts('abc', True), await self.new._tokenize_texts('abc', True))

    async def test_thread_identity_and_serialization(self):
        active = 0
        maximum = 0
        def fn():
            nonlocal active, maximum
            active += 1
            maximum = max(active, maximum)
            ident = threading.get_ident()
            time.sleep(.005)
            active -= 1
            return ident
        ids = await asyncio.gather(*(self.worker.run(fn) for _ in range(20)))
        self.assertEqual(maximum, 1)
        self.assertEqual(len(set(ids)), 1)
        self.assertNotEqual(ids[0], threading.get_ident())

    async def test_exception_releases_slot(self):
        def bad():
            raise ValueError('original error')
        with self.assertRaisesRegex(ValueError, 'original error'):
            await self.worker.run(bad)
        self.assertEqual(await self.worker.run(lambda: 7), 7)

    async def test_cancel_running_and_waiting(self):
        started, finish, second = threading.Event(), threading.Event(), threading.Event()
        def slow():
            started.set()
            finish.wait(2)
            raise ValueError('abandoned failure')
        first = asyncio.create_task(self.worker.run(slow))
        while not started.is_set():
            await asyncio.sleep(.001)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        waiting = asyncio.create_task(self.worker.run(lambda: self.fail('canceled waiter ran')))
        await asyncio.sleep(.002)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        next_task = asyncio.create_task(self.worker.run(second.set))
        await asyncio.sleep(.01)
        self.assertFalse(second.is_set())
        finish.set()
        await asyncio.wait_for(next_task, 2)
        self.assertTrue(second.is_set())

    async def test_close_rejects_work(self):
        self.worker.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            await self.worker.run(lambda: None)

    async def test_startup_switch_and_dynamic(self):
        path = CANDIDATE/'srt/managers/tokenizer_manager.py'
        tree = ast.parse(path.read_text())
        assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'async_text_tokenizer' for t in n.targets))
        code = compile(ast.Module(body=[assign], type_ignores=[]), str(path), 'exec')
        old = os.environ.get('SGLANG_AX_ASYNC_TOKENIZE')
        try:
            for value, dynamic, tok, enabled in [(None,None,True,True),('0',None,True,False),('1',None,True,True),('1',object(),True,False),('1',None,False,False)]:
                if value is None:
                    os.environ.pop('SGLANG_AX_ASYNC_TOKENIZE',None)
                else:
                    os.environ['SGLANG_AX_ASYNC_TOKENIZE']=value
                obj=NS(tokenizer=object() if tok else None, async_dynamic_batch_tokenizer=dynamic)
                exec(code, {'self':obj,'os':os,'AsyncTextTokenizer':HELPER})
                self.assertEqual(obj.async_text_tokenizer is not None, enabled)
                if enabled:
                    obj.async_text_tokenizer.close()
        finally:
            if old is None:
                os.environ.pop('SGLANG_AX_ASYNC_TOKENIZE',None)
            else:
                os.environ['SGLANG_AX_ASYNC_TOKENIZE']=old

    async def test_header_precedence(self):
        fn = extracted(CANDIDATE/'srt/entrypoints/request_headers.py', {'apply_s1_routing_key'})['apply_s1_routing_key']
        for body, headers, expected in [(None,{},None), ('body',{'x-s1-routing-key':'r'},'body'), ('',{'x-s1-routing-key':'r'},''), (None,{'x-s1-routing-key':'r','x-s1-session-id':'s'},'r'), (None,{'x-s1-session-id':'s'},'s'), (None,{'x-s1-routing-key':'','x-s1-session-id':'s'},'s'), (None,{'x-s1-routing-key':''},None)]:
            obj=NS(routing_key=body,session_id='native',cache_salt='salt',received_time=123)
            fn(obj,headers)
            self.assertEqual(vars(obj),dict(routing_key=expected,session_id='native',cache_salt='salt',received_time=123))

    async def test_generate_handler_timestamp_and_stream(self):
        # Execute real handler body with transport/manager stubs, including SSE.
        helper = extracted(CANDIDATE/'srt/entrypoints/request_headers.py', {'apply_s1_routing_key'})['apply_s1_routing_key']
        async def generate(obj,request):
            yield {'key':obj.routing_key, 'received_time':obj.received_time}
        ns={'apply_s1_routing_key':helper, 'envs':NS(SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=NS(get=lambda:False)), '_global_state':NS(tokenizer_manager=NS(generate_request=generate,create_abort_task=lambda obj:None)), 'orjson_response':lambda x:x, 'dumps_json':lambda x:json.dumps(x).encode(), 'StreamingResponse':lambda body,**kw:body}
        handler=extracted(CANDIDATE/'srt/entrypoints/http_server.py', {'generate_request'},namespace=ns)['generate_request']
        for stream in (False, True):
            obj=NS(routing_key=None,stream=stream)
            req=NS(headers={'x-s1-session-id':'session'},scope={'arena_recv_perf':12.25})
            result=await handler(obj,req)
            if stream:
                chunks=[x async for x in result]
                self.assertEqual(chunks[-1],b'data: [DONE]\n\n')
                result=json.loads(chunks[0][6:])
            self.assertEqual(result,{'key':'session','received_time':12.25})

    async def test_batch_key_propagation(self):
        class Batch:
            def __getattr__(self,name):
                return [None, None]
        fn=extracted(CANDIDATE/'srt/managers/io_struct.py', {'__getitem__'}, 'GenerateReqInput', {'GenerateReqInput':NS})['__getitem__']
        obj=Batch(); obj.routing_key='affinity'; obj.session_id='native'; obj.rid=['a','b']; obj._get_positional_embeds_item=lambda i:None
        obj._get_positional_embed_overrides_item=lambda i:None
        sub=fn(obj,0)
        self.assertEqual(sub.routing_key,'affinity')
        self.assertIs(fn(obj,0),sub)

    async def test_downstream_key_and_time_code_unchanged(self):
        # Only five files may differ; no scheduler, counters, flush or harness edit.
        changed={str(p.relative_to(CANDIDATE)) for p in CANDIDATE.rglob('*.py') if not (BASE/p.relative_to(CANDIDATE)).exists() or p.read_bytes()!=(BASE/p.relative_to(CANDIDATE)).read_bytes()}
        self.assertEqual(changed,{'srt/managers/async_text_tokenizer.py','srt/managers/tokenizer_manager.py','srt/managers/io_struct.py','srt/entrypoints/http_server.py','srt/entrypoints/request_headers.py'})
        for path in changed:
            ast.parse((CANDIDATE/path).read_text())
        old=ast.parse((BASE/'srt/managers/tokenizer_manager.py').read_text())
        new=ast.parse((CANDIDATE/'srt/managers/tokenizer_manager.py').read_text())
        for name in ('_tokenize_one_request','_handle_batch_output'):
            a=next(n for n in ast.walk(old) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
            b=next(n for n in ast.walk(new) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))


async def measure(obj, text):
    interval=.001
    lags=[]
    done=False
    encoding=False
    active_ticks=0
    async def heartbeat():
        nonlocal active_ticks
        prev=time.perf_counter()
        while not done:
            await asyncio.sleep(interval)
            now=time.perf_counter()
            if encoding:
                active_ticks += 1
            lags.append(max(0,now-prev-interval))
            prev=now
    beat=asyncio.create_task(heartbeat())
    await asyncio.sleep(.01)
    start=time.perf_counter()
    encoding=True
    ids,_=await obj._tokenize_texts(text)
    encoding=False
    elapsed=time.perf_counter()-start
    await asyncio.sleep(.01)
    done=True
    await beat
    return ids,{'elapsed_s':elapsed,'max_loop_lag_s':max(lags),'heartbeat_count':len(lags), 'ticks_during_encode':active_ticks, 'p95_loop_lag_s':sorted(lags)[int(.95*(len(lags)-1))]}


async def real_tests(evidence):
    import transformers, tokenizers
    sys.path.insert(0,str(ROOT/'s1-dev/harness'))
    from s1_common import Renderer
    renderer=Renderer(str(ROOT/'s1-dev/glm_tok'))
    tok=renderer.tokenizer
    worker=HELPER()
    stock,new,off=manager(TYPES[0],tok),manager(TYPES[1],tok,worker),manager(TYPES[1],tok)
    requests={f"{r['pack']}:{r['view']}:{r['logical_call_id']}":r for r in map(json.loads,(ROOT/'s1-dev/data/dev-combined-v1/requests.jsonl').read_text().splitlines())}
    largest=(0,None,None)
    count=0
    with (evidence/'token_comparison.jsonl').open('w') as out, gzip.open(ROOT/'s1-dev/data/dev-combined-v1/bodies/dev-combined-v1.jsonl.gz','rt') as bodies:
        for line in bodies:
            body=json.loads(line)
            text=renderer.render(body)
            expected,_=await stock._tokenize_texts(text)
            got,_=await new._tokenize_texts(text)
            disabled,_=await off._tokenize_texts(text)
            assert expected==got==disabled,body['req_id']
            meta=requests[body['req_id']]
            assert meta.get('glm_tokens',len(expected))==len(expected),(body['req_id'],len(expected),meta.get('glm_tokens'))
            out.write(json.dumps({'req_id':body['req_id'],'tokens':len(expected),'utf8_bytes':len(text.encode()),'token_ids_sha256':hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest(),'equal_enabled':True,'equal_disabled':True})+'\n')
            count+=1
            if len(expected)>largest[0]: largest=(len(expected),text,body['req_id'])
            if count%50==0: print(f'Compared {count} real prompts',flush=True)
    # Boundary strings demonstrate why naive prefix token concatenation is unsafe.
    adversarial=['hello','helloworld','中','中文🙂e\u0301','[gMASK]<sop><|assistant|><think>', '\r\n  \t ', 'a\u200db\x00c']
    for text in adversarial:
        assert await stock._tokenize_texts(text)==await new._tokenize_texts(text)==await off._tokenize_texts(text)
    merged=tok.encode('hello',add_special_tokens=False)
    split=tok.encode('hel',add_special_tokens=False)+tok.encode('lo',add_special_tokens=False)
    assert merged!=split
    # Real concurrent arrivals and paired/batched input, using the same instance.
    results=await asyncio.gather(*(new._tokenize_texts(t) for t in adversarial*3))
    for text,result in zip(adversarial*3,results): assert result==await stock._tokenize_texts(text)
    for text,cross in [(adversarial,False),([['query','document']],True),([['a','b'],['中文','answer']],True)]:
        assert await stock._tokenize_texts(text,cross)==await new._tokenize_texts(text,cross)
    bench=[]
    for target in (100000,250000):
        # CPU-only derived stress input, never sent to a model or used as SLO evidence.
        text=(largest[1]*((target//largest[0])+2))
        text=text[:int(len(text)*target/(largest[0]*((target//largest[0])+2)))]
        for rep in range(3):
            paths=[('sync',stock),('thread',new)] if rep%2==0 else [('thread',new),('sync',stock)]
            previous=None
            for mode,obj in paths:
                ids,stats=await measure(obj,text)
                if previous is not None: assert ids==previous
                previous=ids
                bench.append({'mode':mode,'target_tokens':target,'actual_tokens':len(ids),'repeat':rep,**stats})
        print(f'Benchmarked target {target}',flush=True)
    # Direct Rust backend encode releases the GIL if heartbeat executes during it.
    # Use the same worker and measure() via a minimal manager-compatible shim.
    async def rust_text(text):
        result=await worker.run(lambda: tok.backend_tokenizer.encode_batch([text]))
        return result[0].ids,None
    rust=NS(_tokenize_texts=rust_text)
    _,rust_stats=await measure(rust,text)
    worker.close()
    report={'real_prompts':count,'adversarial_prompts':len(adversarial),'concurrent_comparisons':len(results),'batch_pair_cases':3,'all_token_ids_equal':True,'frozen_glm_counts_equal':True,'largest_real_tokens':largest[0],'largest_real_req_id':largest[2],'benchmark':bench,'rust_backend_thread':rust_stats,'prefix_merge_counterexample':{'full':merged,'naive_concat':split},'python':sys.version,'platform':platform.platform(),'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'tokenizer_class':type(tok).__name__,'is_fast':tok.is_fast,'cpu_affinity':sorted(os.sched_getaffinity(0)),'tokenizers_parallelism':os.environ.get('TOKENIZERS_PARALLELISM','unset')}
    (evidence/'real_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('benchmark','cpu_affinity')},indent=2),flush=True)


async def real_benchmark(evidence):
    """Unmodified dataset prompts near 100k and the longest 256k prompt."""
    sys.path.insert(0, str(ROOT/'s1-dev/harness'))
    from s1_common import Renderer
    renderer = Renderer(str(ROOT/'s1-dev/glm_tok'))
    rows = list(map(json.loads, (ROOT/'s1-dev/data/dev-combined-v1/requests.jsonl').read_text().splitlines()))
    choices = [min(rows, key=lambda r: abs(r['glm_tokens']-100000)), max(rows, key=lambda r: r['glm_tokens'])]
    ids = {f"{r['pack']}:{r['view']}:{r['logical_call_id']}":r for r in choices}
    worker = HELPER()
    stock = manager(TYPES[0], renderer.tokenizer)
    new = manager(TYPES[1], renderer.tokenizer, worker)
    results = []
    with gzip.open(ROOT/'s1-dev/data/dev-combined-v1/bodies/dev-combined-v1.jsonl.gz', 'rt') as f:
        for line in f:
            body = json.loads(line)
            if body['req_id'] not in ids:
                continue
            text = renderer.render(body)
            expected, _ = await stock._tokenize_texts(text)
            await new._tokenize_texts(text)  # Warm both execution paths.
            assert len(expected) == ids[body['req_id']]['glm_tokens']
            for rep in range(3):
                modes = [('sync', stock), ('thread', new)]
                if rep % 2:
                    modes.reverse()
                for mode, obj in modes:
                    actual, stats = await measure(obj, text)
                    assert actual == expected
                    results.append(dict(req_id=body['req_id'], tokens=len(actual), mode=mode, repeat=rep, **stats))
    worker.close()
    report = {'input_kind':'unmodified real dataset prompts, both paths warm', 'runs':results}
    (evidence/'real_prompt_benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--real',action='store_true')
    parser.add_argument('--benchmark-real-only',action='store_true')
    parser.add_argument('--evidence',type=Path,default=ROOT/'evidence/T42')
    args=parser.parse_args()
    args.evidence.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='t42-tests-') as d:
        BASE,CANDIDATE,HELPER,TYPES=prepare(Path(d),args.evidence)
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        if not result.wasSuccessful(): sys.exit(1)
        if args.real: asyncio.run(real_tests(args.evidence))
        if args.benchmark_real_only: asyncio.run(real_benchmark(args.evidence))
