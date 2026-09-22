#!/usr/bin/env python3
"""T29 CPU review of actual D0 v1.1 AST; no GPU or serving changes."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(
        n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


class D0Review(unittest.IsolatedAsyncioTestCase):
    async def reduce(self, values, processor=True):
        calls = []
        async def communicate(req):
            calls.append(('timeout', req.timeout_s))
            return values
        obj = NS(auto_create_handle_loop=lambda: calls.append('loop'),
                 flush_cache_communicator=communicate,
                 mm_processor=NS(clear_preprocess_cache=lambda: calls.append('clear'))
                 if processor else None)
        fn = extract(ROOT/'build/d0/b/python/sglang/srt/managers/tokenizer_control_mixin.py',
                     'flush_cache', {'FlushCacheReqInput': NS, 'FlushCacheReqOutput': NS})
        return await fn(obj, 7), calls

    async def test_all_success_clears(self):
        out, calls = await self.reduce([NS(success=True, message='')] * 2)
        self.assertTrue(out.success)
        self.assertEqual(calls, ['loop', ('timeout', 7), 'clear'])

    async def test_mixed_failure_preserves(self):
        out, calls = await self.reduce([NS(success=True, message=''), NS(success=False, message='busy')])
        self.assertFalse(out.success)
        self.assertIn('busy', out.message)
        self.assertNotIn('clear', calls)

    async def test_empty_is_not_success(self):
        out, calls = await self.reduce([])
        self.assertFalse(out.success)
        self.assertEqual(out.message, 'no worker responses')
        self.assertNotIn('clear', calls)

    async def test_no_processor(self):
        out, calls = await self.reduce([NS(success=True, message='')], False)
        self.assertTrue(out.success)
        self.assertNotIn('clear', calls)

    async def test_scope_stamp_precedes_body(self):
        seen = []
        cls = extract(ROOT/'build/d0/b/python/sglang/srt/entrypoints/http_server.py',
                      '_ArenaRecvTimeMiddleware', {'time': NS(perf_counter=lambda: 42.)})
        async def inner(scope, receive, send):
            seen.append(scope['arena_recv_perf'])
            await receive()
        async def receive():
            seen.append('body-read')
        scope = {'type': 'http', 'arena_recv_perf': -999, 'state': {'unrelated': 1}}
        await cls(inner)(scope, receive, None)
        self.assertEqual(seen, [42., 'body-read'])
        self.assertEqual(scope['state'], {'unrelated': 1})

    async def test_non_http_passthrough(self):
        cls = extract(ROOT/'build/d0/b/python/sglang/srt/entrypoints/http_server.py',
                      '_ArenaRecvTimeMiddleware', {'time': NS(perf_counter=lambda: 42.)})
        seen = []
        async def inner(scope, receive, send):
            seen.append(scope.copy())
        await cls(inner)({'type': 'lifespan'}, None, None)
        self.assertNotIn('arena_recv_perf', seen[0])


if __name__ == '__main__':
    unittest.main()
