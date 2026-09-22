#!/usr/bin/env python3
"""T19 CPU HTTP mock tests: real sockets, concurrent SSE/flush/disconnect, no GPU."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.request
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import cap_spot_check as cap
import if_checks as checks
import logits_check as logits
import serving_probe as probe


@contextmanager
def mock_server(fault=None, warm_diff=0.0, cold_noise=0.0):
    condition = threading.Condition()
    state = {'active': set(), 'cache': set(), 'requests': [], 'headers': [], 'flushes': 0,
             'disconnects': 0, 'generated': 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, body, status=200):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            received = time.time()
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['requests'].append((self.path, body))
            state['headers'].append(dict(self.headers))
            if self.path.startswith('/flush_cache'):
                wait = float(parse_qs(urlsplit(self.path).query).get('timeout', ['0'])[0])
                with condition:
                    state['flushes'] += 1
                    if fault == 'busy_success' and state['active']:
                        return self.send_json({'success': True})
                    if fault == 'early_wait_success' and state['active'] and wait:
                        return self.send_json({'success': True})
                    deadline = time.monotonic() + wait
                    while state['active'] and time.monotonic() < deadline:
                        condition.wait(max(0, deadline-time.monotonic()))
                    success = not state['active']
                    if success and fault != 'no_clear':
                        state['cache'].clear()
                return self.send_json({'success': ('true' if fault == 'string_success' else success)},
                                      200 if success else 400)
            if self.path == '/v1/chat/completions':
                text = body['messages'][0]['content']
                match = re.search(r'known=(\d+|[A-D])', text)
                answer = match.group(1) if match else ('25' if 'log_x' in text else '350')
                content = 'Final answer: ' + answer
                if fault == 'wrong_answer':
                    content = 'Final answer: 999'
                if fault == 'missing_marker':
                    content = 'Here is a derivation without a conclusion.'
                if fault == 'reasoning_only':
                    content = ''
                reason = 'length' if fault in ('length', 'false_length') else 'stop'
                return self.send_json({'model': 'default', 'choices': [{'message': {
                    'content': content, 'reasoning_content': '' if fault == 'missing_reasoning' else 'private reasoning'},
                    'finish_reason': reason}], 'usage': {
                    'completion_tokens': body.get('max_tokens', 100) - (1 if fault == 'false_length' else 0)}})
            if self.path != '/generate':
                return self.send_json({}, 404)
            rid = body['rid']
            with condition:
                if rid in state['active']:
                    return self.send_json({'error': 'Duplicate request ID detected'}, 400)
                state['active'].add(rid)
                rank = body.get('routed_dp_rank', 0)
                had_cache = any(r == rank for r, _ in state['cache'])
                cached = 64 if had_cache else 0
                if fault == 'no_hit':
                    cached = 0
                state['cache'].add((rank, body['text']))
                state['generated'] += 1
                generation = state['generated']
            n = body['sampling_params']['max_new_tokens']
            ids = list(range(10, 10+n))
            if fault == 'greedy_drift' and had_cache and n == 32:
                ids[-1] = 9999
            text_parts = [f' word{i}' for i in ids]
            prompt_count = int(body['text'].split(':')[1]) if body['text'].startswith('rendered:') else 256
            if fault == 'bad_prompt_count':
                prompt_count += 1
            meta = {'id': rid, 'prompt_tokens': prompt_count, 'cached_tokens': cached,
                    'completion_tokens': n, 'request_received_ts': received,
                    'prefill_finished_time': received+.001, 'dp_rank': rank}
            if fault == 'bad_dp':
                meta['dp_rank'] = 0
            if fault == 'bad_time':
                meta['request_received_ts'] = received - 1
            if fault == 'bad_order':
                meta['prefill_finished_time'] = received - .1
            if fault == 'missing_time':
                del meta['prefill_finished_time']
            if body.get('return_logprob'):
                k = body['top_logprobs_num']
                # Swap tail token across the top-k cutoff; union must compare it.
                top_ids = list(range(10, 10+k))
                if fault == 'topk_crossing' and had_cache:
                    top_ids[-1] += 1
                shift = warm_diff if had_cache else (cold_noise if generation % 2 else 0)
                lp = lambda token: -.1 * (token-9) + (shift if token == 10 else 0)
                meta['output_token_logprobs'] = [[lp(i), i, None] for i in ids]
                meta['output_top_logprobs'] = [[[lp(i), i, None] for i in top_ids] for _ in ids]
                if 'token_ids_logprob' in body:
                    selected = body['token_ids_logprob']
                    if fault == 'missing_fixed_id':
                        selected = selected[:-1]
                    meta['output_token_ids_logprobs'] = [[[lp(i), i, None] for i in selected] for _ in ids]
            disconnected = False
            try:
                if not body.get('stream'):
                    self.send_json({'text': ''.join(text_parts), 'output_ids': ids, 'meta_info': meta})
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                counts = sorted(set((0, 1, min(2, n), n)))
                previous = 0
                for count in counts:
                    if n >= 1000 and count > 1:
                        time.sleep(.12)
                    m = {**meta, 'completion_tokens': count,
                         'finish_reason': {'type': 'length', 'length': n} if count == n else None}
                    if fault == 'short_stream' and count == n:
                        m['completion_tokens'] -= 1
                    start = 0 if fault in ('cumulative', 'cumulative_text') else previous
                    out_ids = ids[0 if fault == 'cumulative' else previous:count]
                    event = {'text': ''.join(text_parts[start:count]), 'output_ids': out_ids, 'meta_info': m}
                    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
                    self.wfile.flush()
                    previous = count
                if fault != 'no_done':
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                state['disconnects'] += 1
                disconnected = True
            finally:
                with condition:
                    if fault != 'rid_leak' and not (fault == 'disconnect_leak' and disconnected):
                        state['active'].discard(rid)
                    condition.notify_all()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    thread.start()
    try:
        yield probe.Client(f'http://127.0.0.1:{server.server_port}', timeout=3), state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class InterfaceTests(unittest.TestCase):
    def test_aime_chat_success_and_budget_truth(self):
        with mock_server() as (client, state):
            self.assertTrue(checks.check_chat(client)['final_marker'])
            self.assertNotIn('max_tokens', state['requests'][0][1])
        with mock_server('length') as (client, _):
            self.assertEqual(checks.check_chat(client, 240)['finish_reason'], 'length')

    def test_chat_faults_detected(self):
        for fault in ('missing_marker', 'missing_reasoning', 'reasoning_only', 'false_length', 'length'):
            with self.subTest(fault=fault), mock_server(fault) as (client, _):
                with self.assertRaises(probe.CheckFailed):
                    checks.check_chat(client, 240 if fault == 'false_length' else None)

    def test_all_four_lengths_and_done_required(self):
        with mock_server() as (client, _):
            self.assertEqual([r['budget'] for r in checks.exact_lengths(client)['lengths']], [1, 2, 240, 4096])
        for fault in ('no_done', 'short_stream'):
            with self.subTest(fault=fault), mock_server(fault) as (client, _):
                with self.assertRaises(probe.CheckFailed):
                    checks.exact_lengths(client)

    def test_incremental_and_cumulative_rejection(self):
        with mock_server() as (client, _):
            self.assertTrue(checks.incremental(client, 1)['text_matches_reference'])
        for fault in ('cumulative', 'cumulative_text'):
            with self.subTest(fault=fault), mock_server(fault) as (client, _):
                with self.assertRaises(probe.CheckFailed):
                    checks.incremental(client, 1)

    def test_harness_renderer_twenty_frozen_counts(self):
        class Renderer:
            calls = []
            def __init__(self, path):
                self.path = path
            def render(self, body):
                self.calls.append(body)
                return 'rendered:' + str(body['n'])
            def n_tokens(self, text):
                return int(text.split(':')[1])
        rows = {str(i): {'glm_tokens': 100+i} for i in range(30)}
        fake = SimpleNamespace(Renderer=Renderer, load_index=lambda _: (rows, {}, {}),
            materialize_bodies=lambda _, ids: {rid: {'n': rows[rid]['glm_tokens'], 'tools': ['keep-me']} for rid in ids})
        with patch.object(checks, 'harness', return_value=fake), mock_server() as (client, _):
            result = checks.dev_token_counts(client, Path('/unused'))
            self.assertEqual(len(result['requests']), 20)
            self.assertEqual(len({r['req_id'] for r in result['requests']}), 20)
            self.assertTrue(all(b['tools'] == ['keep-me'] for b in Renderer.calls))
        with patch.object(checks, 'harness', return_value=fake), mock_server('bad_prompt_count') as (client, _):
            with self.assertRaises(probe.CheckFailed):
                checks.dev_token_counts(client, Path('/unused'))

    def test_busy_flush_fails_then_waits_and_clears(self):
        with mock_server() as (client, state):
            result = checks.busy_flush(client, 1, 4096)
            self.assertGreater(result['wait_elapsed_s'], .1)
            self.assertEqual(result['after_flush_cached_tokens'], 0)
            self.assertFalse(state['active'])

    def test_busy_flush_false_success_and_cache_leaks_rejected(self):
        for fault in ('busy_success', 'early_wait_success', 'no_clear', 'string_success'):
            with self.subTest(fault=fault), mock_server(fault) as (client, _):
                with self.assertRaises(probe.CheckFailed):
                    checks.busy_flush(client, 1, 4096)

    def test_dp_all_ranks_and_unknown_route(self):
        with mock_server() as (client, _):
            self.assertEqual(checks.dp_flush(client, [0, 1], 1, 4096)['cleared_ranks'], [0, 1])
        with mock_server('bad_dp') as (client, _):
            with self.assertRaises(probe.CheckFailed):
                checks.dp_flush(client, [0, 1], 1, 4096)
        with self.assertRaises(probe.Inconclusive):
            checks.dp_flush(None, [], 1, 4096)

    def test_unknown_headers_on_both_paths(self):
        with mock_server() as (client, state):
            self.assertTrue(checks.headers_check(client)['chat'])
            self.assertEqual(len(state['requests']), 2)
            self.assertTrue(all(any(k.lower() == 'x-s1-t19-unknown' for k in h) for h in state['headers']))

    def test_rid_reuse_completed_and_disconnect(self):
        with mock_server() as (client, state):
            result = checks.rid_reuse(client, 4096, 1)
            self.assertTrue(result['disconnect_reuse'])
            self.assertGreaterEqual(state['disconnects'], 1)
            self.assertEqual(len({b['rid'] for _, b in state['requests']}), 1)

    def test_duplicate_rid_leak_fails(self):
        for fault in ('rid_leak', 'disconnect_leak'):
            with self.subTest(fault=fault), mock_server(fault) as (client, _):
                with self.assertRaises(probe.CheckFailed):
                    checks.rid_reuse(client, 4096, .3)

    def test_same_host_timestamp_checks(self):
        with mock_server() as (client, _):
            self.assertEqual(checks.timing(client, True)['ttft_source'], 'server')
        for fault in ('bad_time', 'bad_order', 'missing_time'):
            with self.subTest(fault=fault), mock_server(fault) as (client, _):
                with self.assertRaises(probe.CheckFailed):
                    checks.timing(client, True)
        with self.assertRaises(probe.Inconclusive):
            checks.timing(None, False)

    def test_blocked_does_not_pass(self):
        result = checks.run_checks(None, ['IF-12'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['checks'][0]['status'], 'blocked')


class LogprobTests(unittest.TestCase):
    pair = {'previous': 'Shared previous full prompt', 'next': 'Shared next full prompt'}

    def test_cold_warm_equal_and_correct_wire_parameters(self):
        with mock_server() as (client, state):
            result = logits.run_comparison([client], self.pair, k=3, wait=1, expected_cached=64)
            self.assertTrue(result['passed'])
            self.assertTrue(result['checkpoint_depth_verified'])
            bodies = [b for path, b in state['requests'] if path == '/generate' and b.get('return_logprob')]
            self.assertTrue(all(b['top_logprobs_num'] == 3 and b['logprob_start_len'] == -1 for b in bodies))
            self.assertTrue(any('token_ids_logprob' in b for b in bodies))
            self.assertEqual(result['servers'][0]['tolerance'], 0)

    def test_noise_calibration_cold_only(self):
        with mock_server(cold_noise=.01) as (client, state):
            result = logits.run_comparison([client], self.pair, calibrate=True, k=3, wait=1)
            self.assertTrue(result['passed'])
            self.assertAlmostEqual(result['servers'][0]['tolerance'], .02)
            self.assertTrue(all(b['text'] == self.pair['next'] for path, b in state['requests'] if path == '/generate'))

    def test_numeric_and_greedy_regressions_fail(self):
        with mock_server(warm_diff=.1) as (client, _):
            self.assertFalse(logits.run_comparison([client], self.pair, k=3, wait=1)['passed'])
        with mock_server('greedy_drift') as (client, _):
            self.assertFalse(logits.run_comparison([client], self.pair, k=3, wait=1)['passed'])

    def test_missing_hit_depth_and_union_values_rejected(self):
        for fault in ('no_hit', 'missing_fixed_id'):
            with self.subTest(fault=fault), mock_server(fault) as (client, _):
                with self.assertRaises(probe.CheckFailed):
                    logits.run_comparison([client], self.pair, k=3, wait=1)
        with mock_server() as (client, _):
            with self.assertRaises(probe.CheckFailed):
                logits.run_comparison([client], self.pair, k=3, wait=1, expected_cached=128)

    def test_top_k_union_includes_rank_crossing(self):
        with mock_server('topk_crossing') as (client, _):
            result = logits.run_comparison([client], self.pair, k=3, wait=1)
            self.assertTrue(result['passed'])
            self.assertEqual(result['servers'][0]['cold_vs_warm']['compared_token_ids'], [10, 11, 12, 13])

    def test_two_servers_compare_both_modes(self):
        with mock_server() as (one, _), mock_server(warm_diff=.2) as (two, _):
            result = logits.run_comparison([one, two], self.pair, k=3, wait=1)
            self.assertFalse(result['passed'])
            self.assertEqual([r['mode'] for r in result['cross_server']], ['cold', 'warm'])

    def test_two_servers_independent_boundary_depth_assertions(self):
        with mock_server() as (one, _), mock_server() as (two, _):
            result = logits.run_comparison([one, two], self.pair, k=3, wait=1,
                                           other_expected_cached=64)
            self.assertTrue(result['passed'])
            self.assertFalse(result['servers'][0]['checkpoint_depth_verified'])
            self.assertTrue(result['servers'][1]['checkpoint_depth_verified'])


class CapabilityTests(unittest.TestCase):
    @staticmethod
    def questions(n=10, kind='aime'):
        answer = '25' if kind == 'aime' else 'B'
        return [{'id': str(i), 'question': 'Test question known='+answer, 'answer': answer, 'kind': kind} for i in range(n)]

    def test_answer_extraction_is_explicit_and_content_only(self):
        self.assertEqual(cap.extract_answer(r'Work 123 then \boxed{025}', 'aime'), '25')
        self.assertEqual(cap.extract_answer('Answer: (B)', 'gpqa'), 'B')
        self.assertEqual(cap.extract_answer('025', 'aime'), '25')
        self.assertIsNone(cap.extract_answer('Some work involves 25 things.', 'aime'))
        self.assertIsNone(cap.extract_answer(None, 'aime'))
        self.assertIsNone(cap.extract_answer(r'\boxed{1234}', 'aime'))

    def test_two_servers_both_suites_without_any_output_clamp(self):
        with mock_server() as (one, state1), mock_server() as (two, state2):
            for suite, n, kind in (('aime', 10, 'aime'), ('gpqa', 20, 'gpqa')):
                result = cap.compare([one, two], self.questions(n, kind), suite, {}, 19)
                self.assertTrue(result['passed'])
                self.assertEqual(result['scores']['candidate'], n)
            for _, body in state1['requests']+state2['requests']:
                self.assertEqual(set(body), {'model', 'stream', 'temperature', 'messages'})
                self.assertEqual(body['model'], 'default')

    def test_regression_reasoning_only_and_length_fail(self):
        for fault in ('wrong_answer', 'reasoning_only', 'length'):
            with self.subTest(fault=fault), mock_server() as (one, _), mock_server(fault) as (two, _):
                self.assertFalse(cap.compare([one, two], self.questions(), 'aime', {}, 19)['passed'])

    def test_one_answer_regression_tolerance(self):
        calls = 0
        def evaluate(client, question):
            nonlocal calls
            calls += 1
            return {'correct': not (client == 'candidate' and question['id'] == '0'), 'interface_ok': True}
        with patch.object(cap, 'evaluate', side_effect=evaluate):
            result = cap.compare(['stock', 'candidate'], self.questions(), 'aime', {}, 19)
        self.assertEqual(calls, 20)
        self.assertTrue(result['passed'])
        self.assertEqual(result['scores'], {'stock': 10, 'candidate': 9})

    def test_tiny_and_single_server_cannot_mark_registry_pass(self):
        with mock_server() as (client, _):
            self.assertFalse(cap.compare([client], self.questions(), 'aime', {}, 19)['passed'])
            self.assertFalse(cap.compare([client, client], self.questions(2), 'tiny', {}, 19)['passed'])

    def test_runtime_aime_loader_seed_and_source_metadata(self):
        rows = [{'row': {'ID': str(i), 'Problem': f'problem {i}', 'Answer': i}, 'truncated_cells': []} for i in range(30)]
        def fetched(url, *_):
            return (json.dumps({'sha': 'revision'} if '/api/' in url else {'rows': rows}).encode(), 'digest')
        with patch.object(cap, 'fetch', side_effect=fetched):
            a, source = cap.load_questions('aime', 10)
            b, _ = cap.load_questions('aime', 10)
        self.assertEqual(a, b)
        self.assertEqual(source['metadata_revision'], 'revision')

    def test_gpqa_loader_preserves_answers_after_option_permutation(self):
        raw = 'Record ID,Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3\n'
        raw += '\n'.join(f'{i},question,correct,wrong1,wrong2,wrong3' for i in range(20))
        with patch.object(cap, 'fetch', return_value=(raw.encode(), 'digest')):
            questions, _ = cap.load_questions('gpqa', 20)
        self.assertEqual(len(questions), 20)
        for q in questions:
            self.assertIn(q['answer']+'. correct', q['question'])

    def test_dataset_writes_outside_arena_cache_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(probe.CheckFailed):
                cap.fetch('https://invalid.example', Path(temp), 'x')

    def test_dataset_redirect_308_on_python310(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                if self.path == '/mirror':
                    self.send_response(308)
                    self.send_header('Location', '/source')
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"id":"fixture"}')
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        thread.start()
        try:
            opener = urllib.request.build_opener(cap.Redirect308())
            with opener.open(f'http://127.0.0.1:{server.server_port}/mirror') as response:
                self.assertEqual(json.load(response)['id'], 'fixture')
        finally:
            server.shutdown(); server.server_close(); thread.join()


class CommandTests(unittest.TestCase):
    def test_all_dry_runs_offline_no_writes_or_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'result.json'
            for script in ('if_checks.py', 'logits_check.py', 'cap_spot_check.py'):
                result = subprocess.run([sys.executable, '-B', str(probe.REPO/'scripts'/script), '--dry-run',
                    '--base-url', 'http://127.0.0.1:1', '--out', str(out)], capture_output=True, text=True,
                    env={**os.environ, 'S1_API_KEY': 'hidden-test-secret', 'PYTHONDONTWRITEBYTECODE': '1'})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('hidden-test-secret', result.stdout+result.stderr)
                self.assertFalse(out.exists())

    def test_cli_real_mock_run_report_and_failure_exit(self):
        with mock_server() as (client, _), tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'result.json'
            cmd = [sys.executable, '-B', str(probe.REPO/'scripts/if_checks.py'), '--base-url', client.base,
                   '--cases', 'IF-12', '--out', str(out)]
            result = subprocess.run(cmd + ['--same-host'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(json.loads(out.read_text())['passed'])
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(out.read_text())['passed'])

    def test_logit_and_cap_cli_with_mock_endpoints(self):
        with mock_server() as (client, _), tempfile.TemporaryDirectory() as temp:
            pair_file = Path(temp)/'pair.json'
            pair_file.write_text(json.dumps(LogprobTests.pair))
            result = subprocess.run([sys.executable, '-B', str(probe.REPO/'scripts/logits_check.py'),
                '--base-url', client.base, '--pair-file', str(pair_file), '--top-k', '3', '--flush-wait', '1'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue(json.loads(result.stdout)['passed'])
            result = subprocess.run([sys.executable, '-B', str(probe.REPO/'scripts/cap_spot_check.py'),
                '--base-url', client.base, '--candidate-url', client.base, '--suite', 'tiny'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)  # explicit tiny diagnostic cannot pass CAP registry
            self.assertTrue(json.loads(result.stdout)['spot_check_passed'])
            self.assertFalse(json.loads(result.stdout)['registry_sample_complete'])


if __name__ == '__main__':
    unittest.main()
