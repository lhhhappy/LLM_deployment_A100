#!/usr/bin/env python3
"""T19 interface checks on an ALREADY RUNNING, isolated L2/T8 server.

Examples: python3 scripts/if_checks.py --dry-run
  python3 scripts/if_checks.py --base-url http://127.0.0.1:8000 --same-host --out runs/if.json
  python3 scripts/if_checks.py --cases IF-09 --dp-ranks 0,1 --out runs/dp.json
IF-06 needs the original harness requirements/tokenizer; full prompts are kept.
IF-02/06 need GLM weights/tokenizer; random L2 weights cannot pass IF-02.
Long exact-length/busy probes use ignore_eos only for interface testing.
IF-12 requires --same-host (same OS realtime clock, no remote proxy).
IF-09 requires the COMPLETE rank list, routed_dp_rank support and dp_rank echo.
No prompts, generated text, API keys or raw remote errors are written to reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import threading
import time
import uuid

from serving_probe import (REPO, CheckFailed, Inconclusive, client_for, events, flush,
                           harness, integer, payload, require, stream, validate_stream, write_report)

DEFAULT_CASES = ['IF-02', 'IF-04', 'IF-05', 'IF-06', 'IF-08', 'IF-10', 'IF-11', 'IF-12']
DESCRIPTIONS = {
    'IF-02': 'AIME-style chat: content/final marker, separate reasoning, truthful finish_reason/budget',
    'IF-04': 'ignore_eos lengths 1,2,240,4096 (repeat separately for MTP on/off)',
    'IF-05': 'incremental SSE text and output IDs; concatenation equals cold nonstream greedy output',
    'IF-06': 'seeded random 20 original dev prompts through harness Renderer, exact frozen glm_tokens',
    'IF-08': 'live long stream: flush timeout=0 -> 400 false, waiting flush -> 200 true, cold after',
    'IF-09': 'all explicitly routed DP ranks: one busy at a time, aggregate failure then zero cached_tokens',
    'IF-10': 'unknown X-S1-* headers accepted on generate and chat',
    'IF-11': 'rid reused after completion and midstream socket disconnect; bounded cleanup retries',
    'IF-12': 'same-host send-to-receive 0..50 ms, ordered timestamps and server TTFT',
}
PROBE = 'Continue this sequence of distinct words: amber birch cobalt dune elm fern granite. '


def check_chat(client, budget=None, headers=None):
    body = {'model': 'default', 'stream': False, 'temperature': 0,
            'messages': [{'role': 'user', 'content':
                'Find the remainder when the sum of the squares of the integers from 1 to 100 '
                'is divided by 1000. Give your final answer as \\boxed{n}.'}]}
    if budget is not None:
        body['max_tokens'] = budget
    reply = client.json('/v1/chat/completions', body, headers=headers)
    require(reply.get('model') == 'default', 'chat model mismatch')
    choices = reply.get('choices')
    require(isinstance(choices, list) and bool(choices), 'chat choices missing')
    choice = choices[0]
    message = choice.get('message', {})
    content, reasoning = message.get('content'), message.get('reasoning_content')
    require(isinstance(content, str) and bool(content.strip()), 'answer content empty')
    require(bool(re.search(r'\\boxed\s*\{[^}]+\}|(?:final\s+answer|answer)\s*[:=]\s*\S+', content, re.I)),
            'final answer marker absent from content')
    require(isinstance(reasoning, str) and bool(reasoning.strip()), 'reasoning_content empty')
    reason = choice.get('finish_reason')
    require(reason in ('stop', 'length'), 'unexpected chat finish reason')
    count = reply.get('usage', {}).get('completion_tokens')
    if reason == 'length':
        require(budget is not None and integer(count) and count == budget,
                'length finish without evidence that requested budget was reached')
    return {'content_present': True, 'reasoning_present': True, 'final_marker': True,
            'finish_reason': reason, 'completion_tokens': count, 'requested_budget': budget}


def exact_lengths(client):
    rows = []
    for n in (1, 2, 240, 4096):
        data, _ = stream(client, payload(PROBE, n))
        _, meta = validate_stream(data, n)
        rows.append({'budget': n, 'completion_tokens': meta['completion_tokens'], 'events': len(data)})
    return {'lengths': rows}


def incremental(client, wait):
    flush(client, wait)
    prompt = PROBE + uuid.uuid4().hex
    data, _ = stream(client, payload(prompt, 64))
    validate_stream(data, 64)
    text_parts = [e.get('text') for e in data]
    require(all(isinstance(t, str) for t in text_parts), 'stream text missing')
    if sum(bool(t) for t in text_parts) < 2:
        raise Inconclusive('need at least two nonempty SSE text chunks; reduce stream interval')
    ids = []
    previous = 0
    ids_available = all(isinstance(e.get('output_ids'), list) for e in data)
    if ids_available:
        for e in data:
            count = e['meta_info']['completion_tokens']
            require(len(e['output_ids']) == count - previous, 'output IDs repeat a prefix instead of being deltas')
            previous = count
            ids.extend(e['output_ids'])
    flush(client, wait)
    reference = client.json('/generate', payload(prompt, 64))
    require(reference.get('meta_info', {}).get('completion_tokens') == 64, 'reference output too short')
    require(''.join(text_parts) == reference.get('text'),
            'SSE text concatenation differs from cold greedy reference (cumulative prefix or nondeterminism)')
    if ids_available:
        require(ids == reference.get('output_ids'), 'stream/reference greedy token IDs differ')
    return {'events': len(data), 'text_matches_reference': True, 'delta_token_ids_checked': ids_available}


def dev_token_counts(client, dev_root, seed=19):
    common = harness(dev_root)
    root = Path(dev_root) / 'data/dev-combined-v1'
    rows, _, _ = common.load_index(str(root))
    require(len(rows) >= 20, 'need at least 20 dev requests')
    selected = random.Random(seed).sample(sorted(rows), 20)
    bodies = common.materialize_bodies(str(root), selected)
    renderer = common.Renderer(str(Path(dev_root) / 'glm_tok'))
    results = []
    for rid in selected:
        require(rid in bodies, 'dev body missing')
        prompt = renderer.render(bodies[rid])
        frozen = rows[rid]['glm_tokens']
        require(renderer.n_tokens(prompt) == frozen, 'local Renderer differs from frozen glm_tokens')
        result = client.json('/generate', payload(prompt, 1))
        count = result.get('meta_info', {}).get('prompt_tokens')
        results.append({'req_id': rid, 'frozen_glm_tokens': frozen, 'server_prompt_tokens': count})
    require(all(r['frozen_glm_tokens'] == r['server_prompt_tokens'] for r in results),
            'server token count differs from frozen glm_tokens (no truncation or extra template allowed)')
    return {'seed': seed, 'requests': results}


class BackgroundStream:
    def __init__(self, client, body):
        self.client, self.body = client, body
        self.ready, self.done = threading.Event(), threading.Event()
        self.data, self.error, self.finished_at = [], None, None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            with self.client.open('/generate', self.body) as response:
                for e in events(response):
                    self.data.append(e)
                    if e['meta_info'].get('completion_tokens', 0) > 0:
                        self.ready.set()
            self.finished_at = time.monotonic()
        except Exception as e:
            self.error = e
        finally:
            self.done.set()
            self.ready.set()

    def start(self):
        self.thread.start()
        require(self.ready.wait(self.client.timeout), 'long stream first token timed out')
        if self.error:
            raise CheckFailed('long stream failed') from None
        if self.done.is_set():
            raise Inconclusive('long stream completed before busy probe; increase --long-tokens')
        return self

    def join(self):
        self.thread.join(self.client.timeout)
        require(not self.thread.is_alive(), 'long stream did not finish')
        if self.error:
            raise CheckFailed('long stream interrupted') from None
        validate_stream(self.data, self.body['sampling_params']['max_new_tokens'])


def busy_flush(client, wait=120, long_tokens=4096, rank=None):
    require(wait > 0, 'busy flush requires positive server wait')
    flush(client, wait)
    prompt = (PROBE * 128) + uuid.uuid4().hex
    live = BackgroundStream(client, payload(prompt, long_tokens, stream=True, rank=rank))
    try:
        live.start()
        # Both flush requests must be issued while generation is still in flight.
        flush(client, 0, success=False)
        if live.done.is_set():
            raise Inconclusive('stream ended before waiting flush; increase --long-tokens')
        started = time.monotonic()
        flush(client, wait)
        returned = time.monotonic()
        live.join()
        # Concurrent sockets can deliver terminal SSE just after flush HTTP response.
        require(live.finished_at <= returned + .050, 'flush reported success while stream was still running')
        require(started < live.finished_at, 'waiting flush was not exercised during stream')
        cold = client.json('/generate', payload(prompt, 1, rank=rank))['meta_info']
        require(cold.get('cached_tokens') == 0, 'successful busy flush did not clear cache')
        if rank is not None:
            require(all(e['meta_info'].get('dp_rank') == rank for e in live.data), 'DP routing not confirmed')
            require(cold.get('dp_rank') == rank, 'DP cold routing not confirmed')
        return {'busy_status': 400, 'wait_status': 200, 'wait_elapsed_s': returned-started,
                'after_flush_cached_tokens': 0, 'rank': rank}
    finally:
        # Drain only our bounded request; never abort unrelated work or abort_all.
        if live.thread.is_alive():
            live.thread.join(client.timeout)


def dp_flush(client, ranks, wait, long_tokens):
    if len(ranks) < 2:
        raise Inconclusive('IF-09 needs complete --dp-ranks and verifiable routed_dp_rank; see tests/T19_USAGE.md')
    rows = []
    # Put each rank in the busy position, including nonzero ranks (D0 aggregation bug).
    for busy_rank in ranks:
        rows.append(busy_flush(client, wait, long_tokens, busy_rank))
    probes = {rank: PROBE * 128 + uuid.uuid4().hex for rank in ranks}
    for rank in ranks:
        for _ in range(2):
            m = client.json('/generate', payload(probes[rank], 1, rank=rank))['meta_info']
            require(m.get('dp_rank') == rank, 'DP routing not confirmed')
        require(m.get('cached_tokens', 0) > 0, 'DP rank cache was not populated')
    flush(client, wait)
    for rank in ranks:
        m = client.json('/generate', payload(probes[rank], 1, rank=rank))['meta_info']
        require(m.get('dp_rank') == rank and m.get('cached_tokens') == 0, 'DP flush left a populated rank')
    return {'busy_ranks': rows, 'cleared_ranks': ranks}


def headers_check(client):
    headers = {'X-S1-T19-Unknown': 'accepted-probe', 'X-S1-Future-Field': '19'}
    data, _ = stream(client, payload(PROBE, 2), headers)
    validate_stream(data, 2)
    # Header acceptance is independent of random L2 model reasoning ability.
    reply = client.json('/v1/chat/completions', {'model': 'default', 'messages': [
        {'role': 'user', 'content': 'Hello.'}], 'stream': False}, headers=headers)
    require(isinstance(reply.get('choices'), list) and bool(reply['choices']), 'chat failed with unknown headers')
    return {'generate': True, 'chat': True}


def rid_reuse(client, long_tokens=4096, cleanup_timeout=5):
    rid = 't19-reuse-' + uuid.uuid4().hex
    for _ in range(2):
        data, _ = stream(client, payload(PROBE, 2, rid=rid))
        _, m = validate_stream(data, 2)
        require(m.get('id') == rid, 'server did not echo reused rid')
    with client.open('/generate', payload(PROBE, long_tokens, stream=True, rid=rid)) as response:
        for e in events(response):
            count = e['meta_info'].get('completion_tokens', 0)
            if count > 0:
                if count >= long_tokens or e['meta_info'].get('finish_reason') is not None:
                    raise Inconclusive('stream finished before disconnect; increase --long-tokens')
                break
        else:
            raise CheckFailed('no event before disconnect')
    # close() tears down the socket; do not flush or wait for normal completion.
    deadline = time.monotonic() + cleanup_timeout
    attempts = 0
    while True:
        attempts += 1
        status, reply = client.json_status('/generate', payload(PROBE, 2, rid=rid))
        if status == 200:
            m = reply.get('meta_info', {})
            require(m.get('id') == rid and m.get('completion_tokens') == 2, 'rid reuse returned wrong request')
            return {'completed_reuse': True, 'disconnect_reuse': True, 'attempts': attempts}
        # SGLang may still be processing the disconnect when the new request arrives.
        error_text = json.dumps(reply).lower()
        duplicate = any(x in error_text for x in ('duplicate', 'already exists', 'already in use'))
        require(status in (400, 409) and duplicate, 'rid reuse failed for a non-duplicate reason')
        require(time.monotonic() < deadline, 'disconnected rid was not released within cleanup timeout')
        time.sleep(.05)


def timing(client, same_host=False):
    if not same_host:
        raise Inconclusive('IF-12 requires --same-host and a direct client in the server clock domain')
    data, sent = stream(client, payload(PROBE, 2))
    first, final = validate_stream(data, 2)
    delay = final['request_received_ts'] - sent
    require(0 <= delay <= .050, 'server receive timestamp is outside client send + [0,50ms]')
    return {'receive_minus_send_ms': delay * 1000, 'ttft_source': 'server',
            'server_ttft_s': first['prefill_finished_time'] - final['request_received_ts']}


def run_checks(client, cases, *, dev_root=REPO/'s1-dev', seed=19, same_host=False,
               wait=120, long_tokens=4096, chat_budget=None, dp_ranks=(), cleanup_timeout=5):
    actions = {'IF-02': lambda: check_chat(client, chat_budget), 'IF-04': lambda: exact_lengths(client),
        'IF-05': lambda: incremental(client, wait), 'IF-06': lambda: dev_token_counts(client, dev_root, seed),
        'IF-08': lambda: busy_flush(client, wait, long_tokens),
        'IF-09': lambda: dp_flush(client, dp_ranks, wait, long_tokens), 'IF-10': lambda: headers_check(client),
        'IF-11': lambda: rid_reuse(client, long_tokens, cleanup_timeout), 'IF-12': lambda: timing(client, same_host)}
    rows = []
    for case in cases:
        try:
            rows.append({'case': case, 'status': 'pass', 'details': actions[case]()})
        except Inconclusive as e:
            rows.append({'case': case, 'status': 'blocked', 'reason': str(e)})
        except CheckFailed as e:
            rows.append({'case': case, 'status': 'fail', 'reason': str(e)})
        except Exception:
            rows.append({'case': case, 'status': 'fail', 'reason': 'local/HTTP failure; details suppressed'})
    return {'checks': rows, 'passed': all(r['status'] == 'pass' for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--cases', default=','.join(DEFAULT_CASES))
    p.add_argument('--dev-root', type=Path, default=REPO/'s1-dev')
    p.add_argument('--same-host', action='store_true')
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--long-tokens', type=int, default=4096)
    p.add_argument('--chat-budget', type=int, help='optional explicit IF-02 budget; omitted by default')
    p.add_argument('--flush-wait', type=int, default=180)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--cleanup-timeout', type=float, default=5)
    p.add_argument('--dp-ranks', default='', help='complete comma-separated list of live DP ranks')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    cases = a.cases.split(',')
    if any(c not in DESCRIPTIONS for c in cases):
        p.error('unknown case ID')
    try:
        ranks = [int(r) for r in a.dp_ranks.split(',') if r]
    except ValueError:
        p.error('DP ranks must be integers')
    if a.long_tokens < 2 or not 0 < a.flush_wait < a.timeout or a.cleanup_timeout <= 0:
        p.error('positive limits required; HTTP timeout must exceed flush wait')
    if a.chat_budget is not None and a.chat_budget <= 0:
        p.error('chat budget must be positive')
    if any(r < 0 for r in ranks) or len(set(ranks)) != len(ranks):
        p.error('DP ranks must be unique nonnegative integers')
    if a.dry_run:
        print(json.dumps({'dry_run': True, 'checks': {c: DESCRIPTIONS[c] for c in cases},
                          'note': 'no HTTP, imports of harness, downloads or file writes'}, indent=2))
        return 0
    result = run_checks(client_for(a.base_url, a.timeout), cases, dev_root=a.dev_root, seed=a.seed,
        same_host=a.same_host, wait=a.flush_wait, long_tokens=a.long_tokens, chat_budget=a.chat_budget,
        dp_ranks=ranks, cleanup_timeout=a.cleanup_timeout)
    write_report(result, a.out)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
