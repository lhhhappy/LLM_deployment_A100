#!/usr/bin/env python3
"""Shared stdlib HTTP/SSE helpers for T19 diagnostics; no engine startup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid

from preflight_8gpu import CheckFailed, Client as BaseClient, integer, require, timestamp

REPO = Path(__file__).resolve().parents[1]


class Inconclusive(CheckFailed):
    """A prerequisite was not demonstrated; never count this as a pass."""


class Client(BaseClient):
    def open(self, path, payload=None, headers=None):
        require(path.startswith('/') and not path.startswith('//'), 'invalid endpoint path')
        hdr = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}
        hdr.update(headers or {})
        if self.api_key:
            hdr['Authorization'] = 'Bearer ' + self.api_key
        req = urllib.request.Request(self.base + path, headers=hdr,
            data=json.dumps(payload).encode() if payload is not None else None,
            method='POST' if payload is not None else 'GET')
        sent = time.time()  # wall clock, like SGLang's exported realtime timestamps
        try:
            response = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            response = e
        except Exception:
            raise CheckFailed('HTTP transport failure (details suppressed)') from None
        response.client_send_ts = sent
        return response

    def json_status(self, path, payload=None, headers=None):
        with self.open(path, payload, headers) as response:
            try:
                return response.status, json.load(response)
            except Exception:
                raise CheckFailed('invalid JSON response (details suppressed)') from None

    def json(self, path, payload=None, expected_status=200, headers=None):
        status, body = self.json_status(path, payload, headers)
        require(status == expected_status, 'unexpected HTTP status')
        require(isinstance(body, dict), 'JSON response must be an object')
        return body


def flush(client, wait=120, success=True):
    status, body = client.json_status('/flush_cache?timeout=' + str(wait), {})
    require(status == (200 if success else 400), 'flush HTTP status mismatch')
    require(isinstance(body, dict) and body.get('success') is success,
            'flush requires an exact JSON boolean success')
    return {'success': success, 'http_status': status}


def payload(prompt, tokens, stream=False, rid=None, rank=None, **extra):
    result = {'text': prompt, 'stream': stream, 'rid': rid or ('t19-' + uuid.uuid4().hex),
              'sampling_params': {'temperature': 0, 'max_new_tokens': tokens, 'ignore_eos': True}}
    if rank is not None:
        result['routed_dp_rank'] = rank
    result.update(extra)
    return result


def events(response):
    """Require a framed [DONE], including on early EOF; preserve SSE data deltas."""
    require(response.status == 200, 'generate HTTP status mismatch')
    require('text/event-stream' in response.headers.get('Content-Type', ''), 'SSE content type required')
    lines = []
    for raw in response:
        try:
            line = raw.decode('utf-8').rstrip('\r\n')
        except UnicodeError:
            raise CheckFailed('invalid SSE encoding') from None
        if line.startswith('data:'):
            lines.append(line[5:].lstrip(' '))
        elif not line and lines:
            data, lines = '\n'.join(lines), []
            if data == '[DONE]':
                return
            try:
                event = json.loads(data)
            except ValueError:
                raise CheckFailed('invalid SSE JSON') from None
            require(isinstance(event, dict) and isinstance(event.get('meta_info'), dict), 'SSE missing meta_info')
            require('error' not in event, 'server returned SSE error')
            yield event
    raise CheckFailed('SSE ended without framed [DONE]')


def stream(client, body, headers=None):
    with client.open('/generate', {**body, 'stream': True}, headers) as response:
        sent = response.client_send_ts
        data = list(events(response))
    require(bool(data), 'empty SSE stream')
    return data, sent


def validate_stream(data, tokens):
    first = None
    previous = 0
    baseline = None
    for e in data:
        m = e['meta_info']
        require(all(integer(m.get(k)) for k in ('prompt_tokens', 'cached_tokens', 'completion_tokens')),
                'invalid SSE token counters')
        require(0 <= m['cached_tokens'] <= m['prompt_tokens'] and m['prompt_tokens'] > 0, 'invalid cache count')
        counts = (m['prompt_tokens'], m['cached_tokens'])
        if baseline is None:
            baseline = counts
        require(counts == baseline, 'prompt/cache counters changed midstream')
        require(previous <= m['completion_tokens'] <= tokens, 'nonmonotonic/over-budget completion count')
        previous = m['completion_tokens']
        if first is None and previous > 0:
            first = m
    final = data[-1]['meta_info']
    require(final['completion_tokens'] == tokens, 'ignore_eos output length mismatch')
    require(first is not None and timestamp(first.get('prefill_finished_time')) and
            timestamp(final.get('request_received_ts')), 'required first-token/final timestamps missing')
    require(first['prefill_finished_time'] >= final['request_received_ts'], 'negative server TTFT')
    return first, final


def harness(dev_root):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(dev_root).resolve() / 'harness'))
    import s1_common
    return s1_common


def write_report(report, out=None):
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if out:
        out = Path(out).resolve()
        for folder in ('s1-dev', 'src/sglang', 'llm-challenge-arena-v1'):
            protected = (REPO / folder).resolve()
            require(out != protected and protected not in out.parents, 'output points into read-only input tree')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end='')


def client_for(url, timeout=600, key_env='S1_API_KEY'):
    return Client(url, timeout, os.environ.get(key_env))
