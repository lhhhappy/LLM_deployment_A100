#!/usr/bin/env python3
"""Isolated service cost screening, never an SLO or numerical-equivalence gate.

Cold requests, fixed token IDs and output counts. Decode workloads include MTP
and scheduling: their HTTP wall time is NOT isolated kernel time or TPOT.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import urllib.request


def post(path, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{os.environ.get('PORT', '30000')}" + path,
        json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        return response.read()


def inputs(length, batch):
    return [list_tokens(length, 31000 + length + i) for i in range(batch)]


def flush():
    raw = post('/flush_cache?timeout=30', {}).decode()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get('success') is not True:
        raise ValueError('flush lacks success=true acknowledgement')
    return raw


def list_tokens(length, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(length)]


def validate(response, length, count):
    meta = response['meta_info']
    tokens = meta['output_token_logprobs']
    if meta['prompt_tokens'] != length or meta['completion_tokens'] != count or len(tokens) != count:
        raise ValueError('prompt/output count differs from frozen request')
    if meta.get('cached_tokens') != 0:
        raise ValueError('cold request unexpectedly reused cache')
    if any(not math.isfinite(row[0]) for row in tokens):
        raise ValueError('non-finite output logprob')
    return meta


def generate(ids, count):
    body = {'input_ids': ids, 'return_logprob': True,
            'sampling_params': {'max_new_tokens': count, 'temperature': 0, 'ignore_eos': True}}
    start = time.perf_counter()
    response = json.loads(post('/generate', body))
    elapsed = time.perf_counter() - start
    validate(response, len(ids), count)
    return {'wall_s': elapsed, 'response': response}


def run(root, repeats=3):
    root.mkdir(parents=True, exist_ok=False)
    specs = [(n, 1, 1) for n in (256, 1024, 4096, 8192, 16384)]
    specs += [(256, n, 64) for n in (1, 8, 32)]
    rows = []
    with (root / 'raw.jsonl').open('w') as handle:
        for length, batch, count in specs:
            prompts = inputs(length, batch)
            digest = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
            for repeat in range(repeats + 1):
                flush_receipt = flush()
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(batch) as executor:
                    futures = [executor.submit(generate, ids, count) for ids in prompts]
                    responses = [future.result() for future in futures]
                row = {'input_tokens': length, 'batch': batch, 'output_tokens': count,
                       'repeat': repeat, 'warmup': repeat == 0, 'prompt_sha256': digest,
                       'flush_receipt': flush_receipt, 'batch_wall_s': time.perf_counter() - start,
                       'responses': responses}
                handle.write(json.dumps(row, allow_nan=False) + '\n'); handle.flush()
                rows.append(row)
                print('COST_SAMPLE', length, batch, count, repeat, row['batch_wall_s'], flush=True)
    summary = []
    for length, batch, count in specs:
        selected = [row for row in rows if not row['warmup'] and
                    (row['input_tokens'], row['batch'], row['output_tokens']) == (length, batch, count)]
        samples = [row['batch_wall_s'] for row in selected]
        summary.append({'input_tokens': length, 'batch': batch, 'output_tokens': count,
                        'samples_s': samples, 'median_s': statistics.median(samples),
                        'prompt_sha256': selected[0]['prompt_sha256']})
    (root / 'summary.json').write_text(json.dumps({
        'status': 'COST_SCREEN_COMPLETE', 'repeats': repeats, 'cases': summary,
        'scope': 'Isolated HTTP service wall time; output fingerprints retained. '
                 'No model equivalence, capability, full-load N@SLO or pure-kernel/TPOT verdict.'}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()
    run(args.out_dir)
