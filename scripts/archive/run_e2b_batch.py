#!/usr/bin/env python3
"""T26 bounded sequential E2b batch; owns and stops only its own session groups.

Run on the dev box after sourcing code/e1_env.sh. Inputs are frozen in the new
run directory. No installs, daemons, Trisol, or submission operations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

BASE = Path('/sjtu/linhang/arena')
CASES = ('smoke', 'reminder_heavy', 'strict_append')


def emit(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def gpu_free():
    result = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                      '--format=csv,noheader'], text=True)
    if result.strip():
        raise RuntimeError('GPU 0 occupied; refusing to launch')


def stop(server):
    # Popen(start_new_session=True) is the only source of the target group ID.
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=10)
    for _ in range(30):
        try:
            gpu_free()
            return
        except RuntimeError:
            time.sleep(1)
    raise RuntimeError('GPU 0 still occupied after own server shutdown; inspect, do not kill unrelated processes')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--variants', nargs='+', choices=('control', 'candidate', 'off'),
                   default=['control', 'candidate', 'off'])
    args = p.parse_args()
    root = args.root.resolve()
    if root.parent != BASE/'runs' or not root.name.startswith('E2b_'):
        p.error('dedicated arena/runs/E2b_* path required')
    if (root/'batch_started.json').exists():
        p.error('refusing to rerun or overwrite an existing batch')
    # Fail before GPU startup if the frozen replay dependency set is incomplete.
    for name in ('replay_chains.py', 'make_case_sets.py', 'e2b_trace.py'):
        if not (root/'scripts'/name).is_file():
            p.error('missing frozen dependency: '+name)
    gpu_free()
    (root/'batch_started.json').write_text(json.dumps(dict(at=time.time(), pid=os.getpid())))
    for variant in args.variants:
        gpu_free()
        source = BASE/'code'/('sglang-e2b-control' if variant == 'control' else 'sglang-e2b-candidate')
        env = os.environ.copy()
        env.update(PYTHONPATH=str(source/'python'), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8',
                   E2B_TRACE_ROOT=str(root/variant))
        env.pop('SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS', None)
        if variant != 'off':
            env['SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS'] = '154827,154829'
        trace = root/variant
        trace.mkdir()
        argv = [sys.executable, str(root/'scripts/e2b_trace.py'),
                '--model-path', str(BASE/'models/e1-kimi-linear-4l-qfull'),
                '--tokenizer-path', str(BASE/'s1-dev/glm_tok'),
                '--served-model-name', 'e1-random-kimi-linear',
                '--file-storage-path', str(trace/'storage'),
                '--host', '127.0.0.1', '--port', '31000', '--tp-size', '1',
                '--dtype', 'bfloat16', '--attention-backend', 'triton', '--linear-attn-backend', 'triton',
                '--sampling-backend', 'pytorch', '--page-size', '64', '--schedule-policy', 'fcfs',
                '--mamba-radix-cache-strategy', 'extra_buffer', '--mamba-ssm-dtype', 'float32',
                '--mamba-max-states-per-path', '-1', '--mamba-track-interval', '256',
                '--chunked-prefill-size', '8192', '--context-length', '131072',
                '--mem-fraction-static', '0.25', '--max-running-requests', '8',
                '--max-total-tokens', '131072', '--max-mamba-cache-size', '512',
                '--disable-cuda-graph', '--enable-metrics', '--incremental-streaming-output']
        files = [source/'python/sglang/srt/managers'/name for name in
                 ('schedule_policy.py', 'scheduler.py', 'tokenizer_control_mixin.py')]
        files.append(source/'python/sglang/srt/entrypoints/http_server.py')
        (trace/'launch.json').write_text(json.dumps(dict(argv=argv, source=str(source),
            role_ids=env.get('SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS'),
            hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}), indent=2))
        emit('start '+variant)
        with (trace/'server.log').open('w') as log:
            server = subprocess.Popen(argv, cwd=trace, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            try:
                deadline = time.monotonic()+600
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f'{variant} exited early: {server.returncode}')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:31000/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except (OSError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError('readiness exceeded 600s')
                    time.sleep(2)
                emit('ready '+variant)
                for case in (('smoke',) if variant == 'off' else CASES):
                    replay = [sys.executable, str(root/'scripts/replay_chains.py'),
                              '--dev-root', str(BASE/'s1-dev'), '--case-file', str(root/'cases'/f'{case}.json'),
                              '--max-prompt-tokens', '131072', '--require-json-flush',
                              '--output', str(root/f'{variant}_{case}')]
                    with (trace/f'replay_{case}.log').open('w') as out:
                        subprocess.run(replay, cwd=trace, env=env, stdout=out, stderr=subprocess.STDOUT,
                                       timeout=1800, check=True)
                    emit('completed '+variant+' '+case)
                req = urllib.request.Request('http://127.0.0.1:31000/flush_cache', method='POST')
                with urllib.request.urlopen(req, timeout=60) as response:
                    result = json.load(response)
                    if not flush_success(result):
                        raise RuntimeError(f'final flush failed: {result}')
            finally:
                stop(server)
                emit('stopped '+variant)
    (root/'batch_complete.json').write_text(json.dumps(dict(at=time.time(), status='complete')))
    emit('complete; all owned GPU services stopped')


def flush_success(result):
    # D0 may include an optional explanatory message; only success is required.
    return isinstance(result, dict) and result.get('success') is True


if __name__ == '__main__':
    # Convert bounded supervisor termination into an exception so finally stops
    # the separate service session too. SIGKILL cannot be caught.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
