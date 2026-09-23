#!/usr/bin/env python3
"""T57: collect existing pod logs through pexec_codex; never contact the engine.

Local: python3 scripts/pod/collect_run_logs.py --run 035-s0_n22 --n 22
Use existing SSH proxy: add --proxy. Already on GPU box: add --on-gpu.
Output goes to stdout/stderr. Requires Python 3 and the existing SSH alias GPU.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys


# Executed only by the existing CPU-only reviewer helper in the pod.
READ_CODE = r'''
import collections, datetime, json, pathlib, sys, time
run, level = sys.argv[1:3]
root = pathlib.Path('/tmp/ax/runs') / run
now = time.time()
print('READ_TIME_UTC', datetime.datetime.now(datetime.timezone.utc).isoformat())
print('RUN', run, 'LEVEL', level)
if not root.is_dir():
    print('RUN_DIRECTORY_MISSING', str(root))
    sys.exit(2)
for state in ('running', 'pending', 'done', 'failed', 'cancelled'):
    if (pathlib.Path('/tmp/ax/queue') / state / (run + '.sh')).exists():
        print('QUEUE_STATE', state)
logs = [('job.log', 50), ('server.log', 45), ('exit_code', 4),
        (level + '/run_dev.log', 35), (level + '/analysis.txt', 80),
        (level + '/verdict.json', 140), (level + '/gpu_util.csv', 8)]
for relative, count in logs:
    p = root / relative
    if not p.is_file():
        print('\nMISSING', relative)
        continue
    st = p.stat()
    print('\nFILE', relative, 'bytes', st.st_size,
          'age_seconds', round(now - st.st_mtime, 1))
    with p.open('rb') as f:
        start = max(0, st.st_size - 64000)
        f.seek(start)
        data = f.read(64000)
    lines = data.decode(errors='replace').splitlines()
    if start:
        lines = lines[1:]
    print('\n'.join(lines[-count:]))
raws = sorted((root / level).glob('raw_*.jsonl'))
print('\nRAW_FILE_COUNT', len(raws))
for p in raws:
    count = malformed = 0
    ids, errors = set(), collections.Counter()
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('record is not an object')
            except ValueError:
                malformed += 1
                continue
            count += 1
            ids.add(str(row.get('req_id')))
            if row.get('error_class'):
                errors[str(row['error_class'])] += 1
    print('RAW_SNAPSHOT', json.dumps(dict(file=p.name, rows=count,
          unique_ids=len(ids), malformed_lines=malformed,
          error_classes=dict(errors)), ensure_ascii=False))
print('\nDIAGNOSTIC_ONLY: live files may change; counts and old verdicts are not a validated SLO result.')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='035-s0_n22')
    parser.add_argument('--n', type=int, default=22)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--proxy', action='store_true',
                       help='Use the existing SSH configuration, including its proxy.')
    modes.add_argument('--on-gpu', action='store_true',
                       help='Run on the GPU development box; skip SSH.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.run) or args.n <= 0:
        parser.error('--run must be a simple run directory name; --n must be positive')

    command = shlex.join(['python3', '-c', READ_CODE, args.run, f'N{args.n}'])
    helper = '/sjtu/linhang/arena/repo/scripts/pod/pexec_codex'
    remote = 'cd /sjtu/linhang/arena/repo && ' + shlex.join(['bash', helper, command])
    env = os.environ.copy()
    if args.on_gpu:
        argv = ['bash', '-c', remote]
    else:
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=2']
        if not args.proxy:
            argv += ['-o', 'ProxyCommand=none', '-o', 'ProxyJump=none']
            env = {k: v for k, v in env.items()
                   if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
        argv += ['GPU', remote]
    print('CONNECTION_MODE', 'on-gpu' if args.on_gpu else
          ('configured-proxy' if args.proxy else 'direct'), flush=True)
    try:
        result = subprocess.run(argv, env=env, timeout=90)
    except subprocess.TimeoutExpired:
        print('READ_TIMEOUT: local reader timed out; no engine actions were issued.',
              file=sys.stderr)
        return 124
    print('COLLECT_EXIT_CODE', result.returncode, flush=True)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
