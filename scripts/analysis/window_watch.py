#!/usr/bin/env python3
"""Read-only, retrying snapshots of a queued replay. Never stop a GPU job.

Run through window_watch.sh JOB [interval_s=1500] [window_min=25].
Live raw contains completed requests only: all windows remain open until a
matching VALID complete-data receipt is available. Failed jobs may be complete
SLO failures or incomplete measurements; queue state alone cannot distinguish them.
Heartbeat, PID and lock live in ignored build/scratch/window-watch/JOB/;
evidence/LJOB/window/ contains result snapshots only.
"""
import argparse
import base64
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time

import window_gates as gates

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/pod'))
from watch_queue import atomic_json, run_bounded

TERMINAL = {'done', 'failed', 'cancelled'}
CHUNK = 90000

SNAPSHOT_CODE = r'''
import gzip, hashlib, json, pathlib, re, sys, time
job, want_data = sys.argv[1], sys.argv[2] == '1'
root = pathlib.Path('/tmp/ax/runs') / job
states = [s for s in ('pending','running','done','failed','cancelled')
          if (pathlib.Path('/tmp/ax/queue') / s / (job+'.sh')).exists()]
if len(states) > 1: raise ValueError('ambiguous queue state; retry')
state = states[0] if states else 'unknown'
m = dict(job_state=state, observed_at=time.time(), data=False)
def tail(path, size=65536):
    if not path.is_file(): return ''
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-size)); return f.read().decode(errors='replace')
levels = sorted(root.glob('N[0-9]*'), key=lambda p:p.stat().st_mtime)
h = dict(phase='startup', recent_error_lines=[], observation='file snapshots only')
server = root/'server.log'
if server.is_file():
    t = tail(server)
    h['server_log_age_s'] = round(time.time()-server.stat().st_mtime, 1)
    h['recent_error_lines'] = [line[-400:] for line in t.splitlines()
        if re.search(r'Traceback|CUDA out of memory|OutOfMemoryError|AssertionError|ERROR|Watchdog', line)][-8:]
    batches = [line for line in t.splitlines() if 'Prefill batch' in line or 'Decode batch' in line]
    if batches: h['latest_batch'] = batches[-1][-1000:]
if levels:
    level = levels[-1]
    h['level'] = level.name
    h['phase'] = 'preflight'
    if (level/'warmup.log').is_file():
        h['phase'] = 'warmup'
        h['warmup_tail'] = tail(level/'warmup.log', 1800).splitlines()[-3:]
    if (level/'flush_evidence.json').is_file():
        f = json.loads((level/'flush_evidence.json').read_text())
        h['runner_elapsed_s'] = round(time.time()-f['runner_started_s'], 1)
        if f.get('flush_success') is True:
            h['phase'] = 'measurement'
            h['since_flush_s'] = round(time.time()-f['flush_finished_s'], 1)
    gpu = level/'gpu_util.csv'
    if gpu.is_file(): h['gpu_latest'] = tail(gpu, 4096).splitlines()[-8:]
    if (level/'rundev_exit_code').is_file(): h['phase'] = 'scoring_or_finished'
    checkpoints = sorted(level.glob('checkpoint_*.json'), key=lambda p:p.stat().st_mtime)
    if checkpoints:
        c = json.loads(checkpoints[-1].read_text())
        h['completed_at_checkpoint'] = c.get('n_done')
        h['checkpoint_age_s'] = round(time.time()-c['at_s'], 1)
        raw = level/pathlib.Path(c['raw_file']).name
        if raw.is_file(): h['raw_age_s'] = round(time.time()-raw.stat().st_mtime, 1)
m['health'] = h
def done():
    print('WINDOW_META '+json.dumps(m)); sys.exit(0)
if state in ('unknown','pending') or (state == 'running' and not want_data): done()
cks = sorted(root.glob('N*/checkpoint_*.json'), key=lambda p:p.stat().st_mtime)
if not cks: done()
ck = cks[-1]; level = ck.parent
c = json.loads(ck.read_text())
name = c.get('raw_file')
if not isinstance(name, str) or pathlib.Path(name).name != name or not name.startswith('raw_'):
    raise ValueError('invalid checkpoint raw filename')
flush = json.loads((level/'flush_evidence.json').read_text())
if flush.get('flush_success') is not True: raise ValueError('measurement lacks successful flush')
b = (level/name).read_bytes()
# Preserve only complete JSONL records in the live immutable snapshot.
if state == 'running' and b and not b.endswith(b'\n'): b = b[:b.rfind(b'\n')+1]
if not b: done()
z = gzip.compress(b, compresslevel=1, mtime=0); sha = hashlib.sha256(z).hexdigest()
p = pathlib.Path('/tmp/ax/codex') / ('window_'+job+'_'+sha+'.gz')
if not p.exists():
    tmp=p.with_suffix('.part'); tmp.write_bytes(z); tmp.replace(p)
m.update(data=True, raw=name, n=int(level.name[1:]), checkpoint=c, flush=flush,
         archive=str(p), size=len(z), sha256=sha)
for key, file in [('score','score_formal.json'),('verdict','level_verdict.json'),('summary','summary.json'),
                  ('timed_verdict','timed_verdict.json'),('timed_score','timed_score.json')]:
    if (level/file).is_file(): m[key]=json.loads((level/file).read_text())
if 'timed_verdict' in m:
    m['raw_sha256'] = hashlib.sha256(b).hexdigest()
    # Request ids are in the immutable raw; do not duplicate them in transport metadata.
    m['timed_verdict'].pop('dispatched_req_ids', None)
print('WINDOW_META '+json.dumps(m))
'''

READ_CODE = r'''
import base64, sys
with open(sys.argv[1], 'rb') as f:
    f.seek(int(sys.argv[2])); b=f.read(int(sys.argv[3]))
print('WINDOW_DATA '+base64.b64encode(b).decode('ascii'))
'''


def remote(code, *args):
    command = shlex.join(['python3', '-c', code, *map(str, args)])
    # CPU-only reader; its only pod writes are immutable snapshots under codex/.
    return run_bounded([str(ROOT / 'scripts/pod/pexec_codex'), command], 55, cwd=ROOT)


def marked(output, prefix):
    lines = [s[len(prefix):] for s in output.splitlines() if s.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError('missing or ambiguous transport marker: '+prefix)
    return lines[0]


def download(meta, out, call=remote):
    size, sha = meta['size'], meta['sha256']
    if not isinstance(size, int) or size <= 0 or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('invalid snapshot manifest')
    parts = []
    for start in range(0, size, CHUNK):
        length = min(CHUNK, size-start)
        data = base64.b64decode(marked(call(READ_CODE, meta['archive'], start, length), 'WINDOW_DATA '),
                                validate=True)
        if len(data) != length: raise ValueError('truncated snapshot chunk')
        parts.append(data)
    blob = b''.join(parts)
    if hashlib.sha256(blob).hexdigest() != sha: raise ValueError('snapshot SHA256 mismatch')
    path = out / 'raw.jsonl.part'
    path.write_bytes(gzip.decompress(blob))
    rows = gates.load_raw(path)
    if not rows: raise ValueError('empty raw snapshot')
    path.replace(out / 'raw.jsonl')
    return rows


def is_complete(meta, rows):
    v, s = meta.get('verdict', {}), meta.get('summary', {})
    return (meta['job_state'] in TERMINAL and v.get('status') == 'VALID'
            and v.get('rows') == len(rows) and v.get('raw') == meta['raw']
            and Path(s.get('raw') or '').name == meta['raw'] and s.get('n') == meta['n']
            and 'score' in meta)


def is_drained(meta, rows):
    v, s = meta.get('timed_verdict', {}), meta.get('summary', {})
    return (meta['job_state'] in TERMINAL and v.get('status') == 'DRAINED'
            and v.get('scope') == 'fixed_duration_diagnostic' and v.get('n_completed') == len(rows)
            and v.get('raw') == meta['raw'] and v.get('raw_sha256') == meta.get('raw_sha256')
            and v.get('raw_sha256') is not None and v.get('n') == meta['n']
            and Path(s.get('raw') or '').name == meta['raw'] and s.get('n') == meta['n']
            and 'timed_score' in meta)


def health_alerts(meta):
    h = meta.get('health', {})
    alerts = []
    if h.get('recent_error_lines'): alerts.append('recent server error; inspect preserved lines')
    if meta['job_state'] == 'running' and h.get('phase') == 'measurement':
        if h.get('raw_age_s', h.get('since_flush_s', 0)) > 300:
            alerts.append('no completed record for >300s; check long requests/gaps/engine')
        if h.get('server_log_age_s', 0) > 300:
            alerts.append('server log unchanged for >300s; not proof of engine failure')
    return alerts


def render(meta, rows, out, job, width):
    complete = is_complete(meta, rows)
    drained = is_drained(meta, rows)
    run = gates.windows_for(rows, width, not (complete or drained), gates.score_formal.load_harness())
    run.update(job_state=meta['job_state'], complete_data=complete, raw=meta['raw'])
    if complete or drained:
        score_name = 'score_formal.json' if complete else 'timed_score.json'
        atomic_json(out / score_name, meta['score'] if complete else meta['timed_score'])
        bad = gates.check_score(run['whole'], out / score_name)
        if bad: raise ValueError('whole-run calibration differs: '+str(bad))
        run['closure_basis'] = 'matching_complete_data_verdict' if complete else 'validated_dispatch_census_drained'
    atomic_json(out / 'snapshot.json', meta)
    atomic_json(out / 'window_gates.json', {job: run})
    (out / 'window_gates.txt').write_text(gates.table(run, job)+'\n')
    (out / 'window_gates.html').write_text(gates.render_html([(job, run)], width))
    return gates.summary(run, job) + (' | complete, calibration OK' if complete else
                                    ' | DRAINED DIAGNOSTIC; not full cohort' if drained else
                                    ' | INCOMPLETE, completed requests only')


def notify(state, path):
    if not state.get('notification'): return
    atomic_json(path, state)  # Durable outbox before attempting terminal delivery.
    text = run_bounded([sys.executable, 'scripts/agent_message.py', '--to', 'lead', '--from-agent',
                        'lead', '--text', '[window watcher] '+state['notification']], 25, cwd=ROOT)
    if 'SUBMITTED' not in text: raise ValueError('notification was not submitted')
    state.pop('notification', None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('job')
    ap.add_argument('interval_s', type=float, nargs='?', default=1500)
    ap.add_argument('window_min', type=float, nargs='?', default=25)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--notify', action='store_true')
    args = ap.parse_args(argv)
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', args.job): ap.error('invalid job name')
    if any(not math.isfinite(v) or v <= 0 for v in (args.interval_s, args.window_min)):
        ap.error('interval and window must be finite and positive')
    out = ROOT / 'evidence' / ('L'+args.job) / 'window'
    out.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / 'build' / 'scratch' / 'window-watch' / args.job
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime/'watch.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: ap.error('a watcher already holds this job lock')
    path = runtime / 'watch-state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    while True:
        state.update(pid=os.getpid(), heartbeat=time.time())
        terminal = False
        try:
            if args.notify: notify(state, path)
            due = args.once or time.time() >= state.get('next_report_at', 0)
            meta = json.loads(marked(remote(SNAPSHOT_CODE, args.job, int(due)), 'WINDOW_META '))
            terminal = meta['job_state'] in TERMINAL
            state.update(health='up', job_state=meta['job_state'])
            alerts = health_alerts(meta)
            health = dict(observed_at=meta['observed_at'], job_state=meta['job_state'],
                          **meta.get('health', {}), alerts=alerts)
            atomic_json(out/'health.json', health)
            with (out/'health_history.jsonl').open('a') as f:
                f.write(json.dumps(health, ensure_ascii=False)+'\n')
            if alerts != state.get('alerts', []):
                print(args.job+': HEALTH '+json.dumps(alerts), flush=True)
            if alerts:
                with (out/'alerts.jsonl').open('a') as f:
                    f.write(json.dumps(health, ensure_ascii=False)+'\n')
            state['alerts'] = alerts
            if meta['data']:
                rows = download(meta, out)
                line = render(meta, rows, out, args.job, args.window_min)
                print(line, flush=True)
                state.update(last_report=line, next_report_at=time.time()+args.interval_s)
                if args.notify: state['notification'] = line
            elif terminal:
                line = args.job+': '+meta['job_state']+', no measured raw; no score'
                print(line, flush=True)
                if args.notify: state['notification'] = line
            elif args.once or state.get('last_announced_state') != meta['job_state']:
                print(args.job+': '+meta['job_state']+', waiting for measurement/report', flush=True)
            state['last_announced_state'] = meta['job_state']
            atomic_json(path, state)
            if args.notify: notify(state, path)
        except Exception as e:
            state.update(health='retrying', error=str(e), last_error_at=time.time())
            print(args.job+': monitor retry: '+str(e), flush=True)
            atomic_json(path, state)
            if args.once: return 2
            terminal = False
        atomic_json(path, state)
        if terminal or args.once:
            if terminal: print('WATCH_END '+args.job, flush=True)
            return 0
        time.sleep(60)


if __name__ == '__main__':
    sys.exit(main())
