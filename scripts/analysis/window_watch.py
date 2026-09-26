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
            # Before the first 25-completion checkpoint, establish measurement
            # identity from a complete row dispatched after this successful flush.
            # Never use a preflight raw or infer that no checkpoint means no progress.
            candidates = []
            for raw in level.glob('raw_*.jsonl'):
                if raw.stat().st_mtime < f['flush_finished_s']: continue
                with raw.open() as handle: line = handle.readline()
                if not line.endswith('\n'): continue
                try: first = json.loads(line)
                except ValueError: continue
                if first.get('client_dispatch_at_s', 0) >= f['flush_finished_s']:
                    candidates.append(raw)
            if len(candidates) == 1:
                h['raw_age_s'] = round(time.time()-candidates[0].stat().st_mtime, 1)
                stamps = []
                with candidates[0].open() as handle:
                    for line in handle:
                        if not line.endswith('\n'): break
                        row = json.loads(line)
                        stamps.append(row['client_dispatch_at_s'])
                if stamps:
                    h['first_observed_dispatch_s'] = min(stamps)
                    h['completed_rows'] = len(stamps)
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
    argv = [str(ROOT / 'scripts/pod/pexec_codex'), command]
    for attempt in range(3):
        try:
            return run_bounded(argv, 55, cwd=ROOT)
        except RuntimeError as exc:
            # The Pod exec transport occasionally scans a process which exits
            # before it reads /proc/<pid>/stat. Retry that transport race only.
            message = str(exc)
            if attempt == 2 or 'FileNotFoundError' not in message or '/proc/' not in message:
                raise
            time.sleep(1)


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
    meta['downloaded_raw_sha256'] = hashlib.sha256((out/'raw.jsonl').read_bytes()).hexdigest()
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


def next_check(t0, first_delay, interval, last_deadline=None, report_offsets=None):
    """Anchor reports to dispatch time, never to download duration."""
    if report_offsets:
        if last_deadline is None:
            return t0 + report_offsets[0]
        elapsed = last_deadline - t0
        for offset in report_offsets:
            if offset > elapsed + 1e-6:
                return t0 + offset
        return t0 + report_offsets[-1] + (
            math.floor((elapsed - report_offsets[-1]) / interval) + 1
        ) * interval
    first = t0 + first_delay
    if last_deadline is None: return first
    return first + (max(0, round((last_deadline-first)/interval)) + 1) * interval


def compact_status(state, health, now):
    """A bounded view; progress ticks alone do not count as a new diagnostic."""
    age = max(0, now-state.get('heartbeat', 0))
    stale = age > 180 and state.get('job_state') not in TERMINAL
    monitor = 'stale' if stale else state.get('health', 'unknown')
    view = dict(job_state=state.get('job_state', 'unknown'), monitor=monitor,
                phase=health.get('phase', 'unknown'),
                completed=health.get('completed_rows', health.get('completed_at_checkpoint')),
                alerts=health.get('alerts', []),
                next_report_utc=time.strftime('%H:%M:%S', time.gmtime(state['next_report_at']))
                    if state.get('next_report_at') else None,
                last_report=state.get('last_report'),
                error=str(state.get('error', ''))[:400] if monitor != 'up' else None)
    view['auto_analysis'] = state.get('analysis_brief')
    view['analysis_error'] = state.get('analysis_error')
    # Keep the latest counts available without flooding each poll with unchanged
    # diagnosis. A stale watcher, recovery or new alert always breaks deduplication.
    key = {k: view[k] for k in ('job_state', 'monitor', 'phase', 'alerts', 'last_report', 'error')}
    key['report_deadline'] = state.get('last_scheduled_deadline')
    key['auto_analysis'] = view['auto_analysis']
    key['analysis_error'] = view['analysis_error']
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return view, digest


def show_status(out, runtime, job, changes_only):
    def read(path):
        return json.loads(path.read_text()) if path.is_file() else {}
    view, digest = compact_status(read(runtime/'watch-state.json'), read(out/'health.json'), time.time())
    cursor = runtime/'status-cursor.json'
    if changes_only and read(cursor).get('digest') == digest:
        print(job+': unchanged; latest counts in '+str(out/'health.json'))
    else:
        report = read(out/'window_gates.json').get(job, {})
        if report:
            whole = report['whole']
            view.pop('last_report', None)
            snapshot_at = read(out/'snapshot.json').get('observed_at')
            view['diagnostic'] = dict(rows=report['n_rows'], complete=report.get('complete_data', False),
                snapshot_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(snapshot_at)) if snapshot_at else None,
                ttft={k: dict(n=v['n'], p95=v['p95']) for k, v in whole['gates'].items()},
                tpot=whole['tpot'], errors=whole['errors'], cache=whole['tokens'])
        def rounded(value):
            if isinstance(value, float): return round(value, 4)
            if isinstance(value, dict): return {k: rounded(v) for k, v in value.items()}
            if isinstance(value, list): return [rounded(v) for v in value]
            return value
        print(job+': '+json.dumps(rounded(view), ensure_ascii=False))
        print('evidence: '+str(out))
        if changes_only:
            runtime.mkdir(parents=True, exist_ok=True)
            atomic_json(cursor, dict(digest=digest))
    return 2 if view['monitor'] in ('stale', 'unknown', 'retrying') else 0


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
    ap.add_argument('--status', action='store_true', help='read cached status only; no pod request')
    ap.add_argument('--changes-only', action='store_true', help='with --status, suppress repeated diagnosis')
    ap.add_argument('--first-report-s', type=float,
                    help='first report after this many measured seconds, then interval_s')
    ap.add_argument('--report-at-minutes',
                    help='comma-separated measured-minute offsets, then repeat interval_s after the last')
    ap.add_argument('--baseline-raw', type=Path, help='validated full baseline raw for automatic same-ID reports')
    ap.add_argument('--baseline-label', default='baseline')
    ap.add_argument('--alignment-trace', action='store_true', help='summarize existing bounded alignment log per report')
    ap.add_argument('--opening-analysis', action='store_true', help='automatically analyze and preserve a drained opening probe')
    ap.add_argument('--opening-reference-job', action='append', help='completed opening probe for paired comparison; repeat for multiple baselines')
    args = ap.parse_args(argv)
    if args.changes_only and not args.status: ap.error('changes-only requires status')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', args.job): ap.error('invalid job name')
    if any(not math.isfinite(v) or v <= 0 for v in (args.interval_s, args.window_min)):
        ap.error('interval and window must be finite and positive')
    if args.first_report_s is not None and (not math.isfinite(args.first_report_s) or args.first_report_s <= 0):
        ap.error('first-report-s must be finite and positive')
    report_offsets = None
    if args.report_at_minutes:
        try:
            report_offsets = tuple(float(v) * 60 for v in args.report_at_minutes.split(','))
        except ValueError:
            ap.error('report offsets must be comma-separated numbers of minutes')
        if (not report_offsets or any(not math.isfinite(v) or v <= 0 for v in report_offsets)
                or any(b <= a for a, b in zip(report_offsets, report_offsets[1:]))):
            ap.error('report offsets must be positive and strictly increasing')
        if args.first_report_s is not None:
            ap.error('use either --report-at-minutes or --first-report-s')
    out = ROOT / 'evidence' / ('L'+args.job) / 'window'
    runtime = ROOT / 'build' / 'scratch' / 'window-watch' / args.job
    if args.status: return show_status(out, runtime, args.job, args.changes_only)
    out.mkdir(parents=True, exist_ok=True)
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
            scheduled = report_offsets is not None or args.first_report_s is not None
            due = args.once or time.time() >= state.get('next_report_at', float('inf') if scheduled else 0)
            meta = json.loads(marked(remote(SNAPSHOT_CODE, args.job, int(due)), 'WINDOW_META '))
            terminal = meta['job_state'] in TERMINAL
            state.update(health='up', job_state=meta['job_state'])
            state.pop('error', None)
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
            anchor = meta.get('health', {}).get('first_observed_dispatch_s')
            if scheduled and anchor is not None:
                state['measurement_anchor_s'] = anchor
                state['next_report_at'] = next_check(anchor, args.first_report_s, args.interval_s,
                                                     state.get('last_scheduled_deadline'), report_offsets)
                due = args.once or time.time() >= state['next_report_at']
                if due and not meta['data']:
                    meta = json.loads(marked(remote(SNAPSHOT_CODE, args.job, 1), 'WINDOW_META '))
                    terminal = meta['job_state'] in TERMINAL
            if meta['data']:
                rows = download(meta, out)
                line = render(meta, rows, out, args.job, args.window_min)
                print(line, flush=True)
                state['last_report'] = line
                if scheduled and anchor is not None:
                    state['last_scheduled_deadline'] = state['next_report_at']
                    state['next_report_at'] = next_check(anchor, args.first_report_s, args.interval_s,
                                                         state['last_scheduled_deadline'], report_offsets)
                else:
                    state['next_report_at'] = time.time()+args.interval_s
                if args.notify: state['notification'] = line
            elif terminal:
                line = args.job+': '+meta['job_state']+', no measured raw; no score'
                print(line, flush=True)
                if args.notify: state['notification'] = line
            elif args.once or state.get('last_announced_state') != meta['job_state']:
                print(args.job+': '+meta['job_state']+', waiting for measurement/report', flush=True)
            state['last_announced_state'] = meta['job_state']
            if args.baseline_raw and (out/'snapshot.json').is_file():
                # Also catches up a saved window on monitor restart. Failures are
                # retried without stopping health polling or the inference job.
                saved = json.loads((out/'snapshot.json').read_text())
                analysis_key = [saved['sha256'], str(args.baseline_raw), args.baseline_label, args.alignment_trace]
                if state.get('analysis_key') != analysis_key or state.get('analysis_error'):
                    try:
                        from window_report import build_report
                        result = build_report(out, args.baseline_raw, args.baseline_label,
                                              args.job, remote, args.alignment_trace)
                        state.update(analysis_brief=result['brief'], analysis_path=result['path'])
                        if result.get('alignment_error'):
                            state['analysis_error'] = result['alignment_error']
                        else:
                            state['analysis_key'] = analysis_key
                            state.pop('analysis_error', None)
                    except Exception as exc:
                        state.pop('analysis_brief', None)
                        state['analysis_error'] = str(exc)[:400]
            if args.opening_analysis and terminal and meta.get('data') and is_drained(meta, rows):
                try:
                    from opening_report import build_report
                    result = build_report(out, args.job, meta, remote, args.opening_reference_job)
                    state.update(analysis_brief=result['brief'], analysis_path=result['path'],
                                 analysis_kind='opening', analysis_key=meta['sha256'])
                    state.pop('analysis_error', None)
                except Exception as exc:
                    state['analysis_error'] = 'opening analysis: '+str(exc)[:350]
                    terminal = False  # Retry the read-only analysis; never rerun the GPU job.
            atomic_json(path, state)
            if args.notify: notify(state, path)
        except Exception as e:
            error = str(e)[:400]
            changed = state.get('health') != 'retrying' or state.get('error') != error
            if changed or time.time()-state.get('last_error_print_at', 0) >= 600:
                print(args.job+': monitor retry: '+error, flush=True)
                state['last_error_print_at'] = time.time()
            state.update(health='retrying', error=error, last_error_at=time.time())
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
