#!/usr/bin/env python3
"""Bridge GPU watcher events into the registered current Codex terminal.

Poll cached files only. No pod requests, engine changes, or model calls while idle.
Terminal submission is at-least-once; message receipt must be acknowledged by Codex.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

from window_watch import ROOT, TERMINAL, atomic_json, compact_status, run_bounded


def event_key(state, health, now):
    return compact_status(state, health, now)[1]


def source_sample(state, data, error, now):
    """Debounce transport only; diagnostic and engine health events remain immediate."""
    if error is None:
        state.update(source_health='up', source_failures=0)
        state.pop('source_error', None)
        return data['state'], data['health']
    state.update(source_health='retrying', source_failures=state.get('source_failures', 0)+1,
                 source_error=str(error)[:300])
    if state['source_failures'] < 3: return None
    # A stable event key prevents varying SSH error strings from generating new
    # model turns throughout one outage. Exact latest error stays in state.json.
    return (dict(health='retrying', heartbeat=now,
                 error='GPU watcher cache unavailable after 3+ consecutive reads; inspect bridge state.json'),
            dict(alerts=['notification bridge cannot read GPU watcher']))


def message(job, key, state, health, now):
    view, _ = compact_status(state, health, now)
    brief = state.get('last_report', '暂无诊断快照')
    alerts = '; '.join(view['alerts'])[:300]
    return (f'[N30 watcher event {key[:12]}] {job}: '
            f"状态={view['job_state']} 监控={view['monitor']} 完成={view['completed']}; "
            f'{brief}; alerts={alerts}; error={view["error"]}. '
            '程序触发检查：先读 notes/iterations/codex.md 和缓存摘要，按需取证。'
            '完成简短分析、更新日志后结束本轮，等待下一事件；不因局部FAIL自动停任务。')[:1600]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('job')
    ap.add_argument('--notify-test', action='store_true')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', args.job): ap.error('invalid job')
    runtime = ROOT/'build/scratch/window-notify'/args.job
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime/'watch.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: ap.error('notification bridge already running')
    path = runtime/'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    if args.notify_test:
        state['pending'] = dict(key='test-'+str(int(time.time())), text=
            '[N30 watcher 通路测试] 这是后台程序发给当前Codex会话的消息。收到后请确认送达并结束本轮；'
            '后台已负责普通轮询，后续仅在15/45/75分钟检查点、状态或异常变化时发消息。')
    code = ('import json,pathlib; r=pathlib.Path("/sjtu/linhang/arena/repo"); '
            'j='+repr(args.job)+'; '
            's=json.loads((r/"build/scratch/window-watch"/j/"watch-state.json").read_text()); '
            'h=json.loads((r/"evidence"/("L"+j)/"window/health.json").read_text()); '
            'print(json.dumps({"state":s,"health":h}))')
    command = shlex.join(['python3', '-c', code])

    def deliver():
        if not state.get('pending'): return
        atomic_json(path, state)
        item = state['pending']
        result = run_bounded([sys.executable, str(ROOT/'scripts/agent_message.py'),
            '--to', 'codex', '--from-agent', 'codex', '--text', item['text']], 25, cwd=ROOT)
        if 'SUBMITTED destination=codex' not in result: raise ValueError('no delivery receipt')
        if not item['key'].startswith('test-'):
            state['seen_key'] = item['key']
        state['last_submitted_at'] = time.time()
        state.pop('pending')
        atomic_json(path, state)
        print('SUBMITTED '+item['key'][:12], flush=True)

    while True:
        state.update(pid=os.getpid(), heartbeat=time.time())
        terminal = False
        try:
            deliver()
            try:
                data = json.loads(run_bounded([str(ROOT/'scripts/gssh'), command], 80, cwd=ROOT))
                sample = source_sample(state, data, None, time.time())
            except Exception as exc:
                sample = source_sample(state, None, exc, time.time())
            if sample is not None:
                s, h = sample
                key = event_key(s, h, time.time())
                terminal = s.get('job_state') in TERMINAL and s.get('health') == 'up'
                if 'seen_key' not in state and s.get('health') == 'up':
                    state['seen_key'] = key  # Existing diagnostic is the baseline.
                elif key != state.get('seen_key'):
                    state['pending'] = dict(key=key, text=message(args.job, key, s, h, time.time()))
                    deliver()
                if s.get('next_report_at'): state['next_report_at'] = s['next_report_at']
            state.update(health='up')
            state.pop('error', None)
        except Exception as exc:
            error = str(exc)[:300]
            if state.get('error') != error: print('delivery pending: '+error, flush=True)
            state.update(health='retrying', error=error)
            terminal = False
        atomic_json(path, state)
        if terminal or args.once: return 0 if state['health']=='up' else 2
        time.sleep(60)


if __name__ == '__main__':
    sys.exit(main())
