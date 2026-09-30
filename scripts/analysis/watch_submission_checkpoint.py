#!/usr/bin/env python3
"""Notify the active session at the agreed 20-minute measurement checkpoint.

Read-only: never creates an official attempt or changes the queue. The session
reviews live errors and the frozen configuration before the authorized submission.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

from window_watch import ROOT, atomic_json, marked, remote, run_bounded

CODE = r'''
import json, pathlib, sys, time
root=pathlib.Path('/tmp/ax/runs')/sys.argv[1]
states=[s for s in ('running','pending','done','failed','cancelled')
        if (pathlib.Path('/tmp/ax/queue')/s/(sys.argv[1]+'.sh')).exists()]
if len(states)>1: raise ValueError('ambiguous queue state')
m={'job':sys.argv[1], 'state':states[0] if states else 'unknown', 'observed_at_s':time.time()}
def emit(): print('CHECKPOINT '+json.dumps(m));sys.exit(0)
if m['state'] in ('pending','unknown'):emit()
level=root/'N30'
p=level/'flush_evidence.json'
if not p.exists():emit()
flush=json.loads(p.read_text())
m['flush_success']=flush.get('flush_success') is True
if not m['flush_success']:emit()
m['mechanisms_ok']='MECHANISMS OK' in (root/'job.log').read_text()
p=level/'dispatch_ledger.jsonl'
if not p.exists():emit()
def rows(path):
    b=path.read_bytes()
    if b and not b.endswith(b'\n'): b=b[:b.rfind(b'\n')+1]
    return [json.loads(s) for s in b.splitlines() if s.strip()]
events=rows(p)
starts=[e for e in events if e['event']=='window_started']
if len(starts)!=1: raise ValueError('missing/duplicate measurement start')
start=starts[0]['at_s']
if start < flush['flush_finished_s']:raise ValueError('measurement preceded flush')
m.update(first_dispatch_at_s=start, elapsed_s=time.time()-start)
if m['elapsed_s']<float(sys.argv[2]):emit()
paths=list(level.glob('raw_*.jsonl'))
if len(paths)!=1:raise ValueError('ambiguous measurement raw')
completed=rows(paths[0])
ids=[r['req_id'] for r in completed]
if len(ids)!=len(set(ids)):raise ValueError('duplicate raw request')
sent={e['req_id'] for e in events if e['event']=='dispatch'}
# Ledger and raw are read at slightly different instants; counts are observations.
m.update(n_dispatched=len(sent), n_completed=len(ids), observed_errors=sum(bool(r.get('error')) for r in completed),
         completed_requests_only=True, raw=paths[0].name)
emit()
'''


def decision(meta, seconds):
    if meta.get('state') in ('failed','cancelled'):
        return 'STOPPED: inspect the job before any submission'
    if not meta.get('flush_success') or not meta.get('mechanisms_ok'):
        return 'STOPPED: measurement/checks missing' if meta.get('state')=='done' else None
    if meta.get('elapsed_s', 0)<seconds or 'observed_errors' not in meta:
        return None
    if meta['observed_errors']:
        return 'ERRORS_OBSERVED: inspect request errors before submission'
    if meta.get('n_completed',0)==0:
        return 'NO_COMPLETIONS: inspect progress before submission'
    return 'REVIEW_SUBMISSION: measurement >=20min; completed requests show no errors; inspect engine logs/config'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('job')
    ap.add_argument('--seconds',type=int,default=1200)
    a=ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',a.job) or a.seconds<=0:ap.error('invalid job/duration')
    out=ROOT/'build/scratch/submission-checkpoint'/a.job
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'watch.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return 0
    state_file=out/'state.json'
    state=json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get('delivered'):return 0
    while True:
        try:
            meta=json.loads(marked(remote(CODE,a.job,a.seconds),'CHECKPOINT '))
            state.update(pid=os.getpid(),meta=meta)
            status=decision(meta,a.seconds)
            if status:
                evidence=ROOT/'evidence'/('L'+a.job)/'submission-checkpoint.json'
                evidence.parent.mkdir(parents=True,exist_ok=True)
                atomic_json(evidence,dict(meta,checkpoint_status=status,not_a_full_cohort_verdict=True))
                message='[submission checkpoint] '+a.job+': '+status+'; '+str(evidence.relative_to(ROOT))
                result=run_bounded([sys.executable,'scripts/agent_message.py','--to','lead','--from-agent','lead','--text',message],25,cwd=ROOT)
                if 'SUBMITTED' not in result:raise ValueError('notification not delivered')
                state['delivered']=True
                atomic_json(state_file,state)
                print(message,flush=True)
                return 0
            state['health']='waiting'
        except Exception as exc:
            state.update(health='retrying',error=str(exc))
        state['heartbeat']=time.time()
        atomic_json(state_file,state)
        time.sleep(45)


if __name__=='__main__':sys.exit(main())
