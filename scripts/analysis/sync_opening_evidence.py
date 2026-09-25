#!/usr/bin/env python3
"""Copy completed opening reports from the development host; never access the Pod."""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = '/sjtu/linhang/arena/repo'
REMOTE_ARCHIVE = r'''
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path('/sjtu/linhang/arena/repo'); job=sys.argv[1]
s=json.loads((root/'build/scratch/window-watch'/job/'watch-state.json').read_text())
if not (s.get('job_state') in ('done','failed') and s.get('health')=='up'
        and s.get('analysis_kind')=='opening' and not s.get('analysis_error')):
    sys.exit(75)
base=root/'evidence'/('L'+job)
paths=sorted(p for p in (base/'opening').rglob('*') if p.is_file())
paths += [base/'window'/name for name in ('raw.jsonl','snapshot.json')]
files={}
for p in paths:
    assert not p.is_symlink() and p.resolve().is_relative_to(base.resolve())
    files[str(p.relative_to(root))]=p.read_bytes()
assert any(name.endswith('/comparison.json') for name in files)
manifest={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for name,data in files.items():
        item=tarfile.TarInfo(name); item.size=len(data); archive.addfile(item,io.BytesIO(data))
    data=json.dumps(dict(job=job,files_sha256=manifest,analysis_key=s.get('analysis_key'))).encode()
    item=tarfile.TarInfo('transfer-manifest.json'); item.size=len(data)
    archive.addfile(item,io.BytesIO(data))
'''


def sync(job):
    command=shlex.join(['python3','-c',REMOTE_ARCHIVE,job])
    proc=subprocess.run(['ssh','-n','-o','ConnectTimeout=20','-o','BatchMode=yes',
                         'GPU',command],capture_output=True,timeout=90)
    if proc.returncode==75:
        return False
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors='replace')[-500:])
    with tarfile.open(fileobj=io.BytesIO(proc.stdout),mode='r:gz') as archive:
        members=archive.getmembers()
        assert all(m.isfile() for m in members)
        assert len({m.name for m in members})==len(members)
        payload={m.name:archive.extractfile(m).read() for m in members}
    manifest=json.loads(payload.pop('transfer-manifest.json'))
    assert manifest['job']==job and set(payload)==set(manifest['files_sha256'])
    prefix=PurePosixPath('evidence')/('L'+job)
    for name,data in payload.items():
        relative=PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.is_relative_to(prefix)
        assert hashlib.sha256(data).hexdigest()==manifest['files_sha256'][name]
        target=ROOT/name
        if target.exists():
            assert target.read_bytes()==data, f'local file differs; preserve for review: {name}'
    for name,data in payload.items():
        target=ROOT/name; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temp=target.with_name(target.name+'.sync-tmp')
            temp.write_bytes(data); temp.replace(target)
    manifest['copied_at_s']=time.time()
    (ROOT/prefix/'opening/local-sync.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'SYNCED {job}: {len(payload)} files verified',flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs',nargs='+')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--notify',choices=['claude','codex'])
    args=parser.parse_args()
    assert all(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',j) for j in args.jobs)
    runtime=ROOT/'build/scratch/opening-evidence-sync';runtime.mkdir(parents=True,exist_ok=True)
    locks=[]
    for job in sorted(set(args.jobs)):
        handle=(runtime/(job+'.lock')).open('a')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    pending=dict.fromkeys(args.jobs,False);deadline=time.monotonic()+7200
    while pending:
        for job in list(pending):
            try:
                if not pending[job]:
                    if not sync(job):continue
                    pending[job]=True
                if args.notify:
                    message=f'{job}完整opening分析及window/raw.jsonl已同步本地evidence/L{job}/；逐文件SHA256验证，收据opening/local-sync.json，含所有基线comparison与paired.csv。'
                    subprocess.run([sys.executable,str(ROOT/'scripts/agent_message.py'),
                                    '--to',args.notify,'--from-agent','codex','--text',message],
                                   check=True,timeout=30)
                del pending[job]
            except Exception as exc:
                print(f'RETRY {job}: {exc}',flush=True)
        (runtime/'state.json').write_text(json.dumps(dict(at_s=time.time(),pending=pending))+'\n')
        if not pending:return 0
        if not args.wait or time.monotonic()>=deadline:return 2
        time.sleep(60)


if __name__=='__main__':
    sys.exit(main())
