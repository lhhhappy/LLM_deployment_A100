#!/usr/bin/env python3
"""Deploy selected commit diffs, runtime and jobs to an idle, paused queue.

Copies only selected diffs, never build/engine/trees. Failed publication leaves
the queue paused. Original run directories and job scripts are never overwritten.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def build(out, specs):
    out.mkdir(parents=True, exist_ok=False)
    jobs, runtime = [], []
    for spec in specs:
        name, source = spec.split('=', 1)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.sh', name):
            raise ValueError('invalid queue name')
        path = ROOT / source
        subprocess.run(['bash', '-n', str(path)], check=True)
        text = path.read_text()
        commits = re.findall(r'^G_COMMIT=([0-9a-f]{40})$', text, re.M)
        if len(commits) != 1 or not re.search(r'^G_EXPECT="[^"\n]+"$', text, re.M):
            raise ValueError('job needs a fixed G_COMMIT and nonempty G_EXPECT')
        if 'G_PATCHES=' in text:
            raise ValueError('legacy patch jobs are no longer supported')
        commit = commits[0]
        subprocess.run([str(ROOT/'scripts/engine/export.sh'), commit], check=True,
                       stdout=subprocess.DEVNULL)
        diff = Path('engine') / (commit+'.diff')
        (out/diff).parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'build'/diff, out/diff)
        target = Path('jobs') / name
        (out/target).parent.mkdir(exist_ok=True)
        shutil.copyfile(path, out/target)
        jobs.append(dict(name=name, source=str(target), commit=commit))
    if not jobs or len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('empty or duplicate job list')
    for p in [Path('scripts/pod/lib.sh'), Path('scripts/pod/jobs/dev_ladder_template.sh')]:
        dest = Path('bin')/p
        (out/dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/p, out/dest)
        runtime.append(str(dest))
    for p in sorted((ROOT/'build/verify_kit').iterdir()):
        if p.is_file() and p.name != 'SHA256SUMS':
            dest = Path('verify_kit')/p.name
            (out/dest).parent.mkdir(exist_ok=True)
            shutil.copyfile(p, out/dest)
            runtime.append(str(dest))
    shutil.copyfile(__file__, out/'queue_bundle.py')
    manifest = dict(workflow='engine_commit_v1', jobs=jobs, runtime=runtime)
    (out/'bundle.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def atomic_copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(source.read_bytes())
    tmp.chmod(0o644)
    tmp.replace(dest)


def publish(stage, ax):
    manifest = json.loads((stage/'bundle.json').read_text())
    assert manifest['workflow'] == 'engine_commit_v1'
    queue = ax/'queue'
    assert (queue/'PAUSE').exists(), 'pause the queue before deploying runtime'
    assert not list((queue/'running').glob('*.sh')), 'running job: runtime must remain unchanged'
    for state in ('pending', 'running', 'done', 'failed', 'cancelled'):
        (queue/state).mkdir(exist_ok=True)
    payloads = [(stage/r, ax/r) for r in manifest['runtime']]
    for commit in sorted({j['commit'] for j in manifest['jobs']}):
        assert re.fullmatch('[0-9a-f]{40}', commit)
        r = Path('engine')/(commit+'.diff')
        payloads.append((stage/r, ax/r))
    for src, _ in payloads:
        assert src.is_file(), str(src)
    names = {j['name'] for j in manifest['jobs']}
    pending = {p.name for p in (queue/'pending').glob('*.sh')}
    assert pending <= names, f'unexpected pending jobs: {pending-names}'
    for job in manifest['jobs']:
        src = stage/job['source']
        assert src.is_file()
        existing = [queue/s/job['name'] for s in ('pending','running','done','failed','cancelled')
                    if (queue/s/job['name']).exists()]
        assert len(existing) <= 1
        if existing:
            assert existing[0].parent.name == 'pending', 'job name already used; use a fresh run name'
            assert existing[0].read_bytes() == src.read_bytes(), 'existing job differs'
        else:
            run = ax/'runs'/job['name'][:-3]
            assert not run.exists(), f'existing run directory: {run}'
    for src, dest in payloads:
        atomic_copy(src, dest)
    receipt = dict(manifest, deployed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (ax/'runtime-deployment.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for job in manifest['jobs']:
        name = job['name']
        if not (queue/'pending'/name).exists():
            run = ax/'runs'/name[:-3]
            run.mkdir(parents=True)
            (run/'deployment-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
            os.link(stage/job['source'], queue/'pending'/name)
        print('QUEUED', name, 'engine_commit='+job['commit'], flush=True)
    print('RUNTIME_DEPLOYED engine_commit_v1; queue remains paused for review', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build','publish'])
    p.add_argument('stage', type=Path)
    p.add_argument('jobs', nargs='*')
    p.add_argument('--ax', type=Path, default=Path('/tmp/ax'))
    a = p.parse_args()
    if a.action == 'build': build(a.stage, a.jobs)
    else: publish(a.stage, a.ax)


if __name__ == '__main__': main()
