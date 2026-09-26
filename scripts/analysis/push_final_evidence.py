#!/usr/bin/env python3
"""Sync a drained replay's final raw and analysis to the GPU development host.

The local window watcher owns the authoritative downloaded snapshot. This
helper waits for its closed-data receipt, transfers only that immutable result,
and verifies every file's SHA256 on the development host before recording a
local receipt. It does not access or change the inference Pod.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = '/sjtu/linhang/arena/repo'
TERMINAL = {'done', 'failed'}


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def ready_files(job, opening):
    base = ROOT / 'evidence' / ('L' + job)
    state = read_json(ROOT / 'build/scratch/window-watch' / job / 'watch-state.json')
    snap = read_json(base / 'window/snapshot.json')
    raw = base / 'window/raw.jsonl'
    if not state or not snap or not raw.is_file():
        return None
    if state.get('job_state') not in TERMINAL or state.get('health') != 'up':
        return None
    if snap.get('job_state') not in TERMINAL:
        return None
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if snap.get('downloaded_raw_sha256') != digest:
        raise ValueError(f'{job}: downloaded raw hash mismatch')
    if opening:
        if state.get('analysis_kind') != 'opening' or state.get('analysis_error'):
            return None
        verdict = snap.get('timed_verdict') or {}
        if (verdict.get('status') != 'DRAINED' or verdict.get('raw_sha256') != digest
                or verdict.get('n_completed') != sum(1 for _ in raw.open('rb'))):
            raise ValueError(f'{job}: opening raw does not match drained verdict')
        analysis = base / 'opening/analysis.json'
        if not analysis.is_file():
            return None
        files = sorted(p for p in (base / 'opening').rglob('*')
                       if p.is_file() and p.name != 'devhost-sync.json')
    else:
        verdict = snap.get('verdict') or {}
        if (verdict.get('status') != 'VALID' or verdict.get('rows') != sum(1 for _ in raw.open('rb'))):
            return None
        files = []
    files += [raw, base / 'window/snapshot.json']
    if any(p.is_symlink() or not p.resolve().is_relative_to(base.resolve()) for p in files):
        raise ValueError(f'{job}: unsafe evidence path')
    return files


def sync(job, opening):
    files = ready_files(job, opening)
    if files is None:
        return False
    base = ROOT / 'evidence' / ('L' + job)
    receipt = base / ('opening/devhost-sync.json' if opening else 'window/devhost-sync.json')
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if read_json(receipt) == manifest:
        return True
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    transferred = subprocess.run(
        [str(ROOT / 'scripts/gssh'), f'tar xzf - -C {REMOTE_ROOT}'],
        input=archive.getvalue(), capture_output=True, timeout=660)
    if transferred.returncode:
        raise RuntimeError(f'{job}: transfer failed: {transferred.stderr.decode(errors="replace")[-500:]}')
    code = '''
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.loads(sys.argv[2])
for name,want in manifest.items():
    p=root/name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==want, name
print('SYNC_OK',len(manifest),flush=True)
'''
    command = shlex.join(['python3', '-c', code, REMOTE_ROOT, json.dumps(manifest)])
    checked = subprocess.run([str(ROOT / 'scripts/gssh'), command],
                             capture_output=True, text=True, timeout=660)
    if checked.returncode or 'SYNC_OK ' not in checked.stdout:
        raise RuntimeError(f'{job}: remote hash check failed: {checked.stderr[-500:]}')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'SYNCED {job}: {len(files)} final files, SHA256 verified on development host', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs', nargs='+')
    parser.add_argument('--full-job', action='append', default=[], help='full-cohort job without opening report')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', j) for j in args.jobs):
        parser.error('invalid job name')
    if not set(args.full_job) <= set(args.jobs):
        parser.error('--full-job must name a listed job')
    pending = list(dict.fromkeys(args.jobs))
    while pending:
        for job in pending[:]:
            try:
                if sync(job, job not in args.full_job):
                    pending.remove(job)
            except Exception as exc:
                print(f'RETRY {job}: {exc}', flush=True)
        if not pending:
            return 0
        if not args.wait:
            return 2
        time.sleep(60)


if __name__ == '__main__':
    sys.exit(main())
