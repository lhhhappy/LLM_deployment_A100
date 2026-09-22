#!/usr/bin/env python3
"""T41 CPU validation: real -p3/fuzz=0 application, all-file comparison, py_compile.
Run after scripts/make_120.py. Uses only owned build/p120/verified output.
"""
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'build/p120'
OUT = ROOT / 'evidence/T41'


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def main():
    base = ROOT / 'build/base_exact/sglang'
    before = hashes(base)
    baseline = WORK / 'baseline/sglang'
    candidate = WORK / 'candidate/sglang'
    verified = WORK / 'verified/sglang'
    if verified.parent.exists():
        shutil.rmtree(verified.parent)
    shutil.copytree(baseline, verified)
    proc = subprocess.run(['patch', '--batch', '--fuzz=0', '-p3', '-i',
                           str(ROOT / 'patches/120-sched-protect-chain.patch')],
                          cwd=verified, text=True, capture_output=True)
    (OUT / 'patch_apply.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError('120 apply failed')
    expected, actual = hashes(candidate), hashes(verified)
    assert expected == actual, 'Applied patch differs from generated candidate'
    base_patched = hashes(baseline)
    changed = [p for p, digest in actual.items() if base_patched.get(p) != digest]
    assert sorted(changed) == ['srt/managers/schedule_policy.py', 'srt/managers/scheduler.py'], changed
    files = sorted(verified.rglob('*.py'))
    failures = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(str(exc))
    tools = [ROOT / 'scripts/make_120.py', Path(__file__), ROOT / 'tests/test_sched_protect_chain.py']
    for path in tools:
        py_compile.compile(str(path), doraise=True)
    assert before == hashes(base), 'READ-ONLY BASE CHANGED'
    receipt = dict(python=sys.version, patch_application='PASS',
                   candidate_matches_patch=True, changed_files=changed,
                   compared_files=len(actual), py_compile_files=len(files),
                   tool_compile_files=len(tools), failures=failures,
                   base_unchanged_during_verification=True,
                   changed_sha256={p: actual[p] for p in changed})
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
