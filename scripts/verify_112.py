#!/usr/bin/env python3
"""Verify deterministic generation, exact full patch stack, compilation and reversal."""
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import make_112 as gen


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def main():
    ev = ROOT / 'evidence/T47/112'
    before = hashes(ROOT / 'build/base_exact/sglang')
    target = ROOT / 'patches/112-sm80-indexer-kernels.patch'
    expected = target.read_bytes()
    gen.main()
    assert target.read_bytes() == expected
    work = ROOT / 'build/p112/verify/sglang'
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(ROOT / 'build/base_exact/sglang', work)
    patches = gen.PATCHES + ['112-sm80-indexer-kernels', '120-sched-protect-chain', '130-async-tokenize']
    logs = []
    for patch in patches:
        r = subprocess.run(['patch', '-p3', '--fuzz=0', '--no-backup-if-mismatch', '--batch', '-i', str(ROOT / 'patches' / (patch+'.patch'))],
                           cwd=work, text=True, capture_output=True, check=True)
        logs.append(patch + '\n' + r.stdout)
        assert 'fuzz' not in r.stdout.lower()
        if patch == '112-sm80-indexer-kernels':
            assert hashes(work) == hashes(ROOT / 'build/p112/candidate/sglang')
    files = list(work.rglob('*.py')) + [ROOT / 'scripts/make_112.py', ROOT / 'scripts/verify_112.py',
                                      ROOT / 'scripts/test_sm80_indexer_112.py', ROOT / 'scripts/kernels/sm80_indexer_112.py']
    for f in files:
        py_compile.compile(str(f), doraise=True)
    # Reverse the ENTIRE chain to prove the exact immutable source bytes return.
    for patch in reversed(patches):
        r = subprocess.run(['patch', '-R', '-p3', '--fuzz=0', '--no-backup-if-mismatch', '--batch', '-i', str(ROOT / 'patches' / (patch+'.patch'))],
                           cwd=work, text=True, capture_output=True, check=True)
        logs.append('reverse ' + patch + '\n' + r.stdout)
    assert hashes(work) == before
    assert hashes(ROOT / 'build/base_exact/sglang') == before
    (ev / 'full_stack_apply.log').write_text(''.join(logs))
    receipt = dict(status='PASS', patches=patches, py_compile=len(files),
                   deterministic=True, base_unchanged=True, full_reverse_exact=True,
                   applied_equals_candidate=True, base_files=len(before),
                   patch_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (ev / 'stack_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
