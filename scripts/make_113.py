#!/usr/bin/env python3
"""Reproduce 113 on a private copy of base_exact + 000,101,105,110,111,112.

Kernel source of truth: scripts/kernels/sm80_indexer_113.py. No GPU or image build.
"""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'build/p113'
PATCHES = ['000-interface-compliance', '101-d1v12-on-base',
           '105-role-split-single-partial', '110-sm80-dsa-indexer',
           '111-sm80-fp8-moe-marlin', '112-sm80-indexer-kernels']
SHIM = 'srt/layers/attention/dsa/sm80_deep_gemm.py'
KERNEL = 'srt/layers/attention/dsa/sm80_indexer_kernels.py'


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / '.gitignore').write_text('*\n')
    before, after = WORK / 'baseline/sglang', WORK / 'candidate/sglang'
    for dest in (before, after):
        if dest.exists():
            shutil.rmtree(dest)
    shutil.copytree(ROOT / 'build/base_exact/sglang', before)
    logs = []
    for patch in PATCHES:
        r = subprocess.run(['patch', '-p3', '--fuzz=0', '--no-backup-if-mismatch', '--batch', '-i',
                            str(ROOT / 'patches' / (patch + '.patch'))],
                           cwd=before, check=True, text=True, capture_output=True)
        logs.append(r.stdout)
    shutil.copytree(before, after)
    (after / KERNEL).write_bytes((ROOT / 'scripts/kernels/sm80_indexer_113.py').read_bytes())
    diff = []
    for rel in (KERNEL,):
        old = (before / rel).read_text().splitlines(True) if (before / rel).exists() else []
        diff.extend(difflib.unified_diff(old, (after / rel).read_text().splitlines(True),
                                       'a/python/sglang/' + rel if old else '/dev/null',
                                       'b/python/sglang/' + rel))
    patch = ROOT / 'patches/113-sm80-prefill-indexer.patch'
    patch.write_text(''.join(diff))
    ev = ROOT / 'evidence/T47/113'
    ev.mkdir(exist_ok=True, parents=True)
    (ev / 'generate_apply.log').write_text(''.join(logs))
    (ev / 'generate_receipt.json').write_text(json.dumps({
        'baseline_patches': PATCHES,
        'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256((after / KERNEL).read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print('Generated', patch.relative_to(ROOT))


if __name__ == '__main__':
    main()
