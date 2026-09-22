#!/usr/bin/env python3
"""Verify deterministic generation, exact full patch stack, compilation and reversal."""
import ast
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import make_113 as gen


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def main():
    ev = ROOT / 'evidence/T47/113'
    source112=(ROOT / 'scripts/kernels/sm80_indexer_112.py').read_text()
    source113=(ROOT / 'scripts/kernels/sm80_indexer_113.py').read_text()
    def defs(src):
        return {node.name: ast.get_source_segment(src,node) for node in ast.parse(src).body
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    fixed = {'H', 'D', 'PAGE', 'BQ', 'BK', 'HH', 'DD', 'CLEAN', 'GROUP', 'LOOP', 'BLOCK'}
    for source in (source112, source113):
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in ('_paged', '_ragged', '_prefill', '_unpack_prefill'):
                for arg in node.args.args:
                    is_const = arg.annotation is not None and ast.unparse(arg.annotation) == 'tl.constexpr'
                    assert is_const == (arg.arg in fixed), (node.name, arg.arg)
    d112,d113=defs(source112),defs(source113)
    for name in ('_e4m3_to_bf16','_paged','_ragged','fp8_paged_mqa_logits'):
        assert d112[name] == d113[name], name
    assert d112['fp8_mqa_logits'] == d113['_fp8_mqa_logits_112'].replace('_fp8_mqa_logits_112(', 'fp8_mqa_logits(',1)
    before = hashes(ROOT / 'build/base_exact/sglang')
    target = ROOT / 'patches/113-sm80-prefill-indexer.patch'
    expected = target.read_bytes()
    gen.main()
    assert target.read_bytes() == expected
    work = ROOT / 'build/p113/verify/sglang'
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(ROOT / 'build/base_exact/sglang', work)
    patches = gen.PATCHES + ['113-sm80-prefill-indexer', '140-kda-dual-snapshot', '120-sched-protect-chain', '130-async-tokenize', '150-startup-warmup']
    logs = []
    for patch in patches:
        r = subprocess.run(['patch', '-p3', '--fuzz=0', '--no-backup-if-mismatch', '--batch', '-i', str(ROOT / 'patches' / (patch+'.patch'))],
                           cwd=work, text=True, capture_output=True, check=True)
        logs.append(patch + '\n' + r.stdout)
        assert 'fuzz' not in r.stdout.lower()
        if patch == '113-sm80-prefill-indexer':
            assert hashes(work) == hashes(ROOT / 'build/p113/candidate/sglang')
    files = list(work.rglob('*.py')) + [ROOT / 'scripts/make_113.py', ROOT / 'scripts/verify_113.py',
                                      ROOT / 'scripts/test_sm80_indexer_113.py', ROOT / 'scripts/kernels/sm80_indexer_113.py',
                                      ROOT / 'scripts/bench_sm80_indexer_47.py', ROOT / 'scripts/test_sm80_indexer_cache_47.py']
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
                   decode_source_byte_exact=True, fallback_source_byte_exact=True,
                   runtime_shapes_and_strides=True,
                   applied_equals_candidate=True, base_files=len(before),
                   patch_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (ev / 'stack_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
