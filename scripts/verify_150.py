#!/usr/bin/env python3
"""Verify 11-patch fuzz=0 stack, deterministic generation, syntax and byte-exact reversal."""
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import make_150 as gen

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/T46'


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def main():
    base=ROOT/'build/base_exact/sglang'; original=hashes(base)
    target=ROOT/'patches/150-startup-warmup.patch'; expected=target.read_bytes()
    gen.main(); assert target.read_bytes()==expected
    work=ROOT/'build/p150/verify/sglang'
    if work.exists(): shutil.rmtree(work)
    shutil.copytree(base,work)
    stack=gen.PATCHES+['150-startup-warmup']; logs=[]
    for patch in stack:
        r=subprocess.run(['patch','-p3','--fuzz=0','--no-backup-if-mismatch','--batch','-i',str(ROOT/'patches'/(patch+'.patch'))],
                         cwd=work,text=True,capture_output=True,check=True)
        logs.append(patch+'\n'+r.stdout); assert 'fuzz' not in r.stdout.lower()
    assert hashes(work)==hashes(ROOT/'build/p150/candidate/sglang')
    files=list(work.rglob('*.py'))
    tools=[ROOT/'scripts'/name for name in ('make_150.py','verify_150.py','inventory_warmup_150.py',
              'test_startup_warmup_150.py','test_warmup_cache_150.py','test_warmup_flush_gloo_150.py','summarize_150.py','p150/ax_shapes.py')]
    for f in files+tools: py_compile.compile(str(f),doraise=True)
    for patch in reversed(stack):
        r=subprocess.run(['patch','-R','-p3','--fuzz=0','--no-backup-if-mismatch','--batch','-i',str(ROOT/'patches'/(patch+'.patch'))],
                         cwd=work,text=True,capture_output=True,check=True)
        logs.append('reverse '+patch+'\n'+r.stdout)
    assert hashes(work)==original
    assert hashes(base)==original
    (EV/'full_stack_apply.log').write_text(''.join(logs))
    row=dict(status='PASS',patches=stack,python_source_files=len(files),python_tools=len(tools),
        deterministic=True,applied_equals_candidate=True,full_reverse_exact=True,base_unchanged=True,
        patch_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (EV/'stack_receipt.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row,indent=2))


if __name__=='__main__': main()
