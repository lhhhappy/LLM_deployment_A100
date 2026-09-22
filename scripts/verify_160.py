#!/usr/bin/env python3
"""Verify 12-patch fuzz=0 stack, deterministic generation, syntax and byte-exact reversal."""
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import make_160 as gen

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/T48'


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def main():
    base=ROOT/'build/base_exact/sglang'; original=hashes(base)
    target=ROOT/'patches/160-nextn-sm80.patch'; expected=target.read_bytes()
    gen.main(); assert target.read_bytes()==expected
    work=ROOT/'build/p160/verify/sglang'
    if work.exists(): shutil.rmtree(work)
    shutil.copytree(base,work)
    stack=gen.PATCHES+['160-nextn-sm80']; logs=[]
    for patch in stack:
        r=subprocess.run(['patch','-p3','--fuzz=0','--no-backup-if-mismatch','--batch','-i',str(ROOT/'patches'/(patch+'.patch'))],
                         cwd=work,text=True,capture_output=True,check=True)
        logs.append(patch+'\n'+r.stdout); assert 'fuzz' not in r.stdout.lower()
    assert hashes(work)==hashes(ROOT/'build/p160/candidate/sglang')
    files=list(work.rglob('*.py'))
    tools=[ROOT/'scripts'/name for name in ('make_160.py','verify_160.py',
              'test_mtp_sm80_160.py','test_mtp_resolve_160.py','test_mtp_marlin_160.py',
              'extract_spec_stats_160.py','test_mtp_policy_160.py','p160/ax_mtp_sm80.py')]
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
