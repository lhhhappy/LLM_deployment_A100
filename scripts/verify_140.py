#!/usr/bin/env python3
"""T45: deterministic generator, real fuzz=0 stack, compilation and byte rollback."""
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import tempfile

import make_140 as gen

ROOT=gen.ROOT
EV=ROOT/'evidence/T45'


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def apply(root,name,reverse=False):
    cmd=['patch','-p3','--fuzz=0','--batch','--no-backup-if-mismatch']
    if reverse:cmd+=['-R']
    result=subprocess.run(cmd+['-i',str(ROOT/'patches'/f'{name}.patch')],cwd=root,
                          capture_output=True,text=True,check=True)
    assert 'fuzz' not in result.stdout.lower()
    return name+'\n'+result.stdout


def main():
    base=hashes(ROOT/'build/base_exact/sglang')
    patch=ROOT/'patches/140-kda-dual-snapshot.patch'
    expected=patch.read_bytes();gen.main();assert patch.read_bytes()==expected
    logs=[]
    chain=gen.PATCHES+['140-kda-dual-snapshot','120-sched-protect-chain','130-async-tokenize']
    candidate=ROOT/'build/p140/stack/sglang'
    control=ROOT/'build/p140/control/sglang'
    for d in (candidate,control):
        if d.exists():shutil.rmtree(d)
        shutil.copytree(ROOT/'build/base_exact/sglang',d)
    for name in chain:
        logs.append(apply(candidate,name))
        if name=='140-kda-dual-snapshot':
            assert hashes(candidate)==hashes(ROOT/'build/p140/candidate/sglang')
    for name in [n for n in chain if not n.startswith('140-')]: logs.append(apply(control,name))
    files=list(candidate.rglob('*.py'))
    toolfiles=[ROOT/'scripts'/n for n in ('make_140.py','verify_140.py','test_kda_snapshot_140.py',
               'test_kda_snapshot_cache_140.py','replay_kda_snapshot_140.py')]+list((ROOT/'scripts/p140').glob('*.py'))
    with tempfile.TemporaryDirectory() as out:
        for i,p in enumerate(files+toolfiles):py_compile.compile(str(p),cfile=str(Path(out)/f'{i}.pyc'),doraise=True)
    # Separate reversible copy: retain stack trees for scheduler parity tests.
    rev=ROOT/'build/p140/reverse/sglang'
    if rev.exists():shutil.rmtree(rev)
    shutil.copytree(candidate,rev)
    for name in reversed(chain):logs.append(apply(rev,name,True))
    assert hashes(rev)==base and hashes(ROOT/'build/base_exact/sglang')==base
    receipt=dict(status='PASS',patches=chain,files=len(files),tools=len(toolfiles),
                 deterministic=True,applied_equals_candidate=True,base_unchanged=True,
                 full_reverse_exact=True,patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest())
    (EV/'full_stack_apply.log').write_text(''.join(logs))
    (EV/'stack_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
