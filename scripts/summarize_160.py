#!/usr/bin/env python3
"""Require final T48 evidence and bind it to the delivered source/scripts."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/T48'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    finals=dict(kda='kda_final.log',small='small_final.log',kpool='kpool_final.log',
                dsa='dsa_v1.log',acceptance='acceptance_v3.log',indexer='indexer_v1.log',
                mhc='mhc_v1.log',sharing='sharing_final.log')
    groups={}
    for group,filename in finals.items():
        lines=(EV/'gpu'/filename).read_text().splitlines()
        rows=[json.loads(s) for s in lines if s.startswith('{')]
        assert rows[-1]==dict(group=group,kind='complete',status='PASS'),filename
        assert not any('Traceback (most recent call last)' in s for s in lines),filename
        groups[group]=[r for r in rows if r['kind'] not in ('environment','complete')]
    marlin=(EV/'gpu/marlin_final.log').read_text()
    assert marlin.rstrip().endswith('MARLIN_ALL_PASS') and 'MOE_CLIP10_PASS' in marlin
    dense=[dict(m=int(m),n=int(n),k=int(k),relative_l2=float(e),last_graph_relative_l2=float(g))
           for m,n,k,e,g in re.findall(r'DENSE M=(\d+) N=(\d+) K=(\d+) rel_err=([\deE.+-]+) graph_rel_err=([\deE.+-]+) PASS',marlin)]
    assert len(dense)==9
    resolve=(EV/'gpu/resolve_v4.log').read_text()
    assert '"ignored_remap": true' in resolve and '"max_running_requests": 32' in resolve
    assert 'Ran 10 tests' in (EV/'cpu.log').read_text() and (EV/'cpu.log').read_text().rstrip().endswith('OK')
    source=ROOT/'build/p160/candidate/sglang'
    local={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert local==json.loads((EV/'gpu/remote_source_hashes.json').read_text())
    env=json.loads((EV/'gpu/environment.json').read_text())
    for name in ['test_mtp_sm80_160.py','test_mtp_resolve_160.py','test_mtp_marlin_160.py']:
        assert env['scripts'][name]==digest(ROOT/'scripts'/name),name
    assert env['config_hash']==digest(ROOT/'s1-dev/glm_tok/config.json')
    assert env['runner_sha256']==digest(ROOT/'scripts/run_mtp_sm80_160.sh')
    receipt=json.loads((EV/'stack_receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['patch_sha256']==digest(ROOT/'patches/160-nextn-sm80.patch')
    relocation=json.loads((EV/'gpu/cache_relocation.json').read_text());assert len(relocation)==7
    idle=(EV/'gpu/gpu_final_idle.log').read_text();assert '0, 4 MiB, 0 %' in idle and '1, 4 MiB, 0 %' in idle
    artifacts=[ROOT/'patches/160-nextn-sm80.patch',ROOT/'scripts/p160/ax_mtp_sm80.py',
               ROOT/'scripts/pod/jobs/dev_b160_mtp_n6.sh',ROOT/'scripts/run_mtp_sm80_160.sh']
    artifacts+=list((ROOT/'scripts').glob('*160.py'))
    result=dict(status='PASS_L1_OPERATORS_AND_PACKAGING_ONLY',cpu_tests=10,groups=groups,
                marlin_dense=dense,marlin_clip10_pass=True,actual_server_args_and_draft_config_pass=True,
                candidate_files_matched=len(local),stack=receipt,
                artifact_sha256={str(p.relative_to(ROOT)):digest(p) for p in artifacts},
                evidence_sha256={str(p.relative_to(EV)):digest(p) for p in EV.rglob('*') if p.is_file() and p.name!='summary.json'},
                corrected_cache_directory_violation=dict(builds=7,bytes=sum(r['bytes'] for r in relocation)),
                gpu_idle=True,full_model_loaded=False,tp8_run=False,service_or_slo_verified=False,
                image_built=False,submitted=False,
                limitations=['random activations and reduced expert count','operator graphs, not complete EAGLE runners',
                             'sampling test uses one-hot probabilities, not distribution equivalence',
                             'TileLang mhc static race-analysis warnings retained','AOT package identity with L3 not confirmed'])
    (EV/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','artifact_sha256','evidence_sha256')},indent=2))

if __name__=='__main__':main()
