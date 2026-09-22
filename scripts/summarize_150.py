#!/usr/bin/env python3
"""Bind T46 raw CPU/Gloo/GPU/stack evidence and delivered source hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/T46'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def summarize(path, cold):
    data=rows(path)
    assert data[-1]['status']=='PASS', path
    repeated=[r for r in data if r.get('case','').endswith('-repeat')]
    first=[r for r in data if r.get('case','').endswith('-first')]
    assert len(repeated)==len(first)==12
    if cold:
        assert sum(r['disk_hits'] for r in first)==0
        assert sum(r['compiled'] for r in first)>0
    fields=('jit_misses','bench_calls','compiled','disk_hits','cache_files_added','cache_files_changed')
    assert all(r[f]==0 for r in repeated for f in fields)
    novel=next(r for r in data if r.get('case')=='113-novel-nq-601')
    assert novel['jit_misses']>0
    return {'environment':data[0], 'first_totals':{f:sum(r[f] for r in first) for f in fields},
            'repeat_cases':len(repeated),'repeat_totals':{f:sum(r[f] for r in repeated) for f in fields},
            'novel_nq_601':novel}


def main():
    cpu=(EV/'cpu_tests.log').read_text()
    assert 'Ran 21 tests' in cpu and cpu.rstrip().endswith('OK')
    stack=json.loads((EV/'stack_receipt.json').read_text()); assert stack['status']=='PASS'
    gloo=rows(EV/'gloo.log'); assert len(gloo)==2
    assert all(r['status']=='PASS' and [x['failed'] for x in r['results']]==[False,True,True] for r in gloo)
    cold=summarize(EV/'operators_cold5.log',True)
    persistent=summarize(EV/'operators_persistent.log',False)
    delivered=[ROOT/'patches/150-startup-warmup.patch',ROOT/'patches/150-startup-warmup.md',
               ROOT/'scripts/p150/ax_shapes.py']
    delivered += list((ROOT/'scripts').glob('*150.py'))
    remote=json.loads((EV/'remote_source_hashes.json').read_text())
    for rel,sha in remote.items():
        if rel.startswith('sglang/'):
            local=ROOT/'build/p150/candidate'/rel
        elif rel=='ax_shapes.py': local=ROOT/'scripts/p150/ax_shapes.py'
        else: local=ROOT/'scripts'/rel
        assert hashlib.sha256(local.read_bytes()).hexdigest()==sha,rel
    bound=delivered+[p for p in EV.iterdir() if p.is_file() and p.name!='summary.json']
    summary={'status':'PASS_SCOPED_VALIDATION','cpu_tests':21,'gloo_rank_cases':6,
             'stack':stack,'cold_operator_run':cold,'persistent_operator_run':persistent,
             'remote_sources_match':len(remote),
             'limitations':['No full-model/TP8 service test','Finite warmup cannot guarantee zero serving-time JIT',
                            'NT_BUCKET2/eager decode coverage depends on scheduling',
                            'MTP skipped; selected ordinary TP profile only'],
             'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(bound))}}
    (EV/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sha256','stack')},indent=2))


if __name__=='__main__': main()
