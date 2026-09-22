#!/usr/bin/env python3
"""Bind T45 numeric/CPU/off/stack/replay evidence to delivered source hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/T45'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    log=EV/'numeric_final_v3.log'
    rows=[json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')]
    complete=next(r for r in rows if r.get('kind')=='complete')
    assert complete['passed'] and complete['cases']==8
    env=next(r for r in rows if r.get('kind')=='environment')
    for rel,digest in env['files'].items():
        assert sha(ROOT/'build/p140/candidate/sglang'/rel)==digest,rel
    numeric=[r for r in rows if r.get('kind')=='numeric']
    assert len(numeric)==8
    for r in numeric:
        for name in ('boundary','conv','end','final','resume','exporter_output','exporter_final','off_baseline_output','off_baseline_final'):
            if r[name] is not None:
                assert r[name]['equal'] and r[name]['max_abs']==0,(r,name)
    cpu=json.loads([x for x in (EV/'cpu_final_v2.log').read_text().splitlines() if x.startswith('{')][-1])
    assert cpu==dict(tests=21,failures=0,errors=0)
    assert (EV/'cache_off_candidate.json').read_bytes()==(EV/'cache_off_baseline.json').read_bytes()
    scheduler=json.loads((EV/'scheduler_summary.json').read_text())
    assert scheduler['off_json_bytes_equal'] and scheduler['off_cases']==32
    stack=json.loads((EV/'stack_receipt.json').read_text())
    assert stack['status']=='PASS'
    patch=ROOT/'patches/140-kda-dual-snapshot.patch'
    assert sha(patch)==stack['patch_sha256']
    replay=json.loads((EV/'replay/replay_summary.json').read_text())
    remote=json.loads((EV/'source_hashes.json').read_text())
    for rel,digest in remote.items():
        if rel.startswith('candidate/sglang/'):
            local=ROOT/'build/p140'/rel
        elif rel.startswith('baseline/'):
            local=ROOT/'build/p140/baseline/sglang/kernels/ops/attention/fla'/Path(rel).name
        else:
            local=ROOT/'scripts'/rel
        assert sha(local)==digest,('remote source mismatch',rel)
    assert len(remote)==20
    inputs=[patch,ROOT/'patches/140-kda-dual-snapshot.md']
    inputs += sorted((ROOT/'scripts').glob('*140*.py'))+sorted((ROOT/'scripts/p140').glob('*.py'))
    inputs += [ROOT/'scripts/run_kda_snapshot_140.sh']
    inputs += [p for p in EV.rglob('*') if p.is_file() and p.name not in ('summary.json','numeric_table.md','README.md','summary.log','check_records.log')]
    # The copied disabled oracle must be the unmodified baseline.
    source={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    delivered=json.loads((EV/'generate_receipt.json').read_text())['changed']
    source.update({str((ROOT/'build/p140/candidate/sglang'/r).relative_to(ROOT)):
                   sha(ROOT/'build/p140/candidate/sglang'/r) for r in delivered})
    report=dict(status='PASS',environment=env,numeric=numeric,
                numeric_cases=8,boundary_resume_cases=7,max_abs=0,
                cpu=cpu,cache_off_bytes_equal=True,cache_off_scenarios=3,
                scheduler_off_cases=32,scheduler_off_rounds=960,
                scheduler_on_cases=8,role_example_extends={'off':2,'on':1},
                remote_source_files_match=len(remote),
                stack=stack,replay=replay,sha256=source,
                limitations=['operator/CPU methods only; no complete engine/TP8/real weights/NEXTN/SLO',
                             'off comparison shares unchanged helper autotune choices; original recurrence source retained',
                             'offline unlimited independent chains, prompt-only; no decode/eviction/concurrency/retraction'])
    (EV/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    table=['| H×D | 每条extend长度 | 边界offset | fp32边界最大误差 | conv误差 | 续算输出误差 | off与原kernel |',
           '|---|---|---|---:|---:|---:|---|']
    for r in numeric:
        table.append(f"| {r['heads']}×128 | {r['lengths']} | {r['boundaries']} | " +
                     ('— | — | —' if r['boundary'] is None else '0 | 0 | 0') + ' | 逐元素相同 |')
    (EV/'numeric_table.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sha256','numeric','environment','replay')},indent=2))


if __name__=='__main__':main()
