#!/usr/bin/env python3
"""T47: validate final source identities and all numeric/graph/cache/performance gates."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/T47'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(name):return [json.loads(x) for x in (EV/name).read_text().splitlines() if x.startswith('{')]

def main():
    result=dict(status='PASS',numeric={},compiler={},sources={})
    for n,total,graphs in ((112,88,4),(113,222,10)):
        rr=rows(f'{n}_all.log')
        assert rr[-1]==dict(kind='complete',mode='all',status='PASS')
        env=next(r for r in rr if r['kind']=='environment')
        assert env['source_sha256']==sha(ROOT/f'scripts/kernels/sm80_indexer_{n}.py')
        assert env['oracle_sha256']==sha(ROOT/'build/p110/sm80_deep_gemm.py')
        assert env['test_sha256']==sha(ROOT/f'scripts/test_sm80_indexer_{n}.py')
        if n==113:
            assert env['baseline_sha256']==sha(ROOT/'scripts/kernels/sm80_indexer_112.py')
            assert env['reused_test_sha256']==sha(ROOT/'scripts/test_sm80_indexer_112.py')
        num=[r for r in rr if r['kind']=='numeric'];gg=[r for r in rr if r['kind']=='graph']
        assert len(num)==total and len(gg)==graphs
        assert all(r['max_row_relative_linf']<.01 and r['min_topk_overlap']>=.995 for r in num)
        assert all(r['eager_bit_exact'] and r['replays']==3 for r in gg)
        result['numeric'][n]=dict(comparisons=len(num),max_relative_linf=max(r['max_row_relative_linf'] for r in num),
            max_relative_l2=max(r['max_row_relative_l2'] for r in num),min_topk_overlap=min(r['min_topk_overlap'] for r in num),
            graph_variants=len(gg),graph_replays=sum(r['replays'] for r in gg),environment=env)
        compiler=json.loads((EV/f'{n}/compiler/compiler_receipt.json').read_text())
        assert compiler['source_sha256']==env['source_sha256']
        result['compiler'][n]=compiler
        gen=json.loads((EV/f'{n}/generate_receipt.json').read_text())
        stem='112-sm80-indexer-kernels' if n==112 else '113-sm80-prefill-indexer'
        assert gen['patch_sha256']==sha(ROOT/f'patches/{stem}.patch')
        assert gen['source_sha256']==env['source_sha256']
        history=json.loads((ROOT/f'evidence/T{43 if n==112 else 44}/summary.json').read_text())
        assert history['environment']['source_sha256']==sha(EV/f'sm80_indexer_{n}_v1.py')
        assert history['stack']['patch_sha256']==sha(ROOT/f'patches/drafts/{stem}-v1.patch')
        result['sources'][n]=gen
    paged113=next(k for k in result['compiler'][113]['kernels'] if k['name']=='_paged')
    assert paged113['ptx_sha256']==result['compiler'][112]['paged']['ptx_sha256']
    assert all(k['spills']==0 for k in result['compiler'][113]['kernels'])
    assert all(result['compiler'][112][name]['spills']==0 for name in ('paged','ragged'))
    stack=json.loads((EV/'113/stack_receipt.json').read_text())
    assert stack['status']=='PASS' and len(stack['patches'])==11 and stack['runtime_shapes_and_strides']
    assert stack['patch_sha256']==result['sources'][113]['patch_sha256']
    result['stack']=stack
    cc=rows('cache.log');env=cc[0]
    assert env['source112']==result['sources'][112]['source_sha256']
    assert env['source113']==result['sources'][113]['source_sha256']
    assert cc[-1]['status']=='PASS' and cc[-1]['random_shapes']==50
    assert not any(cc[-1]['random'].values())
    shapes=[r for r in cc if r['kind']=='shape']
    for axis in ('NQ','NK','P','batch'):assert len({r[axis] for r in shapes})==50
    fixed={'H','D','PAGE','BQ','BK','HH','DD','CLEAN','GROUP','LOOP','BLOCK'}
    observed_keys=0
    for row in cc:
        if row['kind']!='jit_miss':continue
        key=ast.literal_eval(row['key'])
        names=list(key['signature'])
        for pos,value in key['constants'].items():
            assert names[pos[0]] in fixed or value==1,(names[pos[0]],value)
        observed_keys+=1
    result['cache']=dict(cc[-1],audited_jit_keys=observed_keys)
    bb=rows('performance.log');env=bb[0]
    for n in (112,113):
        assert env['sources'][f'{n}v2']==result['sources'][n]['source_sha256']
        assert env['sources'][f'{n}v1']==sha(EV/f'sm80_indexer_{n}_v1.py')
    assert bb[-1]['status']=='PASS'
    bench=[r for r in bb if r['kind']=='bench']
    assert len(bench)==16
    assert all(r['change_percent']<=5 for r in bench),[(r['kernel'],r['mode'],r['nk'],r['change_percent']) for r in bench]
    result['benchmarks']=bench
    result['max_regression_percent']=max(r['change_percent'] for r in bench)
    inputs=list((ROOT/'scripts').glob('*47*'))+list((ROOT/'scripts').glob('*sm80_indexer_11[23].py'))+list((ROOT/'scripts/kernels').glob('sm80_indexer_11[23].py'))+list(EV.glob('*_v1.py'))+[ROOT/'build/p110/sm80_deep_gemm.py']
    by_name={p.name:p for p in inputs if p.is_file()}
    remote=(EV/'remote_source_sha256.txt').read_text().splitlines()
    for line in remote:
        digest,name=line.split()
        assert digest==sha(by_name[name]),name
    result['remote_source_hashes_matched']=len(remote)
    files=[p for p in EV.rglob('*') if p.is_file() and p.name not in ('summary.json','performance_table.md','summarize.log')]
    result['evidence_sha256']={str(p.relative_to(EV)):sha(p) for p in files}
    result['tool_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*47*') if p.is_file()}
    (EV/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table=['| 版本 | 场景 | nk | v1 ms | v2 ms | 变化 |','|---|---|---:|---:|---:|---:|']
    for r in bench:
        table.append(f"| {r['kernel']} | {r.get('layout',r['mode'])} | {r['nk']} | {r['v1_ms']:.4f} | {r['v2_ms']:.4f} | {r['change_percent']:+.2f}% |")
    (EV/'performance_table.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('status','numeric','cache','max_regression_percent')},indent=2))

if __name__=='__main__':main()
