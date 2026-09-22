#!/usr/bin/env python3
"""Validate T43 final evidence/source hashes and emit the small delivery receipt."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/T43'
rows=[json.loads(s) for s in (EV/'final_all.log').read_text().splitlines() if s.startswith('{')]
assert rows[-1]==dict(kind='complete',mode='all',status='PASS')
env=next(r for r in rows if r['kind']=='environment')
nums=[r for r in rows if r['kind']=='numeric']
graphs=[r for r in rows if r['kind']=='graph']
benches=[{k:v for k,v in r.items() if k not in ('old_samples','new_samples','kind')}
         for r in rows if r['kind']=='bench']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert env['source_sha256']==sha(ROOT/'scripts/kernels/sm80_indexer_112.py')
assert env['test_sha256']==sha(ROOT/'scripts/test_sm80_indexer_112.py')
assert env['oracle_sha256']==sha(ROOT/'build/p110/sm80_deep_gemm.py')
assert len(nums)==88 and len(graphs)==4 and len(benches)==8, (len(nums),len(graphs),len(benches))
assert all(n['max_row_relative_linf']<1e-2 and n['min_topk_overlap']>=.995 for n in nums)
stack=json.loads((EV/'stack_receipt.json').read_text())
assert stack['status']=='PASS'
assert stack['patch_sha256']==sha(ROOT/'patches/112-sm80-indexer-kernels.patch')
assert sha(ROOT/'scripts/kernels/sm80_indexer_112.py')==sha(ROOT/'build/p112/candidate/sglang/srt/layers/attention/dsa/sm80_indexer_kernels.py')
compiler=json.loads((EV/'compiler/compiler_receipt.json').read_text())
assert compiler['source_sha256']==env['source_sha256']
summary=dict(status='PASS',environment=env,numeric_comparisons=len(nums),
    max_row_relative_linf=max(r['max_row_relative_linf'] for r in nums),
    max_row_relative_l2=max(r['max_row_relative_l2'] for r in nums),
    min_topk_overlap=min(r['min_topk_overlap'] for r in nums),
    graph_variants=graphs,benchmarks=benches,stack=stack,compiler=compiler)
(EV/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
