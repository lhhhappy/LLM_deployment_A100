#!/usr/bin/env python3
"""Verify final T44 source/test/compiler/patch identity and derive report tables."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/T44'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(l) for l in (EV/'final_all.log').read_text().splitlines() if l.startswith('{')]
assert rows[-1]==dict(kind='complete',status='PASS',mode='all')
env=next(r for r in rows if r['kind']=='environment')
assert env['source_sha256']==sha(ROOT/'scripts/kernels/sm80_indexer_113.py')
assert env['baseline_sha256']==sha(ROOT/'scripts/kernels/sm80_indexer_112.py')
assert env['oracle_sha256']==sha(ROOT/'build/p110/sm80_deep_gemm.py')
assert env['test_sha256']==sha(ROOT/'scripts/test_sm80_indexer_113.py')
assert env['reused_test_sha256']==sha(ROOT/'scripts/test_sm80_indexer_112.py')
stack=json.loads((EV/'stack_receipt.json').read_text())
gen=json.loads((EV/'generate_receipt.json').read_text())
compiler=json.loads((EV/'compiler/compiler_receipt.json').read_text())
profile=json.loads((EV/'profile/profile_receipt.json').read_text())
assert stack['status']=='PASS' and stack['decode_source_byte_exact']
assert stack['patch_sha256']==gen['patch_sha256']==sha(ROOT/'patches/113-sm80-prefill-indexer.patch')
assert gen['source_sha256']==compiler['source_sha256']==profile['source_sha256']==env['source_sha256']
oldcompiler=json.loads((ROOT/'evidence/T43/compiler/compiler_receipt.json').read_text())
assert next(k for k in compiler['kernels'] if k['name']=='_paged')['ptx_sha256']==oldcompiler['paged']['ptx_sha256']
num=[r for r in rows if r['kind']=='numeric']
bench=[r for r in rows if r['kind']=='bench']
prefill=[r for r in bench if r['name']=='prefill']
assert len(prefill)==6
assert all(r['new_effective_tflops']>=100 for r in prefill)
summary=dict(status='PASS',environment=env,numeric_comparisons=len(num),
             max_row_relative_linf=max(r['max_row_relative_linf'] for r in num),
             max_row_relative_l2=max(r['max_row_relative_l2'] for r in num),
             min_topk_overlap=min(r['min_topk_overlap'] for r in num),
             graphs=[r for r in rows if r['kind']=='graph'],bench=bench,
             min_effective_tflops=min(r['new_effective_tflops'] for r in prefill),
             stack=stack,compiler=compiler,decode_ptx_byte_exact=True,
             scratch_bytes_8192_190k=2*8192*32*128+2*190000*128)
(EV/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| 场景（nq=8192） | nk | 112 ms | 113 ms | 112 TFLOPS | 113 TFLOPS | 加速 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in prefill:
 lines.append(f"| {r['layout']} | {r['nk']} | {r['old_ms']:.3f} | {r['new_ms']:.3f} | {r['old_effective_tflops']:.1f} | {r['new_effective_tflops']:.1f} | {r['speedup']:.2f}× |")
lines+=['','TFLOPS 为用户指定全宽等效口径 `2*nq*nk*32*128 / 秒 / 1e12`；clean=True 剪枝也计入收益，不能当作实际执行MMA的硬件利用率。JSON另报有效区间pair比例与对应吞吐。计时含每次fp8解码与临时缓冲/最终fp32输出分配，排除JIT和输入生成。','',
        '| decode B6/N1 | nk | 112 ms | 113 ms | 输出 |','|---|---:|---:|---:|---|']
for r in bench:
 if r['name'].startswith('decode'):
  lines.append(f"| {r['name']} | {r['nk']} | {r['old_ms']:.4f} | {r['new_ms']:.4f} | 逐bit同 |")
(EV/'performance_table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['status','numeric_comparisons','max_row_relative_linf','max_row_relative_l2','min_topk_overlap','min_effective_tflops']},indent=2))
