"""CPU-only review reproductions. Never contacts an engine or changes production code."""
import json
import hashlib
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 's1-dev/harness'))
from s1_common import load_index, in_ttft_gate

rows, chains, groups = load_index(str(ROOT / 's1-dev/data/dev-combined-v1'))
counts = {}
for row in rows.values():
    key = row['edge_type'] + '/' + ('head' if row['_idx_in_chain'] == 0 else 'followup')
    v = counts.setdefault(key, dict(n=0, prompt=0, frozen_uncached=0))
    v['n'] += 1
    v['prompt'] += row['glm_tokens']
    v['frozen_uncached'] += row['uncached_expected']

empty = OUT / 'empty_raw.jsonl'
empty.write_text('')
ev = OUT / 'empty_verdict.json'
p = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/pod/verify/analyze_run.py'),
                    str(ROOT / 's1-dev/harness'), str(empty), str(ev)], capture_output=True, text=True)
(OUT / 'empty_analyzer.log').write_text(p.stdout + p.stderr)
empty_result = json.loads(ev.read_text()) if ev.exists() else None

ref = {'case': {'tokens': [1, 2, 3], 'logprobs': [-1., -2., -3.]}}
variants = {
    'truncated': {'case': {'tokens': [1], 'logprobs': [-1.]}},
    'diverged_after_first': {'case': {'tokens': [1, 9, 8], 'logprobs': [-1., -8., -9.]}},
}
rp = OUT / 'num_ref.json'; rp.write_text(json.dumps(ref))
num = {}
for name, candidate in variants.items():
    cp = OUT / ('num_' + name + '.json'); cp.write_text(json.dumps(candidate))
    p = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/pod/verify/numcheck_cmp.py'),
                        str(rp), str(cp)], capture_output=True, text=True)
    num[name] = {'rc': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

summary = {'cohort_edge_roles': counts, 'empty_analyzer': empty_result,
           'num_comparator': num}
(OUT / 'cpu_review.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))

raw_path = OUT / '026_N18_raw.jsonl'
if raw_path.exists():
    raw = raw_path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 722 and len({r['req_id'] for r in records}) == 722
    assert set(r['req_id'] for r in records) == set(rows)
    assert all(not r['error'] for r in records)
    accounting = defaultdict(lambda: dict(n=0, prompt=0, actual=0, frozen=0, positive_excess=0))
    for r in records:
        assert r['idx_in_chain'] == rows[r['req_id']]['_idx_in_chain']
        key = r['edge_type'] + '/' + ('head' if r['idx_in_chain'] == 0 else 'followup')
        v = accounting[key]
        actual = r['prompt_tokens'] - r['cached_tokens']
        frozen = r['uncached_expected']
        v['n'] += 1; v['prompt'] += r['prompt_tokens']
        v['actual'] += actual; v['frozen'] += frozen
        v['positive_excess'] += max(0, actual - frozen)
    for v in accounting.values():
        v['net_excess'] = v['actual'] - v['frozen']
    fast = [r for r in records if in_ttft_gate(r, 'fast_intra')]
    naive = [r for r in records if r['phase'] == 'intra' and r['uncached_expected'] <= 4096 and r['ttft_s'] > 3]
    result = dict(raw_sha256=hashlib.sha256(raw).hexdigest(), n=len(records), errors=0,
                  all_actual=sum(r['prompt_tokens'] - r['cached_tokens'] for r in records),
                  by_edge_role=dict(accounting), fast_correct_count=len(fast),
                  fast_correct_over=sum(r['ttft_s'] > 3 for r in fast), fast_naive_over=len(naive),
                  naive_heads=sum(r['idx_in_chain'] == 0 for r in naive),
                  followup_positive_excess_all=sum(max(0, r['prompt_tokens'] - r['cached_tokens'] - r['uncached_expected'])
                                                  for r in records if r['idx_in_chain'] > 0))
    (OUT / '026_accounting.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
