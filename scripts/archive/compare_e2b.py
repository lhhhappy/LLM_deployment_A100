#!/usr/bin/env python3
"""T26 fail-closed E2b cache comparison; old receipts are never overwritten."""
import argparse
import json
from pathlib import Path

CASES = ('smoke', 'reminder_heavy', 'strict_append')


def quantile(values):
    values = sorted(values)
    return values[min(len(values)-1, int(.95*len(values)))] if values else None


def rows(path):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    keyed = {r['req_id']: r for r in data}
    assert data and len(keyed) == len(data), ('empty or duplicated requests', str(path))
    for r in data:
        assert not r.get('error') and r['output_tokens'] == 4, (r['req_id'], 'request failed/budget changed')
        assert 0 <= r['cached_tokens'] <= r['prompt_tokens'], (r['req_id'], 'bad cache counter')
        assert r['prompt_tokens'] == r['rendered_prompt_tokens'], (r['req_id'], 'render mismatch')
    return keyed


def compare(root, old, case, control_root=None):
    control_root = control_root or root
    groups = {'stock': rows(old/f'stock_{case}/requests.jsonl'),
              'v11': rows(old/f'on_{case}/requests.jsonl'),
              'control': rows(control_root/f'control_{case}/requests.jsonl'),
              'candidate': rows(root/f'candidate_{case}/requests.jsonl')}
    if case == 'smoke':
        groups['off'] = rows(root/'off_smoke/requests.jsonl')
    assert all(list(g) == list(groups['stock']) for g in groups.values()), 'IDs or order differ'
    joined = []
    for rid, r in groups['stock'].items():
        for key in ('prompt_sha256', 'chain_id', 'idx_in_chain', 'rendered_prompt_tokens',
                    'prompt_tokens', 'output_tokens', 'phase', 'uncached_expected'):
            assert all(g[rid][key] == r[key] for g in groups.values()), (rid, key, 'mismatch')
        pred = groups['candidate'][rid]['predictions']['role_conservative']['cached_tokens']
        assert all(g[rid]['predictions']['role_conservative']['cached_tokens'] == pred
                   for g in groups.values()), (rid, 'prediction mismatch')
        joined.append(dict(req_id=rid, chain_id=r['chain_id'], idx_in_chain=r['idx_in_chain'],
            prompt_sha256=r['prompt_sha256'], prompt_tokens=r['prompt_tokens'],
            fast_intra=r['idx_in_chain'] > 0 and r['phase'] not in ('turn_start', 'context_reset')
                and (r['uncached_expected'] or 0) <= 4096,
            role_pred=pred, **{v: g[rid]['cached_tokens'] for v, g in groups.items()}))
    fast = [r for r in joined if r['fast_intra']]
    pairs = {}
    for candidate, baseline in (('control', 'v11'), ('candidate', 'control'),
                                ('candidate', 'v11'), ('candidate', 'stock')):
        pairs[candidate+'_vs_'+baseline] = {
            'better': sum(r[candidate] > r[baseline] for r in joined),
            'equal': sum(r[candidate] == r[baseline] for r in joined),
            'worse': sum(r[candidate] < r[baseline] for r in joined)}
    within = sum(abs(r['candidate']-r['role_pred']) <= 64 for r in joined)
    p95 = {v: quantile([r['prompt_tokens']-r[v] for r in fast]) for v in (*groups, 'role_pred')}
    summary = dict(requests=len(joined), fast_intra_n=len(fast), pairs=pairs,
        fast_uncached_p95=p95, candidate_within_64_role_prediction=within,
        candidate_within_64_fraction=within/len(joined),
        total_uncached={v: sum(r['prompt_tokens']-r[v] for r in joined) for v in groups},
        off_equals_stock=all(r['off'] == r['stock'] for r in joined) if 'off' in groups else None)
    return dict(summary=summary, requests=joined)


def trace_summary(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    startup = [e for e in events if e['kind'] == 'startup']
    assert len(startup) == 1, 'TP1 single startup required'
    flushes = [e for e in events if e['kind'] == 'flush']
    assert flushes and all(e['success'] is True for e in flushes), 'flush missing/failed'
    base = startup[0]['pools']
    return dict(startup_pools=base, truncation_align=startup[0]['truncation_align'],
        successful_flushes=len(flushes), all_flushes_restore_pools=all(e['after'] == base for e in flushes),
        last_flush=flushes[-1], tail_splits=sum(e['kind'] == 'tail_split' for e in events))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--old', type=Path, default=Path('/sjtu/linhang/arena/runs/E2_20260922'))
    p.add_argument('--control-root', type=Path)
    args = p.parse_args()
    assert (args.root/'batch_complete.json').exists(), 'batch not complete'
    data = {case: compare(args.root, args.old, case, args.control_root) for case in CASES}
    traces = {v: trace_summary((args.control_root if v == 'control' and args.control_root else args.root)
                              /v/'scheduler_trace.jsonl') for v in ('control', 'candidate', 'off')}
    reminder = data['reminder_heavy']['summary']
    gates = {
        'D1-01': data['smoke']['summary']['off_equals_stock'],
        'D1-02': reminder['pairs']['candidate_vs_stock']['worse'] == 0
            and reminder['candidate_within_64_fraction'] >= .95
            and reminder['fast_uncached_p95']['candidate'] < reminder['fast_uncached_p95']['stock'],
        'D1-03': data['strict_append']['summary']['pairs']['candidate_vs_stock']['worse'] == 0,
        'D1-08_scoped_N1': traces['candidate']['all_flushes_restore_pools'],
        'zero_regression_vs_control_all_cases': all(c['summary']['pairs']['candidate_vs_control']['worse'] == 0
                                                    for c in data.values())}
    print(json.dumps(dict(cases=data, traces=traces, gates=gates,
        provenance=dict(candidate_root=str(args.root), control_root=str(args.control_root or args.root), old_root=str(args.old)),
        limitations=['Random stand-in, TP1 N1 extra_buffer; not SLO/accuracy proof',
                     'D1-04 remains unresolved; not rerun', 'No 003, DSA, HiCache, lazy, MTP or DP']), indent=2))
