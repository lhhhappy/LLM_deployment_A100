#!/usr/bin/env python3
"""Reproduce descriptive local/formal differences; never predict formal N."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {}


def read(path):
    path = ROOT / path
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def formal(path):
    d = read(path)
    if 'scorecard' in d:
        sc = d['scorecard']
        s = sc['scorewheel_stress']
        assert s == sc['scorewheel_raw_result']['stress']
        assert d['scoringState']['scoreIsFinal'] and sc['scorewheel_gate_passed']
    else:  # Historical normalized status snapshots, not full CLI responses.
        assert d['id'] in (45979, 45980) and d['gate_passed']
        s = d['stress']
    assert d['execStatus'] == 'completed' and s['passed']
    return {'N': s['n_at_slo'], 'metrics': {k: s[k] for k in (
        'fast_intra_p95', 'overall_intra_p95', 'turn_start_p95',
        'chain_start_p95', 'tpot_mean', 'tpot_p95')}}


def local(tag):
    paths = list(ROOT.glob(f'evidence/L{tag}*/N*/score_formal.json'))
    assert len(paths) == 1, paths
    d = read(paths[0].relative_to(ROOT))
    assert d['dev']['coverage'] == 1
    metrics = {k.split('(')[0] + '_p95': v
               for k, v in d['dev']['ttft_p95_by_gate'].items()}
    metrics.update({k: d['tpot'][k] for k in ('tpot_mean', 'tpot_p95')})
    return {'N': d['dev']['config']['N'], 'n': d['dev']['n_attempted'],
            'metrics': metrics}


def main():
    pairs = []
    for attempt, tag, path in (
        (46251, '069', 'evidence/official/attempt-46251-final-20260925.json'),
        (46174, '067', 'evidence/official/attempt-46174-20260924.json'),
        (46173, '068', 'evidence/official/attempt-46173-20260924.json'),
        (45979, '044r', 'evidence/cost-audit-20260924/official-45979.json'),
        (45980, '045r', 'evidence/cost-audit-20260924/official-45980.json'),
    ):
        f, l = formal(path), local(tag)
        delta = {k: {'local_minus_formal': v - f['metrics'][k],
                     'local_over_formal': v / f['metrics'][k],
                     'local_relative_pct': 100 * (v / f['metrics'][k] - 1)}
                 for k, v in l['metrics'].items()}
        pairs.append({'attempt': attempt, 'local_run': tag,
                      'formal': f, 'local': l, 'differences': delta,
                      'same_N': f['N'] == l['N']})
    out = ROOT / 'evidence/official/local-formal-comparison-20260925.json'
    out.write_text(json.dumps({
        'scope': 'descriptive; different datasets, and first three pairs differ in N; not causal or predictive',
        'units': 'TTFT seconds; TPOT seconds per token',
        'pairs': pairs, 'source_sha256': SOURCES,
    }, indent=2) + '\n')
    for p in pairs:
        print(f"{p['attempt']} N{p['formal']['N']} vs {p['local_run']} N{p['local']['N']}")
        for k, d in p['differences'].items():
            print(f"  {k}: {p['formal']['metrics'][k]:.6f} -> {p['local']['metrics'][k]:.6f}; {d['local_relative_pct']:+.2f}%")


if __name__ == '__main__':
    main()
