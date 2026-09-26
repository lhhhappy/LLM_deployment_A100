#!/usr/bin/env python3
"""Metadata-only gap distribution audit; never changes data or calls the engine."""
import argparse
import collections
import json
import math
from pathlib import Path

import longchain as lc


def stats(values):
    xs = sorted(values)
    if not xs:
        return {'n': 0}
    def quantile(q):
        return xs[max(0, math.ceil(len(xs) * q) - 1)]
    return {'n': len(xs), 'p50_s': quantile(.5), 'p90_s': quantile(.9),
            'p95_s': quantile(.95), 'p99_s': quantile(.99), 'max_s': xs[-1],
            'zero_fraction': sum(x == 0 for x in xs)/len(xs),
            'over_10s_fraction': sum(x > 10 for x in xs)/len(xs),
            'over_30s_fraction': sum(x > 30 for x in xs)/len(xs),
            'over_60s_fraction': sum(x > 60 for x in xs)/len(xs)}


def length_bin(n):
    return ('1-4', '5-8', '9-15', '16-30', '31-99', '100+')[sum(n > e for e in (4, 8, 15, 30, 99))]


def audit(root, cohort_path):
    rows = {lc.req_id(r): r for r in lc.read_jsonl(root / 'requests.jsonl')}
    cohort = json.loads(cohort_path.read_text())
    prov_path = root / 'provenance.jsonl'
    prov = {p['req_id']: p for p in lc.read_jsonl(prov_path)} if prov_path.exists() else {}
    groups = collections.defaultdict(list)
    missing = collections.Counter()
    chain_stats = []
    n_pairs = n_long_pairs = 0
    for chain in cohort['chains']:
        rs = [rows[rid] for rid in chain['req_ids']]
        orig = [int(r.get('replay_gap_ms') or 0) for r in rs]
        total = sum(orig)
        eff = orig[:]
        if total > 3600000:
            eff = [round(x * 3600000 / total) for x in orig]
            last = max(i for i, x in enumerate(orig) if x)
            eff[last] += 3600000-sum(eff)
        for i, (rid, row, gap_ms, effective_ms) in enumerate(zip(chain['req_ids'], rs, orig, eff)):
            if row.get('replay_gap_ms') is None:
                missing['all'] += 1
                missing['head' if i == 0 else 'nonhead'] += 1
            gap = effective_ms/1000
            groups['all_effective'].append(gap)
            groups['all_raw'].append(gap_ms/1000)
            kind = prov.get(rid, {}).get('kind', 'original')
            groups['kind/' + kind].append(gap)
            if i == 0:
                groups['head'].append(gap)
                continue
            groups['nonhead'].append(gap)
            groups['phase/' + row['phase']].append(gap)
            groups['length/' + length_bin(len(rs))].append(gap)
            if i > 1:
                n_pairs += 1
                n_long_pairs += eff[i-1] > 30000 and effective_ms > 30000
        chain_stats.append({'chain_id': chain['chain_id'], 'n': len(rs),
                            'sum_raw_s': total/1000, 'sum_effective_s': sum(eff)/1000,
                            'max_gap_s': max(eff)/1000,
                            'long_gaps_over_30s': sum(x > 30000 for x in eff),
                            'capped': total > 3600000})
    return {'root': str(root), 'cohort_sha256': cohort['cohort_sha256'],
            'groups': {k: stats(v) for k, v in sorted(groups.items())},
            'null_gap_count_treated_as_zero_by_harness': dict(missing),
            'capped_chains': sum(c['capped'] for c in chain_stats),
            'chain_total_effective_gap': stats([c['sum_effective_s'] for c in chain_stats]),
            'consecutive_nonhead_pairs': n_pairs, 'consecutive_gaps_both_over_30s': n_long_pairs,
            'chains': chain_stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('evidence/longchain-design-20260924/timing-distribution.json'))
    a = ap.parse_args()
    datasets = {'dev': audit(Path('s1-dev/data/dev-combined-v1'), Path('s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json'))}
    for name in ('full', 'lite'):
        root = Path('data/s1-dev-longchain' + ('-lite' if name == 'lite' else ''))
        datasets[name] = audit(root, root / 'cohort.json')
    profile = Path('evidence/phoenix-longchain-20260924/expanded/joint-events.jsonl')
    events = [e for e in lc.read_jsonl(profile) if e.get('same_system_hash') is True
              and e.get('same_tools_hash') is True and not e.get('model_changed')
              and isinstance(e.get('gap_s'), (int, float)) and e['gap_s'] >= 0
              and isinstance(e.get('prompt_delta_source'), (int, float))]
    phoenix = {}
    for label, es in [('all_eligible', events), ('reported_compression', [e for e in events if e['event_observation'] == 'reported_compression'])]:
        phoenix[label] = {'raw': stats([e['gap_s'] for e in es]),
                          'capped_300s': stats([min(e['gap_s'], 300) for e in es]),
                          'unique_sessions': len({e['session_id'] for e in es})}
    doc = {'method': 'Nearest-rank quantiles; effective gap applies original 3600s per-chain cap. Nonhead excludes first replay request regardless of phase. Null gaps follow harness zero conversion. Phoenix is unweighted observed event sample, not population or formal workload.',
           'datasets': datasets, 'phoenix_eligible_event_sample': phoenix,
           'limits': ['Public dev only exposes prefixes, not the hidden full gap distribution.',
                      'Phoenix end-to-start and competition replay gaps have different decompositions.',
                      'The generator samples linked segments, replaces some compression references and caps at 300s; it does not guarantee marginal or joint gap distribution preservation.',
                      'Lite stratification balances aggregate gap totals, not gap quantiles or autocorrelation.']}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({name: {k: v for k, v in d.items() if k != 'chains'} for name, d in datasets.items()}, ensure_ascii=False, indent=2))
    print('PHOENIX', json.dumps(phoenix, ensure_ascii=False))


if __name__ == '__main__':
    main()
