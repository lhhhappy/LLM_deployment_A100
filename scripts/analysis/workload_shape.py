#!/usr/bin/env python3
"""Compare the shape of a request set with the organizer's public metadata (chain lengths, heads, families, turn
starts, intra sizes, gaps), so a local finding can be checked against what the official workload looks like.

Usage: workload_shape.py [--set DIR ...]   (each DIR holds requests.jsonl and chains.jsonl; default: the organizer's
       s1-dev/data/dev-combined-v1 and cache/s1-dev-longchain-v3)
Per set it prints, from requests.jsonl (per-request rows; for the organizer's set only the dev prefixes have rows):
  chain heads (first request of each chain): new-token distribution, count >50k / >100k, prefix families among heads,
  source-side cache ratio of heads; non-head turn starts: new-token distribution and the share above 8192, split by
  edge type; non-head intra requests above 8192; prompt lengths; replay gaps.
And from chains.jsonl (full chains): requests per chain, turn starts and context resets per chain, uncached share.
All numbers are from the metadata files (measured on the frozen sets), none are run results.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float('nan')


def desc(v):
    return (f"n={len(v)} med={q(v, .5):.0f} mean={statistics.mean(v):.0f} p90={q(v, .9):.0f} "
            f"p99={q(v, .99):.0f} max={max(v):.0f}") if v else 'n=0'


def show(root):
    root = Path(root)
    R = [json.loads(l) for l in open(root / 'requests.jsonl')]
    print(f"====== {root} requests={len(R)}")
    by = collections.defaultdict(list)
    for r in R:
        by[r['chain_id']].append(r)
    for v in by.values():
        v.sort(key=lambda r: r['dispatch_offset_ms'])
    heads = [v[0] for v in by.values()]
    hu = [r['uncached_expected'] for r in heads]
    print(' chain heads new tokens:', desc(hu), ' >50k:', sum(x > 50000 for x in hu), ' >100k:', sum(x > 100000 for x in hu))
    fam = collections.Counter(r.get('prefix_family_id') for r in heads)
    multi = sorted((v for k, v in fam.items() if k and v >= 2), reverse=True)
    print(f" head families (>=2 heads share prefix_family_id): {len(multi)} families, {sum(multi)} heads, sizes {multi[:12]}")
    ratio = [(r.get('source_cached_input_tokens') or 0) / r['source_prompt_tokens'] for r in heads if r.get('source_prompt_tokens')]
    if ratio:
        print(f" head source-side cache ratio: median {q(ratio, .5):.2f}, >=0.5: {sum(x >= .5 for x in ratio)}/{len(ratio)}")
    turn = [r for v in by.values() for i, r in enumerate(v) if i > 0 and r['phase'] == 'turn_start']
    if turn:
        u = [r['uncached_expected'] for r in turn]
        print(' non-head turn starts new tokens:', desc(u), f' >8192: {sum(x > 8192 for x in u)} ({sum(x > 8192 for x in u) / len(u):.0%})',
              ' warm(cached>=50%) and >8192:', sum(1 for r in turn if r['uncached_expected'] > 8192 and r['uncached_expected'] < 0.5 * r['glm_tokens']))
        for e, rs in collections.defaultdict(list, {e: [r for r in turn if r['edge_type'] == e] for e in {r['edge_type'] for r in turn}}).items():
            print(f"    {e:20s}", desc([r['uncached_expected'] for r in rs]))
    intra = [r for v in by.values() for i, r in enumerate(v) if i > 0 and r['phase'] == 'intra']
    if intra:
        u = [r['uncached_expected'] for r in intra]
        print(' non-head intra new tokens:', desc(u), f' >8192: {sum(x > 8192 for x in u)} ({sum(x > 8192 for x in u) / len(u):.1%})',
              ' <=4096 (fast):', sum(x <= 4096 for x in u))
    gl = [r['glm_tokens'] for r in R]
    print(' prompt length:', desc(gl))
    gaps = [(r.get('replay_gap_ms') or 0) / 1000 for r in R if r.get('replay_gap_ms')]
    if gaps:
        print(' replay gaps (s):', desc(gaps), ' >60 s:', sum(x > 60 for x in gaps))
    chains = root / 'chains.jsonl'
    if chains.exists():
        C = [json.loads(l) for l in open(chains)]
        n = [c['n_requests'] for c in C]
        tot = sum(n)
        print(f" full chains: {len(C)} chains, {tot} requests; requests per chain {desc(n)}")
        for lo, hi in ((1, 4), (5, 20), (21, 50), (51, 10 ** 6)):
            cs = [c for c in C if lo <= c['n_requests'] <= hi]
            print(f"    length {lo}-{hi if hi < 10 ** 6 else 'max'}: chains {len(cs)}, requests {sum(c['n_requests'] for c in cs) / tot:.0%}, "
                  f"logical tokens {sum(c['sum_glm_tokens'] for c in cs) / sum(c['sum_glm_tokens'] for c in C):.0%}, "
                  f"uncached {sum(c['sum_uncached_expected'] for c in cs) / max(1, sum(c['sum_uncached_expected'] for c in C)):.0%}")
        ts = [c['phases'].get('turn_start', 0) for c in C]
        print(f"    turn starts per chain {desc(ts)}; chains with a context_reset {sum(c['phases'].get('context_reset', 0) > 0 for c in C)}; "
              f"uncached/logical tokens overall {sum(c['sum_uncached_expected'] for c in C) / sum(c['sum_glm_tokens'] for c in C):.3f}; "
              f"output budget per request med {q([c['max_output_i_sum'] / c['n_requests'] for c in C], .5):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', action='append', default=None)
    a = ap.parse_args()
    for root in a.set or [ROOT / 's1-dev/data/dev-combined-v1', ROOT / 'cache/s1-dev-longchain-v3']:
        show(root)


if __name__ == '__main__':
    main()
