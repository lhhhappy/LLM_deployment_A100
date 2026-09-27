#!/usr/bin/env python3
"""v5 = v4 with re-apportioned output budgets (metadata only; bodies, prompts, cohort and chain totals unchanged).

Why: v4 apportions each chain's synthesized output total across its synthesized requests in proportion to public
templates' outputs. This variant explores a heavier output tail. Source-history actual outputs use a different
tokenizer from GLM replay budgets, so matching their quantiles would not establish faithful workload calibration.

Rule (deterministic, --seed): for every chain, the original (public) requests keep their real outputs; the synthesized
requests draw budgets from the public per-request output distribution (all 722 real outputs) raised to the power
--gamma (heavier tail), scaled so that the chain's synthesized total equals v4's (including its recorded source
aggregate conflicts), capped at --cap tokens per request (the harness request timeout is 1200 s: a 16k-token output at 60 ms/token
still finishes) with the excess redistributed inside the chain, minimum 2, integer sums preserved exactly.

  python3 -B scripts/longchain/rebudget.py --parent cache/s1-dev-longchain-v4 --public s1-dev/data/dev-combined-v1 \
      --out cache/s1-dev-longchain-v5 --gamma 1.6 --cap 16384 --seed 20270101

The fresh output keeps the parent's cohort order and links its body shards. Metadata provenance and the complete
manifest are refreshed. --set optionally changes the label without changing order. --metadata-only permits missing
assets in a local metadata snapshot; that result stays explicitly incomplete and is not replay acceptance.
"""
import argparse, json, os, random, statistics, hashlib, collections
if __package__:
    from .longchain_metadata import checked_output, publish
else:
    from longchain_metadata import checked_output, publish

def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else None

def apportion_exact(draws, total, cap, floor=2):
    """Integer budgets with sum == total, proportional to draws, each within [floor, cap]."""
    n = len(draws)
    if total < floor * n: floor = max(1, total // n)
    vals = [None] * n; free = list(range(n)); remaining = total
    for _ in range(n + 2):
        s = sum(draws[i] for i in free)
        if not free or s <= 0: break
        scale = remaining / s
        raw = {i: draws[i] * scale for i in free}
        clipped = False
        for i in list(free):
            if raw[i] >= cap: vals[i] = cap; remaining -= cap; free.remove(i); clipped = True
            elif raw[i] <= floor: vals[i] = floor; remaining -= floor; free.remove(i); clipped = True
        if not clipped:
            # largest-remainder rounding on the free ones
            fl = {i: int(raw[i]) for i in free}; rem = remaining - sum(fl.values())
            order = sorted(free, key=lambda i: raw[i] - fl[i], reverse=True)
            for i in free: vals[i] = fl[i]
            for i in order[:max(0, rem)]: vals[i] += 1
            remaining = 0; free = []
            break
    for i in range(n):
        if vals[i] is None: vals[i] = floor
    diff = total - sum(vals)
    # settle any residue from clipping on the largest non-capped entries
    if diff:
        order = sorted(range(n), key=lambda i: -vals[i])
        for i in order:
            if diff == 0: break
            room = (cap - vals[i]) if diff > 0 else (vals[i] - floor)
            step = max(-room, min(room, diff)) if diff > 0 else max(-room, diff)
            vals[i] += step; diff -= step
    return vals

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parent', required=True); ap.add_argument('--public', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--gamma', type=float, default=1.6); ap.add_argument('--cap', type=int, default=16384)
    ap.add_argument('--seed', type=int, default=20270101); ap.add_argument('--max-context', type=int, default=524288)
    ap.add_argument('--set', help='optional derived set label; default preserves the parent label')
    ap.add_argument('--metadata-only', action='store_true', help='allow missing parent assets; output remains explicitly incomplete')
    a = ap.parse_args()
    checked_output(a.parent, a.out)
    rows = [json.loads(l) for l in open(os.path.join(a.parent, 'requests.jsonl'), encoding='utf-8')]
    public = [json.loads(l) for l in open(os.path.join(a.public, 'requests.jsonl'), encoding='utf-8')]
    base = sorted(r['max_output_i'] for r in public if r.get('max_output_i'))
    chains = {c['chain_id']: c for c in (json.loads(l) for l in open(os.path.join(a.parent, 'chains.jsonl'), encoding='utf-8'))}
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get('split') == 'synthetic': by[r['chain_id']].append(i)
    before = [r['max_output_i'] for r in rows]
    rng = random.Random(a.seed); changed = 0
    for cid in sorted(by):
        idx = by[cid]
        total = sum(rows[i]['max_output_i'] for i in idx)
        draws = [rng.choice(base) ** a.gamma for _ in idx]
        caps = [min(a.cap, a.max_context - 256 - rows[i]['glm_tokens']) for i in idx]
        cap = max(2, min(caps))  # one cap per chain keeps the rule simple; every request stays inside its context room
        vals = apportion_exact(draws, total, cap)
        assert sum(vals) == total, (cid, sum(vals), total)
        for i, v in zip(idx, vals):
            assert v >= 1 and rows[i]['glm_tokens'] + v <= a.max_context - 256
            if rows[i]['max_output_i'] != v: changed += 1
            rows[i]['max_output_i'] = v
    for cid, c in chains.items():
        s = sum(r['max_output_i'] for r in rows if r['chain_id'] == cid)
        assert s == c['max_output_i_sum'], (cid, s, c['max_output_i_sum'])
    after = [r['max_output_i'] for r in rows]
    stats = lambda v: dict(p50=q(v, .5), p90=q(v, .9), p95=q(v, .95), p99=q(v, .99), max=max(v), mean=round(statistics.mean(v)))
    operation = {'kind': 'rebudget-output-tail-sensitivity',
           'rule': __doc__.strip().split('\n\n')[2],
           'params': dict(gamma=a.gamma, cap=a.cap, seed=a.seed, max_context=a.max_context),
           'public_requests_sha256': hashlib.sha256(open(os.path.join(a.public, 'requests.jsonl'), 'rb').read()).hexdigest(),
           'changed_requests': changed, 'output_budget_stats': {'before': stats(before), 'after': stats(after),
           'synthetic_after': stats([rows[i]['max_output_i'] for idx in by.values() for i in idx])}}
    man = publish(a.parent, a.out, rows, operation=operation, name=a.set, metadata_only=a.metadata_only)
    print('changed', changed, 'of', len(rows), '| before', stats(before), '| after', stats(after))
    print('status', man['status'], '| requests sha256', man['artifacts']['requests.jsonl'])

if __name__ == '__main__':
    main()
