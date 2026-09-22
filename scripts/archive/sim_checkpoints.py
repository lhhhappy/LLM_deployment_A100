"""Estimate real uncached tokens per request under different KDA-state checkpoint policies
(no eviction assumed). Uses frozen glm_tokens / glm_lcp_with_prev only."""
import json, collections, sys
R = [json.loads(l) for l in open(sys.argv[1])]
byc = collections.defaultdict(list)
for r in R: byc[r['chain_id']].append(r)
A = 64          # mamba cache chunk alignment
def fl(x): return x // A * A
def simulate(policy, chunk=8192):
    out = []
    for cid, rs in byc.items():
        rs.sort(key=lambda r: r['dispatch_offset_ms'])
        ckpts = {0}
        for i, r in enumerate(rs):
            L = r['glm_tokens']; lcp = r['glm_lcp_with_prev'] if i else 0
            hit = max(c for c in ckpts if c <= lcp)
            out.append((r, L - hit))
            # checkpoints produced while prefilling [hit, L)
            pos = hit
            while pos + chunk < L:            # chunked-prefill chunk ends
                pos += chunk; ckpts.add(fl(pos))
            ckpts.add(fl(L))                  # end of prompt (aligned)
            if policy in ('branch', 'boundary') and lcp > hit:
                ckpts.add(fl(lcp))            # branching point (engine tracks fork)
            if policy == 'boundary':
                # next turn's LCP = start of trailing reminder; model as a checkpoint at
                # the NEXT request's lcp if it lies inside this prompt (role-boundary oracle)
                if i + 1 < len(rs):
                    nl = rs[i+1]['glm_lcp_with_prev']
                    if hit < nl <= L: ckpts.add(fl(nl))
    return out
def q(xs, p): xs = sorted(xs); return xs[min(len(xs)-1, int(len(xs)*p))]
for pol in ('end_only', 'branch', 'boundary'):
    o = simulate(pol)
    intra = [(r, u) for r, u in o if r['phase'] == 'intra' and r['chain_index'] is not None]
    fast = [(r, u) for r, u in intra if r['uncached_expected'] <= 4096]
    exp = sum(r['uncached_expected'] for r, _ in o); act = sum(u for _, u in o)
    print(f"{pol:9s} total uncached actual/expected = {act/exp:5.2f}x | "
          f"fast_intra n={len(fast)} actual-uncached p50={q([u for _,u in fast],.5)} p90={q([u for _,u in fast],.9)} "
          f"p95={q([u for _,u in fast],.95)} frac>8k={sum(u>8192 for _,u in fast)/len(fast):.2f} | "
          f"all intra p95={q([u for _,u in intra],.95)}")
