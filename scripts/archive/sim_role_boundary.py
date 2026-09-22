"""Non-oracle D1 check (v2: named runtime policies per Codex D1 §8.5).
Policies: stock | role_conservative (skip role when a branch track would compete) |
role_over_branch (role replaces branch in that forward; end kept via 2nd chunk) |
role_all (branch, role and end all kept; needs multiple chunks)

Original doc: Non-oracle D1 check (answers Codex F7). For every prompt, compute candidate checkpoint
positions from the CURRENT prompt only (last <|user|> token, last <|observation|> token), then
ask: does the next request's real token LCP reach that position (checkpoint <= LCP, same path)?
Then re-simulate uncached tokens with a one-track-per-extend stock model vs. stock + role boundary.
Runs on the GPU box CPU (tokenizer only); no serving."""
import sys, json, gzip, collections
sys.path.insert(0, 'harness')
from s1_common import Renderer
root = 'data/dev-combined-v1'
R = [json.loads(l) for l in open(f'{root}/requests.jsonl')]
B = {}
with gzip.open(f'{root}/bodies/dev-combined-v1.jsonl.gz', 'rt') as f:
    for l in f:
        b = json.loads(l); B[b['req_id']] = b
rend = Renderer('glm_tok'); tk = rend.tokenizer
USER = tk.convert_tokens_to_ids('<|user|>'); OBS = tk.convert_tokens_to_ids('<|observation|>')
byc = collections.defaultdict(list)
for r in R: byc[r['chain_id']].append(r)
A = 64
fl = lambda x: x // A * A
def q(xs, p): xs = sorted(xs); return xs[min(len(xs)-1, int(len(xs)*p))] if xs else None
POLICIES = ('stock', 'role_conservative', 'role_over_branch', 'role_all')
reach = collections.Counter(); res = {p: [] for p in POLICIES}; stats = {p: collections.Counter() for p in POLICIES}
for cid, rs in byc.items():
    rs.sort(key=lambda r: r['dispatch_offset_ms'])
    T = [tk.encode(rend.render(B['%s:%s:%s' % (r['pack'], r['view'], r['logical_call_id'])]),
                   add_special_tokens=False) for r in rs]
    for pol in POLICIES:
        store = [(0, None)]
        for i, (r, t) in enumerate(zip(rs, T)):
            hit = 0
            for L, j in store:
                if L > hit and (j is None or (len(t) >= L and T[j][L-1] == t[L-1] and T[j][:L] == t[:L])):
                    hit = L
            lcp = 0
            if i:
                p = T[i-1]; m = min(len(p), len(t))
                while lcp < m and p[lcp] == t[lcp]: lcp += 1
            res[pol].append((r, i, len(t) - hit, r['uncached_expected']))
            end = fl(len(t))
            branch = fl(lcp) if (lcp > hit and fl(lcp) > hit) else None
            cands = [k for k in range(hit, len(t)) if t[k] in (USER, OBS)]
            role = fl(cands[-1]) if cands else None
            if role is not None and not (hit < role < end): role = None
            pos = hit
            while pos + 8192 < len(t):
                pos += 8192; store.append((fl(pos), i))
            if pol == 'stock' or role is None:
                store.append((branch if branch is not None else end, i))
            elif pol == 'role_conservative':
                if branch is not None:
                    store.append((branch, i)); stats[pol]['skipped_branch_conflict'] += 1
                else:
                    store += [(role, i), (end, i)]; stats[pol]['taken'] += 1
            elif pol == 'role_over_branch':
                store += [(role, i), (end, i)]; stats[pol]['taken'] += 1
            elif pol == 'role_all':
                store += [(role, i), (end, i)] + ([(branch, i)] if branch is not None else []); stats[pol]['taken'] += 1
        # reach statistics (policy-independent)
    for i in range(len(rs) - 1):
        t, n = T[i], T[i+1]
        m = min(len(t), len(n)); lcp = 0
        while lcp < m and t[lcp] == n[lcp]: lcp += 1
        users = [k for k, x in enumerate(t) if x == USER]
        lu = users[-1] if users else -1
        reach['pairs'] += 1
        reach['next_lcp >= last<|user|>'] += lcp >= lu
        reach['next_lcp == last<|user|>'] += lcp == lu
        reach['next_lcp == prev_len'] += lcp == len(t)
print(dict(reach))
for pol in POLICIES:
    fast = [u for r, i, u, e in res[pol] if i > 0 and r['phase'] not in ('turn_start', 'context_reset') and e <= 4096]
    intra = [u for r, i, u, e in res[pol] if i > 0 and r['phase'] not in ('turn_start', 'context_reset')]
    print(f"{pol:18s} {dict(stats[pol])} fast_intra n={len(fast)} real-uncached p50={q(fast,.5)} p90={q(fast,.9)} p95={q(fast,.95)} max={max(fast)} | overall_intra p95={q(intra,.95)}")
