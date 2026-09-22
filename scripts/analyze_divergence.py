"""Render consecutive prompts of each chain, tokenize, locate the token LCP and print
what text sits at the divergence (prev tail that got replaced vs new continuation)."""
import sys, json, gzip, collections
sys.path.insert(0, 'harness')
from s1_common import Renderer
root = 'data/dev-combined-v1'
R = [json.loads(l) for l in open(f'{root}/requests.jsonl')]
B = {}
with gzip.open(f'{root}/bodies/dev-combined-v1.jsonl.gz', 'rt') as f:
    for l in f:
        b = json.loads(l)
        B[b.get('logical_call_id') or b.get('req_id')] = b
print('bodies', len(B), list(next(iter(B.values())).keys()))
rend = Renderer('glm_tok'); tk = rend.tokenizer
byc = collections.defaultdict(list)
for r in R: byc[r['chain_id']].append(r)
cache = {}
def toks(r):
    k = '%s:%s:%s' % (r['pack'], r['view'], r['logical_call_id'])
    if k not in cache:
        b = B[k]
        cache[k] = tk.encode(rend.render(b), add_special_tokens=False)
    return cache[k]
shown = 0; stats = collections.Counter(); prevtail = []
for cid, rs in byc.items():
    rs.sort(key=lambda r: r['dispatch_offset_ms'])
    for i in range(1, len(rs)):
        p, c = rs[i-1], rs[i]
        if c['phase'] != 'intra' or c['edge_subtype'] != 'reminder-replaced': continue
        a, b = toks(p), toks(c)
        n = 0
        m = min(len(a), len(b))
        while n < m and a[n] == b[n]: n += 1
        stats['n'] += 1; stats['lcp_match_frozen'] += (n == c['glm_lcp_with_prev'])
        prevtail.append(len(a) - n)
        if shown < 4:
            shown += 1
            print('=' * 100, '\n', cid, 'prev_len', len(a), 'cur_len', len(b), 'lcp', n, 'frozen_lcp', c['glm_lcp_with_prev'])
            print('--- common tail before divergence:\n', repr(tk.decode(a[max(0, n-60):n])))
            print('--- PREV after divergence (replaced):\n', repr(tk.decode(a[n:n+400])))
            print('--- CUR after divergence:\n', repr(tk.decode(b[n:n+300])))
print(stats)
