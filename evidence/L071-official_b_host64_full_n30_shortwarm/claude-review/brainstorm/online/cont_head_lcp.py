#!/usr/bin/env python3
"""Other-chain donors only. For every idx0 chain head that is not cold (cont_head: append-only / compact-rebuild edge),
find the best message-level shared prefix with ANY other request body in the local long-chain set.
Token LCP is estimated by char fraction of the rendered JSON (label: 推算).
Output: cont_head_lcp.csv (one row per cont_head)."""
import gzip, json, hashlib, csv, os, collections
ROOT = '/workspace/Agentic_science_challenge'
OUT = os.path.dirname(os.path.abspath(__file__))
meta = {}
for l in open(f'{ROOT}/data/s1-dev-longchain/requests.jsonl'):
    r = json.loads(l); meta[r['pack'] + ':canon:' + r['logical_call_id']] = r
# idx within chain by dispatch order
bychain = collections.defaultdict(list)
for k, r in meta.items(): bychain[r['chain_id']].append(k)
idx = {}
for c, ks in bychain.items():
    ks.sort(key=lambda k: meta[k]['dispatch_offset_ms'] or 0)
    for i, k in enumerate(ks): idx[k] = i
heads = {k for k, r in meta.items() if idx[k] == 0 and r['edge_type'] not in ('chain-head', 'system-tools-changed')}
print('cont_heads', len(heads))

def chain_hashes(d):
    st = hashlib.sha1((str(d['system']) + '\x00' + str(d['tools'])).encode()).digest()[:8]
    msgs = d['messages']
    if isinstance(msgs, str):
        try: msgs = json.loads(msgs)
        except Exception: msgs = eval(msgs)
    h = st; out = []; cum = len(str(d['system'])) + len(str(d['tools'])); cums = []
    for m in msgs:
        s = json.dumps(m, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha1(h + s.encode()).digest()[:8]
        cum += len(s); out.append(h); cums.append(cum)
    return st, out, cums

head_chain = {}
# pass 1: head chains
with gzip.open(f'{ROOT}/data/s1-dev-longchain/bodies/s1-dev-longchain.jsonl.gz', 'rt') as f:
    for l in f:
        d = json.loads(l)
        if d['req_id'] in heads:
            head_chain[d['req_id']] = chain_hashes(d)
lookup = {}
for k, (st, hs, cums) in head_chain.items():
    for i, h in enumerate(hs): lookup.setdefault(h, []).append((k, i))
best = {k: (0, None) for k in heads}   # (n_msgs matched, req_id)
with gzip.open(f'{ROOT}/data/s1-dev-longchain/bodies/s1-dev-longchain.jsonl.gz', 'rt') as f:
    for l in f:
        d = json.loads(l); rid = d['req_id']
        st, hs, cums = chain_hashes(d)
        rc = meta[rid]['chain_id'] if rid in meta else None
        for i, h in enumerate(hs):
            for (k, j) in lookup.get(h, ()):
                if k == rid or meta[k]['chain_id'] == rc: continue   # only other chains can donate
                if j + 1 > best[k][0]: best[k] = (j + 1, rid)
rows = []
for k in sorted(heads):
    st, hs, cums = head_chain[k]
    n, rid = best[k]
    frac = (cums[n - 1] / cums[-1]) if n > 0 else 0.0
    m = meta[k]; pt = m['glm_tokens']
    src = meta.get(rid) if rid else None
    rows.append(dict(req_id=k, chain_id=m['chain_id'], prompt_tokens=pt, uncached_expected=m['uncached_expected'],
        msgs=len(hs), best_match_msgs=n, best_match_req=rid, best_match_chain=(src or {}).get('chain_id'),
        best_match_idx=idx.get(rid) if rid else None,
        est_lcp_tokens=int(pt * frac), est_min_uncached=pt - int(pt * frac)))
with open(f'{OUT}/cont_head_lcp.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import statistics
mu = [r['est_min_uncached'] for r in rows]
print('est minimal achievable uncached p50', statistics.median(mu), 'sum', sum(mu), 'vs expected sum', sum(r['uncached_expected'] for r in rows))
print('n with est_min_uncached>16k', sum(1 for x in mu if x > 16384))
