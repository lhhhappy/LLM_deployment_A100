#!/usr/bin/env python3
"""Opening wave (arrivals in first 30 s) of local N30 runs: how many tokens are unique if
simultaneous requests could share prefixes (trie size), vs what the engine actually prefilled.
LCP at message granularity incl. a system+tools level; tokens by char fraction (推算)."""
import gzip, json, glob, hashlib, os, sys
ROOT = '/workspace/Agentic_science_challenge'; OUT = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1] if len(sys.argv) > 1 else '071'
d = {'069': 'L069-official_b_pace_off_host64_full_n30_shortwarm', '071': 'L071-official_b_host64_full_n30_shortwarm'}[run]
rs = [json.loads(l) for l in open(glob.glob(f'{ROOT}/evidence/{d}/N30/raw_*.jsonl')[0])]
t0 = min(r['t_recv_s'] for r in rs)
wave = sorted([r for r in rs if r['t_recv_s'] - t0 < 30], key=lambda r: r['t_recv_s'])
ids = {r['req_id'] for r in wave}
bodies = {}
with gzip.open(f'{ROOT}/data/s1-dev-longchain/bodies/s1-dev-longchain.jsonl.gz', 'rt') as f:
    for l in f:
        if any(i in l[:200] for i in ids):
            b = json.loads(l)
            if b['req_id'] in ids: bodies[b['req_id']] = b
        if len(bodies) == len(ids): break
def seq(b):
    st = str(b['system']) + '\x00' + str(b['tools'])
    msgs = b['messages']
    if isinstance(msgs, str):
        try: msgs = json.loads(msgs)
        except Exception: msgs = eval(msgs)
    parts = [st] + [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in msgs]
    hs, cum, h, c = [], [], b'', 0
    for p in parts:
        h = hashlib.sha1(h + p.encode()).digest()[:8]; c += len(p); hs.append(h); cum.append(c)
    return hs, cum
S = {r['req_id']: seq(bodies[r['req_id']]) for r in wave}
uniq = 0; tot = 0; seen = []
for r in wave:
    hs, cum = S[r['req_id']]; pt = r['prompt_tokens']
    best = 0
    for (hs2, cum2) in seen:
        k = 0
        while k < min(len(hs), len(hs2)) and hs[k] == hs2[k]: k += 1
        best = max(best, k)
    lcp_tok = int(pt * cum[best - 1] / cum[-1]) if best else 0
    uniq += pt - lcp_tok; tot += pt; seen.append((hs, cum))
act = sum(r['prompt_tokens'] - r['cached_tokens'] for r in wave)
print(json.dumps(dict(run=run, wave_requests=len(wave), prompt_tokens=tot, engine_uncached=act,
      trie_unique_tokens_est=uniq, dedup_headroom_tokens_est=act - uniq), ensure_ascii=False))
