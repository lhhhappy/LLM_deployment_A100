#!/usr/bin/env python3
"""Steady-state chain-start reading of one run, optionally paired with a reference on the same request IDs.

The chain gate counts requests over 30 s, but locally the steady state rarely crosses it, so the judging quantity
for steady-state mechanisms is the distribution just under the line: how many chain-bucket requests (first request
of a chain, or a context reset; the harness's phase_gate) arriving after the opening land in 10-15 / 15-20 / 20-30 /
>30 s, and how long the 100k+ cold heads take after batch entry (t_exec_start_s -> t_first_token_s) versus how long
they waited. TPOT>0.10 counts and the four gates are printed alongside (harness definitions, same-ID when paired).

  python3 -B scripts/analysis/steady_chain.py <job-or-raw> [--ref <job-or-raw>] [--opening-s 60] [--giant 100000]

<job-or-raw> is a run directory name under evidence/ (L<job>/N*/raw_*.jsonl) or a raw file path. Measured values
only; a window run is a diagnostic, not a verdict.
"""
import argparse, glob, os, statistics, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 's1-dev/harness'))
import s1_common as sc
from miss_profile import load

def resolve(x):
    if os.path.isfile(x): return x
    for pat in (f'evidence/L{x}/N*/raw_*.jsonl', f'evidence/L{x}*/N*/raw_*.jsonl'):
        fs = sorted(glob.glob(os.path.join(ROOT, pat)))
        if fs: return fs[0]
    sys.exit(f'no raw for {x}')

def bucket(v):
    g = sc.phase_gate(v)
    return 'chain' if g == sc.P95_CHAIN_START_S else 'turn' if g == sc.P95_TURN_START_S else 'intra'

def gates(D, ids):
    c = collections.Counter()
    for k in ids:
        v = D[k]; b = bucket(v); t = v.get('ttft_s') or 0
        if b == 'chain' and t > 30: c['chain'] += 1
        elif b == 'turn' and t > 15: c['turn'] += 1
        elif b == 'intra' and t > 5: c['overall'] += 1
        if sc.in_ttft_gate(v, 'fast_intra') and t > 3: c['fast'] += 1
    tp = [D[k]['tpot_s'] for k in ids if D[k].get('tpot_s') is not None]
    return c, sum(x > 0.10 for x in tp), (1000 * statistics.mean(tp) if tp else 0.0)

def tbin(t): return '>30' if t > 30 else '20-30' if t > 20 else '15-20' if t > 15 else '10-15' if t > 10 else '<=10'

def reading(D, ids, opening_s, giant):
    t0 = min(v['t_recv_s'] for v in D.values())
    steady = [D[k] for k in ids if bucket(D[k]) == 'chain' and D[k]['t_recv_s'] - t0 >= opening_s]
    opening = [D[k] for k in ids if bucket(D[k]) == 'chain' and D[k]['t_recv_s'] - t0 < opening_s]
    bins = collections.Counter(tbin(v.get('ttft_s') or 0) for v in steady)
    giants = []
    for v in sorted(steady, key=lambda v: -(v['prompt_tokens'] - (v.get('cached_tokens') or 0))):
        unc = v['prompt_tokens'] - (v.get('cached_tokens') or 0)
        if unc < giant: break
        wait = (v.get('t_exec_start_s') or 0) - v['t_recv_s']; ex = (v.get('t_first_token_s') or 0) - (v.get('t_exec_start_s') or 0)
        giants.append((v.get('req_id'), v['t_recv_s'] - t0, unc, v.get('ttft_s') or 0, wait, ex))
    return opening, steady, bins, giants

def organizer_split(D, ids, opening_s):
    """Chain-bucket items split by the organizer's phase label of the head. The organizer's chains are session
    segments cut at system/tool changes, so a chain head is a session_start (p50 18.6k tokens) or a cold
    mid-session row (intra/turn_start, p90 88k, up to 257k); the second group is where the misses are. The
    session_start-plus-reset subset is printed as the light part of the bucket, not as the online population."""
    items = [(rid, D[rid]) for rid in ids if bucket(D[rid]) == 'chain']
    def kind(v):
        return ('head:' + str(v.get('phase'))) if v.get('idx_in_chain') == 0 else 'reset'
    tot = {}; miss = {}
    for _, v in items:
        k = kind(v); tot[k] = tot.get(k, 0) + 1
        if (v.get('ttft_s') or 0) > 30: miss[k] = miss.get(k, 0) + 1
    org = [(rid, v) for rid, v in items if v.get('phase') in ('session_start', 'context_reset')]
    t0 = min((v.get('client_dispatch_at_s') or 0) for v in D.values())
    def stats(grp):
        if not grp: return 'n=0'
        tt = sorted((v.get('ttft_s') or 0) for _, v in grp)
        w = sorted(((v.get('t_exec_start_s') or 0) - (v.get('t_recv_s') or 0)) for _, v in grp)
        e = sorted(((v.get('t_first_token_s') or 0) - (v.get('t_exec_start_s') or 0)) for _, v in grp)
        q = lambda a, x: a[min(len(a) - 1, int(x * len(a)))]
        return (f'n={len(grp)} over30={sum(1 for t in tt if t > 30)} ttft p50/p95={q(tt, .5):.1f}/{q(tt, .95):.1f}'
                f' wait p95={q(w, .95):.1f} exec p95={q(e, .95):.1f}')
    op = [(r, v) for r, v in org if (v.get('client_dispatch_at_s') or 0) - t0 < opening_s]
    st = [(r, v) for r, v in org if (v.get('client_dispatch_at_s') or 0) - t0 >= opening_s]
    return tot, miss, stats(op), stats(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run'); ap.add_argument('--ref'); ap.add_argument('--opening-s', type=float, default=60.0)
    ap.add_argument('--giant', type=int, default=100000)
    a = ap.parse_args()
    A = load(resolve(a.run)); B = load(resolve(a.ref)) if a.ref else None
    ids = sorted(set(A) & set(B)) if B else sorted(A)
    ca, ta, ma = gates(A, ids)
    label = f'{a.run} vs {a.ref} (common {len(ids)})' if B else f'{a.run} ({len(ids)} requests)'
    print(f'== {label}')
    if B:
        cb, tb, mb = gates(B, ids)
        print(f'   gates ref->run: chain {cb["chain"]}->{ca["chain"]} turn {cb["turn"]}->{ca["turn"]} overall {cb["overall"]}->{ca["overall"]} fast {cb["fast"]}->{ca["fast"]} | TPOT>0.10 {tb}->{ta} mean {mb:.1f}->{ma:.1f} ms')
    else:
        print(f'   gates: chain {ca["chain"]} turn {ca["turn"]} overall {ca["overall"]} fast {ca["fast"]} | TPOT>0.10 {ta} mean {ma:.1f} ms')
    oa, sa, ba, ga = reading(A, ids, a.opening_s, a.giant)
    order = ['<=10', '10-15', '15-20', '20-30', '>30']
    line = lambda b: ' '.join(f'{k}:{b.get(k, 0)}' for k in order)
    print(f'   run  opening chain items {len(oa)} (over 30 s: {sum(1 for v in oa if (v.get("ttft_s") or 0) > 30)}) | steady chain items {len(sa)} TTFT bins {line(ba)}')
    if B:
        ob, sb, bb, gb = reading(B, ids, a.opening_s, a.giant)
        print(f'   ref  opening chain items {len(ob)} (over 30 s: {sum(1 for v in ob if (v.get("ttft_s") or 0) > 30)}) | steady chain items {len(sb)} TTFT bins {line(bb)}')
        gref = {g[0]: g for g in gb}
        print(f'   steady cold heads >= {a.giant} uncached (run vs ref): ttft = wait + after-entry')
        for rid, t, unc, ttft, wait, ex in ga:
            r = gref.get(rid)
            rs = f' | ref ttft {r[3]:5.1f} = {r[4]:4.1f} + {r[5]:5.1f}' if r else ''
            print(f'     t={t:6.0f}s unc={unc:7d} ttft {ttft:5.1f} = {wait:4.1f} + {ex:5.1f}{rs}')
    else:
        print(f'   steady cold heads >= {a.giant} uncached: ttft = wait + after-entry')
        for rid, t, unc, ttft, wait, ex in ga: print(f'     t={t:6.0f}s unc={unc:7d} ttft {ttft:5.1f} = {wait:4.1f} + {ex:5.1f}')

    for tag, D_ in (('run', A),) + ((('ref', B),) if B else ()):
        tot, miss, sop, sst = organizer_split(D_, ids, a.opening_s)
        print(f'   {tag}  chain misses by head kind {miss} of {tot}')
        print(f'   {tag}  light subset (session_start heads + resets): opening {sop} | steady {sst}')

if __name__ == '__main__':
    main()
