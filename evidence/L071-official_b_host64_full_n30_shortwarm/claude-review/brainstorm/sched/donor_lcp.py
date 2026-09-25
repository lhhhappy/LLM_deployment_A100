#!/usr/bin/env python3
"""Who could have supplied the prefix of each chain_start request, and was it ready in time?

ESTIMATE, not a token-exact LCP: no tokenizer package is installed on this CPU box, so each
frozen body is serialized (system, tools, messages in order) and prefix-hashed every CK chars.
The shared-prefix length between two requests is the last matching checkpoint; it is converted to
tokens with the target's own tokens/char ratio. Calibrated below against the two token-exact LCPs
in final-analysis/summary.json (lc295<-QSdTYVbowNG 252,115; z9 llm:2<-llm:1 222,738).

For each chain_start request B in a run:
  lcp_any    = longest shared prefix with ANY other request of the replay (any time)
  lcp_ready  = longest shared prefix with a request whose prefill finished before B's first batch
  lcp_later  = longest shared prefix with a request that was received before B's first batch
               but whose prefill had NOT finished by then (concurrent sibling)
This bounds reuse that timing could have given; it does not prove a valid hybrid checkpoint existed.
"""
import collections, csv, gzip, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 's1-dev/harness'))
from s1_common import in_ttft_gate  # noqa: E402

CK = 1024
RUNS = {
    '071': ROOT / 'evidence/L071-official_b_host64_full_n30_shortwarm/N30',
    '069': ROOT / 'evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30',
}


def serialize(b):
    parts = [b.get('system') or '', json.dumps(b.get('tools'), sort_keys=True, ensure_ascii=False)]
    for m in b.get('messages') or []:
        parts.append(json.dumps(m, sort_keys=True, ensure_ascii=False))
    return '\x00'.join(parts)


def fingerprints(s):
    h = hashlib.blake2b(digest_size=8)
    out = []
    enc = s.encode('utf-8', 'surrogatepass')
    # checkpoint on character boundaries measured in bytes of the utf-8 stream (close enough)
    for i in range(CK, len(enc) + 1, CK):
        h.update(enc[i - CK:i])
        out.append(int.from_bytes(h.copy().digest(), 'little'))
    return out, len(enc)


def main():
    raws = {}
    for run, d in RUNS.items():
        rows = [json.loads(s) for s in next(d.glob('raw_*.jsonl')).read_text().splitlines() if s.strip()]
        raws[run] = {r['req_id']: r for r in rows}
    ids = set(raws['071'])
    assert ids == set(raws['069']) and len(ids) == 5601
    cache = Path('/tmp/claude-0/-workspace-Agentic-science-challenge/e4faf351-6a00-4cf3-89bb-765b4c17abe2/scratchpad/fp_cache.json.gz')  # large, kept out of evidence
    if cache.exists():
        fp = json.loads(gzip.open(cache, 'rt').read())
    else:
        fp = {}
        with gzip.open(ROOT / 'data/s1-dev-longchain/bodies/s1-dev-longchain.jsonl.gz', 'rt', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                b = json.loads(line)
                rid = b.get('req_id')
                if rid in ids and rid not in fp:
                    f, n = fingerprints(serialize(b))
                    fp[rid] = [f, n]
        assert set(fp) == ids, len(fp)
        with gzip.open(cache, 'wt') as fh:
            fh.write(json.dumps(fp))
    index = collections.defaultdict(list)
    for rid, (f, n) in fp.items():
        for k, d in enumerate(f):
            index[(k, d)].append(rid)

    def tok(rid, k_matched, run):
        f, n = fp[rid]
        return int(round(k_matched * CK * raws[run][rid]['prompt_tokens'] / max(n, 1)))

    def best(rid, run, cond):
        f, _ = fp[rid]
        for k in range(len(f) - 1, -1, -1):
            cands = [x for x in index[(k, f[k])] if x != rid and cond(raws[run][x])]
            if cands:
                return k + 1, cands
        return 0, []

    # calibration against token-exact LCPs
    calib = []
    for tgt, src, exact in (('scimaster:canon:lc_20260924_295:llm:0001', 'scimaster:canon:QSdTYVbowNG_k_R8lOG7T:llm:1', 252115),
                            ('scimaster:canon:z9-JbTlHSeEgAkoO6zYyY:llm:2', 'scimaster:canon:z9-JbTlHSeEgAkoO6zYyY:llm:1', 222738)):
        a, b = fp[tgt][0], fp[src][0]
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        calib.append(dict(target=tgt, source=src, exact_tokens=exact, est_tokens=tok(tgt, k, '071')))

    out = []
    for run in RUNS:
        R = raws[run]
        for rid, B in R.items():
            if not in_ttft_gate(B, 'chain_start'):
                continue
            ex = B['t_exec_start_s']
            k_any, c_any = best(rid, run, lambda x: True)
            k_rdy, c_rdy = best(rid, run, lambda x: x['t_first_token_s'] < ex)
            k_cur, c_cur = best(rid, run, lambda x: x['t_recv_s'] < ex <= x['t_first_token_s'])
            # earliest-served request that covers the any-time maximum
            first_any = min(c_any, key=lambda x: R[x]['t_first_token_s']) if c_any else ''
            out.append(dict(
                run=run, req_id=rid, bad=B['ttft_s'] > 30, ttft_s=round(B['ttft_s'], 2),
                arr_min=round((B['client_dispatch_at_s'] - min(r['client_dispatch_at_s'] for r in R.values())) / 60, 2) if False else '',
                prompt=B['prompt_tokens'], cached=B['cached_tokens'], unc=B['prompt_tokens'] - B['cached_tokens'],
                exp=B['uncached_expected'],
                lcp_any_est=tok(rid, k_any, run), lcp_ready_est=tok(rid, k_rdy, run), lcp_concurrent_est=tok(rid, k_cur, run),
                donor_any=first_any,
                donor_any_ft_minus_B_ex_s=round(R[first_any]['t_first_token_s'] - ex, 2) if first_any else '',
                donor_any_recv_minus_B_recv_s=round(R[first_any]['t_recv_s'] - B['t_recv_s'], 2) if first_any else '',
                donor_any_unc=(R[first_any]['prompt_tokens'] - R[first_any]['cached_tokens']) if first_any else '',
                donor_concurrent=c_cur[0] if c_cur else '',
            ))
    t0 = {run: min(r['client_dispatch_at_s'] for r in raws[run].values()) for run in RUNS}
    for o in out:
        o['arr_min'] = round((raws[o['run']][o['req_id']]['client_dispatch_at_s'] - t0[o['run']]) / 60, 2)
    with open(HERE / 'chain-start-donors.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    summ = {'label': 'ESTIMATE (char-serialized prefix hash, CK=%d chars)' % CK, 'calibration': calib,
            'note': 'lcp_any/any_gap include the request\'s OWN later chain successors (which contain its prompt); they are NOT donors. Use lcp_ready / lcp_concurrent.'}
    for run in RUNS:
        O = [o for o in out if o['run'] == run]
        for grp, sel in (('all', O), ('bad', [o for o in O if o['bad']]),
                         ('first_min', [o for o in O if o['arr_min'] < 1])):
            def s(key):
                return sum(o[key] for o in sel)
            summ['%s_%s' % (run, grp)] = dict(
                n=len(sel), n_bad=sum(o['bad'] for o in sel), unc_sum=s('unc'), cached_sum=s('cached'),
                # tokens a ready donor could have supplied beyond what was actually cached
                ready_gap_sum=sum(max(0, o['lcp_ready_est'] - o['cached']) for o in sel),
                # tokens that only a concurrent (not yet finished) sibling could have supplied
                concurrent_only_sum=sum(max(0, o['lcp_concurrent_est'] - max(o['lcp_ready_est'], o['cached'])) for o in sel),
                n_concurrent_only_ge_16k=sum((o['lcp_concurrent_est'] - max(o['lcp_ready_est'], o['cached'])) >= 16384 for o in sel),
                n_ready_gap_ge_16k=sum((o['lcp_ready_est'] - o['cached']) >= 16384 for o in sel),
                any_gap_sum=sum(max(0, o['lcp_any_est'] - o['cached']) for o in sel),
            )
    (HERE / 'chain-start-donors-summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
