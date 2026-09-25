#!/usr/bin/env python3
"""ESTIMATE for every request: longest shared prompt prefix with a request whose prefill had
finished before this request's first batch (lcp_ready). Same char-serialized prefix hash as
donor_lcp.py (calibrated there to within 0.1% on two token-exact pairs).

  ready_gap = max(0, lcp_ready - cached_tokens): recompute that an earlier-finished request could
              have supplied (cache capacity / checkpoint / eviction domain), upper bound;
  cold_in_replay = unc - ready_gap: recompute no earlier-finished request could have supplied.
Then: were the long requests that occupied the prefill lane during bad requests' waits
cold-in-replay or cache-loss recomputes?
"""
import collections, csv, gzip, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import donor_lcp as D  # noqa: E402
import sched_wait as SW  # noqa: E402

CACHE = Path('/tmp/claude-0/-workspace-Agentic-science-challenge/e4faf351-6a00-4cf3-89bb-765b4c17abe2/scratchpad/fp_cache.json.gz')


def main():
    fp = json.loads(gzip.open(CACHE, 'rt').read())
    index = collections.defaultdict(list)
    for rid, (f, n) in fp.items():
        for k, d in enumerate(f):
            index[(k, d)].append(rid)
    out, summ = [], {'label': 'ESTIMATE (prefix hash); ready = donor prefill finished before first batch'}
    for run in SW.RUNS:
        R, t0 = SW.load(run)
        # sort donors by finish so the time filter is a bisect-free comparison
        lr = {}
        for rid, r in R.items():
            f, n = fp[rid]
            ex = r['ex']
            k_best = 0
            for k in range(len(f) - 1, -1, -1):
                if any(x != rid and R[x]['ft'] < ex for x in index[(k, f[k])]):
                    k_best = k + 1
                    break
            lcp = int(round(k_best * D.CK * r['prompt_tokens'] / max(n, 1)))
            lcp = min(lcp, r['prompt_tokens'])
            gap = max(0, lcp - r['cached_tokens'])
            gap = min(gap, r['unc'])
            lr[rid] = (lcp, gap)
            if gap >= 1024:  # keep the CSV small; totals below use every request
                out.append(dict(run=run, req_id=rid, unc=r['unc'], cached=r['cached_tokens'],
                                lcp_ready_est=lcp, ready_gap_est=gap))
        unc = sum(r['unc'] for r in R.values())
        gap = sum(v[1] for v in lr.values())
        by_gate = {}
        for g, _ in SW.GATES:
            sel = [rid for rid, r in R.items() if g in r['gates']]
            by_gate[g] = dict(unc=sum(R[x]['unc'] for x in sel), ready_gap=sum(lr[x][1] for x in sel))
        # lane occupants of bad chain_start waits (from bad-wait-sched.csv)
        occ = []
        for row in csv.DictReader(open(HERE / 'bad-wait-sched.csv')):
            if row['run'] == run and 'chain_start' in row['over'] and row['top_occ_id']:
                o = row['top_occ_id']
                occ.append(dict(bad=row['req_id'], occ=o, occ_unc=R[o]['unc'], occ_ready_gap=lr[o][1],
                                occ_gate='|'.join(R[o]['gates'])))
        # all long-lane seconds split by occupant type (whole run)
        lane_cold = lane_loss = 0.0
        for r in R.values():
            if r['unc'] > SW.LONG:
                dur = r['ft'] - r['ex']
                frac = lr[r['req_id']][1] / max(r['unc'], 1)
                lane_loss += dur * frac
                lane_cold += dur * (1 - frac)
        summ[run] = dict(total_unc=unc, ready_gap_total=gap, ready_gap_frac=round(gap / unc, 3), by_gate=by_gate,
                         chain_bad_top_occupants=occ,
                         n_top_occ_ready_gap_ge_half=sum(o['occ_ready_gap'] >= o['occ_unc'] / 2 for o in occ),
                         long_lane_seconds_cold=round(lane_cold, 1), long_lane_seconds_cacheloss=round(lane_loss, 1))
    with open(HERE / 'ready-donor-all.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (HERE / 'ready-donor-summary.json').write_text(json.dumps(summ, indent=1))
    for run in SW.RUNS:
        s = dict(summ[run])
        s.pop('chain_bad_top_occupants')
        print(run, json.dumps(s))
    for o in summ['071']['chain_bad_top_occupants']:
        print(o)


if __name__ == '__main__':
    main()
