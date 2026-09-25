#!/usr/bin/env python3
"""Split each bad request's queue wait into lane-busy / KV-full / other seconds (071 and 069).

Queue window = [t_exec_start - queue_time, t_exec_start] (scheduler waiting queue -> first batch).
Sampled every 0.1 s:
  lane : some OTHER request with unc > 8192 is inside its [exec_start, first_token] interval
         (i.e. the single chunked-prefill lane is taken; measured from raw timestamps)
  kv   : lane not taken and the latest TP0 log line (<= t) shows full token usage >= 0.90
         (locked KV = capacity - available - evictable; source pool_stats_observer.py@759a6eb)
  other: neither (short work, decode pacing, host restore, eligibility rules ... not separable here)
This is a timing co-occurrence split, NOT proof that the lane or KV was the refusal reason.
"""
import bisect, csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sched_wait as SW  # noqa: E402

KV_FULL = 0.90


def main():
    rows_out, summ = [], {'label': 'measured timestamps; class split is co-occurrence, not causal',
                          'kv_full_threshold': KV_FULL}
    for run in SW.RUNS:
        R, t0 = SW.load(run)
        log = SW.load_log(run)
        ts_u = [(ts, kv['full token usage']) for ts, k, kv in log if 'full token usage' in kv]
        # log stamps are whole seconds (floor); treat the sample as valid from ts to ts+1
        tsl = [t for t, _ in ts_u]
        longs = sorted((r['ex'], r['ft'], r['req_id']) for r in R.values() if r['unc'] > SW.LONG)
        starts = [s for s, _, _ in longs]

        def lane_busy(t, me):
            i = bisect.bisect_right(starts, t)
            for j in range(max(0, i - 3), i):
                s, e, rid = longs[j]
                if s <= t < e and rid != me:
                    return True
            return False

        def usage(t):
            i = bisect.bisect_right(tsl, t) - 1
            return ts_u[i][1] if i >= 0 else 0.0

        run_rows = []
        for r in R.values():
            if not r['over']:
                continue
            a, z = r['qstart'], r['ex']
            n = max(1, int((z - a) / 0.1))
            lane = kv = other = 0
            for i in range(n):
                t = a + (i + .5) * (z - a) / n
                if lane_busy(t, r['req_id']):
                    lane += 1
                elif usage(t) >= KV_FULL:
                    kv += 1
                else:
                    other += 1
            dt = (z - a) / n
            cls = max((('lane', lane), ('kv', kv), ('other', other)), key=lambda x: x[1])[0]
            row = dict(run=run, req_id=r['req_id'], over='|'.join(r['over']), arr_min=round(r['arr_min'], 2),
                       ttft_s=round(r['ttft_s'], 2), queue_s=round(r['q'], 2), exec_to_first_s=round(r['ft'] - r['ex'], 2),
                       unc=r['unc'], exp=r['exp'], lane_s=round(lane * dt, 2), kvfull_s=round(kv * dt, 2),
                       other_s=round(other * dt, 2), dominant=cls,
                       need_s=round(r['ttft_s'] - max(l for g, l in SW.GATES if g in r['over']), 2))
            run_rows.append(row)
        rows_out += run_rows
        s = {}
        for g, lim in SW.GATES:
            P = [x for x in run_rows if g in x['over'].split('|')]
            if not P:
                continue
            tot = sum(x['queue_s'] for x in P)
            s[g] = dict(n_bad=len(P), queue_s_sum=round(tot, 1),
                        lane_s_sum=round(sum(x['lane_s'] for x in P), 1),
                        kvfull_s_sum=round(sum(x['kvfull_s'] for x in P), 1),
                        other_s_sum=round(sum(x['other_s'] for x in P), 1),
                        n_dominant={c: sum(x['dominant'] == c for x in P) for c in ('lane', 'kv', 'other')},
                        n_dominant_first_min={c: sum(x['dominant'] == c and x['arr_min'] < 1 for x in P) for c in ('lane', 'kv', 'other')})
        # whole-run share of time with KV >= threshold (measurement window only)
        tt = [t for t, u in ts_u if t >= t0]
        span = tt[-1] - tt[0]
        full = sum(1 for (t, u), (t2, _) in zip(ts_u, ts_u[1:]) if t >= t0 and u >= KV_FULL)
        s['log_lines_with_usage_ge_0.90_frac'] = round(full / max(1, len([1 for t in tt])), 3)
        summ[run] = s
    with open(HERE / 'wait-class.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    (HERE / 'wait-class-summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
