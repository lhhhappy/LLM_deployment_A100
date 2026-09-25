#!/usr/bin/env python3
"""Scheduling view of bad requests in 071 (122 on) and 069 (122 off), N30 long-chain.

Read-only. Every number is derived from raw_*.jsonl (server timestamps) and the TP0
"Prefill batch"/"Decode batch" lines of server.log (1-second resolution).

Definitions (all measured fields, derived arithmetic labelled in outputs):
  recv   = t_recv_s (HTTP entry);  ex = t_exec_start_s (= forward_entry, first prefill batch);
  ft     = t_first_token_s (prefill finished);  q = queue_time_s (waiting queue -> first batch)
  unc    = prompt_tokens - cached_tokens (actual recompute);  exp = uncached_expected (frozen label)
  "long" = unc > 8192 (cannot finish in one 8192 chunk -> becomes THE chunked request)
  lane interval of a long request = [ex, ft]  (includes interleaved decode; not pure GPU time)
Gate membership and allowed counts come from the original harness / score_formal.
"""
import calendar, collections, csv, json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 's1-dev/harness'))
sys.path.insert(0, str(ROOT / 'scripts'))
from s1_common import in_ttft_gate  # noqa: E402
from score_formal import allowed_over  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNS = {
    '071': ROOT / 'evidence/L071-official_b_host64_full_n30_shortwarm/N30',
    '069': ROOT / 'evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30',
}
GATES = (('fast_intra', 3.0), ('overall_intra', 5.0), ('turn_start', 15.0), ('chain_start', 30.0))
EXPECT_OVER = {'071': {'fast_intra': 199, 'overall_intra': 147, 'turn_start': 3, 'chain_start': 30},
               '069': {'fast_intra': 222, 'overall_intra': 195, 'turn_start': 7, 'chain_start': 31}}
LONG = 8192
LOGRE = re.compile(r'^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) TP0\] (Prefill|Decode) batch, (.*)$')


def load(run):
    d = RUNS[run]
    raw = next(d.glob('raw_*.jsonl'))
    rows = [json.loads(s) for s in raw.read_text().splitlines() if s.strip()]
    ids = [r['req_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 5601, (run, len(ids))
    assert not any(r.get('error') for r in rows)
    t0 = min(r['client_dispatch_at_s'] for r in rows)
    for r in rows:
        r['recv'], r['ex'], r['ft'] = r['t_recv_s'], r['t_exec_start_s'], r['t_first_token_s']
        r['q'] = r['queue_time_s']
        r['qstart'] = r['ex'] - r['q']
        r['unc'] = r['prompt_tokens'] - r['cached_tokens']
        r['exp'] = r['uncached_expected']
        r['deficit'] = r['unc'] - r['exp']
        r['arr_min'] = (r['client_dispatch_at_s'] - t0) / 60
        r['gates'] = [g for g, _ in GATES if in_ttft_gate(r, g)]
        r['over'] = [g for g, lim in GATES if in_ttft_gate(r, g) and r['ttft_s'] > lim]
        assert abs((r['ft'] - r['recv']) - r['ttft_s']) < 1e-3, r['req_id']
    for g, lim in GATES:
        n = sum(g in r['gates'] for r in rows)
        k = sum(g in r['over'] for r in rows)
        assert k == EXPECT_OVER[run][g], (run, g, k)
    return {r['req_id']: r for r in rows}, t0


def load_log(run):
    """TP0 per-line samples: (epoch_s, kind, dict)."""
    out = []
    with open(RUNS[run] / 'server.log', errors='replace') as fh:
        for line in fh:
            m = LOGRE.match(line.rstrip('\n'))
            if not m:
                continue
            ts = calendar.timegm(tuple(int(x) for x in m.groups()[:6]))
            kv = {}
            for part in m.group(8).split(', '):
                if ':' in part:
                    k, v = part.split(':', 1)
                    try:
                        kv[k.strip().lstrip('#')] = float(v.strip())
                    except ValueError:
                        pass
            out.append((ts, m.group(7), kv))
    return out


def union_len(ivs):
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(ivs):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def clip(s, e, a, b):
    return max(s, a), min(e, b)


def analyze(run):
    R, t0 = load(run)
    log = load_log(run)
    rows = list(R.values())
    longs = [r for r in rows if r['unc'] > LONG]
    # lane concurrency check: how often do two long lane intervals overlap?
    ev = sorted([(r['ex'], 1) for r in longs] + [(r['ft'], -1) for r in longs])
    conc, t_prev, dur = 0, None, collections.Counter()
    for t, d in ev:
        if t_prev is not None:
            dur[conc] += t - t_prev
        conc += d
        t_prev = t
    lane_conc = {str(k): round(v, 1) for k, v in sorted(dur.items())}

    bad = [r for r in rows if r['over']]
    per = []
    for b in bad:
        a, z = b['qstart'], b['ex']
        W = max(z - a, 1e-9)
        lane_iv, lane_iv_later, occ = [], [], collections.Counter()
        for y in longs:
            if y is b:
                continue
            s, e = clip(y['ex'], y['ft'], a, z)
            if e > s:
                lane_iv.append((s, e))
                if y['recv'] > b['recv']:
                    lane_iv_later.append((s, e))
                occ[y['req_id']] += e - s
        over_short = over_long = over_long_tok = 0
        over_long_cachehit = 0
        for x in rows:
            if x is b or not (x['recv'] > b['recv'] and x['ex'] < b['ex']):
                continue
            if x['unc'] > LONG:
                over_long += 1
                over_long_tok += x['unc']
                over_long_cachehit += x['cached_tokens'] > b['cached_tokens']
            else:
                over_short += 1
        # work-in-window estimate (uniform spread of each request's unc over its [ex, ft])
        work = 0.0
        for y in rows:
            if y is b:
                continue
            s, e = clip(y['ex'], y['ft'], a, z)
            if e > s:
                work += y['unc'] * (e - s) / max(y['ft'] - y['ex'], 1e-3)
        smp = [kv for ts, k, kv in log if a - 1 <= ts <= z + 1]
        pre = [kv for ts, k, kv in log if a - 1 <= ts <= z + 1 and 'new-token' in kv]
        top = occ.most_common(1)
        topr = R[top[0][0]] if top else None
        per.append(dict(
            run=run, req_id=b['req_id'], over='|'.join(b['over']), arr_min=round(b['arr_min'], 2),
            ttft_s=round(b['ttft_s'], 2), queue_s=round(b['q'], 2),
            recv_to_exec_s=round(b['ex'] - b['recv'], 2), exec_to_first_s=round(b['ft'] - b['ex'], 2),
            unc=b['unc'], exp=b['exp'], cached=b['cached_tokens'], deficit=b['deficit'],
            lane_busy_frac=round(union_len(lane_iv) / W, 3),
            lane_busy_by_later_frac=round(union_len(lane_iv_later) / W, 3),
            n_long_occupants=len(occ),
            top_occ_id=top[0][0] if top else '', top_occ_s=round(top[0][1], 2) if top else 0,
            top_occ_unc=topr['unc'] if topr else '', top_occ_deficit=topr['deficit'] if topr else '',
            top_occ_later=(topr['recv'] > b['recv']) if topr else '',
            top_occ_gate='|'.join(topr['gates']) if topr else '',
            overtakers_short=over_short, overtakers_long=over_long, overtakers_long_unc=over_long_tok,
            overtakers_long_more_cached=over_long_cachehit,
            work_in_wait_tok_est=int(work), work_rate_tok_s_est=int(work / W),
            log_prefill_tok_in_wait=int(sum(kv['new-token'] for kv in pre)),
            max_token_usage=max((kv.get('full token usage', 0) for kv in smp), default=''),
            max_mamba_usage=max((kv.get('mamba usage', 0) for kv in smp), default=''),
            max_running=max((kv.get('running-req', 0) for kv in smp), default=''),
            max_queue=max((kv.get('queue-req', 0) for kv in smp), default=''),
        ))
    return R, t0, log, per, lane_conc


def main():
    res = {}
    allp = []
    for run in RUNS:
        R, t0, log, per, lane_conc = analyze(run)
        res[run] = dict(R=R, t0=t0, log=log, per=per, lane_conc=lane_conc)
        allp += per
    with open(OUT / 'bad-wait-sched.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(allp[0]))
        w.writeheader()
        w.writerows(allp)

    summ = {'note': 'measured from raw + TP0 log; *_est columns are estimates (uniform spread)'}
    for run, d in res.items():
        R = d['R']
        rows = list(R.values())
        s = {'lane_concurrency_seconds_long_intervals': d['lane_conc']}
        gates = {}
        for g, lim in GATES:
            n = sum(g in r['gates'] for r in rows)
            k = sum(g in r['over'] for r in rows)
            gates[g] = dict(n=n, over=k, allowed=allowed_over(n), slack=allowed_over(n) - k)
        s['gates'] = gates
        for g in ('chain_start', 'fast_intra', 'overall_intra', 'turn_start'):
            P = [p for p in d['per'] if g in p['over'].split('|')]
            if not P:
                continue
            def med(key):
                v = sorted(p[key] for p in P)
                return v[len(v) // 2]
            s[g] = dict(
                n_bad=len(P),
                median_lane_busy_frac=med('lane_busy_frac'),
                median_lane_busy_by_later_frac=med('lane_busy_by_later_frac'),
                n_lane_busy_ge_0_8=sum(p['lane_busy_frac'] >= .8 for p in P),
                n_lane_mostly_later=sum(p['lane_busy_by_later_frac'] >= .5 for p in P),
                n_with_long_overtaker=sum(p['overtakers_long'] > 0 for p in P),
                sum_overtakers_short=sum(p['overtakers_short'] for p in P),
                sum_overtakers_long=sum(p['overtakers_long'] for p in P),
                n_top_occ_is_cache_loss=sum(isinstance(p['top_occ_deficit'], int) and p['top_occ_deficit'] > 8192 for p in P),
                n_top_occ_later=sum(p['top_occ_later'] is True for p in P),
                max_token_usage_seen=max(p['max_token_usage'] for p in P if p['max_token_usage'] != ''),
                max_running_seen=max(p['max_running'] for p in P if p['max_running'] != ''),
                median_work_rate_tok_s_est=med('work_rate_tok_s_est'),
            )
        # arrival histogram for chain_start
        cs = [r for r in rows if 'chain_start' in r['gates']]
        bins = [0, 1, 2, 5, 10, 20, 40, 80, 1e9]
        hist = []
        for lo, hi in zip(bins, bins[1:]):
            sel = [r for r in cs if lo <= r['arr_min'] < hi]
            hist.append(dict(min_from=lo, min_to=hi if hi < 1e9 else 'end', n=len(sel),
                             over=sum('chain_start' in r['over'] for r in sel),
                             unc_sum=sum(r['unc'] for r in sel)))
        s['chain_start_arrival_hist'] = hist
        s['chain_start_bad_arrival_min'] = sorted(round(r['arr_min'], 2) for r in cs if 'chain_start' in r['over'])
        # near-miss arithmetic
        badcs = sorted((r for r in cs if 'chain_start' in r['over']), key=lambda r: r['ttft_s'])
        s['chain_start_bad_ttft_sorted'] = [round(r['ttft_s'], 1) for r in badcs]
        s['chain_start_30_45'] = sum(30 < r['ttft_s'] <= 45 for r in cs)
        s['chain_start_25_30'] = sum(25 < r['ttft_s'] <= 30 for r in cs)
        s['rescuable_by_order_only'] = sum((r['ft'] - r['ex']) < 29.0 for r in badcs)
        # cold-open wave: work arriving vs lane throughput in first minutes
        t0s = min(r['recv'] for r in rows)
        wave = []
        for m in (0.5, 1, 2, 3, 5):
            arrived = sum(r['unc'] for r in rows if r['recv'] < t0s + 60 * m)
            done = sum(kv['new-token'] for ts, k, kv in d['log'] if 'new-token' in kv and t0s - 1 <= ts < t0s + 60 * m)
            wave.append(dict(minutes=m, uncached_arrived_tok=arrived, prefill_log_tok_done=int(done),
                             backlog_tok=int(arrived - done)))
        s['cold_open_wave'] = wave
        # prefill throughput during busy seconds (log): tokens per second when >=1 prefill line
        per_sec = collections.Counter()
        for ts, k, kv in d['log']:
            if 'new-token' in kv and ts >= t0s - 1:
                per_sec[ts] += kv['new-token']
        mins = collections.defaultdict(int)
        for ts, v in per_sec.items():
            mins[int((ts - t0s) // 60)] += v
        s['prefill_tok_per_min_first10'] = [mins[i] for i in range(10)]
        s['prefill_tok_per_min_p50_steady_10_60'] = sorted(mins[i] for i in range(10, 60))[25]
        summ[run] = s

    # paired chain_start comparison
    A, B = res['071']['R'], res['069']['R']
    ids = sorted({r for r in A if 'chain_start' in A[r]['over']} | {r for r in B if 'chain_start' in B[r]['over']},
                 key=lambda i: A[i]['arr_min'])
    with open(OUT / 'chain-start-paired.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['req_id', 'arr_min_071', 'ttft_071', 'queue_071', 'exec_to_first_071', 'unc_071', 'exp',
                    'arr_min_069', 'ttft_069', 'queue_069', 'exec_to_first_069', 'unc_069', 'bad_071', 'bad_069'])
        for i in ids:
            a, b = A[i], B[i]
            w.writerow([i, round(a['arr_min'], 2), round(a['ttft_s'], 1), round(a['q'], 1), round(a['ft'] - a['ex'], 1), a['unc'], a['exp'],
                        round(b['arr_min'], 2), round(b['ttft_s'], 1), round(b['q'], 1), round(b['ft'] - b['ex'], 1), b['unc'],
                        a['ttft_s'] > 30, b['ttft_s'] > 30])
    summ['paired_chain_start_union'] = len(ids)
    summ['paired_both_bad'] = sum(A[i]['ttft_s'] > 30 and B[i]['ttft_s'] > 30 for i in ids)
    (OUT / 'sched-summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)[:6000])


if __name__ == '__main__':
    main()
