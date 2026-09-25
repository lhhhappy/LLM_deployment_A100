#!/usr/bin/env python3
"""OFFLINE WHAT-IF MODEL (not a measurement): single prefill lane replay with the run's own
capacity trace, to rank ordering policies. Results are 推算 and only as good as the calibration
printed first (policy OBS must reproduce the measured run before other policies mean anything).

Assumptions (all stated in the output JSON):
  A1 open loop: every request arrives at its measured t_recv_s of that run; closed-loop feedback
     (earlier finish -> earlier next request) is ignored.
  A2 work = measured recompute tokens (prompt - cached) of that run; reordering does not change
     cache hits (the replay-true donor analysis says bad chain_start work is mostly cold anyway).
  A3 capacity: in each wall-clock second the lane can process exactly the prefill tokens the real
     run logged in that second (TP0 '#new-token' sum). Decode share / TPOT therefore unchanged by
     construction; the model cannot say anything about TPOT.
     Where the real run had an idle lane but the model has pending work, capacity is still capped
     at the logged amount (conservative for reordering).
  A4 lane rule: at most one request with remaining work > CHUNK may be 'in progress' (the single
     chunked request of SGLang); once started it keeps the lane until done (no preemption).
     Requests whose remaining work <= leftover budget in the tick may finish beside it.
  A5 KV capacity, request-slot limits, host restore and eligibility rules are NOT modelled.
  A6 TTFT_model = prefill completion - t_recv + POST (POST = 0.25 s first-token overhead).
Gate limits/membership from the harness; allowed counts from score_formal.allowed_over.
"""
import calendar, collections, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sched_wait as SW  # noqa: E402

CHUNK = 8192
TICK = 0.25
POST = 0.25
LIM = dict(SW.GATES)


def capacity_trace(log, t_start):
    per_sec = collections.Counter()
    for ts, k, kv in log:
        if 'new-token' in kv and ts >= t_start - 1:
            per_sec[int(ts)] += kv['new-token']
    return per_sec


def own_limit(r):
    lims = [LIM[g] for g in r['gates']]
    return min(lims) if lims else 5.0


def visible_limit(r):
    """Deadline a server could compute itself (no frozen phase labels): small recompute -> 3 s;
    little of the prompt already cached -> treat as a new conversation (30 s); else 5 s."""
    if r['unc'] <= 4096:
        return 3.0
    if r['cached_tokens'] < 0.5 * r['prompt_tokens']:
        return 30.0
    return 5.0


def simulate(R, cap, policy, rate_est=10000.0, bypass_short=False):
    reqs = sorted(R.values(), key=lambda r: r['recv'])
    t = reqs[0]['recv']
    t_end = max(r['recv'] for r in reqs) + 3600
    i, pending, partial = 0, [], None
    rem = {}
    done = {}
    while (i < len(reqs) or pending or partial) and t < t_end:
        while i < len(reqs) and reqs[i]['recv'] <= t:
            r = reqs[i]
            rem[r['req_id']] = max(r['unc'], 1)
            pending.append(r)
            i += 1
        budget = cap.get(int(t), 0) * TICK
        now = t + TICK

        def key(r):
            if policy == 'OBS':
                return r['ex']
            if policy == 'FCFS':
                return r['recv']
            if policy == 'SRPT':
                return rem[r['req_id']]
            if policy == 'SRPT_aging2000':   # engine 123 score: remaining - 2000 tok/s * waited
                return rem[r['req_id']] - 2000.0 * (t - r['recv'])
            if policy == 'SRPT_aging2000_demote':
                d = r['recv'] + visible_limit(r)
                hopeless = t + rem[r['req_id']] / rate_est > d
                return (hopeless, rem[r['req_id']] - 2000.0 * (t - r['recv']))
            if policy == 'LPM':
                return (-r['cached_tokens'], r['recv'])
            if policy == 'EDF_label':
                return r['recv'] + own_limit(r)
            if policy == 'EDF_visible':
                return r['recv'] + visible_limit(r)
            if policy == 'EDF_visible_demote':
                d = r['recv'] + visible_limit(r)
                hopeless = t + rem[r['req_id']] / rate_est > d
                return (hopeless, d)
            raise ValueError(policy)
        if policy == 'OBS':
            cand = [r for r in pending if r['ex'] <= now]   # observed admission order and times
        else:
            cand = pending
        cand = sorted(cand, key=key)
        if bypass_short:
            # short requests (<=4096 left) ride beside the continuing chunk before it takes the rest
            for r in [r for r in cand if rem[r['req_id']] <= 4096]:
                if rem[r['req_id']] <= budget:
                    budget -= rem[r['req_id']]
                    rem[r['req_id']] = 0
                    done[r['req_id']] = now
                    pending.remove(r)
            cand = [r for r in cand if r['req_id'] not in done]
        if partial is not None:
            use = min(budget, rem[partial['req_id']])
            rem[partial['req_id']] -= use
            budget -= use
            if rem[partial['req_id']] <= 0:
                done[partial['req_id']] = now
                partial = None
        for r in cand:
            if budget <= 0:
                break
            rid = r['req_id']
            if partial is not None and r is partial:
                continue
            if rem[rid] <= budget:
                budget -= rem[rid]
                rem[rid] = 0
                done[rid] = now
                pending.remove(r)
            elif partial is None:
                partial = r
                pending.remove(r)
                use = budget
                rem[rid] -= use
                budget = 0
        t = now
    out = {}
    for r in reqs:
        f = done.get(r['req_id'])
        out[r['req_id']] = (f - r['recv'] + POST) if f is not None else float('inf')
    return out


def gate_counts(R, ttft):
    res = {}
    for g, lim in SW.GATES:
        sel = [rid for rid, r in R.items() if g in r['gates']]
        res[g] = sum(ttft[rid] > lim for rid in sel)
    return res


def main():
    out = {'label': 'OFFLINE MODEL, 推算; see assumptions', 'assumptions': __doc__.split('Assumptions')[1].strip()}
    for run in SW.RUNS:
        R, t0 = SW.load(run)
        log = SW.load_log(run)
        cap = capacity_trace(log, min(r['recv'] for r in R.values()))
        actual = {rid: r['ttft_s'] for rid, r in R.items()}
        allowed = {g: SW.allowed_over(sum(g in r['gates'] for r in R.values())) for g, _ in SW.GATES}
        res = {'measured': gate_counts(R, actual), 'allowed': allowed}
        for bypass in (False, True):
            for pol in ('OBS', 'LPM', 'FCFS', 'SRPT', 'SRPT_aging2000', 'SRPT_aging2000_demote', 'EDF_label', 'EDF_visible', 'EDF_visible_demote'):
                tt = simulate(R, cap, pol, bypass_short=bypass)
                name = pol + ('+shortbypass' if bypass else '')
                gc = gate_counts(R, tt)
                cs = sorted(v for rid, v in tt.items() if 'chain_start' in R[rid]['gates'])
                fs = sorted(v for rid, v in tt.items() if 'fast_intra' in R[rid]['gates'])
                entry = dict(gates=gc, chain_p95=round(cs[int(.95 * len(cs)) - 1], 1),
                             chain_max=round(cs[-1], 1), chain_over60=sum(v > 60 for v in cs),
                             fast_p95=round(fs[int(.95 * len(fs)) - 1], 2),
                             first_min_chain_bad=sum(tt[rid] > 30 for rid, r in R.items()
                                                     if 'chain_start' in r['gates'] and r['arr_min'] < 1),
                             unfinished=sum(v == float('inf') for v in tt.values()))
                if pol == 'OBS':
                    # calibration: per-request agreement with measured TTFT on bad-prone gates
                    errs = sorted(abs(tt[rid] - actual[rid]) for rid, r in R.items() if 'chain_start' in r['gates'])
                    entry['calib_chain_abs_err_p50_p90'] = (round(errs[len(errs) // 2], 2), round(errs[int(.9 * len(errs))], 2))
                    same = sum((tt[rid] > 30) == (actual[rid] > 30) for rid, r in R.items() if 'chain_start' in r['gates'])
                    entry['calib_chain_same_verdict'] = '%d/%d' % (same, sum('chain_start' in r['gates'] for r in R.values()))
                    errs_f = sorted(abs(tt[rid] - actual[rid]) for rid, r in R.items() if 'fast_intra' in r['gates'])
                    entry['calib_fast_abs_err_p50_p90'] = (round(errs_f[len(errs_f) // 2], 2), round(errs_f[int(.9 * len(errs_f))], 2))
                res[name] = entry
                print(run, name, json.dumps(entry), flush=True)
        out[run] = res
    (HERE / 'lane-model.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
