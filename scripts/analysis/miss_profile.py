#!/usr/bin/env python3
"""Profile the TTFT-gate misses of a run request by request, plus a capacity ledger.

Usage:
  miss_profile.py --raw RAW.jsonl [--ref REF_RAW.jsonl] [--metrics metrics.jsonl] [--csv out.csv] [--top N]

Buckets are the harness's (s1_common.in_ttft_gate on the raw rows: idx_in_chain, phase, uncached_expected), so the
counts equal the scorer's. Every miss is classified by what the server could see:
  chain:  family_rider   cached >= 50% of the prompt and new <= 6144  (rode along after a family leader)
          big_cold       new >= 50k        mid_cold  the rest      context_reset  phase == context_reset
  turn / fast / overall:  warm|cold  x  one_round (new <= 8192) | multi_round (new > 8192)
Wait is split with the server stamps when the raw has them: recv -> first execution (queueing) versus first
execution -> first token (compute). With --ref, each miss says whether the same request missed in the reference
(fixed / new / both), and the summary is restricted to the common request IDs.
The capacity ledger (Little's law on the run's own records, all measured from the raw): completions per second X,
and the average number of sessions decoding (D), waiting for the first token (W) and sleeping in replay gaps (S)
over the window from the first dispatch to the last finish; with --metrics also running peak, full-KV usage p95,
evictable KV median and host tier fill.
Calibrated on L081 (N30 full): gate counts equal level_verdict.json (see notes/reports/109-chain-turn-rootcause-0926.md).
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 's1-dev/harness'))
from s1_common import in_ttft_gate  # noqa: E402

LIMITS = {'chain_start': 30.0, 'turn_start': 15.0, 'overall_intra': 5.0, 'fast_intra': 3.0}


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r.get('error'):
            continue
        rows[r['req_id']] = r
    return rows


def bucket_of(r):
    for g in ('chain_start', 'turn_start', 'fast_intra'):
        if in_ttft_gate(r, g):
            return g
    return 'overall_intra' if in_ttft_gate(r, 'overall_intra') else None


def classify(r, bucket):
    prompt, cached = r['prompt_tokens'] or 0, r['cached_tokens'] or 0
    new = prompt - cached
    warm = prompt > 0 and cached >= 0.5 * prompt
    if bucket == 'chain_start':
        if r.get('phase') == 'context_reset':
            return 'context_reset'
        if warm and new <= 6144:
            return 'family_rider'
        return 'big_cold' if new >= 50000 else 'mid_cold'
    return ('warm' if warm else 'cold') + ('_one_round' if new <= 8192 else '_multi_round')


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float('nan')


def wait_split(r):
    t_recv, t_exec, t_tok = r.get('t_recv_s'), r.get('t_exec_start_s'), r.get('t_first_token_s')
    if t_recv and t_exec and t_tok:
        return t_exec - t_recv, t_tok - t_exec
    return None, None


def ledger(rows, metrics=None):
    ok = [r for r in rows.values() if r.get('client_finish_at_s') and r.get('client_first_token_at_s')]
    if not ok:
        return {}
    t0 = min(r['client_dispatch_at_s'] for r in ok)
    t1 = max(r['client_finish_at_s'] for r in ok)
    T = max(t1 - t0, 1e-9)
    decode = sum(r['client_finish_at_s'] - r['client_first_token_at_s'] for r in ok)
    wait = sum(r['client_first_token_at_s'] - r['client_dispatch_at_s'] for r in ok)
    gaps = sum((r.get('effective_replay_gap_ms') or 0) / 1000.0 for r in ok)
    out = dict(window_s=round(T, 1), completions=len(ok), X_req_per_s=round(len(ok) / T, 3),
               D_decoding=round(decode / T, 2), W_waiting_first_token=round(wait / T, 2), S_sleeping_gaps=round(gaps / T, 2),
               output_tokens_per_s=round(sum(r.get('output_tokens') or 0 for r in ok) / T, 1),
               tpot_mean_ms=round(1000 * statistics.mean(r['tpot_s'] for r in ok if r.get('tpot_s') is not None), 2))
    if metrics:
        samples = [json.loads(l) for l in open(metrics)]
        samples = [m for m in samples if t0 <= m.get('t', 0) <= t1]
        if samples:
            out.update(running_peak=max(m.get('num_running_reqs', 0) for m in samples),
                       full_kv_usage_p95=round(q([m.get('full_token_usage', 0) for m in samples], .95), 3),
                       kv_evictable_median=int(q([m.get('kv_evictable_tokens', 0) for m in samples], .5)),
                       queue_nonempty_share=round(sum(m.get('num_queue_reqs', 0) > 0 for m in samples) / len(samples), 2))
            host = [m for m in samples if m.get('hicache_host_total_tokens')]
            if host:
                out['host_fill_p95'] = round(q([m['hicache_host_used_tokens'] / m['hicache_host_total_tokens'] for m in host], .95), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--ref')
    ap.add_argument('--metrics')
    ap.add_argument('--csv')
    ap.add_argument('--top', type=int, default=60, help='misses listed per bucket')
    a = ap.parse_args()
    rows = load(a.raw)
    ref = load(a.ref) if a.ref else None
    ids = sorted(set(rows) & set(ref)) if ref else sorted(rows)
    t0 = min(rows[k]['client_dispatch_at_s'] for k in ids)
    print(f"run={a.raw} requests={len(rows)} {'common with ref=' + str(len(ids)) if ref else ''}")
    out_rows = []
    for g in ('chain_start', 'turn_start', 'overall_intra', 'fast_intra'):
        lim = LIMITS[g]
        members = [k for k in ids if bucket_of(rows[k]) == g or (g == 'overall_intra' and in_ttft_gate(rows[k], g))]
        if g == 'overall_intra':
            members = [k for k in ids if in_ttft_gate(rows[k], 'overall_intra')]
        elif g == 'fast_intra':
            members = [k for k in ids if in_ttft_gate(rows[k], 'fast_intra')]
        misses = [k for k in members if rows[k]['ttft_s'] > lim]
        ttfts = [rows[k]['ttft_s'] for k in members]
        line = f"== {g}: n={len(members)} over={len(misses)} p50={q(ttfts, .5):.2f} p90={q(ttfts, .9):.2f} p95={q(ttfts, .95):.2f}"
        if ref:
            ref_miss = [k for k in members if ref[k]['ttft_s'] > lim]
            line += f" | ref over={len(ref_miss)} fixed={len(set(ref_miss) - set(misses))} new={len(set(misses) - set(ref_miss))}"
        print(line)
        classes = Counter(classify(rows[k], g) for k in misses)
        if misses:
            print('   miss classes:', dict(classes))
        if g == 'turn_start' and members:
            small = [k for k in members if (rows[k]['prompt_tokens'] - rows[k]['cached_tokens']) <= 8192]
            print(f"   turn split: new<=8192 n={len(small)} over={sum(rows[k]['ttft_s'] > lim for k in small)}; "
                  f"new>8192 n={len(members) - len(small)} over={sum(rows[k]['ttft_s'] > lim for k in set(members) - set(small))}")
        for k in sorted(misses, key=lambda k: rows[k]['client_dispatch_at_s'])[:a.top]:
            r = rows[k]
            new = (r['prompt_tokens'] or 0) - (r['cached_tokens'] or 0)
            wait, exe = wait_split(r)
            status = ''
            if ref:
                status = 'both' if ref[k]['ttft_s'] > lim else 'new'
            print(f"   {k[-28:]:28s} t={r['client_dispatch_at_s'] - t0:7.1f} prompt={r['prompt_tokens']:7d} cached={r['cached_tokens']:7d} "
                  f"new={new:6d} {classify(r, g):16s} wait={wait if wait is None else round(wait, 1)!s:>6} exec={exe if exe is None else round(exe, 1)!s:>5} "
                  f"ttft={r['ttft_s']:6.1f}" + (f" ref={ref[k]['ttft_s']:6.1f} {status}" if ref else ''))
            out_rows.append(dict(req_id=k, bucket=g, cls=classify(r, g), t=round(r['client_dispatch_at_s'] - t0, 1),
                                 prompt=r['prompt_tokens'], cached=r['cached_tokens'], new=new, wait=wait, exec=exe,
                                 ttft=r['ttft_s'], ref_ttft=ref[k]['ttft_s'] if ref else None))
        if ref:
            for k in sorted(set(k for k in members if ref[k]['ttft_s'] > lim) - set(misses))[:a.top]:
                r = rows[k]
                out_rows.append(dict(req_id=k, bucket=g, cls=classify(r, g), t=round(r['client_dispatch_at_s'] - t0, 1),
                                     prompt=r['prompt_tokens'], cached=r['cached_tokens'], new=r['prompt_tokens'] - r['cached_tokens'],
                                     wait=None, exec=None, ttft=r['ttft_s'], ref_ttft=ref[k]['ttft_s'], fixed=True))
    led = ledger({k: rows[k] for k in ids}, a.metrics)
    print('== capacity ledger (measured from this raw):', json.dumps(led))
    if ref:
        print('== reference ledger:', json.dumps(ledger({k: ref[k] for k in ids})))
    if a.csv:
        keys = ['req_id', 'bucket', 'cls', 't', 'prompt', 'cached', 'new', 'wait', 'exec', 'ttft', 'ref_ttft', 'fixed']
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in out_rows:
                w.writerow({k: row.get(k) for k in keys})
        print('csv:', a.csv)


if __name__ == '__main__':
    main()
