#!/usr/bin/env python3
"""Per-5-min arrival windows for local N30 runs: bad counts per gate, prefill work arriving,
structural (cont_head) vs other positive cache deficit. Also chain p95 excluding opening minutes.
All numbers 实测 from raw_*.jsonl; cont_head deficit is 'structural' per cont_head_lcp.csv (推算 LCP)."""
import json, glob, csv, os, collections, math
ROOT = '/workspace/Agentic_science_challenge'; OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = {'068': 'evidence/L068-official_b_pace_off_full_n30_shortwarm',
        '069': 'evidence/L069-official_b_pace_off_host64_full_n30_shortwarm',
        '071': 'evidence/L071-official_b_host64_full_n30_shortwarm'}
cont = {r['req_id']: int(r['est_min_uncached']) for r in csv.DictReader(open(f'{OUT}/cont_head_lcp.csv'))}
def gate(r):
    if r['idx_in_chain'] == 0 or r.get('phase') == 'context_reset': return 'chain', 30.0
    if r.get('phase') == 'turn_start': return 'turn', 15.0
    return ('fast', 3.0) if (r.get('uncached_expected') or 0) <= 4096 else ('slow_intra', 5.0)
def p95(a):
    a = sorted(a); 
    if not a: return None
    k = 0.95 * (len(a) - 1); f = math.floor(k); c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)
def cp_allowed(n, alpha=0.05, p0=0.05):
    # largest k such that one-sided CP lower bound <= p0 (scipy-free via beta quantile bisection)
    from math import comb
    def lower(k):
        if k == 0: return 0.0
        # lower bound L: P(X>=k | p=L) = alpha
        lo, hi = 0.0, 1.0
        for _ in range(60):
            m = (lo + hi) / 2
            tail = sum(comb(n, i) * m**i * (1-m)**(n-i) for i in range(k, n+1))
            if tail > alpha: hi = m
            else: lo = m
        return lo
    k = 0
    while k <= n and lower(k + 1) <= p0: k += 1
    return k
res = {}
rows_out = []
for run, d in RUNS.items():
    rs = [json.loads(l) for l in open(glob.glob(f'{ROOT}/{d}/N30/raw_*.jsonl')[0])]
    t0 = min(r['t_recv_s'] for r in rs)
    W = collections.defaultdict(lambda: collections.Counter())
    for r in rs:
        g, lim = gate(r); m = (r['t_recv_s'] - t0) / 60; w = int(m // 5) * 5
        unc = r['prompt_tokens'] - r['cached_tokens']; dfc = unc - (r['uncached_expected'] or 0)
        W[w]['n'] += 1; W[w]['unc_act'] += unc
        if r['req_id'] in cont: W[w]['deficit_structural'] += max(0, dfc)
        else: W[w]['deficit_other'] += max(0, dfc)
        if r['ttft_s'] > lim: W[w]['bad_' + g] += 1
        W[w]['wait_sum'] += r['t_exec_start_s'] - r['t_recv_s']
    for w in sorted(W):
        c = W[w]
        rows_out.append(dict(run=run, win_min=w, n=c['n'], unc_act_M=round(c['unc_act']/1e6, 3),
            deficit_other_M=round(c['deficit_other']/1e6, 3), deficit_structural_M=round(c['deficit_structural']/1e6, 3),
            bad_chain=c['bad_chain'], bad_fast=c['bad_fast'], bad_slow_intra=c['bad_slow_intra'], bad_turn=c['bad_turn'],
            mean_wait_s=round(c['wait_sum']/c['n'], 2)))
    cs = [r for r in rs if gate(r)[0] == 'chain']
    S = {}
    for cut in [0, 1, 6, 10]:
        sub = [r for r in cs if (r['t_recv_s'] - t0)/60 >= cut]
        over = sum(1 for r in sub if r['ttft_s'] > 30)
        S[f'from_{cut}min'] = dict(n=len(sub), over=over, p95=round(p95([r['ttft_s'] for r in sub]), 2), allowed_cp=cp_allowed(len(sub)))
    tot_def = sum(max(0, (r['prompt_tokens']-r['cached_tokens'])-(r['uncached_expected'] or 0)) for r in rs)
    str_def = sum(max(0, (r['prompt_tokens']-r['cached_tokens'])-(r['uncached_expected'] or 0)) for r in rs if r['req_id'] in cont)
    neg = sum(min(0, (r['prompt_tokens']-r['cached_tokens'])-(r['uncached_expected'] or 0)) for r in rs)
    S['gross_pos_deficit_M'] = round(tot_def/1e6, 3); S['cont_head_pos_deficit_M'] = round(str_def/1e6, 3)
    S['cont_head_est_unavoidable_M'] = round(sum(cont.values())/1e6, 3)
    S['neg_deficit_M'] = round(neg/1e6, 3)
    # opening wave: arrivals < 0.5 min
    ow = [r for r in rs if (r['t_recv_s'] - t0) < 30]
    S['opening_wave'] = dict(n=len(ow), n_chain=sum(1 for r in ow if gate(r)[0]=='chain'),
        prompt_M=round(sum(r['prompt_tokens'] for r in ow)/1e6, 3), unc_act_M=round(sum(r['prompt_tokens']-r['cached_tokens'] for r in ow)/1e6, 3),
        over30=sum(1 for r in ow if r['ttft_s'] > 30), max_ttft=round(max(r['ttft_s'] for r in ow), 1),
        last_first_token_min=round((max(r['t_first_token_s'] for r in ow)-t0)/60, 2))
    # service throughput in first 5 minutes: uncached tokens whose first token landed in [0,300s)
    done5 = [r for r in rs if r['t_first_token_s'] - t0 < 300]
    S['uncached_prefilled_first5min_tok_per_s'] = round(sum(r['prompt_tokens']-r['cached_tokens'] for r in done5)/300)
    res[run] = S
    print('==', run, json.dumps(S, ensure_ascii=False))
with open(f'{OUT}/window_decomp.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
json.dump(res, open(f'{OUT}/window_decomp_summary.json', 'w'), indent=1, ensure_ascii=False)
