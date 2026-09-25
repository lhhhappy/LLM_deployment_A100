#!/usr/bin/env python3
"""Red-team table for chain_start failures in local N30 runs (read-only).

For each run: classify every chain_start request (harness phase_gate: idx0 or context_reset)
into cold / systools / cont_head / ctx_reset, and record wait (recv->exec_start),
service (exec_start->first token), expected vs actual uncached, arrival minute.
Writes per-request CSV and prints per-run summaries.
"""
import json, glob, csv, os, sys, collections
ROOT = '/workspace/Agentic_science_challenge'
OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = {
  '067': 'evidence/L067-official_b_full_n30_shortwarm',
  '068': 'evidence/L068-official_b_pace_off_full_n30_shortwarm',
  '069': 'evidence/L069-official_b_pace_off_host64_full_n30_shortwarm',
  '071': 'evidence/L071-official_b_host64_full_n30_shortwarm',
}
req_meta = {}
for l in open(f'{ROOT}/data/s1-dev-longchain/requests.jsonl'):
    r = json.loads(l); req_meta[r['logical_call_id']] = r

def ctype(r):
    if r['idx_in_chain'] == 0:
        et = r.get('edge_type')
        if et == 'chain-head': return 'cold_head'
        if et == 'system-tools-changed': return 'systools_head'
        return 'cont_head'   # idx0 but expects cross-chain prefix (append-only / compact-rebuild)
    return 'ctx_reset'

def is_cs(r):
    return r['idx_in_chain'] == 0 or r.get('phase') == 'context_reset'

def gate_limit(r):
    if is_cs(r): return 30.0
    if r.get('phase') == 'turn_start': return 15.0
    return 3.0 if (r.get('uncached_expected') or 0) <= 4096 else 5.0

rows_all = []
summ = {}
for run, d in RUNS.items():
    raw = glob.glob(f'{ROOT}/{d}/N30/raw_*.jsonl')[0]
    rs = [json.loads(l) for l in open(raw)]
    t0 = min(r['t_recv_s'] for r in rs)
    for r in rs:
        r['_min'] = (r['t_recv_s'] - t0) / 60
        r['_wait'] = r['t_exec_start_s'] - r['t_recv_s']
        r['_svc'] = r['t_first_token_s'] - r['t_exec_start_s']
        r['_unc_act'] = r['prompt_tokens'] - r['cached_tokens']
        r['_deficit'] = r['_unc_act'] - (r['uncached_expected'] or 0)
        r['_bad'] = r['ttft_s'] > gate_limit(r)
    cs = [r for r in rs if is_cs(r)]
    S = collections.OrderedDict()
    S['n_cs'] = len(cs); S['n_cs_bad'] = sum(r['_bad'] for r in cs)
    for t in ['cold_head', 'systools_head', 'cont_head', 'ctx_reset']:
        sub = [r for r in cs if ctype(r) == t]
        bad = [r for r in sub if r['_bad']]
        S[t] = dict(n=len(sub), bad=len(bad),
                    bad_big_deficit=sum(1 for r in bad if r['_deficit'] > 16384),
                    bad_wait_gt_svc=sum(1 for r in bad if r['_wait'] > r['_svc']),
                    bad_svc_gt30=sum(1 for r in bad if r['_svc'] > 30))
    bad = [r for r in cs if r['_bad']]
    S['bad_arrival_min_sorted'] = sorted(round(r['_min'], 1) for r in bad)
    S['bad_first20min'] = sum(1 for r in bad if r['_min'] < 20)
    S['bad_wait_sum_s'] = round(sum(r['_wait'] for r in bad), 1)
    S['bad_svc_sum_s'] = round(sum(r['_svc'] for r in bad), 1)
    S['bad_deficit_gt16k'] = sum(1 for r in bad if r['_deficit'] > 16384)
    S['bad_no_deficit'] = sum(1 for r in bad if r['_deficit'] <= 4096)
    # would-fail-anyway proxy: service alone > 30 s or expected uncached > 250k
    S['bad_svc_alone_gt30'] = sum(1 for r in bad if r['_svc'] > 30)
    # all requests bad by gate
    S['bad_all_gates'] = sum(r['_bad'] for r in rs)
    S['tpot_mean'] = round(sum(r['tpot_s'] for r in rs if r.get('tpot_s')) / sum(1 for r in rs if r.get('tpot_s')), 5)
    S['run_extra_uncached_M'] = round(sum(max(0, r['_deficit']) for r in rs) / 1e6, 3)
    S['run_actual_uncached_M'] = round(sum(r['_unc_act'] for r in rs) / 1e6, 3)
    S['run_expected_uncached_M'] = round(sum(r['uncached_expected'] or 0 for r in rs) / 1e6, 3)
    summ[run] = S
    for r in cs:
        rows_all.append(dict(run=run, req_id=r['req_id'], type=ctype(r), phase=r.get('phase'), bad=int(r['_bad']),
            arr_min=round(r['_min'], 2), ttft=round(r['ttft_s'], 2), wait=round(r['_wait'], 2), svc=round(r['_svc'], 2),
            prompt=r['prompt_tokens'], unc_exp=r['uncached_expected'], unc_act=r['_unc_act'], deficit=r['_deficit']))

with open(f'{OUT}/chainstart_by_run.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys())); w.writeheader(); w.writerows(rows_all)
json.dump(summ, open(f'{OUT}/chainstart_summary.json', 'w'), indent=1, ensure_ascii=False)
for k, v in summ.items():
    print('==', k); print(json.dumps(v, ensure_ascii=False))
