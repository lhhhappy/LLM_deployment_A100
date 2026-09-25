#!/usr/bin/env python3
"""Official-result facts (read-only, from evidence/official/*.json and cost-audit 45979):
1) slo_attainment is an exact fraction with denominator dividing 1788 in every official stress result
   -> fixed judged cohort per level (1788 requests), independent of N and team.
2) tpm_all by N across all teams (ceiling ~1.75-1.79M/min from N14 up).
3) Per-gate envelope at passing levels, and our own ladder."""
import json, glob, collections, statistics as st, os
ROOT = '/workspace/Agentic_science_challenge'; OUT = os.path.dirname(os.path.abspath(__file__))
def stress_of(x):
    return ((x.get('scorecard') or {}).get('scorewheel_raw_result') or {}).get('stress') or \
           ((x.get('resultsJson') or {}).get('scorewheel_raw_result') or {}).get('stress') or x.get('stress')
res = {}
allatt = json.load(open(f'{ROOT}/evidence/official/all_att_2026-09-23.json'))
S = [(x['id'], stress_of(x)) for x in allatt if stress_of(x)]
ours = {}
for f in ['attempt-46173-20260924.json', 'attempt-46174-20260924.json', 'attempt-46251-final-20260925.json']:
    ours[f.split('-')[1]] = stress_of(json.load(open(f'{ROOT}/evidence/official/{f}')))
ours['45979'] = stress_of(json.load(open(f'{ROOT}/evidence/cost-audit-20260924/official-45979.json')))
vals = [s['slo_attainment'] for _, s in S if s.get('slo_attainment') is not None] + [s['slo_attainment'] for s in ours.values()]
ok1788 = sum(1 for v in vals if abs(round(v * 1788) / 1788 - v) < 1e-12)
need3576 = sum(1 for v in vals if abs(round(v * 1788) / 1788 - v) >= 1e-12 and abs(round(v * 3576) / 3576 - v) < 1e-12)
res['slo_attainment_denominator'] = dict(n_values=len(vals), exact_over_1788=ok1788, needing_3576=need3576,
    note='all exact k/1788 -> judged cohort per level is fixed at 1788 requests (推算 from exact fractions)')
res['ours'] = {k: dict(n=s['n_at_slo'], chain=s['chain_start_p95'], turn=s['turn_start_p95'], overall=s['overall_intra_p95'],
               fast=s['fast_intra_p95'], tpot_mean=s['tpot_mean'], tpot_p95=s['tpot_p95'], tpm_all=s['tpm_all'],
               tpm_decode=s['tpm_decode'], fails_of_1788=round(1788 * (1 - s['slo_attainment']))) for k, s in ours.items()}
byN = collections.defaultdict(list)
for _, s in S:
    if s.get('n_at_slo'): byN[s['n_at_slo']].append(s)
res['tpm_all_by_N_all_teams'] = {n: dict(k=len(v), median=round(st.median([s['tpm_all'] for s in v])), max=round(max(s['tpm_all'] for s in v)))
                                 for n, v in sorted(byN.items())}
res['gate_envelope_passing'] = {n: {g: dict(median=round(st.median([s[g] for s in v]), 3), max=round(max(s[g] for s in v), 3))
                                    for g in ['chain_start_p95', 'turn_start_p95', 'overall_intra_p95', 'fast_intra_p95', 'tpot_p95']}
                                for n, v in sorted(byN.items()) if n >= 14}
json.dump(res, open(f'{OUT}/official_facts.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False)[:3500])
