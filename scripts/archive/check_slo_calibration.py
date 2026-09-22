#!/usr/bin/env python3
"""T25 CPU MODEL OUTPUT: fixed R10 candidates, strict + estimated CP ladders.

No fitting, serving, output truncation or frozen-label scheduling. Save small
per-level summaries, integer exceed counts, and server-proxy confusion counts.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import sim_closed_loop as sim
import sim_envelope_fit as fit
from score_formal import allowed_over, binomial_lower, METHOD


def estimated_summary(summary, records):
    gates = {}
    for name, selector, limit in sim.TTFT_GATE_SPECS:
        vals = [r['ttft_s'] for r in records if sim.in_ttft_gate(r, selector)]
        n, k = len(vals), sum(x > limit for x in vals)
        gates[name] = dict(n=n, over_limit=k, allowed_over=allowed_over(n),
                           rate_ci_lower=binomial_lower(k, n), limit_s=limit,
                           pass_estimated=bool(n and k <= allowed_over(n)))
    # Keep every non-TTFT dev gate + the strict TPOT gate unchanged.
    failures = [g for g in summary['failed_gates'] if g not in gates]
    failures += [g for g, d in gates.items() if not d['pass_estimated']]
    return dict(summary, label='MODEL OUTPUT / estimated CP; not official verdict',
                estimated_ttft=gates, method=METHOD,
                failed_gates=failures, model_pass_dev_plus_tpot=not failures)


def allowance_table():
    rows = []
    for n in (1, 20, 65, 100, 314, 388, 808, 9023):
        k = allowed_over(n)
        rows.append(dict(n=n, allowed_over=k, allowed_rate=k/n,
                         lower_at_allowed=binomial_lower(k, n),
                         lower_at_next=binomial_lower(k+1, n) if k < n else None))
    return dict(label='estimated', method=METHOD, rows=rows,
                example_61s=dict(n=808, n_at_20s=758, n_at_61s=50,
                                 p95_s=sim.q([20.0]*758+[61.0]*50, .95),
                                 lower=binomial_lower(50,808),
                                 pass_estimated=50 <= allowed_over(808)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-report', type=Path, default=sim.REPO/'evidence/T18_calibration/refined/envelope_fit.json')
    ap.add_argument('--candidates', default='6,17,18,19,31,42,43')
    ap.add_argument('--levels', default='2,6,10,14,18,22,26,30')
    ap.add_argument('--out-dir', type=Path, default=sim.REPO/'evidence/T25_slo/calibration')
    args = ap.parse_args()
    output=args.out_dir.resolve()
    for forbidden in (sim.REPO/'s1-dev', sim.REPO/'src/sglang', sim.REPO/'llm-challenge-arena-v1'):
        if output == forbidden or forbidden in output.parents:
            ap.error('output directory is read-only')
    source = json.loads(args.fit_report.read_text())
    recipe=source['arguments']
    if (recipe['output_mode'] != 'budget' or recipe['output_scale'] != 1 or
        recipe['cache_input'] or recipe['output_input'] or recipe['chain_gap_cap_s'] != 3600
        or recipe['seed'] != 20260922):
        ap.error('requires frozen R10 nominal workload recipe')
    candidates={c['id']:c for c in source['candidates']}
    ids=[int(x) for x in args.candidates.split(',')]
    levels=sorted(int(x) for x in args.levels.split(','))
    dev=sim.load_workload(sim.REPO/'s1-dev/data/dev-combined-v1',
                          sim.HARNESS/'g0a/samples_v3/cohort_dev-combined-v1.json')
    arms=[('fcfs','stock'),('spf-upstream','stock'),('spf-upstream','role_conservative'),
          ('edf','stock'),('edf','role_conservative'),('least-slack','stock'),
          ('edf-chain-weighted','stock'),('edf-chain-weighted','role_conservative')]
    report=dict(label='MODEL OUTPUT — R10 fixed parameters; formal-mix WHAT-IF only',
                source_sha256=sim.sha256(args.fit_report), levels=levels, candidates=ids,
                scripts_sha256={p:sim.sha256(sim.REPO/p) for p in
                                ('scripts/sim_closed_loop.py','scripts/score_formal.py',
                                 'scripts/check_slo_calibration.py')},
                allowance=allowance_table(), workloads=[], comparisons=[])
    for workload in (dev, sim.formal_mix(dev)):
        profiles, meta=sim.prepare_profiles(workload)
        confusion={}
        for policy in sim.POLICIES:
            counts=Counter()
            for ch in workload.chains:
                for i,rid in enumerate(ch['req_ids']):
                    proxy='chain_start' if i==0 else ('fast_proxy' if profiles[rid][policy]<=4096 else 'overall_proxy')
                    row=workload.rows[rid]
                    true=sim.category(row)
                    if true=='intra':
                        true='fast_intra' if row['uncached_expected']<=4096 else 'slow_intra'
                    counts[(true,proxy)]+=1
            confusion[policy]={' -> '.join(k):v for k,v in sorted(counts.items())}
        report['workloads'].append(dict(workload.metadata, profiles=meta, proxy_confusion=confusion))
        for cid in ids:
            for scheduler,policy in arms:
                engine=sim.Engine(**dict(candidates[cid]['engine'], scheduler=scheduler))
                strict, estimated=[], []
                for n in levels:
                    result=sim.simulate(workload,profiles,n,engine,policy)
                    summary=fit.compact(result['summary'])
                    strict.append(summary)
                    estimated.append(estimated_summary(summary,result['requests']))
                entry=dict(candidate_id=cid, mix=workload.metadata['mix'], scheduler=scheduler,
                           policy=policy, engine=asdict(engine),
                           strict=dict(sim.ladder_summary(strict),runs=strict),
                           estimated=dict(sim.ladder_summary(estimated),runs=estimated))
                report['comparisons'].append(entry)
                sim.write_json(output/'comparison.json',report)
                print(f"MODEL OUTPUT C{cid} {entry['mix']} {scheduler} {policy}: "
                      f"strict first={entry['strict']['first_failing_N']} {entry['strict']['first_binding_gates']}; "
                      f"estimated first={entry['estimated']['first_failing_N']} {entry['estimated']['first_binding_gates']}", flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
