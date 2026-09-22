#!/usr/bin/env python3
"""T20 CPU cross-check of frozen R10 candidates; every timing is MODEL OUTPUT.

Compare legacy SPF, #40024 budget reservation, stock HRRN/LPM, and FCFS with
identical cache proxies/parameters. No new fit. Use --candidates for a quick
subset; --levels controls the full strict ladder. Does not start an engine.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import sim_closed_loop as sim
import sim_envelope_fit as fit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-report', type=Path, default=sim.REPO / 'evidence/T18_calibration/refined/envelope_fit.json')
    ap.add_argument('--candidates', default='6,17,18,19,31,42,43')
    ap.add_argument('--levels', default='2,6,10,14,18,22,26,30')
    ap.add_argument('--out-dir', type=Path, default=sim.REPO / 'evidence/T20_d2/calibration')
    args = ap.parse_args()
    output = args.out_dir.resolve()
    for forbidden in (sim.REPO / 's1-dev', sim.REPO / 'src/sglang', sim.REPO / 'llm-challenge-arena-v1'):
        if output == forbidden or forbidden in output.parents:
            ap.error('output directory is read-only')
    source = json.loads(args.fit_report.read_text())
    ids = [int(x) for x in args.candidates.split(',')]
    levels = sorted(int(x) for x in args.levels.split(','))
    candidates = {c['id']: c for c in source['candidates']}
    if any(i not in candidates for i in ids):
        ap.error('candidate absent from report')
    assumptions = source['arguments']
    if (assumptions['output_mode'] != 'budget' or assumptions['output_scale'] != 1 or
            assumptions['cache_input'] or assumptions['output_input'] or
            assumptions['chain_gap_cap_s'] != 3600 or assumptions['seed'] != 20260922):
        ap.error('recipe requires frozen R10 nominal workload assumptions')
    dev = sim.load_workload(sim.REPO / 's1-dev/data/dev-combined-v1',
                            sim.HARNESS / 'g0a/samples_v3/cohort_dev-combined-v1.json')
    w = sim.formal_mix(dev)
    profiles, profile_meta = sim.prepare_profiles(w)
    report = dict(label='MODEL OUTPUT — fixed R10 parameters, formal-mix WHAT-IF; not deployment evidence',
                  selected_ids=ids, levels=levels, workload=w.metadata, profiles=profile_meta,
                  source_sha256=sim.sha256(args.fit_report),
                  scripts_sha256={p: sim.sha256(sim.REPO / p) for p in
                                  ('scripts/sim_closed_loop.py', 'scripts/check_d2_sim_calibration.py')},
                  comparisons=[])
    for i in ids:
        # Freeze T20's five-policy recipe as new simulator policies are added.
        for scheduler in ("fcfs", "spf", "spf-upstream", "hrrn", "lpm"):
            engine = sim.Engine(**dict(candidates[i]['engine'], scheduler=scheduler))
            for policy in sim.POLICIES:
                runs = [fit.compact(sim.simulate(w, profiles, n, engine, policy)['summary']) for n in levels]
                entry = dict(sim.ladder_summary(runs), candidate_id=i, policy=policy,
                             engine=asdict(engine), runs=runs)
                report['comparisons'].append(entry)
                sim.write_json(output / 'comparison.json', report)
                print(f"MODEL C{i} {scheduler} {policy}: ceiling={entry['contiguous_pass_through_N']} "
                      f"first_fail={entry['first_failing_N']} {entry['first_binding_gates']}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
