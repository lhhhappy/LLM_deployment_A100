#!/usr/bin/env python3
"""CPU-only, gap-aware 8xA100 session budget estimate; starts no jobs/services.

python3 -B scripts/plan_8gpu_session.py --budget-minutes 240 --out plan.json
Use unmodified dev cohort, build_gap_plan(cap=3600s), and the harness FIFO chain
queue on N closed-loop slots. Calibrate one service-time scale to reference 35m
at N=6. Request work proxy = max_output_i + uncached_expected/32; the conversion
is an assumption, not measured throughput. Concurrency service slowdown defaults
to max(1,N/6)**0.5; exponent 0/1 are sensitivity scenarios, not confidence bounds.
The 35m reference is interpreted as measure-only; startup/preflight/warmup/flush
are extra and may double-count undocumented reference overhead. Stock and D1
have equal service cost unless --d1-service-factor supplies measured evidence.
21m model startup is task.md's example, not a guarantee for our engine. A new
policy block needs a restart and new warmup. Budget selection keeps an ordered
prefix of requested runs, with reserve; no claimed pass/fail or capacity result.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import sys

from make_case_sets import COHORT, REPO, load_data


def makespan(chains, rows, gaps, n, scale, slowdown=1.0, uncached_per_output=32):
    slots = [0.0] * n
    for chain in chains:
        work = sum((rows[rid].get("max_output_i") or 512) +
                   rows[rid]["uncached_expected"]/uncached_per_output for rid in chain["req_ids"])
        duration = sum(gaps[chain["chain_id"]])/1000 + work*scale*slowdown
        start = heapq.heappop(slots)
        heapq.heappush(slots, start+duration)
    return max(slots, default=0.0)


def calibrate(chains, rows, gaps, reference_n, reference_s, ratio=32):
    minimum = makespan(chains, rows, gaps, reference_n, 0, uncached_per_output=ratio)
    if minimum > reference_s:
        raise ValueError(f"reference runtime {reference_s:.1f}s is below exact gap-only FIFO wall {minimum:.1f}s")
    lo, hi = 0.0, 1.0
    while makespan(chains, rows, gaps, reference_n, hi, uncached_per_output=ratio) < reference_s:
        hi *= 2
    for _ in range(80):
        mid = (lo+hi)/2
        if makespan(chains, rows, gaps, reference_n, mid, uncached_per_output=ratio) < reference_s:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def parse_runs(spec):
    result = []
    for block in spec.split():
        policy, levels = block.split(":", 1)
        if policy not in ("stock", "D1"):
            raise ValueError("policies must be stock or D1")
        for part in levels.split(","):
            n = int(part)
            if n < 2 or (n-2) % 4:
                raise ValueError("levels must be arena rungs 2,6,10,14,...")
            result.append((policy, n))
    if not result:
        raise ValueError("empty session")
    return result


def make_plan(args):
    rows, _, _ = load_data(args.dev_root)
    from s1_loadgen import build_gap_plan
    cohort = json.loads(args.cohort.read_text())
    chains = cohort["chains"]
    gaps, gap_stats = build_gap_plan(chains, rows, 3600*1000)
    reference = cohort["reference_runtime"]
    ref_n, ref_s = reference["concurrency_N"], reference["minutes"]*60
    scale = calibrate(chains, rows, gaps, ref_n, ref_s, args.uncached_per_output)
    def estimate(n, exponent, factor=1):
        return makespan(chains, rows, gaps, n, scale,
                        max(1, n/ref_n)**exponent*factor, args.uncached_per_output)/60
    requested = parse_runs(args.runs)
    levels = []
    for n in sorted(set([2, 6, 10, 14, 18, 22] + [n for _, n in requested])):
        levels.append({"N": n, "measure_minutes": estimate(n, args.slowdown_exponent),
            "ideal_constant_service_minutes": estimate(n, 0),
            "linear_contention_minutes": estimate(n, 1),
            "gap_only_fifo_minutes": makespan(chains, rows, gaps, n, 0)/60})
    runs, elapsed, previous, stopped = [], args.self_test_minutes, None, False
    for policy, n in requested:
        restart = policy != previous
        measurement = estimate(n, args.slowdown_exponent,
                               args.d1_service_factor if policy == "D1" else 1)
        startup = args.startup_minutes if restart else 0
        warmup = args.warmup_minutes if restart else 0
        overhead = startup + warmup + args.preflight_minutes + args.flush_minutes
        cost = measurement + overhead
        candidate = elapsed + cost
        fits = not stopped and candidate*(1+args.reserve_fraction) <= args.budget_minutes
        runs.append({"policy": policy, "N": n, "restart": restart,
            "measurement_minutes": measurement, "startup_minutes": startup,
            "warmup_minutes": warmup, "preflight_minutes": args.preflight_minutes,
            "flush_minutes": args.flush_minutes, "total_minutes": cost,
            "selected": fits, "selected_end_minutes": candidate if fits else None})
        if fits:
            elapsed = candidate
        else:
            stopped = True
        previous = policy
    selected = [r for r in runs if r["selected"]]
    nominal = elapsed if selected else 0
    assumptions = [
        __doc__.strip(),
        "No queue admission wait, image pull, package downloads, or unexpected compilation/retry stalls are modeled; reserve is configurable.",
        "Flush allowance covers all verified flush calls per level, not the server's worst-case wait/retry bound.",
        "Session is a fixed proposed sequence, not adaptive ladder search; use ladder_search.py for actual outcomes."]
    return {"label": "planning_estimate_only", "reference": reference,
        "cohort_sha256": cohort["cohort_sha256"], "gap_plan": gap_stats,
        "calibrated_seconds_per_work_unit": scale,
        "assumptions": assumptions, "parameters": {k: str(v) if isinstance(v, Path) else v
                                                    for k, v in vars(args).items()},
        "ladder_levels": levels, "requested_runs": runs,
        "self_test_minutes": args.self_test_minutes,
        "full_requested_nominal_minutes": args.self_test_minutes+sum(r["total_minutes"] for r in runs),
        "selected_runs": len(selected), "selected_nominal_minutes": nominal,
        "selected_with_reserve_minutes": nominal*(1+args.reserve_fraction),
        "remaining_budget_minutes": args.budget_minutes-nominal*(1+args.reserve_fraction),
        "status": "all_fit" if len(selected) == len(runs) else "prefix_fit" if selected else "no_run_fits"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev-root", type=Path, default=REPO/"s1-dev")
    p.add_argument("--cohort", type=Path, default=COHORT)
    p.add_argument("--budget-minutes", type=float, default=240)
    p.add_argument("--runs", default="stock:6,10,14,18 D1:14,18,22")
    p.add_argument("--startup-minutes", type=float, default=21)
    p.add_argument("--warmup-minutes", type=float, default=5)
    p.add_argument("--preflight-minutes", type=float, default=.5)
    p.add_argument("--flush-minutes", type=float, default=.5)
    p.add_argument("--self-test-minutes", type=float, default=2)
    p.add_argument("--reserve-fraction", type=float, default=.15)
    p.add_argument("--slowdown-exponent", type=float, default=.5)
    p.add_argument("--d1-service-factor", type=float, default=1)
    p.add_argument("--uncached-per-output", type=float, default=32)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, float) and (not math.isfinite(value) or value < 0):
            p.error(f"{key} must be finite and nonnegative")
    if min(a.budget_minutes, a.d1_service_factor, a.uncached_per_output) <= 0:
        p.error("budget, service factor, and work conversion must be positive")
    try:
        plan = make_plan(a)
    except (ValueError, KeyError, OSError) as e:
        p.error(str(e))
    if a.out:
        if a.dev_root.resolve() == a.out.resolve() or a.dev_root.resolve() in a.out.resolve().parents:
            p.error("output must be outside read-only s1-dev")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(plan, indent=2)+"\n")
    print("CPU planning estimate; 35m/N6 reference; cap 3600s per chain; no service launched.")
    print(" N    measure(min)   ideal-service   linear-contention   gap-only")
    for row in plan["ladder_levels"]:
        print(f"{row['N']:2d} {row['measure_minutes']:14.1f} {row['ideal_constant_service_minutes']:15.1f} "
              f"{row['linear_contention_minutes']:19.1f} {row['gap_only_fifo_minutes']:10.1f}")
    for r in plan["requested_runs"]:
        print(f"{'KEEP' if r['selected'] else 'DEFER'} {r['policy']} N={r['N']}: "
              f"{r['total_minutes']:.1f}m including {r['startup_minutes']:.1f}m startup, "
              f"{r['warmup_minutes']:.1f}m warmup, {r['preflight_minutes']:.1f}m preflight, "
              f"{r['flush_minutes']:.1f}m flush")
    print(f"{plan['status']}: {plan['selected_runs']} runs, "
          f"{plan['selected_with_reserve_minutes']:.1f}/{a.budget_minutes:.1f}m with "
          f"{100*a.reserve_fraction:.0f}% reserve; full request "
          f"{plan['full_requested_nominal_minutes']:.1f}m before reserve.")
    print("Assumptions: FIFO chains, fitted output+uncached work, concurrency slowdown exponent "
          f"{a.slowdown_exponent}, D1 service factor {a.d1_service_factor}; overheads are estimates. "
          "Admission/image-pull delays excluded. 21m startup comes from task.md's example.")


if __name__ == "__main__":
    main()
