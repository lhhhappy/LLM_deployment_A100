#!/usr/bin/env python3
"""CPU-only public-envelope fitting for sim_closed_loop.py --envelope-fit.

Other teams' passing submissions form a selection-biased prior, never a stock
trace or validation set. All fitted timings and ladders are MODEL OUTPUTS.
Uses only local public scores; never contacts teams, services or the platform.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import sim_closed_loop as sim

ANCHORS = (6, 10, 14, 18)
METRICS = ("fast_intra_p95", "overall_intra_p95", "turn_start_p95",
           "chain_start_p95", "tpot_mean", "tpot_p95")
CORE = ("fast_intra_p95", "overall_intra_p95", "tpot_mean", "tpot_p95")
FIT_LABEL = "MODEL OUTPUT — public-prior coarse calibration; formal-mix WHAT-IF ONLY"
# Explicit search bounds. No posterior probability or hardware interpretation.
SEARCH_SPACE = {
    "prefill_tps": [6000, 100000, "log-uniform"],
    "decode_ms": [1, 30, "log-uniform"],
    "decode_batch_slope": [0, .15, "uniform"],
    "forward_overhead_ms": [0, 4, "uniform"],
    "chunk_tokens": [1024, 2048, 4096, 8192, 16384, 32768],
    "prefill_length_alpha": [0, .5, 1, 1.5],
    "prefill_decode_interval": [0, 1, 4],
}


def public_envelope(attempts, levels=ANCHORS):
    """One canonical stress row per attempt; passing at n, not n_at_slo.

    Do not double-count the duplicate nested raw_result.stress representation.
    Repeated submissions remain repeated as in F35; count distinct authors to
    expose that these are not independent measurements of a single engine.
    """
    if not isinstance(attempts, list):
        raise ValueError("public attempts must be a JSON array")
    groups = {n: [] for n in levels}
    for attempt in attempts:
        card = attempt.get("scorecard") or {}
        stress = card.get("scorewheel_stress")
        if stress is None:
            stress = (card.get("scorewheel_raw_result") or {}).get("stress")
        if not stress or stress.get("passed") is not True or stress.get("n") not in groups:
            continue
        for metric in METRICS:
            v = stress.get(metric)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"invalid passing prior metric {metric}: attempt {attempt.get('id')}")
        groups[stress["n"]].append((attempt, stress))
    result = {}
    for n, pairs in groups.items():
        if not pairs:
            raise ValueError(f"public prior has no passing samples at N={n}")
        result[str(n)] = {
            "n_passing_attempts": len(pairs),
            "n_authors": len({a.get("authorId") for a, _ in pairs}),
            "attempt_ids": [a.get("id") for a, _ in pairs],
            "median": {m: statistics.median(s[m] for _, s in pairs) for m in METRICS},
            "q25": {m: sim.q([s[m] for _, s in pairs], .25) for m in METRICS},
            "q75": {m: sim.q([s[m] for _, s in pairs], .75) for m in METRICS},
        }
    return {"label": "PUBLIC PRIOR INPUT — other teams' passing formal deployments, not our baseline",
            "selection": "passed is true; canonical stress.n; each attempt once; repeated authors retained",
            "levels": result}


def metrics(summary):
    values = {"tpot_mean": summary["tpot_mean_s"], "tpot_p95": summary["tpot_p95_s"]}
    for key, detail in summary["ttft_gate_detail"].items():
        values[key.split("(")[0] + "_p95"] = detail["p95"]
    return {m: values[m] for m in METRICS}


def residuals(runs, prior, factor=2):
    """Equal weight per metric/anchor in log space; no statistical slack fitted."""
    if not math.isfinite(factor) or factor <= 1:
        raise ValueError("fit band factor must be finite and greater than one")
    lookup = {str(r["N"]): metrics(r) for r in runs}
    cells = []
    for n, target in prior["levels"].items():
        if n not in lookup:
            raise ValueError(f"missing fit anchor N={n}")
        for metric, expected in target["median"].items():
            actual = lookup[n][metric]
            if actual is None or actual <= 0 or not math.isfinite(actual):
                raise ValueError(f"unevaluable modeled fit metric: N={n} {metric}")
            ratio = actual / expected
            cells.append(dict(N=int(n), metric=metric, public_prior_s=expected,
                              model_s=actual, residual_s=actual - expected,
                              relative_residual=ratio - 1, log_ratio=math.log(ratio),
                              within_band=1 / factor <= ratio <= factor))
    def aggregate(selected):
        return {"log_rmse": math.sqrt(statistics.mean(c["log_ratio"] ** 2 for c in selected)),
                "max_multiplicative_error": max(math.exp(abs(c["log_ratio"])) for c in selected),
                "in_band": sum(c["within_band"] for c in selected), "total": len(selected),
                "all_in_band": all(c["within_band"] for c in selected)}
    return {"label": FIT_LABEL, "factor": factor, "full": aggregate(cells),
            "core": aggregate([c for c in cells if c["metric"] in CORE]), "cells": cells}


def ttft_first_near_ceiling(runs):
    """Selection condition, not evidence: strict fast/overall first at 18 or 22."""
    ordered = sorted(runs, key=lambda r: r["N"])
    first = next((r for r in ordered if not r["model_pass_dev_plus_tpot"]), None)
    return bool(first and first["N"] in (18, 22) and first["failed_gates"] and
                all(g.startswith(("fast_intra(", "overall_intra(")) for g in first["failed_gates"]) and
                all(r["tpot_p95_le_0_10"] for r in ordered if r["N"] <= first["N"]))


def candidate_engines(base, count, seed, supplied=()):
    sim.integer(count, "fit samples")
    if not isinstance(supplied, (list, tuple)):
        raise ValueError("fit candidates must be a JSON array of Engine overrides")
    try:
        engines = [replace(base, **override) for override in supplied]
    except TypeError as exc:
        raise ValueError(f"invalid Engine candidate override: {exc}") from exc
    rng = random.Random(seed)
    for _ in range(count):
        values = {}
        for name, space in SEARCH_SPACE.items():
            if space[-1] == "log-uniform":
                values[name] = math.exp(rng.uniform(math.log(space[0]), math.log(space[1])))
            elif space[-1] == "uniform":
                values[name] = rng.uniform(space[0], space[1])
            else:
                values[name] = rng.choice(space)
        engines.append(replace(base, **values))
    if not engines:
        raise ValueError("envelope fit needs at least one candidate")
    return engines


def compact(summary):
    # Keep complete gate diagnostics; discard the potentially large batch histogram.
    return {k: v for k, v in summary.items() if k not in ("engine_stats", "engine")}


def run_fit(args, dev, cache, outputs):
    sim.integer(args.fit_compare_top, "fit compare top")
    if not math.isfinite(args.fit_band_factor) or args.fit_band_factor <= 1:
        raise ValueError("fit band factor must be finite and greater than one")
    levels = sorted(sim.csv_values(args.levels, int))
    if not set((*ANCHORS, 22)).issubset(levels):
        raise ValueError("envelope fit levels must include 6,10,14,18,22")
    prior = public_envelope(json.loads(args.envelope_fit.read_text()))
    workload = sim.formal_mix(dev, args.seed)
    profiles, profile_meta = sim.prepare_profiles(workload, cache, outputs, args.output_mode,
        args.output_scale, args.stock_fast_factor, args.stock_slow_factor)
    base = sim.Engine(prefill_tps=float(args.prefill_rates), decode_ms=float(args.decode_ms),
        scheduler=args.fit_scheduler,
        decode_batch_slope=args.decode_batch_slope, decode_speedup=args.decode_speedup,
        forward_overhead_ms=args.forward_overhead_ms, chunk_tokens=args.chunk_tokens,
        page_tokens=args.page_tokens, prefill_length_alpha=args.prefill_length_alpha,
        frontend_ms=args.frontend_ms, prefill_decode_interval=args.prefill_decode_interval,
        max_running=args.max_running, d1_extra_forward_equivalents=args.d1_extra_forward_equivalents)
    supplied = json.loads(args.fit_candidates.read_text()) if args.fit_candidates else []
    engines = candidate_engines(base, args.fit_samples, args.seed, supplied)
    # Public deployments' scheduler is unknown: condition on the explicit assumption.
    # Cache policy differences and the alternate scheduler are held out.
    if any(e.scheduler != args.fit_scheduler for e in engines):
        raise ValueError("candidate scheduler must match --fit-scheduler")
    report = {"label": FIT_LABEL, "prior": prior, "search_space": SEARCH_SPACE,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "workload": dict(workload.metadata, profile_model=profile_meta),
        "input_sha256": {str(p): sim.sha256(p) for p in
            (args.envelope_fit, Path(__file__), Path(sim.__file__), args.fit_candidates,
             args.cache_input, args.output_input) if p},
        "fit_policy": "stock", "fit_scheduler": args.fit_scheduler, "candidates": [], "comparisons": []}
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index, engine in enumerate(engines):
        runs = [compact(sim.simulate(workload, profiles, n, engine, "stock")["summary"])
                for n in sorted(set((*ANCHORS, 22)))]
        fit = residuals(runs, prior, args.fit_band_factor)
        candidate = {"id": index, "engine": asdict(engine), "fit": fit, "runs": runs,
                     "ttft_first_18_22": ttft_first_near_ceiling(runs)}
        report["candidates"].append(candidate)
        if (index + 1) % 16 == 0 or index + 1 == len(engines):
            sim.write_json(output / "envelope_fit.json", report)
            best = min(report["candidates"], key=lambda c: c["fit"]["full"]["log_rmse"])
            print(f"MODEL OUTPUT envelope candidates={index+1}/{len(engines)}; "
                  f"best full log-RMSE={best['fit']['full']['log_rmse']:.4f}; "
                  f"band cells={best['fit']['full']['in_band']}/24", flush=True)
    candidates = report["candidates"]
    report["counts"] = {"candidates": len(candidates),
        "full_band": sum(c["fit"]["full"]["all_in_band"] for c in candidates),
        "core_band": sum(c["fit"]["core"]["all_in_band"] for c in candidates),
        "ttft_first_18_22": sum(c["ttft_first_18_22"] for c in candidates),
        "full_band_and_ttft_first": sum(c["fit"]["full"]["all_in_band"] and
                                       c["ttft_first_18_22"] for c in candidates),
        "core_band_and_ttft_first": sum(c["fit"]["core"]["all_in_band"] and
                                       c["ttft_first_18_22"] for c in candidates)}
    # Preserve best mismatches if no full fit exists; never silently call them calibrated.
    classes = {
        "best_full_residual": sorted(candidates, key=lambda c: c["fit"]["full"]["log_rmse"]),
        "best_core_residual": sorted(candidates, key=lambda c: c["fit"]["core"]["log_rmse"]),
        "ttft_condition_only": sorted((c for c in candidates if c["ttft_first_18_22"]),
                                      key=lambda c: c["fit"]["core"]["log_rmse"]),
        "core_band_and_ttft": sorted((c for c in candidates if c["ttft_first_18_22"] and
                                      c["fit"]["core"]["all_in_band"]),
                                     key=lambda c: c["fit"]["core"]["log_rmse"]),
        "full_band_and_ttft": sorted((c for c in candidates if c["ttft_first_18_22"] and
                                      c["fit"]["full"]["all_in_band"]),
                                     key=lambda c: c["fit"]["full"]["log_rmse"]),
    }
    selected = {}
    for name, group in classes.items():
        for c in group[:args.fit_compare_top]:
            selected.setdefault(c["id"], []).append(name)
    report["selected"] = selected
    for index, reasons in selected.items():
        for scheduler in ("fcfs", "spf"):
            engine = replace(engines[index], scheduler=scheduler)
            for policy in sim.POLICIES:
                runs = [compact(sim.simulate(workload, profiles, n, engine, policy)["summary"])
                        for n in levels]
                report["comparisons"].append(dict(sim.ladder_summary(runs), candidate_id=index,
                    selection_reasons=reasons, engine=asdict(engine), policy=policy, runs=runs))
        print(f"MODEL OUTPUT compared candidate={index}, classes={reasons}", flush=True)
        sim.write_json(output / "envelope_fit.json", report)
    sim.write_json(output / "envelope_fit.json", report)
    return 0
