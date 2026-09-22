#!/usr/bin/env python3
"""Reproduce T18 region ladders and local identifiability/sensitivity checks.

CPU only. Fixed public templates and model profiles, never server measurements.
Run after the documented refined --envelope-fit command in R10.
"""
import argparse
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import sim_closed_loop as sim
import sim_envelope_fit as fit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-report", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--supplemental-only", action="store_true",
                    help="after main checks: compare qualifying local perturbations and co-binding cold cases")
    args = ap.parse_args()
    output = args.out_dir.resolve()
    for forbidden in (sim.REPO / "s1-dev", sim.REPO / "src/sglang", sim.REPO / "llm-challenge-arena-v1"):
        if output == forbidden or forbidden in output.parents:
            ap.error(f"output directory is read-only: {output}")
    source = json.loads(args.fit_report.read_text())
    selected = [c for c in source["candidates"] if c["fit"]["full"]["all_in_band"] and c["ttft_first_18_22"]]
    if not selected:
        ap.error("fit report has no full-band, TTFT-first candidates")
    # This reproduction script intentionally fixes the R10 workload assumptions.
    assumptions = source["arguments"]
    if (assumptions["output_mode"] != "budget" or assumptions["output_scale"] != 1 or
            assumptions["cache_input"] or assumptions["output_input"] or
            assumptions["chain_gap_cap_s"] != 3600 or assumptions["seed"] != 20260922):
        ap.error("this R10 check recipe expects default profiles, gaps and seed")
    dev = sim.load_workload(sim.REPO / "s1-dev/data/dev-combined-v1",
                            sim.HARNESS / "g0a/samples_v3/cohort_dev-combined-v1.json")
    nominal = sim.formal_mix(dev)
    profiles, _ = sim.prepare_profiles(nominal)
    ladder_levels = (2, 6, 10, 14, 18, 22, 26, 30)
    report = {"label": fit.FIT_LABEL, "source_sha256": sim.sha256(args.fit_report),
              "script_sha256": sim.sha256(__file__), "selected_ids": [c["id"] for c in selected],
              "nominal": [], "local_parameter_checks": [], "sensitivity": []}

    def ladder(w, p, e, policy):
        runs = [fit.compact(sim.simulate(w, p, n, e, policy)["summary"]) for n in ladder_levels]
        return dict(sim.ladder_summary(runs), engine=asdict(e), policy=policy, runs=runs)

    def save():
        sim.write_json(output / "region_checks.json", report)

    if args.supplemental_only:
        existing = json.loads((output / "region_checks.json").read_text())
        supplements = [(c["name"], c["engine"]) for c in existing["local_parameter_checks"]
                       if c["fit"]["full"]["all_in_band"] and c["ttft_first_18_22"] and
                       c["name"] != "decode_speedup_equivalent"]
        for c in source["candidates"]:
            last = c["runs"][-1]
            if (c["fit"]["full"]["all_in_band"] and
                    all(r["model_pass_dev_plus_tpot"] for r in c["runs"] if r["N"] <= 18) and
                    last["N"] == 22 and last["tpot_p95_le_0_10"] and
                    any(g.startswith("fast_intra(") for g in last["failed_gates"]) and
                    any(g.startswith("chain_start(") for g in last["failed_gates"])):
                supplements.append((f"co_binding_{c['id']}", c["engine"]))
        extra = {"label": fit.FIT_LABEL, "comparisons": [], "mechanism_diagnostic": []}
        for name, params in supplements:
            for scheduler in ("fcfs", "spf"):
                e = sim.Engine(**dict(params, scheduler=scheduler))
                for policy in sim.POLICIES:
                    extra["comparisons"].append(dict(ladder(nominal, profiles, e, policy), variant=name))
            sim.write_json(output / "supplemental_checks.json", extra)
            print(f"MODEL OUTPUT supplemental comparison={name}", flush=True)
        best = min(selected, key=lambda c: c["fit"]["full"]["log_rmse"])
        for policy in sim.POLICIES:
            result = sim.simulate(nominal, profiles, 14, sim.Engine(**best["engine"]), policy)
            extra["mechanism_diagnostic"].append(dict(result["summary"],
                queue_time_p95_s=sim.q([r["queue_time_s"] for r in result["requests"]], .95)))
        sim.write_json(output / "supplemental_checks.json", extra)
        return

    for c in selected:
        for scheduler in ("fcfs", "spf"):
            e = sim.Engine(**dict(c["engine"], scheduler=scheduler))
            for policy in sim.POLICIES:
                report["nominal"].append(dict(ladder(nominal, profiles, e, policy), candidate_id=c["id"]))
        print(f"MODEL OUTPUT region comparison candidate={c['id']}", flush=True)
        save()

    best = min(selected, key=lambda c: c["fit"]["full"]["log_rmse"])
    base = sim.Engine(**best["engine"])
    mutations = {
        "prefill_low": {"prefill_tps": base.prefill_tps * .875},
        "prefill_high": {"prefill_tps": base.prefill_tps * 1.125},
        "decode_low": {"decode_ms": base.decode_ms * .75},
        "decode_high": {"decode_ms": base.decode_ms * 1.25},
        "slope_zero": {"decode_batch_slope": 0}, "slope_double": {"decode_batch_slope": .08},
        "overhead_zero": {"forward_overhead_ms": 0}, "overhead_double": {"forward_overhead_ms": 4},
        "chunk_half": {"chunk_tokens": 8192}, "chunk_double": {"chunk_tokens": 32768},
        "alpha_low": {"prefill_length_alpha": 1}, "alpha_high": {"prefill_length_alpha": 1.4},
        "interval_half": {"prefill_decode_interval": 16}, "interval_high": {"prefill_decode_interval": 48},
        "decode_speedup_equivalent": {"decode_ms": base.decode_ms * 2, "decode_speedup": 2},
    }
    for name, changes in mutations.items():
        e = replace(base, **changes)
        runs = [fit.compact(sim.simulate(nominal, profiles, n, e, "stock")["summary"])
                for n in (*fit.ANCHORS, 22)]
        report["local_parameter_checks"].append(dict(name=name, base_candidate=best["id"], engine=asdict(e),
            fit=fit.residuals(runs, source["prior"]), ttft_first_18_22=fit.ttft_first_near_ceiling(runs), runs=runs))
    save()
    print("MODEL OUTPUT local parameter checks complete", flush=True)

    # Re-sampling probes template-order sensitivity; it does not sample real formal data.
    for seed in (20260923, 20260924):
        w = sim.formal_mix(dev, seed)
        p, _ = sim.prepare_profiles(w)
        for c in selected:
            e = sim.Engine(**c["engine"])
            for policy in sim.POLICIES:
                report["sensitivity"].append(dict(ladder(w, p, e, policy), candidate_id=c["id"],
                                                   variant=f"template_seed_{seed}"))
        print(f"MODEL OUTPUT template seed={seed} complete", flush=True)
        save()
    # Upper output budgets vs source-model completions: no output truncation in a service.
    source_profiles, _ = sim.prepare_profiles(nominal, output_mode="source")
    conservative = copy.deepcopy(profiles)
    for rid, p in conservative.items():
        row = nominal.rows[rid]
        factor = (3332 / 3144 if sim.in_ttft_gate(row, "fast_intra") else
                  15082 / 14033 if sim.in_ttft_gate(row, "overall_intra") else 1)
        p["role_conservative"] = min(row["glm_tokens"], round(p["role_conservative"] * factor))
    for c in selected:
        e = sim.Engine(**c["engine"])
        for policy in sim.POLICIES:
            report["sensitivity"].append(dict(ladder(nominal, source_profiles, e, policy),
                candidate_id=c["id"], variant="source_output_lengths"))
        report["sensitivity"].append(dict(ladder(nominal, conservative, e, "role_conservative"),
            candidate_id=c["id"], variant="D1_F24_tail_inflation_proxy"))
        report["sensitivity"].append(dict(ladder(nominal, profiles,
            replace(e, d1_extra_forward_equivalents=2), "role_conservative"),
            candidate_id=c["id"], variant="D1_two_extra_overheads"))
        save()
    print("MODEL OUTPUT workload and D1 proxy sensitivities complete", flush=True)
    report["read_only_hashes_match"] = all(sim.sha256(p) == h for p, h in nominal.metadata["input_sha256"].items())
    save()


if __name__ == "__main__":
    main()
