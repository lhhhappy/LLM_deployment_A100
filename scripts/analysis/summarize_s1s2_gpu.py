#!/usr/bin/env python3
"""Recompute the review's single-GPU summary from complete preserved probe logs.

This validates diagnostic checks, not formal harness completeness or SLO gates.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = ("moe-validation-v3", "humming-cache-v2", "host-lifetime-v2")
    logs, sources = {}, {}
    for name in names:
        path = args.logs / f"{name}.log"
        # Native CUDA/NCCL messages are preserved in the original log; only
        # JSON records are used in the machine-readable diagnostic summary.
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
        done = [row for row in rows if row.get("kind") == "done"]
        if len(done) != 1 or not done[0].get("ok"):
            raise ValueError(f"{name}: missing successful completion")
        if int((args.logs / f"{name}.exit").read_text()) != 0:
            raise ValueError(f"{name}: process exited unsuccessfully")
        if any(row.get("ok") is False for row in rows):
            raise ValueError(f"{name}: a check failed")
        logs[name] = rows
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    checks = [r for r in logs[names[0]] if r["kind"] == "check"]
    numerics = [r for r in logs[names[0]] if r["kind"] == "numerics"]
    timing = [r for r in logs[names[1]] if r["kind"] == "humming_cache_timing"]
    lifetime = [r for r in logs[names[2]] if r["kind"] == "host_lifetime"]
    if len(checks) != 57 or len(numerics) != 7:
        raise ValueError("unexpected MoE diagnostic coverage")
    if {r["M"] for r in timing} != {1, 128, 2048, 8192} or len(timing) != 4:
        raise ValueError("missing or duplicated timing shape")
    expected = {(layout, repeat) for layout in ("layer_first", "page_first", "page_first_direct")
                for repeat in range(3)}
    if len(lifetime) != 9 or {(r["layout"], r["repeat"]) for r in lifetime} != expected:
        raise ValueError("missing or duplicated host lifetime case")
    summary = {
        "status": "DIAGNOSTIC",
        "scope": "single A100, random TP8 per-rank MoE shapes and small real pinned host pools; not TP8/full model/SLO",
        "source_log_sha256": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "moe": {"checks_passed": len(checks), "checks_by_kind": dict(Counter(r["what"] for r in checks)),
                "max_humming_relative_l2_vs_fp32": max(r["humming"] for r in numerics)},
        "humming_timing": [{k: r[k] for k in ("M", "median_metadata_pair_us", "median_layer_wall_ms")}
                           for r in timing],
        "host_lifetime": lifetime,
        "limitations": ["eager layer speedup was observed only at M=1 in this paired run",
                        "MoE CUDA graph checks do not validate mechanism 170's full-model graphs",
                        "no model weights, MTP acceptance or multi-rank collectives",
                        "only before/after GPU occupancy snapshots; no continuous isolation/power trace"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"moe_checks": len(checks), "host_cases": len(lifetime), "timing_shapes": len(timing)}))


if __name__ == "__main__":
    main()
