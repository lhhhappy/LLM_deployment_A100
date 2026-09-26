#!/usr/bin/env python3
"""Summarize closed 2026-09-26 DCP development receipts, without an SLO claim.

This reads existing comparator results; it does not replace tensor comparison.
Usage: python3 scripts/analysis/dcp_receipt.py evidence/dcp-mtp-20260926
"""
import argparse
import json
from pathlib import Path


def summarize(root):
    def read(name):
        return json.loads((root / name).read_text())

    comparisons = {}
    for name in ("mtp", "accept", "compact", "host"):
        source = f"phase3/compare_{name}13.json"
        result = read(source)
        comparisons[name] = {
            "source": source,
            "passed": result["passed"],
            "observations": len(result["records"]),
            "max_valid_row_relative_linf": max(
                r["max_row_relative_linf"] for r in result["records"]
            ),
            "max_global_relative_linf": max(
                r["relative_linf"] for r in result["records"]
            ),
            "identical_output_tokens": all(r["same_tokens"] for r in result["responses"]),
            "tolerance": result["tolerance"],
        }

    arms = {}
    for name in ("dcp", "accept_dcp", "compact", "host_ref", "host_dcp"):
        path = f"phase3/mtp_tp2_{name}13"
        arm = {"source": path, "summary": read(path + "/summary.json")}
        arm["capacity"] = {
            role: read(path + f"/capacity_{role}0.json") for role in ("target", "draft")
        }
        if arm["summary"]["controlled_proposals"]:
            arm["acceptance"] = read(path + "/acceptance_verdict.json")
        if arm["summary"]["host_restore_and_flush"]:
            restore = read(path + "/host_restore_verdict.json")
            arm["restore"] = {k: v for k, v in restore.items() if k != "records"}
            flush = read(path + "/flush.json")
            arm["flush"] = {
                "success": flush["flush_result"]["success"],
                "cached_tokens_after_flush": flush["after_flush_meta"]["cached_tokens"],
            }
        arms[name] = arm

    sparse = read("phase2/sparse10.json")
    performance = {}
    for width in (2, 4, 8):
        cases = [c for c in sparse["cases"] if c["width"] == width and c["rows"] in (136, 152)]
        base = [c["baseline"]["graph"]["p50_ms"] for c in cases]
        candidate = [c["candidate"]["graph"]["p50_ms"] for c in cases]
        ratio = [b / c for b, c in zip(base, candidate)]
        performance[width] = {
            "cases": len(cases), "baseline_p50_ms_range": [min(base), max(base)],
            "candidate_p50_ms_range": [min(candidate), max(candidate)],
            "ratio_range": [min(ratio), max(ratio)],
        }
    move = read("phase3/move13.json")
    return {
        "validity": "CLOSED_DEVELOPMENT_DIAGNOSTICS_NOT_SLO",
        "generator": "python3 scripts/analysis/dcp_receipt.py evidence/dcp-mtp-20260926",
        "source_verification": read("phase3/fix5-source-verification.json"),
        "comparisons": comparisons, "arms": arms,
        "move": {"source": "phase3/move13.json", "cases": len(move["cases"]), "passed": move["passed"]},
        "sparse": {
            "source": "phase2/sparse10.json", "cases": len(sparse["cases"]),
            "boundary_contracts": len(sparse["boundary_contracts"]), "passed": sparse["passed"],
            "max_row_relative_linf": max(c["numerical"]["max_row_relative_linf"] for c in sparse["cases"]),
            "graph_performance": performance,
        },
        "archive": read("phase3/local-receipt.json"),
        "limitations": [
            "Scaled weights, TP2; no full-checkpoint TP8 quality or SLO result.",
            "Controlled proposals test the real verifier but not natural acceptance rate or speed.",
            "Sparse timings exclude TP8 communication and the rest of the service.",
            "Independent participant review remains pending.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.root), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
