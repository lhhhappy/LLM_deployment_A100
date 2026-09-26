#!/usr/bin/env python3
"""Compare two complete dcp_mtp_probe arms, including sensitive DSA traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def compare(test: Path, ref: Path, tolerance: float):
    for arm in (test, ref):
        if not (arm / "summary.json").is_file():
            raise ValueError(f"Incomplete MTP arm: {arm}")
    responses = [json.loads((arm / "responses.json").read_text()) for arm in (test, ref)]
    response_checks = []
    if len(responses[0]) != len(responses[1]):
        raise ValueError("Different fixture counts")
    for actual, expected in zip(*responses):
        a, b = actual["result"], expected["result"]
        tokens_a, tokens_b = a.get("output_ids"), b.get("output_ids")
        if tokens_a is None or tokens_b is None:
            raise ValueError("Missing output_ids; cannot establish identical MTP histories")
        response_checks.append({"case": expected["case"],
                                "same_fixture": actual["case"] == expected["case"] and actual["prompt_length"] == expected["prompt_length"],
                                "same_tokens": tokens_a == tokens_b,
                                "test_accept_histogram": a["meta_info"].get("spec_correct_drafts_histogram"),
                                "ref_accept_histogram": b["meta_info"].get("spec_correct_drafts_histogram")})
    files_a = {p.name for p in (test / "trace").glob("*.pt")}
    files_b = {p.name for p in (ref / "trace").glob("*.pt")}
    records, phases = [], {}
    missing = sorted(files_a ^ files_b)
    for name in sorted(files_a & files_b):
        a, b = (torch.load(arm / "trace" / name) for arm in (test, ref))
        same_shape = a["shape"] == b["shape"] and a["attn"].shape == b["attn"].shape
        same_meta = all(a[k] == b[k] for k in ("role", "rank", "mode", "layer"))
        same_inputs = all(torch.equal(a[k], b[k]) for k in ("positions", "seq_lens"))
        finite = bool(torch.isfinite(a["attn"]).all() and torch.isfinite(b["attn"]).all())
        max_ref = b["attn"].abs().max().item()
        max_abs = (a["attn"] - b["attn"]).abs().max().item() if same_shape and finite else None
        relative = max_abs / max(max_ref, 1e-30) if max_abs is not None else None
        passed = same_shape and same_meta and same_inputs and finite and relative <= tolerance
        record = {"file": name, "mode": b["mode"], "role": b["role"],
                  "same_shape": same_shape, "same_metadata": same_meta,
                  "same_positions_and_lengths": same_inputs, "finite": finite,
                  "ref_max": max_ref, "max_abs": max_abs, "relative_linf": relative,
                  "passed": passed}
        records.append(record)
        key = f"{b['role']}:{b['mode']}"
        phase = phases.setdefault(key, {"observations": 0, "max_relative_linf": 0., "max_ref_signal": 0., "passed": True})
        phase["observations"] += 1
        phase["max_relative_linf"] = max(phase["max_relative_linf"], relative or 0.)
        phase["max_ref_signal"] = max(phase["max_ref_signal"], max_ref)
        phase["passed"] &= passed
    required = {"target:EXTEND", "target:TARGET_VERIFY", "draft:DECODE", "draft:DRAFT_EXTEND_V2"}
    missing_phases = sorted(required - phases.keys())
    # Refuse a vacuous pass from absent or collapsed attention signals.
    signals_present = all(v["max_ref_signal"] > 1e-8 for v in phases.values())
    passed = bool(records) and not missing and not missing_phases and signals_present
    passed &= all(r["passed"] for r in records)
    passed &= all(r["same_tokens"] and r["same_fixture"] for r in response_checks)
    return {"validity": "MTP_NUMERICAL_DIAGNOSTIC", "passed": passed,
            "tolerance": tolerance, "test": str(test), "reference": str(ref),
            "missing_trace_files": missing, "missing_phases": missing_phases,
            "signals_present": signals_present, "phases": phases,
            "responses": response_checks, "records": records,
            "limitations": "Scaled model and observed acceptance only; no TP8, performance, or SLO claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args()
    result = compare(args.test, args.reference, args.tolerance)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
