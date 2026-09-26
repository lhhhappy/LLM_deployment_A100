#!/usr/bin/env python3
"""Independently recheck frozen DCP model traces on CPU; never runs the model.

This deliberately checks every stored row, including rejected draft rows.
Use only on trusted project-generated Torch trace files. The result is a
numerical diagnostic, not an SLO result or proof of final-source execution.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def audit(test, reference):
    summaries = [json.loads((p / "summary.json").read_text()) for p in (test, reference)]
    files = [{p.name for p in (root / "trace").glob("*.pt")} for root in (test, reference)]
    if files[0] != files[1] or len(files[0]) != 504:
        raise ValueError("This frozen fixture requires the same 504 traces in both arms")
    rows, phases, hashes = [], {}, []
    for name in sorted(files[0]):
        paths = [p / "trace" / name for p in (test, reference)]
        a, b = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        hashes.extend({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths)
        same_meta = all(a[k] == b[k] for k in ("role", "rank", "mode", "layer", "shape", "rows"))
        same_inputs = all(torch.equal(a[k], b[k]) for k in ("seq_lens", "positions"))
        x, y = a["attn"].float(), b["attn"].float()
        same_shape = x.shape == y.shape
        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        rel = None
        if same_shape and finite and x.numel():
            rel = float(((x - y).abs().flatten(1).amax(1)
                         / y.abs().flatten(1).amax(1).clamp_min(1e-30)).max())
        passed = bool(same_meta and same_inputs and same_shape and finite
                      and y.abs().max() > 1e-6 and rel is not None and rel <= .01)
        phase = f"{a['role']}:{a['mode']}"
        phases[phase] = phases.get(phase, 0) + 1
        rows.append({"file": name, "phase": phase, "same_metadata": same_meta,
                     "same_inputs": same_inputs, "finite": finite,
                     "max_row_relative_linf": rel, "passed": passed,
                     "test_graph_replay": a.get("replay_id"),
                     "test_graph_key": a.get("graph_key")})
    required = {"draft:EXTEND", "draft:DECODE", "draft:DRAFT_EXTEND_V2",
                "target:EXTEND", "target:TARGET_VERIFY"}
    outputs = [json.loads((p / "responses.json").read_text()) for p in (test, reference)]
    same_outputs = len(outputs[0]) == len(outputs[1]) == 4 and all(
        a["case"] == b["case"] and a["prompt_length"] == b["prompt_length"]
        and a["result"]["output_ids"] == b["result"]["output_ids"]
        for a, b in zip(*outputs))
    graph_required = bool(summaries[0].get("graph_phase_rank_replays"))
    graph_coverage = {}
    for phase in ("draft:DECODE", "draft:DRAFT_EXTEND_V2", "target:TARGET_VERIFY"):
        observed = [r for r in rows if r["phase"] == phase]
        graph_coverage[phase] = bool(observed) and all(
            r["test_graph_replay"] is not None and r["test_graph_key"] is not None for r in observed)
    return {"validity": "CPU_RECHECK_OF_AUTHOR_GPU_TRACES",
            "test": str(test), "reference": str(reference),
            "passed": required.issubset(phases) and same_outputs and all(r["passed"] for r in rows)
                      and (not graph_required or all(graph_coverage.values())),
            "records": rows, "source_files": hashes, "phases": phases,
            "same_outputs": same_outputs, "graph_required": graph_required,
            "graph_coverage": graph_coverage,
            "max_row_relative_linf": max(r["max_row_relative_linf"] or 0 for r in rows),
            "acceptance_histograms": [r["result"]["meta_info"].get("spec_correct_drafts_histogram")
                                      for r in outputs[0]],
            "limits": "Stored samples only; old source, low addresses, TP2, no HiCache or acceptance>1 claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.test, args.reference)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "source_files")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
