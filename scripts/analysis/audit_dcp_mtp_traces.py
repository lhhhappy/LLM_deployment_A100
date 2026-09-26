#!/usr/bin/env python3
"""Independently recheck frozen DCP model traces on CPU; never runs the model.

By default this checks every stored row, including rejected draft rows.
For newer records --accepted-rows independently rebuilds the acceptance mask
from the retained per-request lengths; it still reports rejected-row errors.
Use only on trusted project-generated Torch trace files. The result is a
numerical diagnostic, not an SLO result or proof of final-source execution.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def audit(test, reference, expected_traces=504, expected_cases=4, accepted_rows=False):
    summaries = [json.loads((p / "summary.json").read_text()) for p in (test, reference)]
    files = [{p.name for p in (root / "trace").glob("*.pt")} for root in (test, reference)]
    if files[0] != files[1] or len(files[0]) != expected_traces:
        raise ValueError(f"This fixture requires the same {expected_traces} traces in both arms")
    if any(s["trace_files"] != expected_traces for s in summaries):
        raise ValueError("Summary and retained trace counts disagree")
    rows, phases, hashes = [], {}, []
    for name in sorted(files[0]):
        paths = [p / "trace" / name for p in (test, reference)]
        a, b = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        hashes.extend({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths)
        same_meta = all(a[k] == b[k] for k in ("role", "rank", "mode", "layer", "shape", "rows"))
        same_inputs = all(torch.equal(a[k], b[k]) for k in ("seq_lens", "positions"))
        x_all, y_all = a["attn"].float(), b["attn"].float()
        def mask(record):
            if not accepted_rows or record["mode"] != "DRAFT_EXTEND_V2":
                return torch.ones(len(record["attn"]), dtype=torch.bool)
            width, front = record["draft_window_width"], record["num_front_tokens"]
            counts = record["accepted_tokens"]
            assert width > 0 and len(counts) * width == record["shape"][0]
            assert bool(((counts + front >= 0) & (counts + front <= width)).all())
            full = (torch.arange(width)[None, :] < counts[:, None] + front).reshape(-1)
            expected = full[record["rows"]]
            assert torch.equal(record["valid_rows"], expected), "Captured acceptance mask disagrees with lengths"
            return expected
        ma, mb = mask(a), mask(b)
        same_meta = same_meta and torch.equal(ma, mb) and a.get("case") == b.get("case")
        x, y = x_all[ma], y_all[mb]
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
                     "checked_rows": int(mb.sum()), "rejected_rows": int((~mb).sum()),
                     "all_rows_max_abs": float((x_all - y_all).abs().max()),
                     "test_graph_replay": a.get("replay_id"),
                     "test_graph_key": a.get("graph_key")})
    required = {"draft:EXTEND", "draft:DECODE", "draft:DRAFT_EXTEND_V2",
                "target:EXTEND", "target:TARGET_VERIFY"}
    outputs = [json.loads((p / "responses.json").read_text()) for p in (test, reference)]
    same_outputs = len(outputs[0]) == len(outputs[1]) == expected_cases and all(
        a["case"] == b["case"] and a["prompt_length"] == b["prompt_length"]
        and a.get("prompt_sha256") == b.get("prompt_sha256")
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
            "accepted_rows_only": accepted_rows,
            "limits": "Stored samples only; use source, acceptance and restore receipts to establish coverage. No TP8/SLO claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-traces", type=int, default=504)
    parser.add_argument("--expected-cases", type=int, default=4)
    parser.add_argument("--accepted-rows", action="store_true")
    args = parser.parse_args()
    result = audit(args.test, args.reference, args.expected_traces, args.expected_cases, args.accepted_rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "source_files")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
