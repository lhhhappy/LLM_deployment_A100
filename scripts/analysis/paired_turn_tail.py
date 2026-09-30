#!/usr/bin/env python3
"""Audit turn-start p95 and per-request churn between two complete replay raw files.

Uses the original dev harness bucket and quantile through score_formal.load_harness().
The two raw files must contain the same turn-start request IDs. Official attempts do
not expose request rows, so this tool only diagnoses local replays.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import score_formal  # noqa: E402


def selected(path, scorer):
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if scorer.in_ttft_gate(row, "turn_start"):
                rid = row["req_id"]
                if rid in rows:
                    raise ValueError(f"duplicate turn-start request: {rid}")
                rows[rid] = row
    return rows


def describe(rows, scorer):
    vals = sorted(r["ttft_s"] for r in rows.values())
    return {
        "n": len(vals),
        "p50_s": scorer.q(vals, 0.5),
        "p95_s": scorer.q(vals, 0.95),
        "over_15s": sum(v > 15 for v in vals),
        "mean_s": sum(vals) / len(vals),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    scorer = score_formal.load_harness()
    a, b = selected(args.baseline, scorer), selected(args.candidate, scorer)
    if set(a) != set(b):
        raise ValueError(f"turn-start ID sets differ: baseline={len(a)}, candidate={len(b)}")
    ids = sorted(a)
    changed = []
    for rid in ids:
        before, after = a[rid], b[rid]
        row = {
            "req_id": rid,
            "baseline_ttft_s": before["ttft_s"],
            "candidate_ttft_s": after["ttft_s"],
            "baseline_queue_s": before.get("queue_time_s"),
            "candidate_queue_s": after.get("queue_time_s"),
            "baseline_cached_tokens": before.get("cached_tokens"),
            "candidate_cached_tokens": after.get("cached_tokens"),
        }
        if (before["ttft_s"] > 15) != (after["ttft_s"] > 15):
            changed.append(row)
    deltas = sorted(b[rid]["ttft_s"] - a[rid]["ttft_s"] for rid in ids)
    result = {
        "scope": "local complete same-ID turn-start bucket; not official request-level data",
        "baseline": {"raw": str(args.baseline), "sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(), **describe(a, scorer)},
        "candidate": {"raw": str(args.candidate), "sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(), **describe(b, scorer)},
        "paired": {
            "n": len(ids),
            "repaired_over_15s": sum(a[rid]["ttft_s"] > 15 >= b[rid]["ttft_s"] for rid in ids),
            "new_over_15s": sum(b[rid]["ttft_s"] > 15 >= a[rid]["ttft_s"] for rid in ids),
            "regressed_more_than_3s": sum(d > 3 for d in deltas),
            "improved_more_than_3s": sum(d < -3 for d in deltas),
            "delta_ttft_p50_s": scorer.q(deltas, 0.5),
            "delta_ttft_p95_s": scorer.q(deltas, 0.95),
            "crossed_15s": changed,
        },
    }
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
