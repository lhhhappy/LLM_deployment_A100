#!/usr/bin/env python3
"""Repair v5/v5g publication and derive an explicitly diagnostic tail-only arm; reuse bodies."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

if __package__:
    from .longchain_metadata import digest, publish, read_rows, req_id
else:
    from longchain_metadata import digest, publish, read_rows, req_id


def tail_only(base, stretched, heads):
    """Keep ordinary v5 waits; admit only long, already-tail synthetic stretches.

    This is a controlled sensitivity rule, not a reconstruction of source event
    positions. P50/P90 cannot move because only original values above P90 grow.
    The source raw >60s count is an upper-bound check, never a fitting target.
    """
    if [req_id(r) for r in base] != [req_id(r) for r in stretched]:
        raise ValueError("gap proposal ID/order mismatch")
    gaps = sorted((r.get("replay_gap_ms") or 0) for r in base if req_id(r) not in heads)
    threshold = gaps[min(len(gaps)-1, int(.9 * len(gaps)))]
    result, changed = deepcopy(base), 0
    for original, proposal, row in zip(base, stretched, result):
        old, new = original.get("replay_gap_ms") or 0, proposal.get("replay_gap_ms") or 0
        if (req_id(row) not in heads and row.get("split") == "synthetic"
                and old > threshold and new > max(60000, old)):
            if new > 310000:
                raise ValueError("duration proposal exceeds the per-step replay cap")
            row["replay_gap_ms"] = new
            if "gap_regap_factor" in proposal:
                row["gap_regap_factor"] = proposal["gap_regap_factor"]
            changed += 1
    after = sorted((r.get("replay_gap_ms") or 0) for r in result if req_id(r) not in heads)
    for quantile in (.5, .9):
        i = min(len(gaps)-1, int(quantile * len(gaps)))
        if after[i] != gaps[i]:
            raise ValueError("ordinary waiting quantile changed")
    if len(after) == 5290 and sum(g > 60000 for g in after) > 340:
        raise ValueError("processed long-gap count exceeds the supplied raw source upper bound")
    return result, dict(kind="tail-only-gap-sensitivity", threshold_ms=threshold, changed_requests=changed,
        rule="synthetic non-head; old gap > v5 P90; duration proposal > max(old gap,60s); keep all other gaps",
        limitation="event placement remains synthetic; no claim of recovered source timing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("v5", "v5g", "v5g-tail"))
    args = parser.parse_args()
    parent = args.source_root / "s1-dev-longchain-v4"
    original = read_rows(parent / "requests.jsonl")
    cohort = json.loads((parent / "cohort.json").read_text())
    heads = {c["req_ids"][0] for c in cohort["chains"]}
    receipt_path = args.out_root / "publication.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    previous = None
    base_v5 = None
    for version in ("v5", "v5g", "v5g-tail"):
        source = args.source_root / ("s1-dev-longchain-" + ("v5g" if version == "v5g-tail" else version))
        rows = read_rows(source / "requests.jsonl")
        tail_receipt = None
        if version == "v5":
            base_v5 = deepcopy(rows)
        elif version == "v5g-tail":
            rows, tail_receipt = tail_only(base_v5, rows, heads)
        if [req_id(r) for r in rows] != [req_id(r) for r in original]:
            raise ValueError("request identity/order mismatch")
        changes = Counter(k for a, b in zip(original, rows) for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        if version == "v5" and changes.keys() - {"max_output_i"}:
            raise ValueError("v5 changes more than output budgets")
        if previous is not None:
            delta = {k for a, b in zip(previous, rows) for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
            if delta - {"replay_gap_ms", "gap_regap_factor"}:
                raise ValueError("v5g differs from v5 beyond the recorded gap transform")
        previous = rows
        if args.variant and version != args.variant:
            continue
        name = "s1-dev-longchain-" + version + "-review-0927"
        out = args.out_root / name
        operation = dict(kind="repair-existing-" + version,
            source_requests_sha256=digest(source / "requests.jsonl"),
            source_manifest_sha256=digest(source / "manifest.json"),
            gap_scope="inherited v4" if version == "v5" else
                "duration-based sensitivity; imputed gaps scaled per chain; not recovered source timing",
            quality="output budgets and raw source outputs use different tokenizers; not formal workload calibration")
        if tail_receipt:
            operation.update(tail_receipt)
            operation["gap_scope"] = "tail only; ordinary v5 waiting retained"
        manifest = publish(parent, out, rows, name=name, operation=operation)
        if read_rows(out / "requests.jsonl") != rows:
            raise ValueError("published request values differ from frozen input")
        if json.loads((out / "cohort.json").read_text())["chains"] != cohort["chains"]:
            raise ValueError("cohort changed")
        gaps = sorted((r.get("replay_gap_ms") or 0) / 1000 for r in rows if req_id(r) not in heads)
        q = lambda p: gaps[min(len(gaps) - 1, int(p * len(gaps)))]
        receipts[version] = dict(root=str(out.resolve()), set=name, rows=len(rows), chains=len(cohort["chains"]),
            status=manifest["status"], request_values_equal_to_existing=version != "v5g-tail",
            request_values_equal_to_plan=True, changes_from_v4=dict(changes), tail_rule=tail_receipt,
            requests_sha256=manifest["artifacts"]["requests.jsonl"], cohort_sha256=cohort["cohort_sha256"],
            gap_seconds=dict(count=len(gaps), p50=q(.5), p90=q(.9), p95=q(.95), p99=q(.99),
                             mean=sum(gaps)/len(gaps), over60=sum(g > 60 for g in gaps), total=sum(gaps)),
            body_refs={p: manifest["artifacts"][p] for p in manifest["artifacts"] if p.startswith("bodies/")})
        print(version, "PUBLISHED", receipts[version]["gap_seconds"], flush=True)
    receipt_path.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
