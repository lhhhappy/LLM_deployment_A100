#!/usr/bin/env python3
"""Audit one complete dev raw JSONL against the frozen cohort (CPU only).

Usage: python3 scripts/analysis/review_raw.py RAW [--out REPORT.json]

This explains cache accounting and TTFT populations; it is not the formal SLO
verdict. Use score_formal.py / level_verdict.py for that decision.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "s1-dev/harness"))
from s1_common import in_ttft_gate, load_index  # noqa: E402


def edge_role(row, idx):
    return row["edge_type"] + "/" + ("head" if idx == 0 else "followup")


def review(raw_path, data_root):
    rows, _, _ = load_index(str(data_root))
    raw = raw_path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    by_id = {}
    duplicate = []
    for record in records:
        rid = record["req_id"]
        if rid in by_id:
            duplicate.append(rid)
        by_id[rid] = record
    missing = sorted(set(rows) - set(by_id))
    extra = sorted(set(by_id) - set(rows))
    mismatched = sorted(
        rid for rid in set(rows) & set(by_id)
        if by_id[rid].get("idx_in_chain") != rows[rid]["_idx_in_chain"]
        or by_id[rid].get("edge_type") != rows[rid]["edge_type"]
        or by_id[rid].get("phase") != rows[rid]["phase"]
        or by_id[rid].get("uncached_expected") != rows[rid]["uncached_expected"]
    )
    cohort = defaultdict(lambda: dict(n=0, prompt=0, frozen_uncached=0))
    for row in rows.values():
        item = cohort[edge_role(row, row["_idx_in_chain"])]
        item["n"] += 1
        item["prompt"] += row["glm_tokens"]
        item["frozen_uncached"] += row["uncached_expected"]
    result = dict(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        n=len(records),
        expected_n=len(rows),
        duplicate_req_ids=sorted(set(duplicate)),
        missing_req_ids=missing,
        extra_req_ids=extra,
        cohort_mismatches=mismatched,
        errors=sum(bool(r.get("error")) for r in records),
        cohort_edge_roles=dict(cohort),
    )
    result["complete"] = not (duplicate or missing or extra or mismatched)
    if not result["complete"]:
        return result

    accounting = defaultdict(lambda: dict(n=0, prompt=0, actual=0, frozen=0, positive_excess=0))
    for rid, r in by_id.items():
        actual = r["prompt_tokens"] - r["cached_tokens"]
        frozen = rows[rid]["uncached_expected"]
        item = accounting[edge_role(rows[rid], rows[rid]["_idx_in_chain"])]
        item["n"] += 1
        item["prompt"] += r["prompt_tokens"]
        item["actual"] += actual
        item["frozen"] += frozen
        item["positive_excess"] += max(0, actual - frozen)
    for item in accounting.values():
        item["net_excess"] = item["actual"] - item["frozen"]
    fast = [r for r in records if in_ttft_gate(r, "fast_intra")]
    naive = [r for r in records if r["phase"] == "intra"
             and r["uncached_expected"] <= 4096 and r["ttft_s"] > 3]
    result.update(
        all_actual=sum(r["prompt_tokens"] - r["cached_tokens"] for r in records),
        by_edge_role=dict(accounting),
        fast_correct_count=len(fast),
        fast_correct_over=sum(r["ttft_s"] > 3 for r in fast),
        fast_naive_over=len(naive),
        naive_heads=sum(r["idx_in_chain"] == 0 for r in naive),
        followup_positive_excess_all=sum(
            max(0, r["prompt_tokens"] - r["cached_tokens"] - rows[r["req_id"]]["uncached_expected"])
            for r in records if r["idx_in_chain"] > 0
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="one complete dev raw_*.jsonl")
    parser.add_argument("--data-root", type=Path, default=ROOT / "s1-dev/data/dev-combined-v1")
    parser.add_argument("--out", type=Path, help="optional report path; stdout is always printed")
    args = parser.parse_args()
    result = review(args.raw, args.data_root)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    print(output, end="")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
