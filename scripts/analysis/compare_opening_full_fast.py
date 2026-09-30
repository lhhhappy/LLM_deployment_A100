#!/usr/bin/env python3
"""Compare one opening and one full replay pair on the same fast request IDs.

Example:
  python3 scripts/analysis/compare_opening_full_fast.py \
    --opening-base evidence/L074-official_0925a_opening_n26/window/raw.jsonl \
    --opening-candidate evidence/L078-opening_q1b_cold6144_n26/window/raw.jsonl \
    --full-base evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/raw_s1-dev-longchain_N30_1790280416.jsonl \
    --full-candidate evidence/L081-cap6144_full_n30/window/analysis/20260925T160429Z-e1169707/raw.jsonl.gz \
    --output /tmp/fast-fourway.json

The full candidate may be an open window. This script compares completed common
IDs only and never assigns a full-run verdict.
"""

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "s1-dev/harness"))
from s1_common import in_ttft_gate  # noqa: E402


def read_raw(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        return {row["req_id"]: row for row in map(json.loads, file) if not row.get("error")}


def wait_seconds(row):
    start = row.get("t_exec_start_s")
    recv = row.get("t_recv_s")
    return None if start is None or recv is None else start - recv


def bucket(row):
    work = row["prompt_tokens"] - row["cached_tokens"]
    return "<=2048" if work <= 2048 else "2049-4096" if work <= 4096 else ">4096"


def compare(base, candidate, fast_ids):
    old = {rid for rid in fast_ids if base[rid]["ttft_s"] > 3.0}
    new = {rid for rid in fast_ids if candidate[rid]["ttft_s"] > 3.0}
    gained = new - old
    fixed = old - new
    waits = [wait_seconds(candidate[rid]) - wait_seconds(base[rid]) for rid in gained
             if wait_seconds(candidate[rid]) is not None and wait_seconds(base[rid]) is not None]
    buckets = {name: sum(bucket(candidate[rid]) == name for rid in gained)
               for name in ("<=2048", "2049-4096", ">4096")}
    return {
        "fast_count": len(fast_ids),
        "base_over_3s": len(old), "candidate_over_3s": len(new),
        "fixed": len(fixed), "new": len(gained), "persistent": len(old & new),
        "new_same_cached_tokens": sum(base[rid]["cached_tokens"] == candidate[rid]["cached_tokens"]
                                      for rid in gained),
        "new_actual_uncached_bucket": buckets,
        "new_median_recv_to_exec_delta_s": statistics.median(waits) if waits else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("opening-base", "opening-candidate", "full-base", "full-candidate"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name.replace("-", "_")) for name in
             ("opening-base", "opening-candidate", "full-base", "full-candidate")}
    records = {name: read_raw(path) for name, path in paths.items()}
    common = set.intersection(*(set(rows) for rows in records.values()))
    fast_ids = {rid for rid in common if in_ttft_gate(records["full-base"][rid], "fast_intra")}
    result = {
        "scope": "four-way completed common IDs; full candidate may be open; diagnostic only",
        "input_paths": {name: str(path) for name, path in paths.items()},
        "common_request_ids": len(common),
        "fast_request_ids": len(fast_ids),
        "opening": compare(records["opening-base"], records["opening-candidate"], fast_ids),
        "full_window": compare(records["full-base"], records["full-candidate"], fast_ids),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
