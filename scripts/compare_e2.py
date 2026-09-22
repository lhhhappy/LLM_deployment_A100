#!/usr/bin/env python3
"""Join E2 stock/off/on by immutable request ID and prompt hash; no HTTP writes."""
import argparse
import json
from pathlib import Path


def quantile(values, p=.95):
    values = sorted(values)
    # Exact unchanged harness s1_common.q convention (not interpolated percentile).
    return values[min(len(values) - 1, int(p * len(values)))] if values else None


def rows(path):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    result = {r["req_id"]: r for r in data}
    assert len(data) == len(result), "duplicate ID"
    return result


def compare(root, case):
    groups = {v: rows(root / f"{v}_{case}" / "requests.jsonl")
              for v in ("stock", "off", "on")}
    assert groups["stock"].keys() == groups["off"].keys() == groups["on"].keys()
    joined = []
    for rid, stock in groups["stock"].items():
        triplet = [groups[v][rid] for v in ("stock", "off", "on")]
        for key in ("prompt_sha256", "rendered_prompt_tokens", "prompt_tokens", "output_tokens"):
            assert len({r[key] for r in triplet}) == 1, (rid, key)
        assert not any(r.get("error") for r in triplet)
        on = triplet[2]
        joined.append({"req_id": rid, "chain_id": stock["chain_id"],
                       "idx_in_chain": stock["idx_in_chain"],
                       "prompt_sha256": stock["prompt_sha256"],
                       "prompt_tokens": stock["prompt_tokens"],
                       "fast_intra": stock["idx_in_chain"] > 0 and
                           stock["phase"] not in ("turn_start", "context_reset") and
                           (stock["uncached_expected"] or 0) <= 4096,
                       **{v + "_cached": groups[v][rid]["cached_tokens"] for v in groups},
                       "role_pred_cached": on["predictions"]["role_conservative"]["cached_tokens"],
                       "stock_pred_cached": on["predictions"]["stock"]["cached_tokens"]})
    fast = [r for r in joined if r["fast_intra"]]
    summary = {"requests": len(joined), "fast_intra_n": len(fast),
               "off_equals_stock": sum(r["off_cached"] == r["stock_cached"] for r in joined),
               "on_better": sum(r["on_cached"] > r["stock_cached"] for r in joined),
               "on_equal": sum(r["on_cached"] == r["stock_cached"] for r in joined),
               "on_worse": sum(r["on_cached"] < r["stock_cached"] for r in joined),
               "on_within_64_role_prediction": sum(abs(r["on_cached"] - r["role_pred_cached"]) <= 64 for r in joined),
               "fast_uncached_p95": {v: quantile([r["prompt_tokens"] - r[v + "_cached"] for r in fast])
                                     for v in ("stock", "off", "on", "role_pred", "stock_pred")},
               "total_uncached": {v: sum(r["prompt_tokens"] - r[v + "_cached"] for r in joined)
                                  for v in ("stock", "off", "on")}}
    return {"summary": summary, "requests": joined}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", type=Path)
    args = p.parse_args()
    print(json.dumps({c: compare(args.run_root, c) for c in
                      ("smoke", "reminder_heavy", "strict_append", "stock20")}, indent=2))
