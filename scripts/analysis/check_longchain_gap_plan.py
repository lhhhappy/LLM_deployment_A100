#!/usr/bin/env python3
"""Report effective waits by calling the unchanged S1 harness gap planner."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "sum_ms": sum(values),
            "mean_ms": sum(values) / len(values),
            **{f"p{int(p * 100)}_ms": values[min(len(values) - 1, int(p * len(values)))]
               for p in (.5, .9, .95, .99)},
            "max_ms": values[-1], "over60s": sum(v > 60000 for v in values),
            "over300s": sum(v > 300000 for v in values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.cap_seconds < 0:
        parser.error("cap must be nonnegative; zero means uncapped")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.harness_dir.resolve()))
    harness = importlib.import_module("s1_loadgen")
    common = importlib.import_module("s1_common")
    results = {}
    for root in args.root:
        rows, _, _ = common.load_index(str(root))
        cohort = json.loads((root / "cohort.json").read_text())
        chains = cohort["chains"]
        ids = [rid for c in chains for rid in c["req_ids"]]
        if len(ids) != len(set(ids)) or set(ids) != set(rows):
            raise ValueError(f"cohort/request coverage mismatch: {root}")
        plan, stats = harness.build_gap_plan(chains, rows, args.cap_seconds * 1000)
        original_heads, effective_heads, original_tail, effective_tail = [], [], [], []
        for chain in chains:
            original = [int(rows[rid].get("replay_gap_ms") or 0) for rid in chain["req_ids"]]
            effective = plan[chain["chain_id"]]
            expected = min(sum(original), args.cap_seconds * 1000) if args.cap_seconds else sum(original)
            if len(original) != len(effective) or min(effective) < 0 or sum(effective) != expected:
                raise ValueError(f"gap plan invariant failed: {chain['chain_id']}")
            original_heads.append(original[0])
            effective_heads.append(effective[0])
            original_tail.extend(original[1:])
            effective_tail.extend(effective[1:])
        results[root.name] = {"root": str(root.resolve()),
            "requests_sha256": hashlib.sha256((root / "requests.jsonl").read_bytes()).hexdigest(),
            "cohort_id": cohort["cohort_sha256"], "harness_stats": stats,
            "original_nonheads": distribution(original_tail),
            "effective_nonheads": distribution(effective_tail),
            "original_heads": distribution(original_heads), "effective_heads": distribution(effective_heads)}
    report = {"scope": "Original harness effective waiting only; no restored-source or performance claim",
              "harness_path": harness.__file__,
              "harness_sha256": hashlib.sha256(Path(harness.__file__).read_bytes()).hexdigest(),
              "datasets": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, result in results.items():
        print(name, "compressed_chains", result["harness_stats"]["n_chains_compressed"],
              "effective_nonheads", json.dumps(result["effective_nonheads"]))


if __name__ == "__main__":
    main()
