#!/usr/bin/env python3
"""Incrementally repair an existing long-chain dataset; never regenerate bodies.

Examples:
  repair_metadata.py --parent cache/s1-dev-longchain-v4 --out cache/v5-fixed \
    --set v5-fixed --outputs-from cache/s1-dev-longchain-v5/requests.jsonl
  repair_metadata.py --parent cache/s1-dev-longchain-v4 --out cache/v4-order-v3 \
    --set v4-order-v3 --cohort-from cache/s1-dev-longchain-v3/cohort.json

Gap patches are JSONL, keyed by req_id OR (chain_id, idx_in_chain), with gap_ms.
--gap-basis capped-replay requires already processed organizer-style gaps;
raw-end-to-start-sensitivity and synthetic-sensitivity are explicitly diagnostic.
No distribution, request placement or tool/think decomposition is inferred.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path

if __package__:
    from .longchain_metadata import checked_output, digest, publish, read_rows, req_id
else:
    from longchain_metadata import checked_output, digest, publish, read_rows, req_id


def apply_gaps(rows, patches, basis):
    if basis not in ("capped-replay", "raw-end-to-start-sensitivity", "synthetic-sensitivity"):
        raise ValueError("an explicit gap basis is required; raw gaps are not processed replay gaps")
    by_id = {req_id(r): r for r in rows}
    chains = defaultdict(list)
    for row in rows:
        chains[row["chain_id"]].append(row)
    positions, heads = {}, set()
    for cid, group in chains.items():
        group.sort(key=lambda r: (r.get("dispatch_offset_ms") or 0, r["logical_call_id"]))
        heads.add(req_id(group[0]))
        positions.update({(cid, i): req_id(row) for i, row in enumerate(group)})
    resolved, seen = [], set()
    for patch in patches:
        rid = patch.get("req_id")
        positional = "chain_id" in patch or "idx_in_chain" in patch
        if positional:
            index = patch.get("idx_in_chain")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError("idx_in_chain must be a nonnegative integer")
            found = positions.get((patch.get("chain_id"), index))
            if found is None or rid is not None and rid != found:
                raise ValueError("gap patch position/ID mismatch")
            rid = found
        if rid not in by_id or rid in seen or rid in heads:
            raise ValueError(f"unknown, duplicate or head gap patch: {rid}")
        gap = patch.get("gap_ms")
        if (not isinstance(gap, (int, float)) or isinstance(gap, bool)
                or not math.isfinite(gap) or gap < 0):
            raise ValueError(f"invalid gap_ms: {rid}")
        if basis == "capped-replay" and gap > 310000:
            raise ValueError("capped replay gap exceeds min(tool,300s)+min(think,10s)")
        seen.add(rid)
        resolved.append((rid, gap))
    for rid, gap in resolved:
        row = by_id[rid]
        row.update(replay_gap_ms=gap, gap_valid=True, gap_imputed=True,
                   net_think_ms=None, tool_union_ms=None)
        row.pop("gap_regap_factor", None)
    return len(resolved)


def execute(args):
    parent, out = checked_output(args.parent, args.out)
    original = read_rows(parent / "requests.jsonl")
    edited = deepcopy(original)
    operation = {"kind": "incremental-metadata-repair", "inputs": {},
                 "scope": "diagnostic workload; no formal representativeness claim"}
    if args.outputs_from:
        donor = read_rows(args.outputs_from)
        if len(donor) != len(edited) or [req_id(r) for r in donor] != [req_id(r) for r in edited]:
            raise ValueError("output donor IDs/order differ from parent")
        for row, candidate in zip(edited, donor):
            if {k: v for k, v in row.items() if k != "max_output_i"} != {
                    k: v for k, v in candidate.items() if k != "max_output_i"}:
                raise ValueError("output donor changed fields other than output budget")
            row["max_output_i"] = candidate["max_output_i"]
        operation["inputs"]["outputs"] = dict(path=str(args.outputs_from.resolve()),
                                              sha256=digest(args.outputs_from))
    if args.gap_patch:
        count = apply_gaps(edited, read_rows(args.gap_patch), args.gap_basis)
        operation["inputs"]["gaps"] = dict(path=str(args.gap_patch.resolve()),
            sha256=digest(args.gap_patch), basis=args.gap_basis, applied=count,
            note="source timestamps are not modified; harness still applies its per-chain gap cap")
    elif args.gap_basis:
        raise ValueError("--gap-basis requires --gap-patch")
    result = publish(parent, out, edited, name=args.set, operation=operation,
                     cohort_path=args.cohort_from, metadata_only=args.metadata_only)
    receipt = result["metadata_patches"][-1]
    print(f"{result['status']}: {out}; changed requests={receipt['changed_requests']}; "
          f"fields={receipt['fields']}; missing assets={len(result['missing_artifacts'])}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", required=True)
    parser.add_argument("--outputs-from", type=Path)
    parser.add_argument("--cohort-from", type=Path)
    parser.add_argument("--gap-patch", type=Path)
    parser.add_argument("--gap-basis", choices=("capped-replay", "raw-end-to-start-sensitivity", "synthetic-sensitivity"))
    parser.add_argument("--metadata-only", action="store_true", help="prepare without missing bodies; stays incomplete")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
