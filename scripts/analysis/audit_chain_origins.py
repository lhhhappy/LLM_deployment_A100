#!/usr/bin/env python3
"""Audit replay heads using the original harness and local public body structure.

Reads no remote database and emits no prompt text. Source cache counters are
historical observations, not the cache state of a future replay.
"""
import argparse
import collections
import csv
import gzip
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.s1_dev.resolve()
    data = source / "data/dev-combined-v1"
    spec = importlib.util.spec_from_file_location(
        "s1_common", source / "harness/s1_common.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    rows, chains, groups = harness.load_index(str(data))
    heads = {group[0]["_req_id"]: group[0] for group in groups.values()}
    structures = {}
    with gzip.open(data / "bodies/dev-combined-v1.jsonl.gz", "rt") as stream:
        for line in stream:
            body = json.loads(line)
            rid = body["req_id"]
            if rid not in heads:
                continue
            if rid in structures:
                raise ValueError(f"duplicate head body: {rid}")
            messages = body.get("messages") or []
            structures[rid] = {
                "nonempty_top_level_system": bool(body.get("system")),
                "first_message_role": messages[0].get("role") if messages else None,
                "message_count": len(messages),
                "has_assistant_message": any(m.get("role") == "assistant" for m in messages),
            }
    if structures.keys() != heads.keys():
        raise ValueError("missing head bodies; audit incomplete")
    detail = []
    strata = collections.defaultdict(collections.Counter)
    for rid, row in sorted(heads.items()):
        chain = chains[row["chain_id"]]
        for key in ("session_id", "chain_index"):
            if row[key] != chain[key]:
                raise ValueError(f"request/chain metadata mismatch: {rid}: {key}")
        if row["phase"] == "session_start" and row["chain_index"] == 0:
            origin = "source_session_start_annotation"
        elif row["phase"] != "session_start" and row["chain_index"] > 0:
            origin = "source_mid_session_annotation"
        else:
            origin = "inconsistent_or_unknown"
        item = {
            "req_id": rid,
            "chain_id": row["chain_id"],
            "session_id": row["session_id"],
            "chain_index": row["chain_index"],
            "replay_idx_in_chain": row["_idx_in_chain"],
            "source_phase": row["phase"],
            "source_edge_type": row["edge_type"],
            "origin": origin,
            "gate_seconds": harness.phase_gate(row),
            "source_cached_input_tokens": row.get("source_cached_input_tokens"),
            **structures[rid],
        }
        detail.append(item)
        counter = strata[origin]
        counter["n"] += 1
        counter["nonempty_top_level_system"] += item["nonempty_top_level_system"]
        counter["first_message_role=" + str(item["first_message_role"])] += 1
        counter["has_assistant_message"] += item["has_assistant_message"]
        counter["source_cache_read_positive"] += (item["source_cached_input_tokens"] or 0) > 0
        counter["source_cache_read_missing"] += item["source_cached_input_tokens"] is None
    summary = {
        "scope": "public 722-request prefix sample, not source database or formal replay",
        "source": str(data),
        "serving_requests": len(rows),
        "replay_heads": len(heads),
        "head_phases": dict(collections.Counter(r["phase"] for r in heads.values())),
        "head_edge_types": dict(collections.Counter(r["edge_type"] for r in heads.values())),
        "head_gates_seconds": dict(collections.Counter(harness.phase_gate(r) for r in heads.values())),
        "all_gates_seconds": dict(collections.Counter(harness.phase_gate(r) for r in rows.values())),
        "strata": dict(strata),
        "interpretation": [
            "Origin classification follows source annotations; message roles do not identify origin.",
            "system is a separate body field; the original Renderer prepends it to messages.",
            "Positive source cache reads do not prove prior turns in this session.",
            "Source cache reads are not the actual cache hits of the formal replay.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "heads.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(detail)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
