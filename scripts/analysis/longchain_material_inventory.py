#!/usr/bin/env python3
"""Count public material supply, not independent trajectories or accepted joins."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from longchain import (REPO, canonical, digest, file_digest, json_dump,
                       load_source, reminder, tool_pairing)


def inventory(root):
    rows, chains, bodies, _ = load_source(root)
    users, assistants, groups = {}, {}, {}
    occurrences = collections.Counter()
    for rid, body in bodies.items():
        row = rows[rid]
        session = (row["pack"], row["session_id"])
        messages = body.get("messages", [])
        for i, message in enumerate(messages):
            role = message.get("role")
            if role == "user" and not reminder(message) and message.get("content"):
                bank, kind, material = users, "non_reminder_user", message
            elif role == "assistant" and not message.get("tool_calls") and message.get("content"):
                bank, kind, material = assistants, "assistant_without_calls", message
            elif role == "assistant" and message.get("tool_calls"):
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    j += 1
                material = messages[i:j]
                if tool_pairing(material) == "incomplete":
                    occurrences["incomplete_tool_group"] += 1
                    continue
                bank, kind = groups, "complete_tool_group"
            else:
                continue
            occurrences[kind] += 1
            key = digest(material)
            entry = bank.setdefault(key, {"sessions": set(), "packs": set(),
                                          "chains": set(), "chars": len(canonical(material))})
            entry["sessions"].add(session)
            entry["packs"].add(row["pack"])
            entry["chains"].add(row["chain_id"])
    sessions = {(r["pack"], r["session_id"]) for r in rows.values()}
    # Query-only transplant candidates do not import donor tool schemas. This
    # count deliberately does not claim semantic compatibility or acceptance.
    candidate_pairs = sum(
        1 for u in users.values() for c in chains.values()
        if c["pack"] in u["packs"] and (c["pack"], c["session_id"]) not in u["sessions"])
    banks = {"non_reminder_user": users, "assistant_without_calls": assistants,
             "complete_tool_group": groups}
    return {
        "scope": "All public serving canon bodies; exact JSON deduplication across snapshots.",
        "definitions": {
            "non_reminder_user": "Candidate only: role=user, nonempty, no system-reminder; may include controller/compression messages, not verified human queries.",
            "assistant_without_calls": "Nonempty assistant message without tool_calls; reuse still requires context review.",
            "complete_tool_group": "Assistant calls plus contiguous results; explicit ID matching or implicit-order count matching. Source IDs retained in exact JSON hash; normalized semantic diversity may be lower.",
            "candidate_pairs": "Distinct query candidate x receiving source chain, same pack and different source session. Not accepted continuations or statistically independent examples.",
        },
        "source_files": {str(p.relative_to(root)): file_digest(p) for p in
                         [root / "requests.jsonl", root / "chains.jsonl",
                          *sorted((root / "bodies").rglob("*.jsonl.gz"))]},
        "public_requests": len(rows), "source_chains": len(chains),
        "source_sessions": len(sessions),
        "source_declared_requests": sum(c["n_requests"] for c in chains.values()),
        "distinct_system_tools_bodies": len({digest([b.get("system"), b.get("tools")]) for b in bodies.values()}),
        "snapshot_occurrences": dict(occurrences),
        "unique_materials": {k: len(v) for k, v in banks.items()},
        "unique_materials_by_pack": {pack: {k: sum(pack in x["packs"] for x in v.values())
                                             for k, v in banks.items()}
                                      for pack in sorted({r["pack"] for r in rows.values()})},
        "same_pack_cross_session_query_chain_candidate_pairs": candidate_pairs,
        "human_query_count": None, "accepted_join_count": None,
        "new_dataset_generated": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=REPO / "s1-dev/data/dev-combined-v1")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = inventory(args.source_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.out, result)
    print(canonical({k: v for k, v in result.items() if k not in ("source_files", "definitions")}))
