#!/usr/bin/env python3
"""Summarize opt-in [ax-admission] logs; CPU only, not a latency attribution.

Repeated waiting snapshots are cumulative and are NOT added to admissions.
Separate admission episodes (e.g. retraction) are kept and counted separately.
Optionally join a raw JSONL by req_id. Missing diagnostics stay unknown, not zero.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

MARKER = "[ax-admission] "


def validate(row):
    if not isinstance(row.get("rid"), str) or not row["rid"]:
        raise ValueError("admission row lacks rid")
    counts = row.get("decisions")
    if not isinstance(counts, dict) or any(
        not isinstance(k, str) or type(v) is not int or v < 0 for k, v in counts.items()
    ):
        raise ValueError("invalid decision counters")
    age = row.get("observed_wait_s")
    if type(age) not in (int, float) or not math.isfinite(age) or age < 0:
        raise ValueError("invalid observed wait")
    for key in ("observed_at_s", "queue_entry_at_s"):
        value = row.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid {key}")
    examples = row.get("examples", {})
    if not isinstance(examples, dict):
        raise ValueError("invalid reason examples")
    for reason, context in examples.items():
        if reason not in counts or not isinstance(context, dict):
            raise ValueError("example lacks corresponding decision")
        for key, value in context.items():
            if key in ("blocker_rid", "partial_rid"):
                if not isinstance(value, str) or not value:
                    raise ValueError("invalid blocker identity")
            elif key == "head_added":
                if type(value) is not bool:
                    raise ValueError("invalid head_added")
            else:
                raise ValueError(f"unknown example field: {key}")


def iter_events(lines, max_bytes=16 * 1024 * 1024):
    """Shared format reader. Old unscoped records remain readable, not joinable.

    Cap diagnostic input across concatenated server processes; ordinary log
    lines are streamed and discarded. Each production process is capped at 8 MiB.
    """
    consumed = 0
    for lineno, line in enumerate(lines, 1):
        if MARKER not in line:
            continue
        consumed += len(line.encode('utf-8'))
        if consumed > max_bytes:
            raise ValueError("diagnostic input exceeds byte budget; select one run")
        try:
            obj = json.loads(line.split(MARKER, 1)[1])
            if not isinstance(obj, dict) or obj.get("schema", 1) not in (1, 2):
                raise ValueError("unsupported admission schema")
            event = obj.get("event")
            if event == "admit":
                validate(obj)
            elif event == "waiting":
                sample = obj["sample"]
                if not isinstance(sample, list) or type(obj.get("queue_size")) is not int or obj["queue_size"] < len(sample):
                    raise ValueError("invalid waiting queue size")
                seen = set()
                for row in sample:
                    validate(row)
                    if row["rid"] in seen:
                        raise ValueError("duplicate rid in waiting snapshot")
                    seen.add(row["rid"])
            elif event != "budget_exhausted":
                raise ValueError("unknown admission event")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"admission log line {lineno}: {exc}") from exc
        yield obj


def analyze(lines, raw_rows=()):
    admitted, pending = {}, {}
    snapshots = admits = 0
    budget_exhausted = False
    for obj in iter_events(lines):
        if obj["event"] == "budget_exhausted":
            budget_exhausted = True
        elif obj["event"] == "admit":
            admitted.setdefault(obj["rid"], []).append(obj)
            pending.pop(obj["rid"], None)
            admits += 1
        else:
            for row in obj["sample"]:
                pending[row["rid"]] = row
            snapshots += 1
    if not snapshots and not admits and not budget_exhausted:
        raise ValueError("no admission diagnostics; enable SGLANG_AX_ADMISSION_TRACE=1")
    raw = {}
    for row in raw_rows:
        rid = row["req_id"]
        if rid in raw:
            raise ValueError(f"duplicate raw req_id: {rid}")
        raw[rid] = row
    rows = []
    aggregate = Counter()
    for rid in sorted(admitted.keys() | pending.keys() | raw.keys()):
        episodes = admitted.get(rid, [])
        counts = Counter()
        for ep in episodes:
            counts.update(ep["decisions"])
        aggregate.update(counts)
        pending_row = pending.get(rid)
        rr = raw.get(rid, {})
        rows.append({
            "req_id": rid, "admission_episodes": len(episodes),
            "diagnostic_present": bool(episodes or pending_row),
            "admitted_decisions": dict(counts) if episodes else None,
            "last_pending_decisions": pending_row["decisions"] if pending_row else None,
            "last_pending_observed_wait_s": pending_row["observed_wait_s"] if pending_row else None,
            "ttft_s": rr.get("ttft_s"), "phase": rr.get("phase"),
        })
    return {
        "admission_events": admits, "waiting_snapshots": snapshots,
        "budget_exhausted": budget_exhausted,
        "admitted_decision_counts": dict(aggregate), "rows": rows,
        "limitations": [
            "decision counts are not time shares or a counterfactual latency saving",
            "waiting samples are bounded cumulative observations, not a full queue history",
            "last_pending can be stale after abort; admission means selected, not executed",
            "cache fields reflect the most recent match, not necessarily arrival residency",
            "generic/unscanned reasons do not prove resource infeasibility for each request",
            "this summary joins rid only; use admission_triage.py for first-admission time/episode filtering",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    raw = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()] if args.raw else []
    with args.log.open() as f:
        result = analyze(f, raw)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.csv:
        with args.csv.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=result["rows"][0].keys() if result["rows"] else ["req_id"])
            writer.writeheader()
            for row in result["rows"]:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                                 for k, v in row.items()})
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "limitations")}))


if __name__ == "__main__":
    main()
