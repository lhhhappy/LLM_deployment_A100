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


def validate(row, schema=1):
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
    intervals = row.get("decision_intervals")
    if schema >= 3 and intervals is None:
        raise ValueError("schema 3 lacks decision intervals")
    if intervals is not None:
        validate_intervals(intervals, age, counts)


def validate_intervals(intervals, observed, decisions):
    if not isinstance(intervals, dict):
        raise ValueError("invalid decision intervals")
    count = intervals.get("count")
    if type(count) is not int or count < 0:
        raise ValueError("invalid interval count")
    totals = []
    for name in ("seconds_by_reason", "seconds_by_batch"):
        values = intervals.get(name)
        if not isinstance(values, dict) or any(
            not isinstance(k, str) or not k or type(v) not in (int, float)
            or not math.isfinite(v) or v < 0 for k, v in values.items()
        ):
            raise ValueError(f"invalid {name}")
        if name == "seconds_by_reason" and any(
            k != "unobserved_path" and not decisions.get(k) for k in values
        ):
            raise ValueError("interval reason has no observed decision")
        if name == "seconds_by_batch" and set(values) - {"prefill", "decode", "idle"}:
            raise ValueError("unknown interval batch kind")
        totals.append(sum(values.values()))
    uncovered = intervals.get("uncovered_s")
    if type(uncovered) not in (int, float) or not math.isfinite(uncovered) or uncovered < 0:
        raise ValueError("invalid uncovered interval")
    if not math.isclose(totals[0], totals[1], abs_tol=1e-6, rel_tol=1e-9):
        raise ValueError("reason/batch wall totals differ")
    if not math.isclose(totals[0] + uncovered, observed, abs_tol=1e-6, rel_tol=1e-9):
        raise ValueError("intervals do not account for observed wait")
    if count == 0 and any(totals):
        raise ValueError("nonzero wall time without intervals")


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
            if not isinstance(obj, dict) or obj.get("schema", 1) not in (1, 2, 3):
                raise ValueError("unsupported admission schema")
            event = obj.get("event")
            if event == "admit":
                validate(obj, obj.get("schema", 1))
            elif event == "waiting":
                sample = obj["sample"]
                if not isinstance(sample, list) or type(obj.get("queue_size")) is not int or obj["queue_size"] < len(sample):
                    raise ValueError("invalid waiting queue size")
                seen = set()
                for row in sample:
                    validate(row, obj.get("schema", 1))
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
    latest_waiting = None
    for obj in iter_events(lines):
        if obj["event"] == "budget_exhausted":
            budget_exhausted = True
        elif obj["event"] == "admit":
            admitted.setdefault(obj["rid"], []).append(obj)
            pending.pop(obj["rid"], None)
            admits += 1
        else:
            latest_waiting = {
                "observed_at_s": obj.get("observed_at_s"),
                "queue_size": obj["queue_size"],
                "sampled_requests": len(obj["sample"]),
                "unsampled_requests": obj["queue_size"] - len(obj["sample"]),
                "sample": obj["sample"],
                "scope": "waiting queue at snapshot only; excludes active partial and requests before scheduler receipt",
            }
            # A complete queue snapshot supersedes stale pending identities.
            # Truncated snapshots cannot establish that omitted RIDs departed.
            if obj["queue_size"] == len(obj["sample"]):
                pending.clear()
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
            "last_pending_decision_intervals": pending_row.get("decision_intervals") if pending_row else None,
            "ttft_s": rr.get("ttft_s"), "phase": rr.get("phase"),
        })
    return {
        "admission_events": admits, "waiting_snapshots": snapshots,
        "budget_exhausted": budget_exhausted,
        "admitted_decision_counts": dict(aggregate), "rows": rows,
        "latest_waiting_snapshot": latest_waiting,
        "limitations": [
            "decision counts are not time shares or a counterfactual latency saving",
            "waiting samples are bounded cumulative observations, not a full queue history",
            "last_pending can be stale after abort; admission means selected, not executed",
            "cache fields reflect the most recent match, not necessarily arrival residency",
            "generic/unscanned reasons do not prove resource infeasibility for each request",
            "this summary joins rid only; use admission_triage.py for first-admission time/episode filtering",
            "decision intervals are subsequent scheduler wall time, not sustained blocking causes or GPU service",
            "latest queue snapshot is not an all-arrived, not-yet-first-token service census",
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
