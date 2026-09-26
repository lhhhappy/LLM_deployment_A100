#!/usr/bin/env python3
"""Recheck 125 guard timing against measured request dispatch, using raw data.

This is a diagnostic, not a complete-cohort score. --root supplies the original
workspace containing retained L112/L130b data and the unmodified harness.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys

LIMITS = {"chain_start": 30, "turn_start": 15, "overall_intra": 5, "fast_intra": 3}
RUNS = {"112": "L112-v3_n30_124_125x_117_60m", "130b": "L130b-v3_n30_A_warm15_60m"}


def audit(root, max_slow):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "s1-dev/harness"))
    from s1_common import in_ttft_gate

    def gates(rows):
        return {g: {"n": sum(in_ttft_gate(r, g) for r in rows),
                    "over": sum(in_ttft_gate(r, g) and r["ttft_s"] > limit for r in rows)}
                for g, limit in LIMITS.items()}

    def count_slow(rows):
        return sum(r["tpot_s"] > .1 for r in rows if r.get("tpot_s") is not None)

    result, by_id = {}, {}
    for label, name in RUNS.items():
        directory = root / "evidence" / name / "N30"
        raw_files = list(directory.glob("raw_*.jsonl"))
        if len(raw_files) != 1:
            raise ValueError(f"Ambiguous measured raw: {raw_files}")
        raw = raw_files[0]
        data = raw.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(rows) != len({r["req_id"] for r in rows}) or any(r.get("error") for r in rows):
            raise ValueError("Duplicate IDs or failed requests; inspect before comparing")
        by_id[label] = {r["req_id"]: r for r in rows}
        start = min(r["client_dispatch_at_s"] for r in rows)
        end = max(r["client_finish_at_s"] for r in rows)
        events, counters = [], []
        server = directory / "server.log"
        log_sha = hashlib.sha256()
        with server.open("rb") as stream:
            for number, binary_line in enumerate(stream, 1):
                log_sha.update(binary_line)
                line = binary_line.decode(errors="replace")
                if "[ax-125]" not in line and "[ax-124/125]" not in line:
                    continue
                stamp = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\]", line)
                if not stamp:
                    continue
                at = dt.datetime.strptime(stamp[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
                # Log stamps have second resolution: retain the first partial second.
                if not math.floor(start) <= at <= end:
                    continue
                event = {"line_number": number, "at_utc": stamp[1],
                         "seconds_from_first_dispatch": round(at - start, 3), "line": line.strip()}
                change = re.search(r"relief (on|off): cold backlog (\d+) tokens at (\d+) tok/s, slow (\d+)/(\d+)", line)
                if change:
                    event.update(state=change[1], backlog_tokens=int(change[2]),
                                 rate_tokens_per_second=int(change[3]), ever_slow=int(change[4]), seen=int(change[5]))
                    events.append(event)
                stats = re.search(r"parks=(\d+) relief_rounds=(\d+)", line)
                if stats:
                    event.update(parks=int(stats[1]), relief_rounds=int(stats[2]))
                    counters.append(event)
        trips = [e for e in events if e["state"] == "off" and e["ever_slow"] >= max_slow]
        if not trips:
            raise ValueError(f"No observed guard transition in {label}")
        trip = trips[0]
        trip_at = dt.datetime.strptime(trip["at_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
        after = [r for r in rows if r["client_dispatch_at_s"] >= trip_at]
        later_events = [e for e in events if e["seconds_from_first_dispatch"] > trip["seconds_from_first_dispatch"]]
        later_counters = [e for e in counters if e["seconds_from_first_dispatch"] > trip["seconds_from_first_dispatch"]]
        result[label] = {
            "raw_path": str(raw.relative_to(root)), "raw_sha256": hashlib.sha256(data).hexdigest(),
            "server_path": str(server.relative_to(root)), "server_sha256": log_sha.hexdigest(),
            "requests": len(rows), "first_dispatch_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
            "end_from_first_dispatch_s": end - start, "gate_counts": gates(rows),
            "trip": trip, "events": events,
            "any_later_relief_on": any(e["state"] == "on" for e in later_events),
            "post_trip_relief_counter_values": sorted({e["relief_rounds"] for e in later_counters}),
            "first_and_last_counters": counters[:1] + counters[-1:],
            "arrived_after_trip": {"requests": len(after), "gates": gates(after), "final_tpot_over_010": count_slow(after)},
            "arrived_after_14_minutes": gates([r for r in rows if r["client_dispatch_at_s"] >= start + 840]),
            "final_tpot_over_010": count_slow(rows),
        }
    common = sorted(set(by_id["112"]) & set(by_id["130b"]))
    paired = {label: {"gates": gates([rows[rid] for rid in common]),
                      "final_tpot_over_010": count_slow([rows[rid] for rid in common])}
              for label, rows in by_id.items()}
    return {"validity": "DIAGNOSTIC_RECOUNT_NOT_FORMAL_SCORE", "source_root": str(root),
            "max_slow": max_slow, "log_clock": "UTC, one-second resolution",
            "runs": result, "common_requests": len(common), "paired": paired,
            "limits": ["No per-request ever-slow IDs are logged: the overlap with final slow IDs is unknown.",
                       "The absence of relief after the guard is not a causal estimate of MAX_SLOW=250.",
                       "Periodic admission counters are process-lifetime counters, not reset by measurement flush."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-slow", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.max_slow)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"trips_s": {k: v["trip"]["seconds_from_first_dispatch"] for k, v in result["runs"].items()},
                      "common_requests": result["common_requests"], "paired": result["paired"]}, ensure_ascii=False))
