#!/usr/bin/env python3
"""Require observed DCP local routes, separately from the startup policy gate."""
import argparse
import json
from pathlib import Path


def audit(path, *, tp_size, required, role="target", since=None, until=None):
    latest, first, previous_by_rank = {}, {}, {}
    window = since is not None or until is not None
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if "[ax-dcp-local] " not in line:
            continue
        record = json.loads(line.split("[ax-dcp-local] ", 1)[1])
        if record["role"] != role:
            continue
        rank = record["rank"]
        if not isinstance(rank, int) or rank not in range(tp_size):
            raise ValueError(f"line {lineno}: unexpected rank {rank}")
        previous = previous_by_rank.get(rank)
        for route, counts in record["routes"].items():
            for key in ("batches", "query_tokens", "prefix_tokens"):
                if counts[key] < 0 or (previous and counts[key] < previous["routes"][route][key]):
                    raise ValueError(f"line {lineno}: reset/negative {route} {key}")
        previous_by_rank[rank] = record
        if window:
            observed = record["observed_at_s"]
            if (since is not None and observed < since) or (until is not None and observed > until):
                continue
            first.setdefault(rank, record)
        latest[rank] = record
    counts = {rank: {route: values["batches"] - (first[rank]["routes"][route]["batches"] if window else 0)
                     for route, values in record["routes"].items()} for rank, record in latest.items()}
    missing = [(rank, route) for rank in range(tp_size) for route in required
               if counts.get(rank, {}).get(route, 0) <= 0]
    return dict(passed=not missing, role=role, required=required, missing=missing,
                observed_batches=counts, first=first, latest=latest,
                scope=("differences of snapshots inside the requested window; excludes unsampled boundaries"
                       if window else "cumulative logged snapshots, including warmup; activation proof only"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--role", choices=("target", "draft"), default="target")
    parser.add_argument("--since", type=float, help="measurement start, Unix seconds")
    parser.add_argument("--until", type=float, help="measurement end, Unix seconds")
    parser.add_argument("--require", action="append", choices=("local_short", "local_large", "gather_kv"), required=True)
    args = parser.parse_args()
    if args.tp_size < 1:
        parser.error("tp-size must be positive")
    if args.since is not None and args.until is not None and args.until < args.since:
        parser.error("until must be >= since")
    result = audit(args.log, tp_size=args.tp_size, required=args.require, role=args.role,
                   since=args.since, until=args.until)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
