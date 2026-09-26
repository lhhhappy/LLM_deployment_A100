#!/usr/bin/env python3
"""Who got the prefill lane first in an opening: chain-start service order and paired gate changes.

Usage: python3 scripts/analysis/opening_heads.py base=<raw.jsonl> 124=<raw.jsonl> ... [--first-s 60] [--table-s 60]
The first run is the reference. Reads loadgen raw records only (harness fields: idx_in_chain, phase,
uncached_expected, ttft_s, tpot_s, server stamps); gate membership is s1_common.in_ttft_gate, the scorer's rule.

Per run: misses per TTFT gate over all drained requests; chain misses among requests dispatched in the first
--first-s seconds and in the first 5 minutes; requests above 0.10 s/token. Against the reference, on the request
IDs both runs completed: per gate, misses fixed and new. For the reference and every candidate, a table of chain
starts dispatched in the first --table-s seconds, in the order their prefill started, with the work the engine
actually had (prompt minus cached tokens), dispatch-to-execution wait and TTFT, so an ordering or parking change
shows as who ran first. Measured numbers only; no projection to an official level.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"chain_start": 30.0, "turn_start": 15.0, "overall_intra": 5.0, "fast_intra": 3.0}


def load(path):
    rows = {}
    for line in open(path):
        row = json.loads(line)
        if row.get("error_class") is None and row.get("ttft_s") is not None:
            rows[row["req_id"]] = row
    return rows


def summary(rows, gate, first_s):
    t0 = min(r["client_dispatch_at_s"] for r in rows.values())
    out = {"n": len(rows)}
    for selector, limit in LIMITS.items():
        members = [r for r in rows.values() if gate(r, selector)]
        out[selector] = (sum(r["ttft_s"] > limit for r in members), len(members))
    chain = [r for r in rows.values() if gate(r, "chain_start")]
    for label, horizon in ((f"chain_first_{first_s:g}s", first_s), ("chain_first_300s", 300)):
        early = [r for r in chain if r["client_dispatch_at_s"] - t0 < horizon]
        out[label] = (sum(r["ttft_s"] > 30 for r in early), len(early))
    out["tpot_over_0.10"] = sum((r.get("tpot_s") or 0) > 0.10 for r in rows.values())
    return out


def head_table(rows, gate, table_s):
    t0 = min(r["client_dispatch_at_s"] for r in rows.values())
    heads = [r for r in rows.values() if gate(r, "chain_start") and r["client_dispatch_at_s"] - t0 < table_s]
    heads.sort(key=lambda r: r.get("t_exec_start_s") or float("inf"))
    return [(round((r["prompt_tokens"] - (r.get("cached_tokens") or 0)) / 1000, 1),
             round((r.get("t_exec_start_s") or 0) - r["client_dispatch_at_s"], 1), round(r["ttft_s"], 1), r["req_id"])
            for r in heads]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=path/to/raw.jsonl; the first is the reference")
    ap.add_argument("--first-s", type=float, default=60)
    ap.add_argument("--table-s", type=float, default=60)
    ap.add_argument("--harness-dir", type=Path, default=ROOT / "s1-dev/harness")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(args.harness_dir))
    from s1_common import in_ttft_gate
    runs = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs.append((label, load(path)))
    ref_label, ref = runs[0]
    for label, rows in runs:
        s = summary(rows, in_ttft_gate, args.first_s)
        print(f"== {label}: n={s['n']} " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in s.items() if isinstance(v, tuple))
              + f" tpot_over_0.10={s['tpot_over_0.10']}")
        if label != ref_label:
            common = ref.keys() & rows.keys()
            parts = []
            for selector, limit in LIMITS.items():
                ids = [r for r in common if in_ttft_gate(ref[r], selector)]
                fixed = sum(ref[r]["ttft_s"] > limit >= rows[r]["ttft_s"] for r in ids)
                new = sum(rows[r]["ttft_s"] > limit >= ref[r]["ttft_s"] for r in ids)
                parts.append(f"{selector} {sum(ref[r]['ttft_s'] > limit for r in ids)}->"
                             f"{sum(rows[r]['ttft_s'] > limit for r in ids)} (fixed {fixed}, new {new})")
            print(f"   vs {ref_label} on {len(common)} common IDs: " + "; ".join(parts))
        print(f"   chain starts dispatched in the first {args.table_s:g} s, in prefill-start order: "
              "work(k tokens) wait-to-exec(s) ttft(s)")
        for work, wait, ttft, _ in head_table(rows, in_ttft_gate, args.table_s):
            print(f"     {work:7.1f} {wait:7.1f} {ttft:7.1f}{'  MISS' if ttft > 30 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
