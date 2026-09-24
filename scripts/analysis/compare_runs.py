#!/usr/bin/env python3
"""Same-request comparison of two complete dev levels (CPU only). Base first, candidate second.

  compare_runs.py BASE_DIR CAND_DIR [--pairs evidence/T56/pairs_rendered.json] [--harness-dir s1-dev/harness]
                  [--csv OUT.csv]

Each DIR is an evidence level directory (raw_*.jsonl, server.log, level_verdict.json). Fails closed: both raws must
hold the same request ids, each exactly once, with server timestamps. Per request it pairs TTFT, wait
(exec_start - recv), run (first - exec_start), cached/new tokens and the true-LCP gap, so that a change in a gate can
be split into requests whose own work changed (cache), requests that waited less or more (batch formation /
admission) and requests whose own execution changed (chunking). Engine-side: prefill batch size distribution,
partial-only batches, prefill tokens (whole server log, warmup/preflight included), and the 122 "[ax-pace]" lines. Descriptive, not causal: wait is
not a pure queue timer and log lines have 1 s resolution.
"""
import argparse
import collections
import csv
import glob
import json
import re
import sys
from pathlib import Path

PRE = re.compile(r"Prefill batch[,.] #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#running-req: (\d+), "
                 r"#queue-req: (\d+), #pending-token: (\d+)")
GATES = (("fast_intra", 3.0), ("overall_intra", 5.0), ("turn_start", 15.0), ("chain_start", 30.0))


def load(d):
    raws = glob.glob(str(d / "raw_*.jsonl"))
    if len(raws) != 1:
        sys.exit(f"INVALID: {d} needs exactly one raw_*.jsonl, found {len(raws)}")
    rows = [json.loads(l) for l in open(raws[0]) if l.strip()]
    by = {}
    for r in rows:
        if r["req_id"] in by:
            sys.exit(f"INVALID: duplicate {r['req_id']} in {d}")
        if r.get("error") or r.get("t_exec_start_s") is None or r.get("t_first_token_s") is None:
            sys.exit(f"INVALID: error or missing timestamps for {r['req_id']} in {d}")
        by[r["req_id"]] = r
    verdict = json.loads((d / "level_verdict.json").read_text())
    batches = [tuple(map(int, m.groups())) for m in map(PRE.search, open(d / "server.log", errors="replace")) if m]
    pace = [l.strip() for l in open(d / "server.log", errors="replace") if "[ax-pace]" in l]
    return by, verdict, batches, pace


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--pairs", type=Path, default=root / "evidence/T56/pairs_rendered.json")
    ap.add_argument("--harness-dir", type=Path, default=root / "s1-dev/harness")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness_dir))
    from s1_common import in_ttft_gate  # noqa: E402

    B, vb, bb, _ = load(a.base)
    C, vc, bc, pace = load(a.cand)
    if set(B) != set(C):
        sys.exit(f"INVALID: request sets differ ({len(set(B) ^ set(C))} ids)")
    pairs = {p["req_id"]: p for p in json.loads(a.pairs.read_text())}

    print(f"== {a.base.name if a.base.name.startswith('N') else a.base} vs {a.cand}: {len(B)} requests each, same ids")
    for tag, v in (("base", vb), ("cand", vc)):
        g = " ".join(f"{k.split('(')[0] if '(' in k else k}:{x['over_limit']}/{x['allowed_over']}(p95 {x['p95']:.2f})"
                     for k, x in v["ttft_gates"].items())
        print(f"  {tag}: {v['status']} passed={v['passed']} tpot {v['tpot_mean']:.4f}/{v['tpot_p95']:.4f} | {g}")

    rows = []
    for rid, rb in B.items():
        rc = C[rid]
        p = pairs.get(rid)
        row = dict(req_id=rid)
        for tag, r in (("b", rb), ("c", rc)):
            row[f"{tag}_ttft"] = round(r["ttft_s"], 3)
            row[f"{tag}_wait"] = round(r["t_exec_start_s"] - r["t_recv_s"], 3)
            row[f"{tag}_run"] = round(r["t_first_token_s"] - r["t_exec_start_s"], 3)
            row[f"{tag}_cached"] = r["cached_tokens"]
            row[f"{tag}_tpot"] = round(r["tpot_s"], 4) if r.get("tpot_s") is not None else ""
            row[f"{tag}_lcp_gap"] = "" if p is None else max(0, p["true_lcp"] // 64 * 64 - r["cached_tokens"])
        row["prompt"] = rb["prompt_tokens"]
        row["gate"] = next((g for g, _ in GATES if in_ttft_gate(rb, g)), "")
        rows.append(row)

    for gate, lim in GATES:
        g = [r for r in rows if in_ttft_gate(B[r["req_id"]], gate)]
        ob = {r["req_id"] for r in g if r["b_ttft"] > lim}
        oc = {r["req_id"] for r in g if r["c_ttft"] > lim}
        fixed, broke = ob - oc, oc - ob
        def why(ids):
            c = collections.Counter()
            for r in (x for x in g if x["req_id"] in ids):
                dc = r["c_cached"] - r["b_cached"]
                if abs(dc) >= 1024:
                    c["cache " + ("more" if dc > 0 else "less")] += 1
                elif abs(r["c_wait"] - r["b_wait"]) >= abs(r["c_run"] - r["b_run"]):
                    c["wait"] += 1
                else:
                    c["run"] += 1
            return dict(c)
        print(f"  {gate} (> {lim:.0f} s, per-request limit): base {len(ob)} cand {len(oc)} | fixed {len(fixed)} {why(fixed)}"
              f" | new {len(broke)} {why(broke)}")
        if g:
            for k in ("wait", "run"):
                print(f"      {k}: p50 {pct([r['b_'+k] for r in g], .5):.2f} -> {pct([r['c_'+k] for r in g], .5):.2f}"
                      f" | p95 {pct([r['b_'+k] for r in g], .95):.2f} -> {pct([r['c_'+k] for r in g], .95):.2f}")
    tb = [r["b_tpot"] for r in rows if r["b_tpot"] != ""]
    tc = [r["c_tpot"] for r in rows if r["c_tpot"] != ""]
    print(f"  tpot per request > 0.10: {sum(x > .10 for x in tb)} -> {sum(x > .10 for x in tc)}")
    print(f"  cached tokens total: {sum(r['b_cached'] for r in rows)/1e6:.2f}M -> {sum(r['c_cached'] for r in rows)/1e6:.2f}M;"
          f" requests with |delta cached| >= 1024: {sum(abs(r['c_cached']-r['b_cached']) >= 1024 for r in rows)}")
    for tag, bs in (("base", bb), ("cand", bc)):
        h = collections.Counter("<=2k" if b[1] <= 2048 else "<=4k" if b[1] <= 4160 else "<=8k" if b[1] <= 8256 else ">8k"
                                for b in bs)
        solo = sum(1 for b in bs if b[5] > 0 and b[0] == 1)
        print(f"  {tag} prefill batches {len(bs)} {dict(h)} | tokens {sum(b[1] for b in bs)/1e6:.2f}M | "
              f"partial alone {solo}")
    if pace:
        print("  cand [ax-pace] first/last:", pace[0][-160:], "|", pace[-1][-160:], sep="\n    ")
    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
