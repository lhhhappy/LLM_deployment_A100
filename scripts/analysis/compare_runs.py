#!/usr/bin/env python3
"""Same-request comparison of two complete dev levels (CPU only). Base first, candidate second.

  compare_runs.py BASE_DIR CAND_DIR [--pairs evidence/T56/pairs_rendered.json] [--harness-dir s1-dev/harness]
                  [--csv OUT.csv]

Each DIR is an evidence level directory (raw_*.jsonl, server.log, level_verdict.json). Fails closed: each level's
verdict must be VALID and its raw must hold exactly the frozen cohort's request ids (each once, no errors, server
timestamps present). Gates are judged at raw precision; values are rounded only for display/CSV. A true-LCP pair is
used only if its prompt length, chain/idx and predecessor's prompt length match this run's raw ("metadata-checked";
else unknown). Pairs are not bound to prompt content or a cohort content hash: the gap is an unverified-source
diagnostic, not evidence for cache causality. Per request it pairs TTFT, wait
(exec_start - recv), run (first - exec_start), cached/new tokens and the true-LCP gap, so that a change in a gate can
be split into requests whose own work changed (cache), requests that waited less or more (batch formation /
admission) and requests whose own execution changed (chunking). Engine-side: prefill batch size distribution,
partial-only batches, prefill tokens and "[ax-pace]" lines, all restricted to the measurement window
[first recv, last first token] of the raw (warmup/preflight excluded). Log seconds are whole: lines in the two
boundary seconds are dropped and counted separately. Both runs must carry the same frozen workload identity
(run config cohort_sha256_canonical / workload_hash) and identical per-request metadata; N may differ. Descriptive, not causal: wait is
not a pure queue timer and log lines have 1 s resolution.
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\]")
PRE = re.compile(r"Prefill batch[,.] #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#running-req: (\d+), "
                 r"#queue-req: (\d+), #pending-token: (\d+)")
GATES = (("fast_intra", 3.0), ("overall_intra", 5.0), ("turn_start", 15.0), ("chain_start", 30.0))


def epoch(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


def load(d, cohort_ids):
    raws = glob.glob(str(d / "raw_*.jsonl"))
    if len(raws) != 1:
        sys.exit(f"INVALID: {d} needs exactly one raw_*.jsonl, found {len(raws)}")
    verdict = json.loads((d / "level_verdict.json").read_text())
    if verdict.get("status") != "VALID":
        sys.exit(f"INVALID: {d} level_verdict status is {verdict.get('status')!r}, not VALID")
    rows = [json.loads(l) for l in open(raws[0]) if l.strip()]
    by = {}
    for r in rows:
        if r["req_id"] in by:
            sys.exit(f"INVALID: duplicate {r['req_id']} in {d}")
        if r.get("error") or any(r.get(k) is None for k in ("t_recv_s", "t_exec_start_s", "t_first_token_s", "ttft_s")):
            sys.exit(f"INVALID: error or missing timestamps for {r['req_id']} in {d}")
        by[r["req_id"]] = r
    if set(by) != cohort_ids:
        sys.exit(f"INVALID: {d} raw ids differ from the cohort: {len(cohort_ids - set(by))} missing, "
                 f"{len(set(by) - cohort_ids)} extra")
    if verdict.get("rows") not in (None, len(by)):
        sys.exit(f"INVALID: {d} verdict rows {verdict.get('rows')} != raw rows {len(by)}")
    runs = glob.glob(str(d / "run_*.json"))
    if len(runs) != 1:
        sys.exit(f"INVALID: {d} needs exactly one run_*.json, found {len(runs)}")
    cfg = json.loads(Path(runs[0]).read_text())["config"]
    ident = (cfg.get("cohort_sha256_canonical"), cfg.get("workload_hash"))
    if None in ident:
        sys.exit(f"INVALID: {d} run config lacks the workload identity")
    t_lo, t_hi = min(r["t_recv_s"] for r in by.values()), max(r["t_first_token_s"] for r in by.values())
    # a log line stamped ts covers [ts, ts+1): interior only if that whole second lies inside [t_lo, t_hi]
    batches, pace, outside, edge = [], [], 0, 0
    for line in open(d / "server.log", errors="replace"):
        t = TS.match(line)
        if not t:
            continue
        ts = epoch(t.group(1))
        where = "in" if ts >= t_lo and ts + 1 <= t_hi else "edge" if ts < t_hi and ts + 1 > t_lo else "out"
        m = PRE.search(line)
        if m:
            if where == "in":
                batches.append(tuple(map(int, m.groups())))
            elif where == "edge":
                edge += 1
            else:
                outside += 1
        elif "[ax-pace]" in line and where == "in":
            pace.append(line.strip())
    if not batches:
        sys.exit(f"INVALID: no prefill log lines inside the measurement window of {d}")
    return by, verdict, batches, pace, (outside, edge), ident


def pair_ok(p, r, by):
    """The T56 pair must describe this request and its predecessor as replayed here."""
    if p is None or p.get("prompt") != r["prompt_tokens"] or p.get("chain_id") != r["chain_id"] \
            or p.get("idx") != r["idx_in_chain"]:
        return False
    prev = [x for x in by.values() if x["chain_id"] == r["chain_id"] and x["idx_in_chain"] == r["idx_in_chain"] - 1]
    if len(prev) != 1 or prev[0]["prompt_tokens"] != p.get("previous_prompt"):
        return False
    lcp = p.get("true_lcp")
    return type(lcp) is int and 0 <= lcp <= min(r["prompt_tokens"], prev[0]["prompt_tokens"])


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
    ap.add_argument("--cohort", type=Path, default=root / "s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness_dir))
    from s1_common import in_ttft_gate  # noqa: E402

    cohort = json.loads(a.cohort.read_text())
    cohort_ids = {rid for ch in cohort["chains"] for rid in ch["req_ids"]}
    if len(cohort_ids) != cohort["n_requests"]:
        sys.exit("INVALID: cohort file is inconsistent")
    B, vb, bb, _, ob_out, ib = load(a.base, cohort_ids)
    C, vc, bc, pace, oc_out, ic = load(a.cand, cohort_ids)
    if ib != ic:
        sys.exit(f"INVALID: workload identity differs: {ib} vs {ic}")
    meta = ("prompt_tokens", "uncached_expected", "max_output_i", "phase", "chain_id", "idx_in_chain", "edge_type")
    diff = [rid for rid in B if any(B[rid].get(k) != C[rid].get(k) for k in meta)]
    if diff:
        sys.exit(f"INVALID: {len(diff)} requests differ in replay metadata, e.g. {diff[0]}")
    pairs = {p["req_id"]: p for p in json.loads(a.pairs.read_text())}
    bad_pairs = 0

    print(f"== {a.base.name if a.base.name.startswith('N') else a.base} vs {a.cand}: {len(B)} requests each, same ids")
    for tag, v in (("base", vb), ("cand", vc)):
        g = " ".join(f"{k.split('(')[0] if '(' in k else k}:{x['over_limit']}/{x['allowed_over']}(p95 {x['p95']:.2f})"
                     for k, x in v["ttft_gates"].items())
        print(f"  {tag}: {v['status']} passed={v['passed']} tpot {v['tpot_mean']:.4f}/{v['tpot_p95']:.4f} | {g}")

    rows = []
    for rid, rb in B.items():
        rc = C[rid]
        p = pairs.get(rid)
        ok = {tag: pair_ok(p, r, run) for tag, r, run in (("b", rb, B), ("c", rc, C))}
        bad_pairs += p is not None and not all(ok.values())
        row = dict(req_id=rid)
        for tag, r in (("b", rb), ("c", rc)):  # raw precision here; rounded only when written/printed
            row[f"{tag}_ttft"] = r["ttft_s"]
            row[f"{tag}_wait"] = r["t_exec_start_s"] - r["t_recv_s"]
            row[f"{tag}_run"] = r["t_first_token_s"] - r["t_exec_start_s"]
            row[f"{tag}_cached"] = r["cached_tokens"]
            row[f"{tag}_tpot"] = r["tpot_s"] if r.get("tpot_s") is not None else ""
            row[f"{tag}_lcp_gap"] = max(0, p["true_lcp"] // 64 * 64 - r["cached_tokens"]) if ok[tag] else ""
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
    n_pair = sum(1 for rid in B if rid in pairs)
    print(f"  LCP gap (T56 pairs, unverified source, diagnostic only): metadata-checked {n_pair - bad_pairs}, "
          f"rejected {bad_pairs}, no pair (unknown) {len(B) - n_pair} of {len(B)}; workload identity {ib}")
    print(f"  prefill lines excluded: before/after the window base {ob_out[0]} cand {oc_out[0]}, in boundary seconds "
          f"base {ob_out[1]} cand {oc_out[1]}")
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
            w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
        print(f"wrote {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
