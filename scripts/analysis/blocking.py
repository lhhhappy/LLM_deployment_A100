#!/usr/bin/env python3
"""Who blocks whom: per-request attribution of TTFT and TPOT gate overs to the three levers (CPU only).

  blocking.py RAW.jsonl [--pairs evidence/T56/pairs_attributed.json] [--harness-dir s1-dev/harness] [--csv OUT.csv]

For every request the raw record gives server times: received (t_recv_s), first forward (t_exec_start_s), prefill
finished (t_first_token_s); and client times for the decode window (first token → finish). A request's prefill is
"active" on [t_exec_start_s, t_first_token_s]. With that:

TTFT over (harness buckets via s1_common.in_ttft_gate) is split into
  - queue      = exec_start − recv                        → arrangement (it waited for others)
  - own work   = actual uncached tokens × COST           → cost of its own prefill (ESTIMATE, cost model below)
      of which cache gap = max(0, floor64(true LCP with its same-chain predecessor) − cached) × COST
                                                          → less work was possible (only for pairs in --pairs)
  - interleave = (first − exec_start) − own work, if > 0 → others' work run between its own chunks
  and the requests whose prefill was active during its queue window are its blockers (time split evenly among
  concurrently active blockers).
TPOT over: the others' prefill active inside its decode window (union seconds) and who they were.
Blockers are grouped by kind: chain head (idx 0) vs follow-up, and actual uncached size.

COST = 0.11 s per prefill forward + 65 us per new token (122's defaults from F87/F88; an ESTIMATE, not measured per
request). Server and client clocks are the same host clock on the pod (T56). Timestamps are the engine's own; a
request's active window also contains other requests' decode rounds, so blocker seconds are upper bounds.
"""
import argparse
import collections
import csv
import importlib.util
import json
from pathlib import Path

FIXED_S, PER_TOK_S, CHUNK = 0.11, 65e-6, 16384
GATES = ("fast_intra", "overall_intra", "turn_start", "chain_start")
LIMIT = {"fast_intra": 3.0, "overall_intra": 5.0, "turn_start": 15.0, "chain_start": 30.0}


def load_common(harness):
    spec = importlib.util.spec_from_file_location("s1_common", Path(harness) / "s1_common.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def own_cost(uncached):
    chunks = max(1, -(-uncached // CHUNK))
    return chunks * FIXED_S + uncached * PER_TOK_S


def kind(r):
    u = r["uncached"]
    size = "<8k" if u < 8192 else ("8k-64k" if u < 65536 else ">=64k")
    return ("head" if r["idx_in_chain"] == 0 else "followup") + ":" + size


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def attribute(window, act, exclude):
    """Split [w0,w1] among active prefill intervals; returns (union seconds, {req_idx: seconds})."""
    w0, w1 = window
    if w1 <= w0:
        return 0.0, {}
    pts = {w0, w1}
    cand = []
    for j, (s, e) in act:
        if j != exclude and e > w0 and s < w1:
            cand.append((j, max(s, w0), min(e, w1)))
            pts.update((max(s, w0), min(e, w1)))
    pts = sorted(pts)
    union, share = 0.0, collections.defaultdict(float)
    for a, b in zip(pts, pts[1:]):
        live = [j for j, s, e in cand if s <= a and e >= b]
        if live:
            union += b - a
            for j in live:
                share[j] += (b - a) / len(live)
    return union, share


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--pairs", type=Path, default=Path("evidence/T56/pairs_attributed.json"))
    ap.add_argument("--harness-dir", type=Path, default=Path("s1-dev/harness"))
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    common = load_common(a.harness_dir)
    rows = [json.loads(l) for l in open(a.raw)]
    ids = collections.Counter(r["req_id"] for r in rows)
    if len(rows) != 722 or max(ids.values()) != 1:
        raise SystemExit(f"INVALID: {len(rows)} rows, duplicates={sum(v - 1 for v in ids.values())}")
    need = ("t_recv_s", "t_exec_start_s", "t_first_token_s", "client_first_token_at_s", "client_finish_at_s")
    bad = [r["req_id"] for r in rows if any(r.get(k) is None for k in need) or r.get("error")]
    if bad:
        raise SystemExit(f"INVALID: {len(bad)} rows without timing or with errors")
    lcp = {}
    if a.pairs.exists():
        for p in json.load(open(a.pairs)):
            lcp[p["req_id"]] = p["true_lcp"]
    for r in rows:
        r["uncached"] = max(0, r["prompt_tokens"] - r["cached_tokens"])
        r["gates"] = [g for g in GATES if common.in_ttft_gate(r, g)]
    act = [(i, (r["t_exec_start_s"], r["t_first_token_s"])) for i, r in enumerate(rows)]
    out, blame = [], collections.defaultdict(lambda: collections.defaultdict(float))
    for i, r in enumerate(rows):
        ttft = r["t_first_token_s"] - r["t_recv_s"]
        queue = r["t_exec_start_s"] - r["t_recv_s"]
        execs = r["t_first_token_s"] - r["t_exec_start_s"]
        own = min(own_cost(r["uncached"]), execs)
        gap_tok = max(0, (lcp[r["req_id"]] // 64) * 64 - r["cached_tokens"]) if r["req_id"] in lcp else None
        gap_s = min(own, gap_tok * PER_TOK_S) if gap_tok else 0.0
        inter = max(0.0, execs - own)
        qu, qshare = attribute((r["t_recv_s"], r["t_exec_start_s"]), act, i)
        du, dshare = attribute((r["client_first_token_at_s"], r["client_finish_at_s"]), act, i)
        ttft_over = [g for g in r["gates"] if ttft > LIMIT[g]]
        tpot_over = r.get("tpot_s") is not None and r["tpot_s"] > 0.10
        parts = {"queue(arrangement)": queue, "own work(cost)": own - gap_s, "cache gap(work)": gap_s,
                 "interleave(arrangement)": inter}
        top = max(parts, key=parts.get)
        qk = collections.defaultdict(float)
        for j, s in qshare.items():
            qk[kind(rows[j])] += s
        dk = collections.defaultdict(float)
        for j, s in dshare.items():
            dk[kind(rows[j])] += s
        if ttft_over:
            for j, s in qshare.items():
                blame["ttft_queue"][j] += s
        if tpot_over:
            for j, s in dshare.items():
                blame["tpot_decode"][j] += s
        out.append({"req_id": r["req_id"], "idx": r["idx_in_chain"], "gates": "|".join(r["gates"]),
                    "ttft_over": "|".join(ttft_over), "tpot_over": int(tpot_over), "ttft": round(ttft, 3),
                    "queue": round(queue, 3), "own_cost_est": round(own - gap_s, 3), "cache_gap_s": round(gap_s, 3),
                    "cache_gap_tok": gap_tok, "interleave": round(inter, 3), "primary": top,
                    "uncached": r["uncached"], "out": r["output_tokens"], "tpot": r.get("tpot_s"),
                    "queue_blocked_union": round(qu, 2),
                    "queue_blockers": ";".join(f"{k}={v:.1f}" for k, v in sorted(qk.items(), key=lambda x: -x[1])),
                    "decode_prefill_union": round(du, 2),
                    "decode_blockers": ";".join(f"{k}={v:.1f}" for k, v in sorted(dk.items(), key=lambda x: -x[1]))})
    print(f"== {a.raw.name}: 722 rows complete; cost model {FIXED_S}s/forward + {PER_TOK_S*1e6:.0f}us/token (estimate)")
    for g in GATES:
        v = [o for o in out if g in o["ttft_over"].split("|")]
        if not v:
            print(f"-- {g}: 0 over"); continue
        prim = collections.Counter(o["primary"] for o in v)
        s = {k: sum(o[k] for o in v) for k in ("queue", "own_cost_est", "cache_gap_s", "interleave")}
        tot = sum(s.values()) or 1
        bk = collections.defaultdict(float)
        for o in v:
            for kv in filter(None, o["queue_blockers"].split(";")):
                k, x = kv.rsplit("=", 1); bk[k] += float(x)
        bt = sum(bk.values()) or 1
        print(f"-- {g}: {len(v)} over | primary cause " + ", ".join(f"{k} {n}" for k, n in prim.most_common())
              + " | TTFT seconds: " + ", ".join(f"{k} {x/tot:.0%}" for k, x in s.items()))
        print("   their queue time was spent behind: " + ", ".join(f"{k} {x/bt:.0%}" for k, x in sorted(bk.items(), key=lambda x: -x[1])))
    v = [o for o in out if o["tpot_over"]]
    if v:
        dk = collections.defaultdict(float)
        for o in v:
            for kv in filter(None, o["decode_blockers"].split(";")):
                k, x = kv.rsplit("=", 1); dk[k] += float(x)
        dt = sum(dk.values()) or 1
        frac = sorted(o["decode_prefill_union"] / max(1e-9, o["tpot"] * max(1, o["out"] - 1)) for o in v)
        print(f"-- TPOT>0.10: {len(v)} | share of their decode window with some other prefill active: "
              f"p50 {frac[len(frac)//2]:.0%} | that prefill belonged to: "
              + ", ".join(f"{k} {x/dt:.0%}" for k, x in sorted(dk.items(), key=lambda x: -x[1])))
    for key, title in (("ttft_queue", "TTFT-over queue time"), ("tpot_decode", "TPOT-over decode time")):
        b = sorted(blame[key].items(), key=lambda x: -x[1])[:5]
        if b:
            print(f"-- top blockers by {title} caused:")
            for j, s in b:
                r = rows[j]
                print(f"   {s:7.1f}s  {kind(r):16s} uncached {r['uncached']:>7} prefill {r['t_first_token_s']-r['t_exec_start_s']:.1f}s  {r['req_id'][-40:]}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader(); w.writerows(out)
        print(f"wrote {len(out)} rows to {a.csv}")


if __name__ == "__main__":
    main()
