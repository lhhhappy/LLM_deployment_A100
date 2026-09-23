#!/usr/bin/env python3
"""Bad-case analysis for one dev level (CPU only). Reusable for any run.

  badcase.py RAW.jsonl SERVER.log [--harness-dir s1-dev/harness] [--ref REF_RAW.jsonl] [--csv OUT.csv]

Answers, from the raw records and the engine's own batch log lines:
  1. TPOT: how many requests exceed 0.10 s/token, by output length, prompt length and minute (UTC);
  2. for each of those, the prefill batches logged inside its decode window (count, tokens, >=8192-token chunks);
  3. scheduler state at prefill time: how often a big chunk ran while requests were decoding and nothing waited;
  4. TTFT gate overs (harness buckets via s1_common.in_ttft_gate): queue vs execution split, own uncached tokens vs
     the frozen expectation (large excess = the request lost its cache), and prefill tokens logged while it waited.
Log timestamps are whole seconds (the pod clock matches the raw epoch, T56); window counts are therefore approximate.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] Prefill batch[,.] #new-seq: (\d+), #new-token: (\d+), "
                  r"#cached-token: (\d+).*?#running-req: (\d+), #queue-req: (\d+)")


def epoch(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("log", type=Path)
    ap.add_argument("--harness-dir", type=Path, default=Path(__file__).resolve().parents[2] / "s1-dev/harness")
    ap.add_argument("--ref", type=Path, help="raw of another run (e.g. same config at N18) to compare per request")
    ap.add_argument("--csv", type=Path, help="write one row per TPOT/TTFT bad case")
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness_dir))
    from s1_common import in_ttft_gate  # noqa: E402

    rows = [json.loads(l) for l in a.raw.read_text().splitlines() if l.strip()]
    batches = []
    for line in a.log.read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            t, seqs, new, cached, running, queue = m.groups()
            batches.append((epoch(t), int(seqs), int(new), int(cached), int(running), int(queue)))
    batches.sort()
    t0 = min(r["client_dispatch_at_s"] for r in rows)
    t1 = max(r["client_finish_at_s"] for r in rows)
    meas = [b for b in batches if t0 - 1 <= b[0] <= t1 + 1]
    ref = {}
    if a.ref:
        ref = {r["req_id"]: r for r in (json.loads(l) for l in a.ref.read_text().splitlines() if l.strip())}

    print(f"== {a.raw.name}: {len(rows)} requests, wall {t1 - t0:.0f} s, {len(meas)} prefill batches logged in the window")
    big = [b for b in meas if b[2] >= 8192]
    gap = [b for b in big if b[4] > 0 and b[5] == 0]
    print(f"prefill batches >=8192 new tokens: {len(big)}; of these with requests decoding and an EMPTY queue: {len(gap)} "
          f"({100 * len(gap) / max(1, len(big)):.0f}%)")
    print(f"prefill new tokens in window: {sum(b[2] for b in meas) / 1e6:.2f}M")

    # 1-2. TPOT
    bad = [r for r in rows if (r.get("tpot_s") or 0) > 0.10]
    print(f"\n== TPOT: {len(bad)}/{len(rows)} requests > 0.10 s/token ({100 * len(bad) / len(rows):.1f}%); "
          f"mean {sum(r['tpot_s'] for r in rows if r.get('tpot_s')) / len(rows):.4f}, "
          f"p95 {pct([r['tpot_s'] for r in rows if r.get('tpot_s')], .95):.4f}")
    for name, key, edges in (("output tokens", "output_tokens", (100, 300, 1000)),
                             ("prompt tokens", "prompt_tokens", (16384, 65536, 131072))):
        lo = 0
        for hi in edges + (10 ** 9,):
            grp = [r for r in rows if lo <= r[key] < hi]
            nb = sum(1 for r in grp if (r.get("tpot_s") or 0) > 0.10)
            print(f"  {name} [{lo},{hi if hi < 10 ** 9 else 'inf'}): {nb}/{len(grp)} over" + ("" if not grp else f" ({100 * nb / len(grp):.0f}%)"))
            lo = hi
    by_min = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = dt.datetime.fromtimestamp(r["client_first_token_at_s"], dt.timezone.utc).strftime("%H:%M")
        by_min[k][0] += 1
        by_min[k][1] += (r.get("tpot_s") or 0) > 0.10
    worst = sorted(by_min.items(), key=lambda kv: -kv[1][1])[:5]
    print("  worst minutes (first token, UTC): " + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in worst))
    details = []
    for r in bad:
        w0, w1 = r["client_first_token_at_s"], r["client_finish_at_s"]
        inw = [b for b in meas if w0 - 1 <= b[0] <= w1 + 1]
        d = dict(req=r["req_id"], kind="tpot", tpot=round(r["tpot_s"], 3), out=r["output_tokens"],
                 prompt=r["prompt_tokens"], decode_s=round(w1 - w0, 1), prefill_batches=len(inw),
                 prefill_big=sum(1 for b in inw if b[2] >= 8192), prefill_tokens=sum(b[2] for b in inw))
        if r["req_id"] in ref and ref[r["req_id"]].get("tpot_s"):
            d["ref_tpot"] = round(ref[r["req_id"]]["tpot_s"], 3)
        details.append(d)
    if details:
        print(f"  during their decode: prefill batches p50 {pct([d['prefill_batches'] for d in details], .5)}, "
              f"big chunks p50 {pct([d['prefill_big'] for d in details], .5)}, "
              f"prefill tokens p50 {pct([d['prefill_tokens'] for d in details], .5) / 1e3:.0f}k")
        for d in sorted(details, key=lambda d: -d["tpot"])[:5]:
            print(f"   {d['tpot']:.3f} s/tok out={d['out']} prompt={d['prompt']} decode {d['decode_s']} s: "
                  f"{d['prefill_batches']} prefill batches ({d['prefill_big']} >=8192)" +
                  (f" | ref tpot {d['ref_tpot']}" if "ref_tpot" in d else ""))

    # 4. TTFT gate overs
    print("\n== TTFT gate overs (harness buckets)")
    for gate, lim in (("fast_intra", 3), ("overall_intra", 5), ("turn_start", 15), ("chain_start", 30)):
        grp = [r for r in rows if in_ttft_gate(r, gate)]
        over = [r for r in grp if r["ttft_s"] > lim]
        if not over:
            print(f"  {gate}: 0/{len(grp)} over")
            continue
        q = [r["t_exec_start_s"] - r["t_recv_s"] for r in over]
        x = [r["t_first_token_s"] - r["t_exec_start_s"] for r in over]
        lost = [r for r in over if (r["prompt_tokens"] - r["cached_tokens"]) - r["uncached_expected"] > 4096]
        print(f"  {gate}: {len(over)}/{len(grp)} over | queue p50 {pct(q, .5):.1f}s exec p50 {pct(x, .5):.1f}s | "
              f"lost cache (>4096 over frozen) {len(lost)}")
        for r in over:
            waited = [b for b in meas if r["t_recv_s"] - 1 <= b[0] <= r["t_first_token_s"] + 1]
            details.append(dict(req=r["req_id"], kind=gate, ttft=round(r["ttft_s"], 2),
                                queue=round(r["t_exec_start_s"] - r["t_recv_s"], 2),
                                own_uncached=r["prompt_tokens"] - r["cached_tokens"], frozen=r["uncached_expected"],
                                prefill_tokens=sum(b[2] for b in waited)))
    if a.csv and details:
        keys = sorted({k for d in details for k in d})
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(details)
        print(f"\nwrote {len(details)} rows to {a.csv}")


if __name__ == "__main__":
    main()
