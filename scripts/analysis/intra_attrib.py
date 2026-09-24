#!/usr/bin/env python3
"""Why do intra-chain (cached) requests miss their TTFT gate? CPU only, reusable for any dev level.

  intra_attrib.py RAW.jsonl SERVER.log [--pairs evidence/T56/pairs_rendered.json] [--short 8192] [--cap 4096]
                  [--harness-dir s1-dev/harness] [--csv OUT.csv]

For every request in the harness overall_intra bucket (s1_common.in_ttft_gate) it splits the server TTFT into
wait = exec_start - recv and run = first_token - exec_start (neither is a pure GPU timer), and reads the engine's
own prefill log lines (1 s resolution, TP0) inside the wait window: batches, tokens prefilled for others, and how
many batches carried a chunked partial (#pending-token > 0). Reuse is checked against the true token LCP with the
replay predecessor (T56 pairs), never against the frozen uncached_expected.

Labels (first match, for overs only; they are descriptive, not causal proof):
  own_run      run alone exceeds the gate: its own prefill (plus interleaved decode) is too long
  reuse_gap    true-LCP aligned gap > 4096 tokens: it recomputed history that the predecessor had
  behind_partial  waited, and >= half the prefill batches in its wait window carried a partial of another request;
               split by whether 120 could have admitted it next to the partial (full hit with new <= short and
               new <= chunk - cap) or not (cold, or too big for the leftover budget)
  queue        waited with no dominant partial in the window
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
                  r"#cached-token: (\d+).*?#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+)")


def epoch(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


def pair_ok(p, r, rows):
    """The T56 pair must describe this request and its predecessor as replayed in this raw."""
    if p.get("prompt") != r["prompt_tokens"] or p.get("chain_id") != r["chain_id"] or p.get("idx") != r["idx_in_chain"]:
        return False
    prev = [x for x in rows if x["chain_id"] == r["chain_id"] and x["idx_in_chain"] == r["idx_in_chain"] - 1]
    return len(prev) == 1 and prev[0]["prompt_tokens"] == p.get("previous_prompt")


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("log", type=Path)
    ap.add_argument("--pairs", type=Path, default=root / "evidence/T56/pairs_rendered.json")
    ap.add_argument("--short", type=int, default=8192, help="SGLANG_AX_SCHED_SHORT_TOKENS of the run")
    ap.add_argument("--cap", type=int, default=4096, help="SGLANG_AX_SCHED_COLD_CAP of the run")
    ap.add_argument("--chunk", type=int, default=8192, help="chunked_prefill_size of the run")
    ap.add_argument("--harness-dir", type=Path, default=root / "s1-dev/harness")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness_dir))
    from s1_common import in_ttft_gate  # noqa: E402

    rows = [json.loads(l) for l in a.raw.read_text().splitlines() if l.strip()]
    ids = [r["req_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("INVALID: duplicate req_id in raw")
    pairs = {p["req_id"]: p for p in json.loads(a.pairs.read_text())}
    batches = []
    for line in a.log.read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            t, *v = m.groups()
            batches.append((epoch(t), *map(int, v)))
    if not batches:
        sys.exit("INVALID: no prefill lines with #pending-token in the server log")
    batches.sort()
    times = [b[0] for b in batches]

    import bisect
    out, rejected = [], 0
    for r in rows:
        if not in_ttft_gate(r, "overall_intra"):
            continue
        need = ("t_recv_s", "t_exec_start_s", "t_first_token_s")
        if any(r.get(k) is None for k in need):
            sys.exit(f"INVALID: missing server timestamps for {r['req_id']}")
        fast = in_ttft_gate(r, "fast_intra")
        lim = 3.0 if fast else 5.0
        wait = r["t_exec_start_s"] - r["t_recv_s"]
        run = r["t_first_token_s"] - r["t_exec_start_s"]
        new = r["prompt_tokens"] - r["cached_tokens"]
        p = pairs.get(r["req_id"])
        if p is not None and not pair_ok(p, r, rows):
            p, rejected = None, rejected + 1  # the T56 pair does not describe this replay: gap unknown
        gap = None if p is None else max(0, p["true_lcp"] // 64 * 64 - r["cached_tokens"])
        ideal = None if p is None else r["prompt_tokens"] - p["true_lcp"] // 64 * 64  # new tokens at full reuse
        lo = bisect.bisect_left(times, int(r["t_recv_s"]))
        hi = bisect.bisect_right(times, int(r["t_exec_start_s"]))
        win = batches[lo:hi]
        partial = sum(1 for b in win if b[6] > 0)
        # a partial alone, at most chunk - leftover tokens: the leftover was unused while this request waited
        solo = sum(1 for b in win if b[6] > 0 and b[1] == 1 and b[2] <= a.cap)
        admissible = r["cached_tokens"] > 0 and new <= a.short and new <= a.chunk - a.cap
        over = r["ttft_s"] > lim
        label = ""
        if over:
            if run > lim:
                label = "own_run"
            elif gap is not None and gap > 4096:
                label = "reuse_gap"
            elif win and partial * 2 >= len(win):
                label = "behind_partial:" + ("admissible" if admissible else "not_admissible")
            else:
                label = "queue"
        out.append(dict(req_id=r["req_id"], gate="fast_intra" if fast else "overall_intra", limit=lim,
                        ttft=round(r["ttft_s"], 3), over=int(over), label=label, wait=round(wait, 3), run=round(run, 3),
                        prompt=r["prompt_tokens"], cached=r["cached_tokens"], new=new,
                        frozen_expected=r.get("uncached_expected"), lcp_gap="" if gap is None else gap,
                        new_at_full_reuse="" if ideal is None else ideal,
                        win_batches=len(win), win_partial=partial, win_partial_only=solo,
                        win_tokens=sum(b[2] for b in win), win_max_queue=max((b[5] for b in win), default=0),
                        admissible_next_to_partial=int(admissible)))

    overs = [o for o in out if o["over"]]
    print(f"== {a.raw.name}: overall_intra bucket {len(out)} requests, {len(overs)} over their limit "
          f"(fast 3 s / other intra 5 s)")
    cnt = collections.Counter(o["label"] for o in overs)
    for lab, n in cnt.most_common():
        g = [o for o in overs if o["label"] == lab]
        med = lambda k: sorted(x[k] for x in g)[len(g) // 2]
        print(f"  {lab:32s} {n:3d} | wait p50 {med('wait'):6.2f}s run p50 {med('run'):5.2f}s | new p50 {med('new'):6d} "
              f"| window batches p50 {med('win_batches')}, with partial p50 {med('win_partial')}")
    print("  (labels are descriptive; log lines have 1 s resolution; wait includes decode rounds and scheduling)")
    print(f"  T56 LCP pairs rejected (length/chain/predecessor mismatch with this raw): {rejected}; kept pairs are "
          f"metadata-checked only (not bound to prompt content): reuse_gap and full-reuse counts are diagnostics")
    left = a.chunk - a.cap
    cls = [o for o in out if o["cached"] > 0 and left < o["new"] <= a.short]
    if cls and left > 0:
        ov = [o for o in cls if o["over"]]
        lost = sum(1 for o in ov if o["new_at_full_reuse"] != "" and o["new_at_full_reuse"] <= left)
        wb = sum(o["win_batches"] for o in ov)
        print(f"  hits with {left} < new <= {a.short}: {len(ov)}/{len(cls)} over; of the overs {lost} would have had "
              f"new <= {left} at full reuse per the unverified T56 LCP (diagnostic); in their wait windows "
              f"{sum(o['win_partial_only'] for o in ov)}/{wb} batches ran a partial alone at <= {a.cap} tokens")
    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {len(out)} rows to {a.csv}")


if __name__ == "__main__":
    main()
