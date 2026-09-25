#!/usr/bin/env python3
"""Parse TP0 Prefill/Decode batch lines of a run's server.log into an ordered event table.

Sub-second timing: SGLang metrics_reporter (759a6eb) logs a prefill line AFTER the batch result is
processed; `input throughput = this batch's #new-token / (perf_counter now - previous prefill log)`,
so gap_s = new_token / throughput is the exact wall interval since the previous PREFILL log (it contains
any decode steps run in between). Decode lines are logged every 40 decode forwards
(decode_log_interval=40); gen throughput = generated tokens / interval since the previous DECODE log,
and generated tokens == spec tokens (bs + accepted drafts, metrics_reporter.update_spec_metrics), so
gap_s = spec_tokens / gen_throughput. `spec rounds` = sum of batch sizes over the 40 decode steps.
Usage: parse_log.py RUN_DIR OUT_CSV
"""
import csv, re, sys
from datetime import datetime, timezone

run, out = sys.argv[1], sys.argv[2]
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] (Prefill batch|Decode batch), (.*)$")
KV = re.compile(r"(#?[a-zA-Z][a-zA-Z \-()/]*?): ([-\d.]+|True|False)")
rows = []
with open(f"{run}/server.log", errors="replace") as f:
    for ln, line in enumerate(f, 1):
        m = TS.match(line.rstrip("\n"))
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        kind = "P" if m.group(2).startswith("Prefill") else "D"
        kv = {k.strip(): v for k, v in KV.findall(m.group(3))}
        r = dict(line=ln, ts=ts, kind=kind)
        if kind == "P":
            nt = int(kv["#new-token"]); thr = float(kv["input throughput (token/s)"])
            r.update(new_seq=int(kv["#new-seq"]), new_token=nt, cached=int(kv["#cached-token"]),
                     full_usage=float(kv["full token usage"]), mamba_usage=float(kv["mamba usage"]),
                     running=int(kv["#running-req"]), queue=int(kv["#queue-req"]),
                     pending=int(kv["#pending-token"]), thr=thr, gap_s=(nt / thr if thr > 0 else None))
        else:
            st = int(kv["spec tokens"]); sr = int(kv["spec rounds"]); thr = float(kv["gen throughput (token/s)"])
            r.update(running=int(kv["#running-req"]), full_tokens=int(kv["#full token"]),
                     full_usage=float(kv["full token usage"]), spec_tokens=st, spec_rounds=sr,
                     accept_len=float(kv["accept len"]), queue=int(kv["#queue-req"]), thr=thr,
                     gap_s=(st / thr if thr > 0 else None))
        rows.append(r)
cols = ["line", "ts", "kind", "new_seq", "new_token", "cached", "full_usage", "mamba_usage", "running", "queue",
        "pending", "full_tokens", "spec_tokens", "spec_rounds", "accept_len", "thr", "gap_s"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"{len(rows)} events -> {out}")
