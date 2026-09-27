#!/usr/bin/env bash
# Close one finished window/probe job in one step: fetch its raw records (retries across link drops), print the harness
# p95 per TTFT gate and the drained count, the kanban row (misses per gate on the same request IDs as its reference),
# and the steady-state chain reading (opening vs later chain-bucket requests, giants after batch entry).
#   bash scripts/analysis/close_job.sh <job> <N> [ref-job]
# Measured values only; a window run is a diagnostic, not a verdict (the level itself is INVALID: not full-cohort).
set -uo pipefail
cd "$(dirname "$0")/../.."
job=${1:?job}; n=${2:?N}; ref=${3:-}
for i in 1 2 3; do
  ls evidence/L"$job"/N"$n"/raw_*.jsonl >/dev/null 2>&1 && break
  timeout 1200 python3 -B scripts/analysis/fetch_level.py "$job" "$n" >/dev/null 2>&1
  ls evidence/L"$job"/N"$n"/raw_*.jsonl >/dev/null 2>&1 && break
  echo "[close_job] fetch attempt $i failed; retry in 60 s"; sleep 60
done
python3 - "$job" "$n" <<'PY'
import json, sys, glob
job, n = sys.argv[1], sys.argv[2]
d = f"evidence/L{job}/N{n}"
try:
    s = json.load(open(f"{d}/summary.json"))
    g = s.get("ttft_p95_by_gate", {})
    print(f"== {job} N{n}: wall {s.get('wall_s')} s | harness p95 " + " ".join(f"{k.split('(')[0]}={v:.1f}s" for k, v in g.items()))
except FileNotFoundError:
    print(f"== {job} N{n}: no summary.json")
raws = glob.glob(f"{d}/raw_*.jsonl")
if raws:
    rows = [json.loads(l) for l in open(raws[0])]
    print(f"   raw rows {len(rows)}, errors {sum(1 for r in rows if r.get('error'))}, tpot>0.10 {sum(1 for r in rows if (r.get('tpot_s') or 0) > 0.10)}")
PY
timeout 280 python3 -B scripts/analysis/kanban.py 2>/dev/null | grep -F "| $job " | cut -c1-900
if [ -n "$ref" ]; then timeout 280 python3 -B scripts/analysis/steady_chain.py "$job" --ref "$ref" 2>&1 | tail -45
else timeout 280 python3 -B scripts/analysis/steady_chain.py "$job" 2>&1 | tail -45; fi
