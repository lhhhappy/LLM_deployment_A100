#!/usr/bin/env bash
# One-shot progress report of the 8-GPU pod, meant for a periodic loop (every 30 min): the queue state, and for every
# running job a read-only window_watch snapshot (completed requests only; every window still open, so the numbers are
# diagnostics, not verdicts), plus the job.log lines that decide whether the run is healthy (mechanism and data checks,
# warmup, gate failures, tracebacks, OOM). Each snapshot line is appended to build/scratch/progress/<job>.log so the
# ticks can be compared. Nothing is written on the pod.
#   bash scripts/analysis/pod_progress.sh            # report
# Early stop is a human decision: stop a single test job with scripts/pod/stopjob <job.sh> on the GPU box; never the service.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p build/scratch/progress
status=$(timeout 100 bash scripts/pod/pread status 2>/dev/null | grep -E '^(running|pending):' | cut -c1-400)
[ -n "$status" ] || { echo "pod status unavailable (transport)"; exit 1; }
echo "$status"
running=$(grep '^running:' <<<"$status" | sed 's/^running: *//')
for job in $running; do
  name=${job%.sh}
  echo "== $name"
  timeout 100 bash scripts/pod/pread grep 'MECHANISMS|DATA_READY|DATA INVALID|GATE FAIL|SHORT_WARMUP_END|Traceback|OutOfMemory|out of memory|MISMATCH|exit' "/tmp/ax/runs/$name/job.log" 12 2>/dev/null \
    | grep -v '^exit_code' | cut -c1-180 | sed 's/^/  job.log: /'
  snap=$(timeout 280 bash scripts/analysis/window_watch.sh "$name" --once 2>&1 | tail -1 | cut -c1-500)
  line="$(date -u +%FT%TZ) $snap"
  echo "$line" >> "build/scratch/progress/$name.log"
  echo "  now:  $snap"
  prev=$(tail -n 2 "build/scratch/progress/$name.log" | head -n 1)
  [ "$prev" != "$line" ] && [ -n "$prev" ] && echo "  prev: ${prev#* }"
done
[ -n "$running" ] || echo "(nothing running)"
# Global board: every job of the plan with its reference and per-gate arrows (notes/kanban_plan.json -> notes/kanban.md).
echo "== kanban"
timeout 280 python3 -B scripts/analysis/kanban.py 2>/dev/null | tail -n +4
