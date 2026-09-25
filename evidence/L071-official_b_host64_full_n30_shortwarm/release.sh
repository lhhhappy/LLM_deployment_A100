#!/usr/bin/env bash
set -euo pipefail
cd /sjtu/linhang/arena/repo
source scripts/pod/common.sh
log=/sjtu/linhang/arena/runs/jobs/pod071-data-20260925.log
for task_wait in $(seq 1 240); do
  if grep -q '^DONE rc=0$' "$log"; then break; fi
  if grep -Eq '^DONE rc=[1-9][0-9]*$' "$log"; then echo DATA_TRANSFER_FAILED; exit 2; fi
  sleep 10
done
grep -q '^DONE rc=0$' "$log"
scripts/pod/ppush /tmp/ax/bin build/scratch/pod071-verify.py
bexec 'python3 -B /tmp/ax/bin/build/scratch/pod071-verify.py'
scripts/pod/podq init
for task_wait in $(seq 1 270); do
  state=$(bexec 'r=/tmp/ax/runs/071-official_b_host64_full_n30_shortwarm; if grep -q "^PRELOAD_READY:" "$r/preload.log"; then echo READY; elif kill -0 "$(cat "$r/preload.pid")" 2>/dev/null; then echo WAIT; else echo PRELOAD_FAILED; exit 3; fi')
  if grep -q '^READY$' <<<"$state"; then break; fi
  sleep 10
done
grep -q '^READY$' <<<"$state"
scripts/pod/podq resume
echo REPLAY_RELEASED_071
