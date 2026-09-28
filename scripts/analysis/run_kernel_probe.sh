#!/usr/bin/env bash
# Called from a private tmux socket; each experiment has a persistent exit receipt.
set -uo pipefail
KERNEL_ROOT=/sjtu/linhang/arena/codex/mhc-moe-sm80-0928
PROBE_NAME=${1:?name}; KERNEL_GPU=${2:?gpu}; export KERNEL_GPU
shift 2
[[ "$PROBE_NAME" =~ ^[a-z0-9_-]+$ ]] || exit 2
cd "$KERNEL_ROOT" || exit 2
mkdir -p results
date -u +%FT%TZ > "results/$PROBE_NAME.started"
bash scripts/analysis/run_kernel_devbox.sh "$KERNEL_ROOT" "$@" \
  > "results/$PROBE_NAME.jsonl" 2> "results/$PROBE_NAME.stderr"
PROBE_RC=$?
printf '%s\n' "$PROBE_RC" > "results/$PROBE_NAME.exit"
date -u +%FT%TZ > "results/$PROBE_NAME.finished"
exit "$PROBE_RC"
