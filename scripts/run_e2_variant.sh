#!/usr/bin/env bash
# Sequential internal-case replay for one already-running E2 server variant.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
source /sjtu/linhang/arena/code/e1_env.sh
E2_VARIANT=${1:?Use stock, off, or on}
case "$E2_VARIANT" in stock|off|on) ;; *) exit 2 ;; esac
E2_RUN=/sjtu/linhang/arena/runs/E2_20260922
E2_DEADLINE=$((SECONDS + 600))
until curl --silent --fail --max-time 2 http://127.0.0.1:31000/health >/dev/null; do
    if (( SECONDS >= E2_DEADLINE )); then
        echo 'E2 server readiness timed out' >&2
        exit 1
    fi
    sleep 2
done
E2_FLUSH_ARGS=()
if [[ "$E2_VARIANT" != stock ]]; then E2_FLUSH_ARGS=(--require-json-flush); fi
for E2_CASE in smoke reminder_heavy strict_append; do
    python /sjtu/linhang/arena/code/replay_chains.py \
      --dev-root /sjtu/linhang/arena/s1-dev \
      --case-file "/sjtu/linhang/arena/code/e2_cases/$E2_CASE.json" \
      --max-prompt-tokens 131072 \
      --output "$E2_RUN/${E2_VARIANT}_${E2_CASE}" "${E2_FLUSH_ARGS[@]}"
done
python /sjtu/linhang/arena/code/replay_chains.py \
  --dev-root /sjtu/linhang/arena/s1-dev --num-chains 20 \
  --output "$E2_RUN/${E2_VARIANT}_stock20" "${E2_FLUSH_ARGS[@]}"
