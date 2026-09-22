#!/usr/bin/env bash
# E2 serial numeric-only follow-up after the three controlled replay variants.
# Own tmux sessions only; no changes to source/model weights or other GPU jobs.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
source /sjtu/linhang/arena/code/e1_env.sh
E2_RUN=/sjtu/linhang/arena/runs/E2_20260922
E2_DEADLINE=$((SECONDS + 900))
until [[ -f "$E2_RUN/on_stock20/summary.json" ]] && ! tmux has-session -t arena-e2-replay 2>/dev/null; do
    if (( SECONDS >= E2_DEADLINE )); then exit 1; fi
    sleep 2
done
test ! -e "$E2_RUN/comparison.json"
python /sjtu/linhang/arena/code/compare_e2.py "$E2_RUN" > "$E2_RUN/comparison.json"

stop_owned_server() {
    tmux send-keys -t arena-e2-server C-c
    for E2_I in $(seq 1 30); do
        if ! tmux has-session -t arena-e2-server 2>/dev/null; then return; fi
        sleep 1
    done
    echo 'Own server did not exit after SIGINT; refusing another launch' >&2
    return 1
}

stop_owned_server
for E2_VARIANT in on off; do
    test ! -e "$E2_RUN/trace-$E2_VARIANT"
    tmux new-session -d -s arena-e2-server \
      "E2_TRACE_ROOT=$E2_RUN/trace-$E2_VARIANT bash /sjtu/linhang/arena/code/launch_e2_standin.sh $E2_VARIANT > $E2_RUN/server-trace-$E2_VARIANT.log 2>&1"
    E2_DEADLINE=$((SECONDS + 600))
    until curl --silent --fail --max-time 2 http://127.0.0.1:31000/health >/dev/null; do
        if ! tmux has-session -t arena-e2-server 2>/dev/null; then
            echo 'Trace server exited; see server log' >&2
            exit 1
        fi
        if (( SECONDS >= E2_DEADLINE )); then exit 1; fi
        sleep 2
    done
    python /sjtu/linhang/arena/code/e2_raw_logits.py \
      --comparison "$E2_RUN/comparison.json" --trace-root "$E2_RUN/trace-$E2_VARIANT" \
      --dev-root /sjtu/linhang/arena/s1-dev --variant "$E2_VARIANT"
    stop_owned_server
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
