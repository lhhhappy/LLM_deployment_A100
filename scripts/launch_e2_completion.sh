#!/usr/bin/env bash
# T29 diagnostic only; same E2 qfull/000+001 v1.1, new output, no Trisol.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
source /sjtu/linhang/arena/code/e1_env.sh
E2_TASK_RUN=${1:?Supply a new /sjtu/linhang/arena/runs/E2_T29_* directory}
E2_TASK_CONTEXT=${2:-131072}
E2_TASK_KV=${3:-131072}
case "$E2_TASK_RUN" in /sjtu/linhang/arena/runs/E2_T29_*) ;; *) exit 2 ;; esac
test ! -e "$E2_TASK_RUN/server-started"
if [[ -n $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo 'GPU 0 busy; refusing to start.' >&2
    exit 1
fi
E2_TASK_SOURCE=/sjtu/linhang/arena/code/sglang-e2-v1.1
test "$(git -C "$E2_TASK_SOURCE" rev-parse HEAD)" = 94602c9c2b7cbdb8efd5c52802dac6a1c180089e
mkdir -p "$E2_TASK_RUN/storage"
touch "$E2_TASK_RUN/server-started"
cd "$E2_TASK_RUN"
export E2_COMPLETION_ROOT="$E2_TASK_RUN"
export PYTHONPATH="$E2_TASK_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
exec python /sjtu/linhang/arena/code/e2_completion_trace.py \
  --model-path /sjtu/linhang/arena/models/e1-kimi-linear-4l-qfull \
  --tokenizer-path /sjtu/linhang/arena/s1-dev/glm_tok \
  --served-model-name e1-random-kimi-linear \
  --file-storage-path "$E2_TASK_RUN/storage" \
  --host 127.0.0.1 --port 31000 --tp-size 1 \
  --dtype bfloat16 --attention-backend triton --linear-attn-backend triton \
  --sampling-backend pytorch --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype float32 \
  --mamba-max-states-per-path -1 --mamba-track-interval 256 \
  --chunked-prefill-size 8192 --context-length "$E2_TASK_CONTEXT" \
  --mem-fraction-static 0.25 --max-running-requests 8 \
  --max-total-tokens "$E2_TASK_KV" --max-mamba-cache-size 512 \
  --disable-cuda-graph --enable-metrics --incremental-streaming-output
