#!/usr/bin/env bash
# E2 controlled local A/B; only qfull random weights, no Trisol or submission.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
source /sjtu/linhang/arena/code/e1_env.sh
E2_VARIANT=${1:?Use stock, off, or on}
case "$E2_VARIANT" in
  stock) E2_SOURCE=/sjtu/linhang/arena/code/sglang-v0.5.20 ;;
  off|on) E2_SOURCE=/sjtu/linhang/arena/code/sglang-e2-v1.1 ;;
  *) exit 2 ;;
esac
E2_RUN=/sjtu/linhang/arena/runs/E2_20260922
mkdir -p "$E2_RUN/storage"
cd "$E2_RUN"
test "$(git -C "$E2_SOURCE" rev-parse HEAD)" = 94602c9c2b7cbdb8efd5c52802dac6a1c180089e
if [[ "$E2_VARIANT" == stock ]]; then
    git -C "$E2_SOURCE" diff --quiet HEAD -- python
fi
if [[ -n $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo 'GPU 0 is in use; refusing to launch E2.' >&2
    exit 1
fi
unset SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS
if [[ "$E2_VARIANT" == on ]]; then
    export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
fi
export PYTHONPATH="$E2_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
python -c 'import sglang; print("E2 source:",sglang.__file__,flush=True)'
E2_ENTRY=(python -m sglang.launch_server)
if [[ -n ${E2_TRACE_ROOT:-} ]]; then
    E2_ENTRY=(python /sjtu/linhang/arena/code/e2_serve_trace.py)
fi
"${E2_ENTRY[@]}" \
  --model-path /sjtu/linhang/arena/models/e1-kimi-linear-4l-qfull \
  --tokenizer-path /sjtu/linhang/arena/s1-dev/glm_tok \
  --served-model-name e1-random-kimi-linear \
  --file-storage-path "$E2_RUN/storage" \
  --host 127.0.0.1 --port 31000 --tp-size 1 \
  --dtype bfloat16 --attention-backend triton --linear-attn-backend triton \
  --sampling-backend pytorch --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype float32 \
  --mamba-max-states-per-path -1 --mamba-track-interval 256 \
  --chunked-prefill-size 8192 --context-length 131072 \
  --mem-fraction-static 0.25 --max-running-requests 8 \
  --max-total-tokens 131072 --max-mamba-cache-size 512 \
  --disable-cuda-graph --enable-metrics --incremental-streaming-output
