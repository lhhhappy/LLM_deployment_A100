#!/usr/bin/env bash
# Local random-weight stock baseline only. Run under an arena-owned tmux session.
set -euo pipefail
source /sjtu/linhang/arena/code/e1_env.sh
cd /sjtu/linhang/arena/runs/E1_20260922
E1_SOURCE=/sjtu/linhang/arena/code/sglang-v0.5.20
test "$(git -C "$E1_SOURCE" rev-parse HEAD)" = 94602c9c2b7cbdb8efd5c52802dac6a1c180089e
git -C "$E1_SOURCE" diff --quiet HEAD -- python
if [[ -n $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo 'GPU 0 is in use; refusing to launch E1.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
python -m sglang.launch_server \
  --model-path /sjtu/linhang/arena/models/e1-kimi-linear-4l-qfull \
  --tokenizer-path /sjtu/linhang/arena/s1-dev/glm_tok \
  --served-model-name e1-random-kimi-linear \
  --file-storage-path /sjtu/linhang/arena/runs/E1_20260922/storage \
  --host 127.0.0.1 --port 31000 --tp-size 1 \
  --dtype bfloat16 --attention-backend triton --linear-attn-backend triton \
  --sampling-backend pytorch --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype float32 \
  --mamba-max-states-per-path -1 --mamba-track-interval 256 \
  --chunked-prefill-size 8192 --context-length 65536 \
  --mem-fraction-static 0.25 --max-running-requests 8 \
  --max-total-tokens 131072 --max-mamba-cache-size 512 \
  --disable-cuda-graph --enable-metrics --incremental-streaming-output
