#!/usr/bin/env bash
# Run a probe in an isolated developer-GPU directory, using the existing m0 env.
# Usage: bash run_prefill_devbox.sh ROOT command.py [args...]
set -euo pipefail
PREFILL_ROOT=$(readlink -f "${1:?probe root}")
shift
case "$PREFILL_ROOT" in /sjtu/linhang/arena/*) ;; *) exit 2;; esac
PREFILL_ARENA=/sjtu/linhang/arena
export CUDA_VISIBLE_DEVICES=${PREFILL_GPU:-0}
export CUDA_HOME="$PREFILL_ARENA/env/sgl/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$PREFILL_ARENA/env/m0/bin:$PREFILL_ARENA/env/sgl/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PREFILL_ARENA/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:$CUDA_HOME/lib:$PREFILL_ARENA/cache/T48/deps/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PREFILL_ROOT/engine:$PREFILL_ARENA/cache/T48/deps"
export TMPDIR="$PREFILL_ROOT/cache/tmp" HF_HOME="$PREFILL_ROOT/cache/hf"
export SGLANG_CACHE_DIR="$PREFILL_ROOT/cache/sglang"
export SGLANG_JIT_CACHE_DIR="$PREFILL_ROOT/cache/sglang/jit"
export TRITON_CACHE_DIR="$PREFILL_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PREFILL_ROOT/cache/inductor"
export PYTHONPYCACHEPREFIX="$PREFILL_ROOT/cache/pycache"
export TILELANG_CACHE_DIR="$PREFILL_ROOT/cache/tilelang"
export FLASHINFER_WORKSPACE_BASE="$PREFILL_ROOT/cache/flashinfer"
mkdir -p "$TMPDIR" "$HF_HOME" "$SGLANG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$TILELANG_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE"
cd "$PREFILL_ROOT"
exec "$PREFILL_ARENA/env/m0/bin/python" "$@"
