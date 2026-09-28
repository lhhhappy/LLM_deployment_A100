#!/usr/bin/env bash
# Isolated SM80 probes in the existing developer environment; no Pod activity.
set -euo pipefail
KERNEL_ROOT=$(readlink -f "${1:?probe root}")
shift
case "$KERNEL_ROOT" in /sjtu/linhang/arena/codex/mhc-moe-sm80-0928) ;; *) exit 2;; esac
KERNEL_ARENA=/sjtu/linhang/arena
export CUDA_VISIBLE_DEVICES=${KERNEL_GPU:-0}
export CUDA_HOME="$KERNEL_ARENA/env/sgl/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$KERNEL_ARENA/env/m0/bin:$KERNEL_ARENA/env/sgl/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$KERNEL_ARENA/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:$CUDA_HOME/lib:$KERNEL_ARENA/cache/T48/deps/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$KERNEL_ROOT/engine:$KERNEL_ARENA/runs/ep8/alt0/hk/humming_kernels-0.1.12:$KERNEL_ARENA/cache/T48/deps"
export TMPDIR="$KERNEL_ROOT/cache/tmp" HF_HOME="$KERNEL_ROOT/cache/hf"
export SGLANG_CACHE_DIR="$KERNEL_ROOT/cache/sglang"
export SGLANG_JIT_CACHE_DIR="$KERNEL_ROOT/cache/sglang/jit"
export TRITON_CACHE_DIR="$KERNEL_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$KERNEL_ROOT/cache/inductor"
export PYTHONPYCACHEPREFIX="$KERNEL_ROOT/cache/pycache"
export TILELANG_CACHE_DIR="$KERNEL_ROOT/cache/tilelang"
export FLASHINFER_WORKSPACE_BASE="$KERNEL_ROOT/cache/flashinfer"
export HUMMING_CACHE_DIR="$KERNEL_ROOT/cache/humming"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
mkdir -p "$TMPDIR" "$HF_HOME" "$SGLANG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$TILELANG_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$HUMMING_CACHE_DIR"
cd "$KERNEL_ROOT"
exec "$KERNEL_ARENA/env/m0/bin/python" "$@"
