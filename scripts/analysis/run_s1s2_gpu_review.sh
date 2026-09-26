#!/usr/bin/env bash
# Developer GPU0 only; an isolated source export, existing arena Python/CUDA.
# Usage: bash run_s1s2_gpu_review.sh /sjtu/linhang/arena/runs/review-s1s2-20260926 TEST [ARGS...]
set -euo pipefail
REVIEW_RUN=$(readlink -f "${1:?run directory}")
shift
case "$REVIEW_RUN" in /sjtu/linhang/arena/runs/review-s1s2-*) ;; *) exit 2;; esac
REVIEW_ARENA=/sjtu/linhang/arena
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$REVIEW_ARENA/env/sgl/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$REVIEW_ARENA/env/m0/bin:$REVIEW_ARENA/env/sgl/bin:$PATH"
export LD_LIBRARY_PATH="$REVIEW_ARENA/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:$CUDA_HOME/lib:$REVIEW_ARENA/cache/T48/deps/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REVIEW_RUN/candidate/engine:$REVIEW_ARENA/runs/ep8/alt0/hk/humming_kernels-0.1.12:$REVIEW_ARENA/cache/T48/deps"
export NCCL_CUMEM_ENABLE=0
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$REVIEW_RUN/tmp" HF_HOME="$REVIEW_ARENA/cache/hf"
export HUMMING_CACHE_DIR="$REVIEW_RUN/humming-cache" HUMMING_TMP_DIR="$REVIEW_RUN/humming-tmp"
export SGLANG_CACHE_DIR="$REVIEW_RUN/sglang-cache" SGLANG_JIT_CACHE_DIR="$REVIEW_RUN/sglang-cache/jit"
export TRITON_CACHE_DIR="$REVIEW_RUN/triton-cache" TORCHINDUCTOR_CACHE_DIR="$REVIEW_RUN/inductor-cache"
export TILELANG_CACHE_DIR="$REVIEW_RUN/tilelang-cache" FLASHINFER_WORKSPACE_BASE="$REVIEW_RUN/flashinfer-cache"
mkdir -p "$TMPDIR" "$HUMMING_CACHE_DIR" "$HUMMING_TMP_DIR" "$SGLANG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TILELANG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_MOE_FUSE_SWIGLU=0 HC180_DEVICE=cuda
cd "$REVIEW_RUN/candidate"
exec python -B "$@"
