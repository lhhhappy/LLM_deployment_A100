#!/usr/bin/env bash
# Developer GPU0/1 only. ROOT contains engine/sglang and the two diagnostic scripts.
# Usage: bash run_prefix_producer_dev.sh ROOT ARM FEATURE [MTP=1] [GRAPH=1]
set -euo pipefail
PREFIX_RUN_ROOT=$(readlink -f "${1:?run root}")
PREFIX_ARM=${2:?arm}
case "$PREFIX_RUN_ROOT" in /sjtu/linhang/arena/*) ;; *) exit 2;; esac
case "$PREFIX_ARM" in *[!A-Za-z0-9_-]*|'') exit 2;; esac
PREFIX_ARENA=/sjtu/linhang/arena
export PATH="$PREFIX_ARENA/env/m0/bin:$PREFIX_ARENA/env/sgl/bin:$PATH"
export CUDA_HOME="$PREFIX_ARENA/env/sgl/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX_ARENA/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:$CUDA_HOME/lib:$PREFIX_ARENA/cache/T48/deps/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PREFIX_RUN_ROOT/engine:$PREFIX_ARENA/runs/ep8/alt0/hk/humming_kernels-0.1.12:$PREFIX_ARENA/cache/T48/deps"
export CUDA_VISIBLE_DEVICES=0,1 NCCL_CUMEM_ENABLE=0
export TMPDIR="$PREFIX_ARENA/cache/tmp" HF_HOME="$PREFIX_ARENA/cache/hf"
export SGLANG_CACHE_DIR="$PREFIX_ARENA/cache/sglang" SGLANG_JIT_CACHE_DIR="$PREFIX_ARENA/cache/sglang/jit"
export TRITON_CACHE_DIR="$PREFIX_ARENA/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PREFIX_ARENA/cache/inductor"
export PYTHONPYCACHEPREFIX="$PREFIX_ARENA/cache/pycache"
export TILELANG_CACHE_DIR="$PREFIX_ARENA/cache/tilelang" FLASHINFER_WORKSPACE_BASE="$PREFIX_ARENA/cache/flashinfer"
export HUMMING_CACHE_DIR="$PREFIX_ARENA/cache/humming" HUMMING_TMP_DIR="$PREFIX_ARENA/cache/humming-tmp"
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1
export SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0
export SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=
export SGLANG_AX_PREFIX_PRODUCER=${3:?feature 0 or 1} SGLANG_AX_DEADLINE_FAMILY=0
export SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_DEADLINE_FREEZE_CLASS=1
export SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=4096
export SGLANG_AX_PREFIX_TRACE_S=120 SGLANG_AX_PREFIX_TRACE_ROUNDS=2048
args=()
[[ ${4:-1} = 0 ]] || args+=(--mtp)
[[ ${5:-1} = 0 ]] || args+=(--graph)
python "$PREFIX_RUN_ROOT/engine/prefix_producer_dev.py" \
    --model "$PREFIX_ARENA/runs/dcp-prefill-local-kv-20260926/model-mtp-h16" \
    --output "$PREFIX_RUN_ROOT/$PREFIX_ARM" "${args[@]}"
