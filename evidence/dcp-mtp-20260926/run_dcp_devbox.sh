#!/usr/bin/env bash
# Execute one frozen DCP numerical arm on the 2xA100 developer box.
# Usage: bash run_dcp_devbox.sh ROOT SOURCE ARM DCP GRAPH FILL
set -euo pipefail
DCP_RUN_ROOT=$(readlink -f "${1:?run root}")
DCP_SOURCE=$(readlink -f "${2:?source directory containing sglang}")
DCP_ARM=${3:?arm}; DCP_WIDTH=${4:?DCP width}; DCP_GRAPH=${5:?0 or 1}; DCP_FILL=${6:?filler fraction}
case "$DCP_RUN_ROOT" in /sjtu/linhang/arena/*) ;; *) exit 2;; esac
case "$DCP_SOURCE" in /sjtu/linhang/arena/*) ;; *) exit 2;; esac
DCP_ARENA=/sjtu/linhang/arena
export PATH="$DCP_ARENA/env/m0/bin:$DCP_ARENA/env/sgl/bin:$PATH"
export CUDA_HOME="$DCP_ARENA/env/sgl/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "Missing existing CUDA toolkit" >&2; exit 2; }
export LD_LIBRARY_PATH="$DCP_ARENA/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:$DCP_ARENA/cache/T48/deps/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$DCP_SOURCE:$DCP_ARENA/cache/T48/deps"
export NCCL_CUMEM_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1
export TMPDIR="$DCP_ARENA/cache/tmp" HF_HOME="$DCP_ARENA/cache/hf"
export SGLANG_CACHE_DIR="$DCP_ARENA/cache/sglang" SGLANG_JIT_CACHE_DIR="$DCP_ARENA/cache/sglang/jit"
export TRITON_CACHE_DIR="$DCP_ARENA/cache/triton" TORCHINDUCTOR_CACHE_DIR="$DCP_ARENA/cache/inductor"
export PYTHONPYCACHEPREFIX="$DCP_ARENA/cache/pycache"
export TILELANG_CACHE_DIR="$DCP_ARENA/cache/tilelang" FLASHINFER_WORKSPACE_BASE="$DCP_ARENA/cache/flashinfer"
mkdir -p "$TMPDIR" "$HF_HOME" "$SGLANG_JIT_CACHE_DIR" "$TILELANG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1
export SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0
export SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=
command -v ninja >/dev/null || { echo "Missing JIT build tool ninja" >&2; exit 2; }
if [[ ${DCP_PROBE:-attention} = host_gpu ]]; then
  export HC180_DEVICE=cuda
  cd "$DCP_RUN_ROOT/scripts/tests/hicache180"
  python -m unittest -v test_180_glm_host_pools.TestGlmHostTier.test_dcp_round_trip_and_virtual_indexer_capacity
  exit
fi
if [[ ${DCP_PROBE:-attention} = mtp ]]; then
  graph_args=(); [[ "$DCP_GRAPH" = 0 ]] || graph_args+=(--graph)
  hicache_args=(); [[ ${DCP_HICACHE:-0} = 0 ]] || hicache_args+=(--hicache)
  python "$DCP_RUN_ROOT/dcp_mtp_probe.py" --model "${DCP_MTP_MODEL:-$DCP_RUN_ROOT/model-mtp}" \
    --dcp "$DCP_WIDTH" --output "$DCP_RUN_ROOT/$DCP_ARM" "${hicache_args[@]}" "${graph_args[@]}"
  exit
fi
export AX_CHECK_OUT="$DCP_RUN_ROOT/$DCP_ARM.pt" AX_FILL_FRAC="$DCP_FILL" AX_WELL_SCALED=1
export AX_PREFIX=${AX_PREFIX:-8192,6144} AX_EXT=${AX_EXT:-1000,701} AX_DECODE=${AX_DECODE:-4}
[[ ! -e "$AX_CHECK_OUT" ]] || { echo "Refusing to overwrite $AX_CHECK_OUT" >&2; exit 2; }
graph_args=(); [[ "$DCP_GRAPH" = 1 ]] || graph_args+=(--disable-cuda-graph)
python "$DCP_RUN_ROOT/dcp_check.py" \
  --model-path "$DCP_RUN_ROOT/model" --load-format dummy --tp-size 2 --dcp-size "$DCP_WIDTH" \
  --kv-cache-dtype bfloat16 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --linear-attn-backend triton --mem-fraction-static 0.60 --max-total-tokens 32768 \
  --mamba-radix-cache-strategy extra_buffer --chunked-prefill-size 16384 --disable-custom-all-reduce \
  --max-running-requests 8 --max-mamba-cache-size 64 --cuda-graph-max-bs 2 \
  --batch-size 2 --input-len 8192 --output-len 4 --random-seed 1234 "${graph_args[@]}"
[[ -s "$AX_CHECK_OUT" ]] || { echo "Numerical arm failed: missing $AX_CHECK_OUT" >&2; exit 1; }
