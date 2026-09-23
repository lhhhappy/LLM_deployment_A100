#!/usr/bin/env bash
# [ax] T50/116 rerun script (GPU dev box, 2xA100, TP2). Reproduces the DCP address bug on the stock stack and checks
# patch 116: DCP2 logits (cold prefill, prefix-hit extend, decode) vs the same stack without DCP, plus a HIGH-slot
# variant (filler pushes the requests' virtual locs above the per-rank pool rows).
# Run ON the dev box from the repo copy:  bash scripts/analysis/devbox_dcp_check.sh [cases...]
#   env: T (out dir, default /sjtu/linhang/arena/runs/T50), AX_MEM (0.55), AX_PREFIX/AX_EXT/AX_DECODE (see dcp_check.py)
#   cases (default all): orig_ref orig_dcp orig_dcp_hi fix_ref fix_dcp fix_dcp_hi fix_dcp_graph full_ref full_dcp_hi
# Needs BOTH GPUs free. Each case is one engine start (~2-4 min). Results: $T/<case>.pt, $T/<case>.log, $T/matrix_log.txt
set -uo pipefail
R=/sjtu/linhang/arena/repo; T=${T:-/sjtu/linhang/arena/runs/T50}; mkdir -p $T/src
BASE_STACK="000 101 105 106 110 111 112 113 114 115"
build() {  # build <name> <patch ids...>
  local n=$1; shift; [ -f $T/src/$n/.ok ] && return 0
  rm -rf $T/src/$n; mkdir -p $T/src/$n; cp -a $R/build/base_exact/sglang $T/src/$n/
  for p in "$@"; do patch -p3 -d $T/src/$n/sglang --fuzz=0 -s < $(ls $R/patches/$p-*.patch) || { echo "PATCH FAIL $n $p"; return 1; }; done
  touch $T/src/$n/.ok
}
build orig $BASE_STACK || exit 1
build fix $BASE_STACK 116 || exit 1
build full $BASE_STACK 116 140 120 130 150 160 || exit 1

source /sjtu/linhang/arena/env.sh >/dev/null 2>&1
export CUDA_HOME=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:/sjtu/linhang/arena/env/sgl/bin:/sjtu/linhang/arena/env/m0/bin:$PATH
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:/sjtu/linhang/arena/cache/T48/deps/z3/lib
export NCCL_CUMEM_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export TRITON_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/triton SGLANG_JIT_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/jit SGLANG_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/sgl TILELANG_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/tl
PY=/sjtu/linhang/arena/env/m0/bin/python
MODEL=${AX_MODEL:-/sjtu/linhang/arena/runs/rankprof/p8192/model}
cd $R

run() {  # run <case> <stack> <dcp 1|2> <fill_frac> <graph 0|1>
  local c=$1 s=$2 d=$3 f=$4 g=$5 extra=""
  [ $d = 2 ] && extra="--dcp-size 2"
  [ $g = 0 ] && extra="$extra --cuda-graph-backend-decode disabled"
  echo "== $c stack=$s dcp=$d fill=$f graph=$g" | tee -a $T/matrix_log.txt
  PYTHONPATH=$T/src/$s:/sjtu/linhang/arena/cache/T48/deps AX_FILL_FRAC=$f AX_CHECK_OUT=$T/$c.pt timeout 1500 \
    $PY scripts/analysis/dcp_check.py --model-path $MODEL --load-format dummy --tp-size 2 $extra --page-size 64 \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --mamba-radix-cache-strategy extra_buffer \
    --batch-size 2 --input-len 1024 --output-len 1 --chunked-prefill-size 16384 --mem-fraction-static ${AX_MEM:-0.55} \
    --disable-custom-all-reduce > $T/$c.log 2>&1
  echo "rc=$? $(grep -m1 'AX_CHECK done' $T/$c.log | cut -c1-300)" | tee -a $T/matrix_log.txt
  grep -m3 -E "illegal memory|device-side assert|Error|Traceback" $T/$c.log | cut -c1-240 | tee -a $T/matrix_log.txt
}
cmp() { [ -f $T/$1.pt ] && [ -f $T/$2.pt ] && { echo "-- compare $1 vs $2" | tee -a $T/matrix_log.txt; $PY scripts/analysis/dcp_compare.py $T/$1.pt $T/$2.pt 2>&1 | tee -a $T/matrix_log.txt; }; }

CASES=${*:-"orig_ref orig_dcp orig_dcp_hi fix_ref fix_dcp fix_dcp_hi fix_dcp_graph full_ref full_dcp_hi"}
date | tee -a $T/matrix_log.txt
for c in $CASES; do case $c in
  orig_ref)      run $c orig 1 0 0 ;;
  orig_dcp)      run $c orig 2 0 0 ;;
  orig_dcp_hi)   run $c orig 2 0.6 0 ;;
  fix_ref)       run $c fix 1 0 0 ;;
  fix_dcp)       run $c fix 2 0 0 ;;
  fix_dcp_hi)    run $c fix 2 0.6 0 ;;
  fix_dcp_graph) run $c fix 2 0.6 1 ;;
  full_ref)      run $c full 1 0 0 ;;
  full_dcp_hi)   run $c full 2 0.6 0 ;;
esac; done
cmp orig_dcp orig_ref; cmp orig_dcp_hi orig_ref; cmp fix_ref orig_ref
cmp fix_dcp fix_ref; cmp fix_dcp_hi fix_ref; cmp fix_dcp_graph fix_ref; cmp full_dcp_hi full_ref
echo ALLDONE | tee -a $T/matrix_log.txt
$PY scripts/analysis/devbox_dcp_verdict.py $T   # -> $T/SUMMARY.txt (verdict)
