#!/usr/bin/env bash
# [T52 / patch 170] Dev-box check of GLM breakable prefill CUDA graph (BCG): startup/capture, eager-vs-BCG numerics, chunk cost.
# Runs ON the GPU box (single GPU, TP1, rank-surrogate model, dummy weights). From the local repo:
#   ssh GPU "mkdir -p /sjtu/linhang/arena/runs/T52/kit"; scp patches/*.patch scripts/analysis/{bcg_correctness,bcg_compare}.py \
#       scripts/pod/verify/chunkcost.py scripts/analysis/devbox_bcg_check.sh GPU:/sjtu/linhang/arena/runs/T52/kit/
#   scripts/gjob run t52_bcg 'GPU_ID=1 bash /sjtu/linhang/arena/runs/T52/kit/devbox_bcg_check.sh'
# Arms (sequential, same GPU): eager170 = full stack + 170, --cuda-graph-backend-prefill disabled
#                              bcg170   = full stack + 170, --cuda-graph-backend-prefill breakable
#                              base     = full stack without 170, default flags (KDA rule => eager): 170 must not change eager.
#                              negctl   = eager170 with --chunked-prefill-size 2048 (negative control: comparator must see diffs)
# --skip-server-warmup: the stock warmup prompt is tokenized with the real GLM tokenizer (ids ~154k) but the surrogate
# embedding has vocab 19360 -> index-out-of-bounds device assert in eager AND graph arms (first T52 attempt).
# --enable-metrics: chunkcost.py reads meta_info request_received_ts/prefill_finished_time (0 without it).
# TESTPATCH=t52_test_dummy_init.patch + AX_T52_DUMMY_INIT=1: context-sensitive dummy weights (norms=1, scales=1,
#   matrices +-1/sqrt(fan_in)). REQUIRED for a meaningful numerics test: with stock +-1e-3 dummy init the negative control
#   (chunk 2048 vs 4096) was also bit-identical, i.e. logits only saw the current token.
# T52b arms (need TP>=2, GPU_ID=0,1): eager170sc / bcg170sc = eager170 / bcg170 + --enable-attn-tp-input-scattered.
# Env: TP (default 1), GPU_ID (default 1), ARMS (default "eager170 bcg170 base"), CHUNKCOST=1/0, CORRECTNESS=1/0, SUFFIX (output dir suffix), PORT (default 31952).
# On the 8-card pod: same client scripts; use the real model, TP8, VOCAB_MAX=150000, and the launch flags in patches/170-*.md.
set -uo pipefail
T=${T:-/sjtu/linhang/arena/runs/T52};  # T52b uses T=/sjtu/linhang/arena/runs/T52b
 K=$T/kit; R=/sjtu/linhang/arena/repo
GPU_ID=${GPU_ID:-1}; TP=${TP:-1}; PORT=${PORT:-31952}; ARMS=${ARMS:-"eager170 bcg170 base"}; CHUNKCOST=${CHUNKCOST:-1}; CORRECTNESS=${CORRECTNESS:-1}
MODEL=/sjtu/linhang/arena/runs/rankprof/p8192/model
STACK="000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 114-indexer-row-shard 115-sm80-sparse-attn-many-heads 140-kda-dual-snapshot 120-sched-protect-chain 130-async-tokenize 150-startup-warmup 160-nextn-sm80"
build() {  # build <dir> <patches...>
  rm -rf $1; mkdir -p $1; cp -a $R/build/base_exact/sglang $1/
  for p in "${@:2}"; do patch -p3 -d $1/sglang --fuzz=0 -s --no-backup-if-mismatch < $K/$p.patch || { echo "PATCH FAIL $p"; exit 2; }; done
}
EXTRA_P=${TESTPATCH:+${TESTPATCH%.patch}}   # e.g. TESTPATCH=t52_test_dummy_init.patch (test-only, both trees)
build $T/src/s170 $STACK 170-glm-bcg-prefill $EXTRA_P
build $T/src/s0 $STACK $EXTRA_P
[ -n "${TESTPATCH:-}" ] && export AX_T52_DUMMY_INIT=1
echo "trees built (fuzz=0)"

source /sjtu/linhang/arena/env.sh >/dev/null 2>&1
export CUDA_HOME=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:/sjtu/linhang/arena/env/sgl/bin:/sjtu/linhang/arena/env/m0/bin:$PATH
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:/sjtu/linhang/arena/cache/T48/deps/z3/lib
export NCCL_CUMEM_ENABLE=0 CUDA_VISIBLE_DEVICES=$GPU_ID SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_AX_KDA_DUAL_SNAPSHOT=1 SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
export TRITON_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/triton SGLANG_JIT_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/jit SGLANG_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/sgl TILELANG_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/tl
PY=/sjtu/linhang/arena/env/m0/bin/python
COMMON="--model-path $MODEL --load-format dummy --tp-size $TP --page-size 64 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
 --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype float32 --disable-custom-all-reduce --chunked-prefill-size 4096
 --mem-fraction-static 0.8 --context-length 131072 --random-seed 42 --schedule-policy lpm --skip-server-warmup --enable-metrics --host 127.0.0.1 --port $PORT"

for arm in $ARMS; do
  O=$T/$arm${SUFFIX:-}; rm -rf $O; mkdir -p $O
  case $arm in
    eager170) SRC=$T/src/s170; EXTRA="--cuda-graph-backend-prefill disabled" ;;
    bcg170)   SRC=$T/src/s170; EXTRA="--cuda-graph-backend-prefill breakable" ;;
    base)     SRC=$T/src/s0;   EXTRA="" ;;
    eager170sc) SRC=$T/src/s170; EXTRA="--cuda-graph-backend-prefill disabled --enable-attn-tp-input-scattered" ;;
    bcg170sc)   SRC=$T/src/s170; EXTRA="--cuda-graph-backend-prefill breakable --enable-attn-tp-input-scattered" ;;
    negctl)   SRC=$T/src/s170; EXTRA="--cuda-graph-backend-prefill disabled --chunked-prefill-size 2048" ;;  # comparator must see diffs
  esac
  echo "=== $arm $(date +%T) $EXTRA"
  PYTHONPATH=$SRC:/sjtu/linhang/arena/cache/T48/deps setsid $PY -m sglang.launch_server $COMMON $EXTRA > $O/server.log 2>&1 &
  SPID=$!; echo $SPID > $O/server.pid
  ok=0; for i in $(seq 1 300); do
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SPID 2>/dev/null || break; sleep 5; done
  echo "ready=$ok after ~$((i*5))s"
  if [ $ok = 1 ]; then
    [ "$CORRECTNESS" = 1 ] && PORT=$PORT VOCAB_MAX=19000 timeout 3000 $PY $K/bcg_correctness.py $O/correctness.json > $O/correctness.log 2>&1; echo "correctness rc=$?"
    if [ "$CHUNKCOST" = 1 ] && [ $arm != base ] && [ $arm != negctl ]; then
      PORT=$PORT VOCAB_MAX=19000 timeout 3000 $PY $K/chunkcost.py $O 0,98304 512,1024,2048,4096 > $O/chunkcost.log 2>&1; echo "chunkcost rc=$?"
      grep "^FIT\|^CHUNKCOST" $O/chunkcost.log
    fi
  fi
  kill -- -$SPID 2>/dev/null; sleep 10; kill -9 -- -$SPID 2>/dev/null; wait $SPID 2>/dev/null
  grep -iE "prefill cuda graph|Capture|capture_time|bs=\[|Traceback|Error" $O/server.log | grep -v "^\s*$" | head -20 > $O/server_grep.txt
  echo "graph=True prefill lines: $(grep 'Prefill batch' $O/server.log | grep -c 'graph: True')  total prefill lines: $(grep -c 'Prefill batch' $O/server.log)"
done
cd $T
X=${SUFFIX:-}
[ -f eager170$X/correctness.json ] && [ -f bcg170$X/correctness.json ] && $PY $K/bcg_compare.py eager170$X/correctness.json bcg170$X/correctness.json cmp_eager_vs_bcg$X.json | tail -1
[ -f eager170$X/correctness.json ] && [ -f base$X/correctness.json ] && $PY $K/bcg_compare.py base$X/correctness.json eager170$X/correctness.json cmp_base_vs_eager170$X.json | tail -1
[ -f eager170sc$X/correctness.json ] && [ -f bcg170sc$X/correctness.json ] && $PY $K/bcg_compare.py eager170sc$X/correctness.json bcg170sc$X/correctness.json cmp_eagersc_vs_bcgsc$X.json | tail -1
[ -f eager170$X/correctness.json ] && [ -f eager170sc$X/correctness.json ] && $PY $K/bcg_compare.py eager170$X/correctness.json eager170sc$X/correctness.json cmp_eager_vs_eagersc$X.json | tail -1
[ -f eager170$X/correctness.json ] && [ -f negctl$X/correctness.json ] && $PY $K/bcg_compare.py eager170$X/correctness.json negctl$X/correctness.json cmp_negctl_chunk2048$X.json | tail -1
echo ALLDONE
