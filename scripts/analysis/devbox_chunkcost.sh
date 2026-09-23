#!/usr/bin/env bash
# T51: per-chunk extend cost probe + torch-profiler traces on the 2xA100 dev box (GPU0 only), surrogate 8-layer TP8-rank model.
# Surrogate vocab is 19360 (vocab/8): server warmup is skipped and probes draw token ids < 19000.
# Run ON the dev box (e.g. scripts/gjob run t51_x 'bash /sjtu/linhang/arena/repo/scripts/analysis/devbox_chunkcost.sh <cmd>').
#   build    : copy base_exact + apply the patch stack into $T/src
#   serve    : launch sglang (TP1, GPU0, port 31000) in the foreground (run it in its own gjob/tmux window)
#   probe    : chunkcost.py fixed-cost fit  -> $T/chunkcost/
#   mk45     : build the 45-layer surrogate ($T/model45); then serve/probe/profile with MODEL=$T/model45 MEM=0.88 MAXTOK=262144
#   profile  : torch-profiler traces of one extend (c=1024) at P=98304 and P=0 -> $T/prof/<tag>/
# Analysis (on the box, python = /sjtu/linhang/arena/env/m0/bin/python, cwd = repo):
#   scripts/analysis/t51_trace_stats.py <trace.gz> --kernels-out k.json   GPU span/busy/idle(CPU-starved), launches, top kernels, components
#   scripts/analysis/t51_pytree.py <stack trace.gz> '<regex>' [depth] [min_ms]  python call tree (LONGEST=1 -> only longest call)
#   scripts/analysis/t51_pyself.py <stack trace.gz> '<root regex>'        python self time, grouped by framework
# Full T51 sequence (GPU0 only; each server in its own gjob window; stop the server afterwards):
#   build; serve & -> probe; profile                                      (8-layer surrogate)
#   mk45; MODEL=$T/model45 MEM=0.88 MAXTOK=262144 serve & ->
#     OUT=$T/chunkcost45 PS=0,98304,180224 CS=256,1024,2048,4096,8192,16384 probe
#     OUT=$T/prof45 MODES=nostack profile                                  (45-layer surrogate)
set -euo pipefail
R=/sjtu/linhang/arena/repo; T=${T:-/sjtu/linhang/arena/runs/T51}; PORT=${PORT:-31000}
MODEL=${MODEL:-/sjtu/linhang/arena/runs/rankprof/p8192/model}  # model45: MODEL=$T/model45 (build with: mk45)
PATCHES="000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 140-kda-dual-snapshot 120-sched-protect-chain"
source /sjtu/linhang/arena/env.sh >/dev/null 2>&1 || true
export CUDA_HOME=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:/sjtu/linhang/arena/env/sgl/bin:/sjtu/linhang/arena/env/m0/bin:$PATH
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:/sjtu/linhang/arena/cache/T48/deps/z3/lib
export NCCL_CUMEM_ENABLE=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$T/src:/sjtu/linhang/arena/cache/T48/deps SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export TRITON_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/triton SGLANG_JIT_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/jit SGLANG_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/sgl TILELANG_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/tl
export SGLANG_TORCH_PROFILER_DIR=$T/prof
PY=/sjtu/linhang/arena/env/m0/bin/python
case "${1:-}" in
  build)
    rm -rf $T/src; mkdir -p $T/src; cp -a $R/build/base_exact/sglang $T/src/
    for p in $PATCHES; do echo "patch $p"; patch -p3 -d $T/src/sglang --fuzz=0 -s < $R/patches/$p.patch; done
    ( cd $R && md5sum $(for p in $PATCHES; do echo patches/$p.patch; done) ) > $T/src/PATCHES.md5; echo built ;;
  mk45)  # 45-layer per-rank surrogate: [KDA,KDA,KDA,DSA]x11 + KDA = 34 KDA + 11 DSA (same per-rank shapes as p8192)
    mkdir -p $T/model45; cp /sjtu/linhang/arena/runs/rankprof/p8192/model/* $T/model45/
    $PY - $T/model45/config.json <<'PYEOF'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); t = c["text_config"]; L = 45
lt = (["linear_attention"] * 3 + ["deepseek_sparse_attention"]) * 11 + ["linear_attention"]
t.update(num_hidden_layers=L, layer_types=lt, mlp_layer_types=["sparse"] * L)
t["linear_attn_config"].update(kda_layers=[i for i, x in enumerate(lt) if x == "linear_attention"], full_attn_layers=[i for i, x in enumerate(lt) if x != "linear_attention"])
if "indexer_types" in t: t["indexer_types"] = ["full"] * L
json.dump(c, open(p, "w"), indent=1); print("model45", lt.count("linear_attention"), "KDA", lt.count("deepseek_sparse_attention"), "DSA")
PYEOF
    ;;
  serve)
    exec $PY -m sglang.launch_server --model-path $MODEL --load-format dummy --tp-size 1 --port $PORT --host 127.0.0.1 \
      --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --page-size 64 --mamba-radix-cache-strategy extra_buffer \
      --chunked-prefill-size 16384 --disable-custom-all-reduce --mem-fraction-static ${MEM:-0.80} --context-length 262144 --max-total-tokens ${MAXTOK:-1048576} --skip-server-warmup --enable-metrics ${EXTRA_ARGS:-} ;;
  probe)
    OUT=${OUT:-$T/chunkcost}; mkdir -p $OUT; cd $R
    VOCAB_MAX=19000 PORT=$PORT $PY scripts/pod/verify/chunkcost.py $OUT ${PS:-0,32768,98304,180224} ${CS:-256,512,1024,2048,4096,8192,16384} ;;
  profile)
    cd $R; PORT=$PORT $PY scripts/analysis/t51_profile_extend.py ${OUT:-$T/prof} ${PPS:-98304,0} ${PC:-1024} ;;
  *) sed -n 2,10p "$0" ;;
esac
