#!/usr/bin/env bash
# M0 on the GPU box (2xA100): does the submitted code (build/l3_0922f = base + 000 + 101) serve a
# >2048-token prompt on sm80 with the default DSA backends, and with explicit fa3?
# Usage: scripts/m0/run_m0.sh default|fa3   (run from /sjtu/linhang/arena/repo, in tmux)
set -uo pipefail
VARIANT=${1:-default}
ARENA=/sjtu/linhang/arena
REPO=$ARENA/repo
OUT=$ARENA/runs/m0/$VARIANT-$(date +%H%M%S); mkdir -p "$OUT"
source "$REPO/scripts/e1_env.sh"
export PATH=$ARENA/env/m0/bin:$PATH
export PYTHONPATH=$REPO/build/l3_0922f          # exact submitted B code, not the pip-installed sglang
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
unset SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND     # image default
MODEL=$ARENA/models/m0-glm5-dsa-4l
PORT=30100
EXTRA=()
case "$VARIANT" in
  fa3) EXTRA=(--dsa-prefill-backend fa3 --dsa-decode-backend fa3) ;;
  nograph) EXTRA=(--disable-cuda-graph) ;;
  fa3-nograph) EXTRA=(--dsa-prefill-backend fa3 --dsa-decode-backend fa3 --disable-cuda-graph) ;;
esac
{ echo "variant=$VARIANT"; python -c "import sglang,sys;print('sglang from',sglang.__file__)"; pip list 2>/dev/null | grep -iE '^(sglang-kernel|sgl-deep-gemm|flashinfer-python|tilelang|torch) '; } > "$OUT/env.txt" 2>&1
python -m sglang.launch_server --model-path "$MODEL" --load-format dummy --host 127.0.0.1 --port $PORT \
  --tp-size 2 --served-model-name default --enable-metrics --incremental-streaming-output --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer --reasoning-parser glm45 --tool-call-parser glm47 \
  --schedule-policy lpm --mem-fraction-static 0.8 "${EXTRA[@]}" > "$OUT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "SERVER_EXITED_DURING_STARTUP" > "$OUT/result.txt"; break; }
  sleep 5
done
if kill -0 $PID 2>/dev/null; then
  python "$REPO/scripts/m0/probe.py" --port $PORT --out "$OUT" > "$OUT/probe.log" 2>&1
  echo "probe_exit=$?" >> "$OUT/result.txt"
fi
kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null
grep -E "Set DSA backends|NotImplementedError|Traceback|Error|error" "$OUT/server.log" | head -40 > "$OUT/server_errors.txt"
echo "done $OUT"
