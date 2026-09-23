# Prepared by W22/T48; execute only after Claude reviews/approves 8-GPU testing.
set -euo pipefail
N=6
source "$AX/bin/scripts/pod/lib.sh"
# Unique source/config identity: never reuse a plain b113 engine.
NAME=b160_mtp_s3_k1_d4_mr32_n6
prepare_src "$NAME" 000-interface-compliance.patch 101-d1v12-on-base.patch \
  105-role-split-single-partial.patch 110-sm80-dsa-indexer.patch \
  111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch \
  113-sm80-prefill-indexer.patch 140-kda-dual-snapshot.patch \
  120-sched-protect-chain.patch 130-async-tokenize.patch \
  150-startup-warmup.patch 160-nextn-sm80.patch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1
export SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=
export SGLANG_AX_SCHED_PROTECT=0 SGLANG_AX_ASYNC_TOKENIZE=0
export SGLANG_MAMBA_SSM_DTYPE=float32
export SGLANG_OPT_FUSED_KDA_VERIFY=0   # fused verify writes conv_state in-kernel (race; upstream fix #39524 not in base) — keep off
# 32 permits N22/26 follow-ups but saves up to ~1.10 GiB/rank of four-step scratch
# versus implicit 48 (if main-pool capacity permits those request counts).
# This is a request cap, not the number of KV/SSM cache slots.
# No ax_shapes: 150 explicitly skips speculative decoding. Harness warmup stays.
ensure_engine "$NAME" --schedule-policy lpm \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype bfloat16 --linear-attn-backend triton \
  --linear-attn-verify-backend triton \
  --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 32 --cuda-graph-max-bs 32
# Keep the live log's actual location even if ensure_engine reused this same profile.
# lib.sh currently only copies engine_current.log once; that is not a live log.
ENGINE_PID=$(pgrep -f "^python3 -m sglang.launch_server.*--port $PORT " | head -1)
LIVE_LOG=$(readlink "/proc/$ENGINE_PID/fd/1")
[ -f "$LIVE_LOG" ] || { echo 'LIVE_SERVER_LOG_NOT_FOUND'; exit 2; }
START_LINE=$(( $(wc -l < "$LIVE_LOG") + 1 ))
S1=$AX/s1/s1-dev
cd "$S1"
set +e
S1_HARNESS_DIR="$S1/harness" python3 run_dev.py --base-url "http://127.0.0.1:$PORT" --set dev-combined-v1 \
  --root "$S1/data/dev-combined-v1" --cohort "$S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json" \
  --tok-dir /mnt/models --out "$RUN_DIR/dev" --n "$N" ${DEV_EXTRA:-}
rc=$?
set -e
sed -n "${START_LINE},\$p" "$LIVE_LOG" > "$RUN_DIR/spec_harness.log"
python3 "$AX/bin/scripts/extract_spec_stats_160.py" "$RUN_DIR/spec_harness.log" \
  --draft-tokens 4 --out "$RUN_DIR/spec_stats.json"
# Includes preflight/warmup; use harness measurement timestamps to slice a pure
# measurement window before comparing acceptance. Never average rounded intervals.
python3 -c 'import json,sys; print("SUMMARY",json.dumps(json.load(open(sys.argv[1])),ensure_ascii=False)[:1500])' "$RUN_DIR/dev/summary.json" || true
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo ENGINE_DEAD_AFTER_RUN; exit 3; }
exit "$rc"
