N=10
G_NAME=b120v2
G_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 120-sched-protect-chain.patch"
G_ARGS="--chunked-prefill-size 16384 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200"
G_ENV="SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192"
# Generic dev-set run. Wrapper sets: G_NAME (unique src name), G_PATCHES, G_ARGS (extra launch args), G_ENV (env
# assignments), N. Same harness/flags as the other dev jobs; verifies engine liveness afterwards.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_NAME" $G_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
grep -h "KV Cache is allocated\|max_total_num_tokens" $AX/engine_current.log | tail -2 | cut -c1-200
S1=$AX/s1/s1-dev
cd $S1 && S1_HARNESS_DIR=$S1/harness python3 run_dev.py --base-url http://127.0.0.1:$PORT --set dev-combined-v1 \
  --root $S1/data/dev-combined-v1 --cohort $S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json \
  --tok-dir /mnt/models --out $RUN_DIR/dev --n $N ${DEV_EXTRA:-}
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo ENGINE_DEAD_AFTER_RUN; exit 3; }
python3 $AX/verify_kit/analyze_run.py $S1/harness $(ls -t $RUN_DIR/dev/raw_*.jsonl | head -1) | head -12
python3 $AX/verify_kit/logstat.py $RUN_DIR/server.log $PORT 2>/dev/null | head -8
