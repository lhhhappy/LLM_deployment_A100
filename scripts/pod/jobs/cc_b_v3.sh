G_NAME=b140v3
G_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 140-kda-dual-snapshot.patch 120-sched-protect-chain.patch"
G_ARGS="--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 3"
G_ENV="SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1"
# Cost-model probe job (~4 min + engine restart if config differs). Wrapper sets G_NAME, G_PATCHES, G_ARGS, G_ENV.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_NAME" $G_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
python3 $AX/verify_kit/chunkcost.py $RUN_DIR
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || echo ENGINE_DEAD_AFTER_PROBE
