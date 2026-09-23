# Cost-model probe job (~4 min + engine restart if config differs). Wrapper sets G_NAME, G_PATCHES, G_ARGS, G_ENV.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_NAME" $G_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
python3 $AX/verify_kit/chunkcost.py $RUN_DIR
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || echo ENGINE_DEAD_AFTER_PROBE
