CP_NAME=b114
CP_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 114-indexer-row-shard.patch"
CP_ARGS=""
# Cold-prefill probe: start/reuse engine VARIANT, then single-request cold prefills (20k/60k/190k) + profile of 60k,
# + capability smoke subset (correctness guard for structural variants). Wrapper sets:
#   CP_NAME (unique src name), CP_PATCHES (patch list), CP_ARGS (extra launch args), CP_ENV (extra env assignments)
source $AX/bin/scripts/pod/lib.sh
prepare_src "$CP_NAME" $CP_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${CP_ENV:-}" ] && export $CP_ENV
ensure_engine "$CP_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $CP_ARGS || exit 1
grep -h "KV Cache is allocated\|max_total_num_tokens\|row-shard active\|cp_size\|dcp" $AX/engine_current.log | head -6
python3 $AX/verify_kit/coldprobe.py $RUN_DIR 20000,60000,190000 60000
python3 $AX/verify_kit/component_table.py $RUN_DIR/prof_60000 2>/dev/null | head -14
