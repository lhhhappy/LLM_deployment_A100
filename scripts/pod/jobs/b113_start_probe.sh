# 000+101+105+110+111+112 (fused sm80 indexer kernels); DSA backends tilelang. Starts/reuses the engine and probes
# 500..20000-token prompts + /flush_cache. Leaves the engine running for the next job.
source $AX/bin/scripts/pod/lib.sh
prepare_src b113 000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0   # mHC large-batch path: tilelang instead of DeepGEMM (sm80)
ensure_engine b113 --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang || exit 1
grep -n "Set DSA backends\|max_running_requests\|KV Cache is allocated\|mamba" $RUN_DIR/server.log | head -8
python3 $AX/bin/scripts/m0/probe.py --port $PORT --out $RUN_DIR
