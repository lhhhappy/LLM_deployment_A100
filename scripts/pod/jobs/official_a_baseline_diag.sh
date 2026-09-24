# 051: diagnose formal A repeat instability before testing either operator patch.
# Reuses the unchanged baseline engine; cold, serialized one-token repetitions and
# separate 48-token repeats. No SLO/ability/numerical-equivalence verdict is inferred.
G_NAME=off_a_kernelref
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096"

source "$AX/bin/scripts/pod/lib.sh"
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export $G_ENV
unset SGLANG_AX_PACE_TPOT SGLANG_AX_KDA_FUSE_PROJ SGLANG_AX_MOE_FUSE_SWIGLU
printf '%s\n' 'DIAGNOSTIC_ONLY formal_A baseline repeat; no candidate patch'
echo "d46832a6decafaa2bc62562845c09bc1b1b10d9dd7034a6535dd60eb356898ed  $AX/verify_kit/numcheck_baseline_diag.py" | sha256sum -c - || exit 2
prepare_src "$G_NAME" $G_PATCHES || exit 2
( cd "$AX/patches" && sha256sum $G_PATCHES ) | sed 's/^/PATCH_SHA /'
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 3
curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$RUN_DIR/server_info.json" || exit 3
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$RUN_DIR/gpu_memory.csv" || exit 3
python3 "$AX/verify_kit/numcheck_baseline_diag.py" "$RUN_DIR/diag" > "$RUN_DIR/diag.log" 2>&1 || { tail -30 "$RUN_DIR/diag.log"; exit 4; }
cat "$RUN_DIR/diag.log"
echo 'BASELINE_DIAGNOSTIC_COMPLETE: inspect repeat differences; this is not an acceptance gate'
