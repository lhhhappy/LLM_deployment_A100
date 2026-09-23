# T52b numerics fingerprint: 170+114 stack, prefill backend=breakable, scatter=[], chunk 4096. Diff vs /tmp/ax/runs/029a-num_eager_scat/num/numcheck.json.
source $AX/bin/scripts/pod/lib.sh
prepare_src b170r114 000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 114-indexer-row-shard.patch 140-kda-dual-snapshot.patch 120-sched-protect-chain.patch 170-glm-bcg-prefill.patch || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1 SGLANG_AX_INDEXER_ROW_SHARD=1
ensure_engine b170r114 --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --chunked-prefill-size 4096 --mem-fraction-static 0.75  --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 3 --cuda-graph-backend-prefill breakable || exit 1
python3 $AX/verify_kit/numcheck.py $RUN_DIR/num
[ -n "/tmp/ax/runs/029a-num_eager_scat/num/numcheck.json" ] && python3 $AX/verify_kit/numcheck_cmp.py /tmp/ax/runs/029a-num_eager_scat/num/numcheck.json $RUN_DIR/num/numcheck.json
