# Formal A + frozen 122 at dev N22; single-variable comparison with 047.
# 122 SHA256 cefdb2688cbc291742fc3c3ad188e343420fad01407d172f164ca6746d712d3b
# An SLO FAIL is evidence at this N, not a prerequisite blocking independent N26 studies.
G_NAME=off_a_122_n22
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 122-tpot-paced-prefill.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_PACE_TPOT=0.085"
LADDER_UP="22"
unset SGLANG_AX_PACE_TPOT SGLANG_AX_KDA_FUSE_PROJ
echo "EXPERIMENT formal_A_plus_122_dev_N22 reference=047 no_profiler"
sha256sum "$AX/verify_kit/level_verdict.py" "$AX/verify_kit/run_dev_checked.py" \
  "$AX/verify_kit/score_formal.py" "$AX/s1/s1-dev/run_dev.py" \
  "$AX/s1/s1-dev/data/dev-combined-v1/requests.jsonl" \
  "$AX/s1/s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json" || exit 2
echo "cefdb2688cbc291742fc3c3ad188e343420fad01407d172f164ca6746d712d3b  $AX/patches/122-tpot-paced-prefill.patch" | sha256sum -c - || exit 2
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
