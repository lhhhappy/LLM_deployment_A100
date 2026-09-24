# Formal A + 171, full original dev N22; single-variable comparison to 047.
# An SLO FAIL is evidence at this N, not a prerequisite blocking independent N26 studies.
G_NAME=off_a_171_n22
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch 171-kda-bf16-proj-fusion.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_KDA_FUSE_PROJ=1"
LADDER_UP="22"
unset SGLANG_AX_PACE_TPOT SGLANG_AX_MOE_FUSE_SWIGLU SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER NUMREF
SMOKE_GATE=0  # Full harness preflight and task gates remain; no local output-repeat gate.
echo "EXPERIMENT formal_A_plus_171_dev_N22 reference=047 no_profiler no_output_equality_gate"
sha256sum "$AX/verify_kit/level_verdict.py" "$AX/verify_kit/run_dev_checked.py" \
  "$AX/verify_kit/score_formal.py" "$AX/s1/s1-dev/run_dev.py" \
  "$AX/s1/s1-dev/data/dev-combined-v1/requests.jsonl" \
  "$AX/s1/s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json" || exit 2
echo "08eee808be8632776721c90769162a3a30622b8e73fd806565a0c94cdb10026e  $AX/patches/171-kda-bf16-proj-fusion.patch" | sha256sum -c - || exit 2
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
