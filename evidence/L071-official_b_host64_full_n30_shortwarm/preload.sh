#!/usr/bin/env bash
set -eu
AX=/tmp/ax
RUN_DIR=$AX/runs/071-official_b_host64_full_n30_shortwarm
export AX RUN_DIR
#!/usr/bin/env bash
# L070: paired 122 test against L069, keeping host64 and all other settings.
# Re-enable fixed paced prefill (tau .085); this also replaces fixed interval/cold cap.
G_COMMIT=759a6ebb8e31723519ad5daf438e26e24b32501a
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="120=on 122=on 123=off 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_PACE_TPOT=0.085 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
unset G_MEASURE_SECONDS
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="30"
G_DATA_ROOT="$AX/data/s1-dev-longchain"
G_DATA_SET=s1-dev-longchain
G_COHORT="$G_DATA_ROOT/cohort.json"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0

# Dev self-test on one engine: run the levels in LADDER_UP (e.g. "22" or "22 26"); each level is scored by
# verify_kit/level_verdict.py (complete data + harness scorer + task.md rules); stop at the first failure. Wrapper sets:
#   G_COMMIT (engine commit id, exported by scripts/engine/export.sh), G_ARGS, G_ENV, G_EXPECT, optional LADDER.
#   G_EXPECT lists the effective mechanism states the job relies on, e.g. "120=on 122=off 180=on"; the engine's
#   "[ax] mechanisms:" line must match every entry ("off" also matches "off:<reason>") or the job measures nothing.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_COMMIT" || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
ensure_engine "$G_COMMIT" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1

printf "PRELOAD_READY: full replay still waits for verified dataset\n"
