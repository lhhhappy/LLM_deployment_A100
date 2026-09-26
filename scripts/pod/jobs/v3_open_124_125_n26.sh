#!/usr/bin/env bash
# Base + 124 + 125: backlog relief spends TPOT while cold backlog > 30 s of prefill (cold chunks 8192, one decode
# round per prefill instead of two; stops for the level after 40 requests above 0.09 s/token). Single change vs 124.
# N26 opening probe on v3 (the organizer's 311 public chain heads open every run): 600-second admission, drain all
# admitted; stack2 7c6cb634, 46364's configuration otherwise (cold cap 6144). Judged relative to the other v3 opening
# probes on the same request IDs: chain misses (first 5 minutes and window), turn misses, requests above 0.10 s/token.
G_COMMIT=7c6cb6349f088de3af0e4d32440bdfbac6ee7941
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="117=off 118=off 119=off 120=on 122=off 123=off 124=on 125=on 126=off 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=6144 SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_BACKLOG_RELIEF=1 SGLANG_AX_BACKLOG_COLD_CAP=8192 SGLANG_AX_BACKLOG_INTERVAL=1 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_MEASURE_SECONDS=600
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="26"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0
source "$AX/bin/scripts/pod/jobs/v3_data.sh"
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
