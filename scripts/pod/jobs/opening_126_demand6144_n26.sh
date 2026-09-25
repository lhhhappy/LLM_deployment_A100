#!/usr/bin/env bash
# L0xx/126: against L078 (fixed cold cap 6144 on 759a6eb) the cold cap is 126's demand-sized cap: floor 4096,
# maximum 6144, room kept for the waiting complete short hits. Engine 37e90023 + 126; 123 off (its off path
# equals 759a6eb). N26, 600s admission and drain; paired with 074 and 078 on the same request IDs.
G_COMMIT=8f62c5f2b2cf921b2ceb234eacd632b048442fd6
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="120=on 122=off 123=off 126=on 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_SCHED_COLD_CAP_MAX=6144 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_MEASURE_SECONDS=600
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="26"
G_DATA_ROOT="$AX/data/s1-dev-longchain"
G_DATA_SET=s1-dev-longchain
G_COHORT="$G_DATA_ROOT/cohort.json"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE SGLANG_AX_CHUNK_ALIGNMENT_TRACE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0
python3 - "$G_DATA_ROOT" <<'DATA' || exit 2
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
c=json.loads((root/'cohort.json').read_text())
assert c['set']=='s1-dev-longchain' and c['n_chains']==311 and c['n_requests']==5601
ids=[rid for chain in c['chains'] for rid in chain['req_ids']]
assert len(c['chains'])==311 and len(ids)==len(set(ids))==5601
assert sum(bool(s.strip()) for s in (root/'requests.jsonl').open())==5601
print('DATA_READY full chains=311 requests=5601 N26 admission_seconds=600 drain_all_admitted=true warmup=rep16-v1',flush=True)
DATA
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
