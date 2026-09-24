#!/usr/bin/env bash
# Same source commit: formal A parameters, then memory 0.87, then memory 0.87 + HiCache.
# N30, first-dispatch + 70 minutes closes admission, then drains every admitted request.
# This is a duration-limited diagnostic on the full dataset, not full-cohort SLO validation.
G_COMMIT=c92acd57a61eb6f9eed3222cc048877eef7963d9
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="120=on 122=off 123=off 140=off 180=off spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_MEASURE_SECONDS=4200
LADDER_UP="30"
G_DATA_ROOT="$AX/data/s1-dev-longchain"
G_DATA_SET=s1-dev-longchain
G_COHORT="$G_DATA_ROOT/cohort.json"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
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
print('DATA_READY full chains=311 requests=5601 N30 admission_seconds=4200 drain=true',flush=True)
DATA
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
