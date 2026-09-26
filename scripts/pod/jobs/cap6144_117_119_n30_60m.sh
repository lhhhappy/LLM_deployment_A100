#!/usr/bin/env bash
# 119 on top of 117: L084 (081 config + 117 Humming MoE) with --enable-attn-tp-input-scattered and
# SGLANG_AX_SCATTER_MIN_TOKENS=1025 (scatter only extends above the custom all-reduce size). Engine claude/stack
# (37e90023 + 117 + 119). Capability smoke gate on. N30, 60-minute admission then drain; paired with 084 and 081.
G_COMMIT=928b9e186e19c3fdecb07971e31a26578b6d2ec1
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="117=on 119=on 120=on 122=off 123=off 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87 --enable-attn-tp-input-scattered"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=6144 SGLANG_AX_SCATTER_MIN_TOKENS=1025 SGLANG_AX_SM80_FP8_MOE_HUMMING=1 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_MEASURE_SECONDS=3600
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="30"
G_DATA_ROOT="$AX/data/s1-dev-longchain"
G_DATA_SET=s1-dev-longchain
G_COHORT="$G_DATA_ROOT/cohort.json"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=1
python3 - "$G_DATA_ROOT" <<'DATA' || exit 2
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
c=json.loads((root/'cohort.json').read_text())
assert c['set']=='s1-dev-longchain' and c['n_chains']==311 and c['n_requests']==5601
ids=[rid for chain in c['chains'] for rid in chain['req_ids']]
assert len(c['chains'])==311 and len(ids)==len(set(ids))==5601
assert sum(bool(s.strip()) for s in (root/'requests.jsonl').open())==5601
print('DATA_READY full chains=311 requests=5601 N30 admission_seconds=3600 drain_all_admitted=true warmup=rep16-v1',flush=True)
DATA
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
