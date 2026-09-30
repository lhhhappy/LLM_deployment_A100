#!/usr/bin/env bash
# N42 chain diagnostic; frozen 47266 baseline, one factor per candidate.
G_COMMIT=bf6b66faf3ffbe10e593e858d3bbedd1c48cefb2
G_EXPECT="132=on 131_sync=rank0 131_chunk=auto 117=on 118=on 119=off 120=on 122=off 123=off 124=on 125=on 126=off 140=off 180=on spec=- dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1 128p=on 131=on 128=off dcp_local=off dcp_local_max=0 dcp_local_large=0"
G_ARGS='--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --max-running-requests 48 --cuda-graph-max-bs 48 --max-mamba-cache-size 400 --enable-attn-tp-input-scattered --prefill-decode-interval 2 --chunked-prefill-size 24576 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-prefill-tokens 24576'
G_ENV='IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD=4096 SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_AX_BACKLOG_COLD_CAP=24576 SGLANG_AX_BACKLOG_GATE=0.10 SGLANG_AX_BACKLOG_HIGH_S=15 SGLANG_AX_BACKLOG_INTERVAL=0 SGLANG_AX_BACKLOG_LOW_S=5 SGLANG_AX_BACKLOG_MAX_SLOW=80 SGLANG_AX_BACKLOG_RELIEF=1 SGLANG_AX_CHAIN_RISK_CHUNK=0 SGLANG_AX_CHAIN_RISK_INTERVAL=1 SGLANG_AX_DCP_LOCAL_EXTEND=0 SGLANG_AX_DEADLINE_CHAIN_FIRST=1 SGLANG_AX_DEADLINE_FAMILY=0 SGLANG_AX_DEADLINE_FREEZE_CLASS=1 SGLANG_AX_DEADLINE_LOAD=1.05 SGLANG_AX_DEADLINE_MAX_WAIT_S=600 SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S=120 SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_DEADLINE_WARM_S=15 SGLANG_AX_DSA_SPARSE_TRITON=0 SGLANG_AX_DSA_SPARSE_TRITON_PREFILL=1 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_KDA_PREFILL_CPU_LENGTH=1 SGLANG_AX_KDA_PREFILL_PREPARE=1 SGLANG_AX_KDA_PREFILL_STATE_BV16=1 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_PREFIX_PRODUCER=1 SGLANG_AX_PREFIX_TRACE_ROUNDS=0 SGLANG_AX_PREFIX_TRACE_S=0 SGLANG_AX_SCHED_COLD_CAP=24576 SGLANG_AX_SCHED_COLD_CAP_MAX=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=2048 SGLANG_AX_SM80_FP8_MOE_HUMMING=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_MOE_DOWN_TUNE=1 SGLANG_AX_SM80_MOE_REDUCE=0 SGLANG_AX_SM80_MOE_UP_TUNE=1 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_OPT_USE_TOPK_V2=0'
G_MEASURE_SECONDS=3600
G_WARMUP_PROFILE=rep16-v1
LADDER_UP=42
SMOKE_GATE=1
unset LADDER_DOWN G_TIMED_PROMOTE_FROM G_TIMED_PROMOTE_REFERENCE G_CHAIN_START_INTERVAL_S
unset SGLANG_AX_PARK_MAX_ROUNDS SGLANG_AX_PARK_MAX_S SGLANG_AX_DCP_LOCAL_EXTEND_LARGE_MAX SGLANG_AX_DCP_LOCAL_EXTEND_MAX_TOKENS SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
G_DATA_ROOT="$AX/codex/data/s1-dev-longchain-v5g-tail-rot150"
G_DATA_SET=s1-dev-longchain-v5g-tail-rot150
G_COHORT="$G_DATA_ROOT/cohort.json"
python3 - "$G_DATA_ROOT" "${G_MEASURE_SECONDS:-full}" "$LADDER_UP" <<'DATA' || exit 2
import hashlib, json, os, sys
from pathlib import Path
root, seconds, level = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = json.loads((root/'cohort.json').read_text())
assert c['set'] == 's1-dev-longchain-v5g-tail-rot150' and c['n_chains'] == 311 and c['n_requests'] == 5601
assert c['cohort_sha256'] == 'b78593bdea138f58'
ids = [rid for chain in c['chains'] for rid in chain['req_ids']]
assert len(ids) == len(set(ids)) == 5601
assert hashlib.sha256((root/'requests.jsonl').read_bytes()).hexdigest() == '6170fd8204db4de8be4f99861d8cfa20ba36eab8b3317b8a3cd1343c5aef8ebb'
shards = [os.path.join(dp, f) for dp, _, files in os.walk(root/'bodies') for f in files if f.endswith('.jsonl.gz')]
assert shards and all(os.path.isfile(p) for p in shards), 'body shards not visible to harness os.walk'
print('DATA_READY v5g-tail-rot150 chains=311 requests=%d N%s admission_seconds=%s warmup=rep16-v1' % (len(ids), level, seconds), flush=True)
DATA

source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
