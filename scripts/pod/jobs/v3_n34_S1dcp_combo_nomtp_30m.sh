#!/usr/bin/env bash
# 2026-09-27 (fable): S5 candidate WITHOUT MTP. Same as v3_n34_S1dcp_combo_30m (engine 791453ca: 128p + local-extend ON, S1 settings,
# dcp 2, running/graph 48, KDA pool 418, N34, 1800 s admission then drain, smoke gate) with the five speculative flags removed
# (spec=-). 130eze2 (46757 engine without MTP) showed KV pool 2.22M -> 3.13M, steady-state overall/fast misses halved,
# TPOT>0.10 +19 at the same minute; the user judged that cost acceptable and wants no-MTP tried online. Reference: 130ezf
# (same engine with MTP) and 130ez1. Judge on same-ID counts; this is the S5b candidate configuration.
G_COMMIT=791453ca86545a04f9a2d6c943d6f1f7feb4b6a7
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="117=on 118=off 119=off 120=on 122=off 123=off 124=on 125=on 126=off 140=off 180=on spec=- dcp=2 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1 128p=on 128=off dcp_local=on dcp_local_max=512 dcp_local_large=0"
G_ARGS="--dcp-size 2 --kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --max-running-requests 48 --cuda-graph-max-bs 48 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 418"
G_ENV="SGLANG_AX_DCP_LOCAL_EXTEND=1 SGLANG_AX_PREFIX_PRODUCER=1 SGLANG_AX_DEADLINE_FAMILY=0 SGLANG_AX_PREFIX_TRACE_S=120 SGLANG_AX_PREFIX_TRACE_ROUNDS=2048 SGLANG_AX_DEADLINE_FREEZE_CLASS=1 SGLANG_AX_DCP_COMPACT_TOPK=1 SGLANG_AX_DSA_SPARSE_TRITON=0 SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=6144 SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_DEADLINE_WARM_S=15 SGLANG_AX_BACKLOG_RELIEF=1 SGLANG_AX_BACKLOG_COLD_CAP=8192 SGLANG_AX_BACKLOG_INTERVAL=0 SGLANG_AX_BACKLOG_HIGH_S=15 SGLANG_AX_BACKLOG_LOW_S=5 SGLANG_AX_BACKLOG_MAX_SLOW=80 SGLANG_AX_BACKLOG_GATE=0.10 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
G_MEASURE_SECONDS=1800
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="34"
unset SGLANG_AX_DCP_LOCAL_EXTEND_LARGE_MAX SGLANG_AX_DCP_LOCAL_EXTEND_MAX_TOKENS LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=1
# v3 data (5601 requests, cohort 13b346fde05bd592), checked by hash before any engine starts.
G_DATA_ROOT="$AX/data/s1-dev-longchain-v3"
G_DATA_SET=s1-dev-longchain-v3
G_COHORT="$G_DATA_ROOT/cohort.json"
python3 - "$G_DATA_ROOT" "${G_MEASURE_SECONDS:-full}" "$LADDER_UP" <<'DATA' || exit 2
import hashlib, json, os, sys
from pathlib import Path
root, seconds, level = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = json.loads((root/'cohort.json').read_text())
assert c['set'] == 's1-dev-longchain-v3' and c['n_chains'] == 311 and c['n_requests'] == 5601
assert c['cohort_sha256'] == '13b346fde05bd592'
ids = [rid for chain in c['chains'] for rid in chain['req_ids']]
assert len(ids) == len(set(ids)) == 5601
assert hashlib.sha256((root/'requests.jsonl').read_bytes()).hexdigest() == '31f4d7521b623dedd43e9d44efbcc3af0890cea138d0ee9f7c9ee7b20a777305'
shards = [os.path.join(dp, f) for dp, _, files in os.walk(root/'bodies') for f in files if f.endswith('.jsonl.gz')]
assert shards and all(os.path.isfile(p) for p in shards), 'body shards not visible to harness os.walk'
print('DATA_READY v3 chains=311 requests=%d N%s admission_seconds=%s warmup=rep16-v1' % (len(ids), level, seconds), flush=True)
DATA
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
