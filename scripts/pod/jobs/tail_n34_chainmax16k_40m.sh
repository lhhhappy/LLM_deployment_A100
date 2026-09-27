#!/usr/bin/env bash
# 2026-09-27 (fable): CHAIN-MAX arm (user: spend fast/TPOT margin, protect chain starts). S6 base (S1 settings, no DCP, running 32,
# no MTP, 128p + freeze + 131 interval 1) on engine c1fa4877 plus, all at once: 132 chain-first order
# (SGLANG_AX_DEADLINE_CHAIN_FIRST=1), steady-state cold chunk cap 6144 -> 16384 (SCHED_COLD_CAP, BACKLOG_COLD_CAP,
# --chunked-prefill-size 16384: a 250k head pays 15 chunk fixed costs instead of 41), short-hit reserve 8192 -> 2048, and the
# 124 cost model load factor 1.27 -> 1.05 (no-MTP giants run 20-23.5 s, the old constants said 27 s and demoted them as hopeless).
# v5g, N26, 2400 s admission then drain. Same-ID references: S6 (130ezmd) and S1 + no MTP (130ezmc). Judge: chain misses and
# steady chain TTFT bins first; report fast/overall/turn and TPOT>0.10 as the price.
G_COMMIT=c1fa4877d9337d78c122a930d53bd5946c1d7a81
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="117=on 118=off 119=off 120=on 122=off 123=off 124=on 125=on 126=off 140=off 180=on spec=- dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1 128p=on 131=on 128=off dcp_local=off dcp_local_max=512 dcp_local_large=0"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --max-running-requests 48 --cuda-graph-max-bs 48 --prefill-decode-interval 2 --chunked-prefill-size 16384 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_DEADLINE_CHAIN_FIRST=1 SGLANG_AX_DEADLINE_LOAD=1.05 SGLANG_AX_CHAIN_RISK_INTERVAL=1 SGLANG_AX_PREFIX_PRODUCER=1 SGLANG_AX_DEADLINE_FAMILY=0 SGLANG_AX_PREFIX_TRACE_S=120 SGLANG_AX_PREFIX_TRACE_ROUNDS=2048 SGLANG_AX_DEADLINE_FREEZE_CLASS=1 SGLANG_AX_DSA_SPARSE_TRITON=0 SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=2048 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=16384 SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_DEADLINE_WARM_S=15 SGLANG_AX_BACKLOG_RELIEF=1 SGLANG_AX_BACKLOG_COLD_CAP=16384 SGLANG_AX_BACKLOG_INTERVAL=0 SGLANG_AX_BACKLOG_HIGH_S=15 SGLANG_AX_BACKLOG_LOW_S=5 SGLANG_AX_BACKLOG_MAX_SLOW=80 SGLANG_AX_BACKLOG_GATE=0.10 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
G_MEASURE_SECONDS=2400
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="34"
unset SGLANG_AX_DCP_LOCAL_EXTEND_LARGE_MAX SGLANG_AX_DCP_LOCAL_EXTEND_MAX_TOKENS LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=1
# v5g-tail data (5601 requests, cohort ff1dccae1087a798, requests sha 6170fd82), checked by hash before any engine starts.
G_DATA_ROOT="$AX/codex/longchain-repair-0927/data/s1-dev-longchain-v5g-tail-review-0927"
G_DATA_SET=s1-dev-longchain-v5g-tail-review-0927
G_COHORT="$G_DATA_ROOT/cohort.json"
python3 - "$G_DATA_ROOT" "${G_MEASURE_SECONDS:-full}" "$LADDER_UP" <<'DATA' || exit 2
import hashlib, json, os, sys
from pathlib import Path
root, seconds, level = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = json.loads((root/'cohort.json').read_text())
assert c['set'] == 's1-dev-longchain-v5g-tail-review-0927' and c['n_chains'] == 311 and c['n_requests'] == 5601
assert c['cohort_sha256'] == 'ff1dccae1087a798'
ids = [rid for chain in c['chains'] for rid in chain['req_ids']]
assert len(ids) == len(set(ids)) == 5601
assert hashlib.sha256((root/'requests.jsonl').read_bytes()).hexdigest() == '6170fd8204db4de8be4f99861d8cfa20ba36eab8b3317b8a3cd1343c5aef8ebb'
shards = [os.path.join(dp, f) for dp, _, files in os.walk(root/'bodies') for f in files if f.endswith('.jsonl.gz')]
assert shards and all(os.path.isfile(p) for p in shards), 'body shards not visible to harness os.walk'
print('DATA_READY v5g-tail chains=311 requests=%d N%s admission_seconds=%s warmup=rep16-v1' % (len(ids), level, seconds), flush=True)
DATA
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
