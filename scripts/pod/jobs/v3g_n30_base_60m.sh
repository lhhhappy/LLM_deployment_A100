#!/usr/bin/env bash
# Same as v3_n30_base_60m.sh (111) in every setting; the single change is the data: v3g (gap-realistic v3).
# Question: with realistic idle time between requests (fewer sessions awake at once, as online), does the N30
# steady state still hit the KV wall, and do the opening mechanisms keep their relative gains?
G_COMMIT=7c6cb6349f088de3af0e4d32440bdfbac6ee7941
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="117=off 118=off 119=off 120=on 122=off 123=off 124=off 125=off 126=off 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=6144 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_MEASURE_SECONDS=3600
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="30"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0
# v3g data: v3 with the imputed replay gaps rescaled per chain to the organizer's real chain durations
# (scripts/longchain/regap.py; per-request gap mean 5.5 -> 24.4 s, same prompts/budgets/order, cohort 13b346fde05bd592),
# checked by hash before any engine starts. Bodies are the parent's (bodies/ links to s1-dev-longchain-v3/bodies).
G_DATA_ROOT="$AX/data/s1-dev-longchain-v3g"
G_DATA_SET=s1-dev-longchain-v3g
G_COHORT="$G_DATA_ROOT/cohort.json"
python3 - "$G_DATA_ROOT" "${G_MEASURE_SECONDS:-full}" "$LADDER_UP" <<'DATA' || exit 2
import hashlib, json, os, sys
from pathlib import Path
root, seconds, level = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = json.loads((root/'cohort.json').read_text())
assert c['set'] == 's1-dev-longchain-v3g' and c['n_chains'] == 311 and c['n_requests'] == 5601
assert c['cohort_sha256'] == '13b346fde05bd592'
ids = [rid for chain in c['chains'] for rid in chain['req_ids']]
assert len(ids) == len(set(ids)) == 5601
assert hashlib.sha256((root/'requests.jsonl').read_bytes()).hexdigest() == '9341d23878b796a42802b18b18ecb0e3cbb88f6155ca5b5c910bfd363c65ac37'
shards = [os.path.join(dp, f) for dp, _, files in os.walk(root/'bodies') for f in files if f.endswith('.jsonl.gz')]
assert shards and all(os.path.isfile(p) for p in shards), 'body shards not visible to harness os.walk'
print('DATA_READY v3g chains=311 requests=%d N%s admission_seconds=%s warmup=rep16-v1' % (len(ids), level, seconds), flush=True)
DATA
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
