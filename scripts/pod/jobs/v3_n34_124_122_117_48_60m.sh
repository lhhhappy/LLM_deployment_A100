#!/usr/bin/env bash
# N34 combination with 122 instead of 125x: the single change versus v3_n34_124_125x_117_48_60m is that the TPOT
# paced prefill budget (122, tau .085 as official 46174) replaces the opening-only backlog relief (125). 122 yields
# the round to prefill whenever a prefill waits and inserts decode rounds only when a decoder runs out of TPOT slack,
# all the time; 125 does so only while the cold backlog is large. Question: does spending TPOT slack in the steady
# state (Fable 2026-09-26: more sessions decoding at once, fewer waiting) beat opening-only relief at N34?
G_COMMIT=7c6cb6349f088de3af0e4d32440bdfbac6ee7941
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="117=on 118=off 119=off 120=on 122=on 123=off 124=on 125=off 126=off 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 48 --cuda-graph-max-bs 48 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=6144 SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_PACE_TPOT=0.085 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
G_MEASURE_SECONDS=3600
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="34"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0
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
