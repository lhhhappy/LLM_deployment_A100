#!/usr/bin/env bash
# 2026-09-26 (fable): N34 arm of the DCP pair on engine b261cbd9 (same as the N30 pair): running/graph 48 as Codex's w1_n34 job,
# everything else as dcp_stack2_w1_n30_60m. Question: does DCP push the N34 wall back (KV pool use, queue, four gates).
# 2026-09-26 (fable): N30 variant of Codex's paired DCP job (W1), for the level we submit at: running/graph 32 as 112/130, LADDER_UP=30,
# engine b261cbd95cb47c232e15c9ff826652cf1e6fdde5 = e4d7ca68 (84dcca0e + 115/180 DCP with NextN) + b261cbd9 (180 guard: unresolved NextN topk). Everything else as the
# N34 pair (candidate A scheduling, warm 5 s, MTP 3/1/4, host64, 418 KDA slots, SMOKE_GATE=1). W2 vs W1 = the DCP effect at N30.
# DCP candidate pair: same e4d7ca68 engine, candidate A scheduling (117/124/125),
# warm budget 5 s and starvation 120 s, MTP 3/1/4, host64, FP32 KDA,
# running/graph48, exactly 418 KDA slots. Only --dcp-size differs.
# Compact-topk is requested in both arms; W1 retains the baseline path.
# Compared as the complete DCP implementation, not component attribution.
# 3600 s admission + drain is a diagnostic, never a full N34 score.
# Prepared only: TP8 capacity/correctness, queue coordination and raw review
# precede promotion. Old 130/136/137 results do not replace the paired W1 arm.
# 130e5's 10 s warm starvation policy was rejected; it is disabled explicitly.
G_COMMIT=b261cbd95cb47c232e15c9ff826652cf1e6fdde5
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="117=on 118=off 119=off 120=on 122=off 123=off 124=on 125=on 126=off 128=off 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
G_ARGS="--dcp-size 1 --kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 48 --cuda-graph-max-bs 48 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 418"
G_ENV="SGLANG_AX_DCP_COMPACT_TOPK=1 SGLANG_AX_DSA_SPARSE_TRITON=0 SGLANG_AX_DEADLINE_WARM_S=5 SGLANG_AX_DEADLINE_FAST_S=3 SGLANG_AX_DEADLINE_MAX_WAIT_S=120 SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S=120 SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=6144 SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_BACKLOG_RELIEF=1 SGLANG_AX_BACKLOG_COLD_CAP=8192 SGLANG_AX_BACKLOG_INTERVAL=0 SGLANG_AX_BACKLOG_HIGH_S=15 SGLANG_AX_BACKLOG_LOW_S=5 SGLANG_AX_BACKLOG_MAX_SLOW=80 SGLANG_AX_BACKLOG_GATE=0.10 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
G_MEASURE_SECONDS=3600
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="34"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
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
