#!/usr/bin/env bash
# 2026-09-27 (fable): exploration on the submitted DCP2 engine at N34: K3 size-tiered warm budget (one-round warm 15 s, multi-round 5 s),
# engine f2b6425e = e464d8ab + 9be15822 cherry-picked (default-neutral mechanism; CPU suites OK). Single change versus v3_n34_S1dcp_w2_60m.
# 2026-09-26 (fable): S1 (46676) settings + DCP: engine e464d8abe63f60f93fcaf3d0fc5a86c81d70747e = f546934e + the three DCP patches (115 NextN pools/phases, 180 host
# pools, 180 guard) carried byte-identical from Codex's branch; S1 env (warm 15 s, MAX_SLOW 80) + DCP_COMPACT_TOPK=1, --dcp-size 2,
# KDA pool pinned at 418 slots; N34 with running/graph 48. W2 vs W1 = DCP on the submission engine; the next submission candidate.
# 2026-09-26 evening: A' 60-minute N30 window on engine f546934eff67f7b1d9f9cc5516c5a89377aaeb71 (84dcca0e + review fixes 110/117/124/180, see
# v3_open_A_warm15_fixes_n30.sh); single engine change versus 130b (4f9d1f0b). Confirmation run for the next submission.
# Candidate A (112: 124 + aggressive 125 + 117, v3, N30, 3600-second admission then drain) with one change:
# SGLANG_AX_DEADLINE_WARM_S=15 SGLANG_AX_DEADLINE_WARM_MULTI_S=5. 124 cannot see the harness buckets and gives a warm request with more than 4096
# uncached tokens a 5 s budget; turn starts (15 s in the harness) and shared-prefix chain starts (30 s) of that
# shape were judged hopeless after 5 s and waited until the 120 s starvation bound (112 vs 111: turn 12->24,
# 6 new turn misses waited 120.8-124.2 s; notes/program-n30-v3.md). Judged against 112 on the same request IDs:
# turn misses and waits near 120 s first, then chain, overall, fast.
G_COMMIT=f2b6425eb74b4a417b19156bb3b39ddc2f283423
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="117=on 118=off 119=off 120=on 122=off 123=off 124=on 125=on 126=off 140=off 180=on spec=EAGLE dcp=2 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
G_ARGS="--dcp-size 2 --kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 48 --cuda-graph-max-bs 48 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 418"
G_ENV="SGLANG_AX_DCP_COMPACT_TOPK=1 SGLANG_AX_DSA_SPARSE_TRITON=0 SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=6144 SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_DEADLINE_WARM_S=15 SGLANG_AX_DEADLINE_WARM_MULTI_S=5 SGLANG_AX_BACKLOG_RELIEF=1 SGLANG_AX_BACKLOG_COLD_CAP=8192 SGLANG_AX_BACKLOG_INTERVAL=0 SGLANG_AX_BACKLOG_HIGH_S=15 SGLANG_AX_BACKLOG_LOW_S=5 SGLANG_AX_BACKLOG_MAX_SLOW=80 SGLANG_AX_BACKLOG_GATE=0.10 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"
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
