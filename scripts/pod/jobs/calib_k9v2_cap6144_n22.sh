#!/usr/bin/env bash
# Calibration: the configuration of official attempt 46364 (online 0925a, 759a6eb, host64, 122 off, cold cap 6144;
# the same as L081) on the chain-prefix set K=9 (first 9 requests of each of the 311 long chains: 1865 requests,
# 1.050e8 prompt tokens, close to an official level's ~1788 requests / ~1.05e8 tokens), N22, full level. Target:
# 46364 at N22 (chain p95 41.97 s, turn 10.66, fast 1.86, overall 3.15, TPOT mean .0226 / p95 .0438, slo .958).
# The K=9 root is derived on the pod by scripts/longchain/make_prefix_set.py and checked by hash.
G_COMMIT=759a6ebb8e31723519ad5daf438e26e24b32501a
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="120=on 122=off 123=off 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=6144 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
unset G_MEASURE_SECONDS
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="22"
G_DATA_ROOT="$AX/data/s1-dev-longchain-k9v2"
G_DATA_SET=s1-dev-longchain
G_COHORT="$G_DATA_ROOT/cohort.json"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0
if [ ! -d "$G_DATA_ROOT" ]; then
  python3 "$AX/bin/scripts/longchain/make_prefix_set.py" --src "$AX/data/s1-dev-longchain" --dst "$G_DATA_ROOT" \
    --k 9 --harness-dir "$AX/s1/s1-dev/harness" || exit 2
fi
python3 - "$G_DATA_ROOT" <<'DATA' || exit 2
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1])
c=json.loads((root/'cohort.json').read_text())
assert c['set']=='s1-dev-longchain' and c['prefix_k']==9 and c['n_chains']==311 and c['n_requests']==1865
assert c['cohort_sha256']=='3400cc9513f2d1bb'
ids=[rid for chain in c['chains'] for rid in chain['req_ids']]
assert len(ids)==len(set(ids))==1865
req=(root/'requests.jsonl').read_bytes()
assert hashlib.sha256(req).hexdigest()=='5f5095b663473e5a74139fb4fd7c601a6e307639d6ac46088cb1364dbc2a82ea'
assert sum(bool(s.strip()) for s in req.decode().splitlines())==1865
import os
shards=[os.path.join(dp,f) for dp,_,files in os.walk(root/'bodies') for f in files if f.endswith('.jsonl.gz')]
assert shards and all(os.path.isfile(p) for p in shards), 'body shards not visible to harness os.walk'
print('DATA_READY prefix9 chains=311 requests=1865 N22 full_replay=true warmup=rep16-v1',flush=True)
DATA
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
