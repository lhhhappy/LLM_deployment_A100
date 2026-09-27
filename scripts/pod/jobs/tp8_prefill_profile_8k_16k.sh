#!/usr/bin/env bash
# One cold frozen 49k-token request, first at 8k then at 16k prefill chunks.
# Profiles are local cost evidence only. Runs after 081 and before pending 082.
set -euo pipefail
G_COMMIT=759a6ebb8e31723519ad5daf438e26e24b32501a
G_EXPECT="120=on 122=off 123=off 140=off 180=on spec=EAGLE dcp=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
RID='biomaster:canon:member-v5-subagent-cfae8adc9bc2962e50c7a6e0:llm:1'
DATA_ROOT="$AX/data/s1-dev-longchain"
JOB_DIR="$RUN_DIR"
source "$AX/bin/scripts/pod/lib.sh"
prepare_src "$G_COMMIT"
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export $G_ENV
unset SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
export S1_ENGINE_URL="http://127.0.0.1:${PORT:-30000}"

run_case() {
  local tag=$1 chunk=$2 sampler
  RUN_DIR="$JOB_DIR/$tag"
  mkdir -p "$RUN_DIR/warmup"
  ensure_engine "$G_COMMIT" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS --chunked-prefill-size "$chunk"
  local mechanisms
  mechanisms=$(grep -h '\[ax\] mechanisms:' "$RUN_DIR/server.log" | tail -1)
  [[ -n "$mechanisms" ]] || { echo 'MECHANISMS INVALID: no mechanism line'; return 2; }
  for want in $G_EXPECT; do
    local key=${want%%=*} value=${want#*=} actual
    actual=$(grep -o " $key=[^ ]*" <<<" $mechanisms" | head -1 | cut -d= -f2)
    case "$actual" in "$value"|"$value":*) ;; *) echo "MECHANISMS MISMATCH $key expected=$value got=${actual:-missing}"; return 2 ;; esac
  done
  echo "PROFILE_CONFIG tag=$tag chunk=$chunk commit=$G_COMMIT args=[$G_ARGS] env=[$G_ENV] mechanisms=[$mechanisms]"
  ( cd "$AX/s1/s1-dev" && python3 -B "$AX/verify_kit/short_warmup_loadgen.py" \
      --loadgen "$AX/s1/s1-dev/harness/s1_loadgen.py" -- \
      --root "$DATA_ROOT" --set s1-dev-longchain --cohort-file "$DATA_ROOT/cohort.json" \
      --tok-dir /mnt/models --out-dir "$RUN_DIR/warmup" --n 8 --warmup \
  ) > "$RUN_DIR/warmup.log" 2>&1
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 100 \
      > "$RUN_DIR/gpu_memory_mib.csv" 2> "$RUN_DIR/gpu_memory.err" &
  sampler=$!
  if ! python3 -B "$AX/verify_kit/tp8_prefill_profile.py" \
      --root "$DATA_ROOT" --tok-dir /mnt/models --rid "$RID" \
      --base-url "$S1_ENGINE_URL" --out-dir "$RUN_DIR" --tag "$tag"; then
    kill "$sampler" 2>/dev/null || true
    return 2
  fi
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  python3 -B "$AX/verify_kit/prof_ledger.py" \
      "$RUN_DIR"/traces/*TP-0*.trace.json.gz "$RUN_DIR"/traces/*TP-1*.trace.json.gz \
      --json "$RUN_DIR/ledger-rank0-rank1.json" > "$RUN_DIR/ledger-rank0-rank1.txt"
  python3 - "$RUN_DIR/ledger-rank0-rank1.json" "$chunk" <<'PY'
import json,sys
reports=json.load(open(sys.argv[1]))
expected=int(sys.argv[2])
assert len(reports)==2, f'expected two TP rank ledgers, got {len(reports)}'
for report in reports:
    full=report.get('classes',{}).get('extend>=8k',{})
    n=full.get('n',0)
    mean=full.get('mean_new_toks',0)
    assert n >= 2 and 0.9*expected <= mean <= expected, (
        f'profile did not measure {expected}-token blocks: '
        f"trace={report.get('trace')} n={n} mean_new_toks={mean}")
PY
  python3 - "$RUN_DIR/gpu_memory_mib.csv" "$RUN_DIR/gpu_memory_peak.json" <<'PY'
import json,sys
samples={}
for line in open(sys.argv[1]):
    parts=[p.strip() for p in line.split(',')]
    if len(parts)!=2:continue
    try:index,used=map(int,parts)
    except ValueError:continue
    samples.setdefault(index,[]).append(used)
assert set(samples)==set(range(8)),f'incomplete GPU memory samples: {list(samples)}'
json.dump({str(i):{'baseline_mib':v[0],'peak_mib':max(v),'delta_mib':max(v)-v[0],
                   'samples':len(v)} for i,v in sorted(samples.items())},open(sys.argv[2],'w'),indent=2)
PY
  echo "PROFILE_DONE tag=$tag chunk=$chunk"
}

run_case tp8_8k 8192
run_case tp8_16k 16384
echo 'PROFILE_ALL_DONE rank0/rank1 ledgers and peak memory recorded; no replay SLO verdict'
