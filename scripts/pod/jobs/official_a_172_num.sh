# 050: real-weight TP8 numerical SCREEN for 172, separate from scheduling and SLO measurement.
# Baseline repetitions calibrate observed stability; candidate repetitions exercise reuse.
# Tests output/logprob fingerprints, real prefix attempts, MTP and decode graph. It does NOT expose
# every layer/state tensor or prove capability gates. numcheck_cmp's 0.5 logprob cutoff is a coarse
# rejection threshold, not a scientific equivalence tolerance; raw diffs require independent review.
G_NAME=off_a_kernelref
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096"

source "$AX/bin/scripts/pod/lib.sh"
ROOT_RUN=$RUN_DIR
BASE_PATCHES=$G_PATCHES
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export NUMCHECK_PREFIX_LEN=65536
export $G_ENV
unset SGLANG_AX_PACE_TPOT SGLANG_AX_MOE_FUSE_SWIGLU SGLANG_AX_KDA_FUSE_PROJ
LENS=33,37,63,65,256,1024,4096,8192,16384
printf '%s\n' "SCREEN_ONLY 172 real TP8; prefix=$NUMCHECK_PREFIX_LEN lengths=$LENS"
echo "2752e70a32a732294b6025bd13ad6226c269c885c25fb96800885da5b669d606  $AX/patches/172-moe-clamped-swiglu.patch" | sha256sum -c - || exit 2
sha256sum "$AX/verify_kit/numcheck.py" "$AX/verify_kit/numcheck_cmp.py"
for arm in ref1 ref2 cand1 cand2; do
  export RUN_DIR=$ROOT_RUN/$arm
  mkdir -p "$RUN_DIR"
  case "$arm" in
    ref*) G_NAME=off_a_kernelref; G_PATCHES=$BASE_PATCHES; unset SGLANG_AX_MOE_FUSE_SWIGLU ;;
    cand*) G_NAME=off_a_kernel172; G_PATCHES="$BASE_PATCHES 172-moe-clamped-swiglu.patch"; export SGLANG_AX_MOE_FUSE_SWIGLU=1 ;;
  esac
  prepare_src "$G_NAME" $G_PATCHES || exit 2
  ( cd "$AX/patches" && sha256sum $G_PATCHES ) | sed 's/^/PATCH_SHA /'
  ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 3
  curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$RUN_DIR/server_info.json" || exit 3
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$RUN_DIR/gpu_memory.csv" || exit 3
  grep -E 'max_total_num_tokens|KV Cache is allocated' "$RUN_DIR/server.log" > "$RUN_DIR/startup_evidence.log" || exit 3
  python3 "$AX/verify_kit/numcheck.py" "$RUN_DIR/num" "$LENS" > "$RUN_DIR/num.log" 2>&1 || { tail -15 "$RUN_DIR/num.log"; exit 5; }
  tail -1 "$RUN_DIR/num.log"
  if [ "$arm" = ref2 ]; then
    python3 "$AX/verify_kit/numcheck_cmp.py" "$ROOT_RUN/ref1/num/numcheck.json" "$RUN_DIR/num/numcheck.json" > "$ROOT_RUN/ref-repeat.txt" || exit 5
    grep -q 'NUMCMP wrong=0/' "$ROOT_RUN/ref-repeat.txt" || { cat "$ROOT_RUN/ref-repeat.txt"; echo "REFERENCE_UNSTABLE"; exit 6; }
  fi
done
for arm in cand1 cand2; do
  python3 "$AX/verify_kit/numcheck_cmp.py" "$ROOT_RUN/ref1/num/numcheck.json" "$ROOT_RUN/$arm/num/numcheck.json" > "$ROOT_RUN/$arm-vs-ref.txt" || exit 5
  cat "$ROOT_RUN/$arm-vs-ref.txt"
  grep -q 'NUMCMP wrong=0/' "$ROOT_RUN/$arm-vs-ref.txt" || { echo "CANDIDATE_REQUIRES_REVIEW"; exit 7; }
done
# Separate diagnostic after all numerical comparisons: prove the fused kernel ran.
# This trace is not used for latency/SLO comparison.
python3 - "$ROOT_RUN" "$PORT" <<'PYPROFILE'
import gzip, json, re, sys, urllib.request
from pathlib import Path
root, port = Path(sys.argv[1]), sys.argv[2]
trace_dir = root / 'path-profile'
trace_dir.mkdir(exist_ok=True)
def post(path, data):
    req = urllib.request.Request(f'http://127.0.0.1:{port}' + path,
                                 json.dumps(data).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return resp.read()
post('/flush_cache', {})
post('/start_profile', {'output_dir': str(trace_dir), 'activities': ['CPU', 'GPU'],
                        'with_stack': False, 'record_shapes': False, 'profile_id': 'moe172_path'})
try:
    body = {'text': 'Explain how a compiler schedules independent instructions. ' * 128,
            'sampling_params': {'max_new_tokens': 1, 'temperature': 0, 'ignore_eos': True}}
    (trace_dir / 'request-result.json').write_bytes(post('/generate', body))
finally:
    (trace_dir / 'stop-response.txt').write_bytes(post('/stop_profile', {}))
counts = {}
for path in trace_dir.glob('*.json.gz'):
    match = re.search(r'TP-(\d+)', path.name)
    if not match:
        continue
    with gzip.open(path, 'rt') as handle:
        events = json.load(handle)['traceEvents']
    counts[int(match[1])] = sum(e.get('cat') == 'kernel' and
                                '_ax172_clamped_swiglu' in e.get('name', '') for e in events)
(trace_dir / 'kernel-counts.json').write_text(json.dumps(counts, indent=2) + '\n')
print('FUSED_KERNEL_COUNTS', counts)
if set(counts) != set(range(8)) or not all(counts.values()):
    raise RuntimeError('172 fused kernel not observed on every TP rank')
PYPROFILE
[ "$?" = 0 ] || exit 9
RUN_DIR=$ROOT_RUN/cand1 PORT=$PORT bash "$AX/verify_kit/cap_smoke_body.sh" > "$ROOT_RUN/candidate-smoke.log" 2>&1 || exit 8
grep CAP_SMOKE "$ROOT_RUN/candidate-smoke.log" || exit 8
echo "NUMERICAL_SCREEN_COMPLETE: inspect raw discrepancies, caches and memory; no SLO or capability verdict"
