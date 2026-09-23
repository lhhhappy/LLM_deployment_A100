# 038 S0-140@N22: exactly job 035 (S0) with ONE change: SGLANG_AX_KDA_DUAL_SNAPSHOT=0 (patch 140's dual snapshot off;
# 101's role-boundary split takes over). Why: (1) measure what 140 is worth at N22 (R20: 61 follow-ups lost the tail state
# anyway); (2) MTP (160) forces 140 off, so this isolates that forced change before the MTP run.
# Verdict: LEVEL line from level_verdict.py (complete data + harness scorer + task.md rules).
G_NAME=s0v2
G_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 140-kda-dual-snapshot.patch drafts/120-sched-protect-chain-v2.patch"
G_ARGS="--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 "
G_ENV="SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=0"
LADDER_UP="22"
# Dev self-test on one engine: run the levels in LADDER_UP (e.g. "22" or "22 26"); each level is scored by
# verify_kit/level_verdict.py (complete data + harness scorer + task.md rules); stop at the first failure. Wrapper sets:
#   G_NAME, G_PATCHES, G_ARGS, G_ENV (same as dev_generic_template.sh), optional LADDER.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_NAME" $G_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
grep -h "KV Cache is allocated\|max_total_num_tokens" $AX/engine_current.log | tail -2 | cut -c1-200
# Print the effective setup (informational; accuracy of the verdict does not depend on it).
( cd $AX/patches && sha256sum $G_PATCHES ) | sed "s/^/PATCH_SHA /" | cut -c1-120
kv=$(grep -oh "max_total_num_tokens=[0-9]*" $AX/engine_current.log | tail -1 | cut -d= -f2)
echo "PRECHECK (info only) kv_tokens=${kv:-?} patches=$(cat $AX/src/$G_NAME/PATCHES | tr ' ' ',') args=[$G_ARGS] env=[${G_ENV:-}]"
# Correctness gates (lenient: only drop clearly broken outputs, e.g. 0/12; local accuracy is stricter than online) BEFORE spending a 35-min level (09-23: a graph mode gave fast but WRONG outputs, 0/12).
if [ "${SMOKE_GATE:-1}" = 1 ]; then
  RUN_DIR=$RUN_DIR PORT=$PORT bash $AX/verify_kit/cap_smoke_body.sh | tee $RUN_DIR/smoke.log | grep CAP_SMOKE
  c=$(grep -o "correct=[0-9]*" $RUN_DIR/smoke.log | cut -d= -f2)
  [ -n "$c" ] && [ "$c" -ge "${SMOKE_MIN:-6}" ] || { echo "GATE FAIL: capability smoke ${c:-?}/12"; exit 6; }
fi
if [ -n "${NUMREF:-}" ]; then   # numeric fingerprint vs a reference engine run (numcheck.json)
  python3 $AX/verify_kit/numcheck.py $RUN_DIR/num >/dev/null && python3 $AX/verify_kit/numcheck_cmp.py $NUMREF $RUN_DIR/num/numcheck.json | tee $RUN_DIR/numcmp.txt | tail -1
  grep -q "NUMCMP wrong=0/" $RUN_DIR/numcmp.txt || { echo "GATE FAIL: numerics differ from reference"; exit 7; }
fi
S1=$AX/s1/s1-dev; first=1
# Start high (target is N22/26): climb LADDER_UP; stop at the first failure. No descending by default (user 09-23:
# a failed N18 means diagnose + fix, lower levels carry no decision value). LADDER_DOWN only if explicitly set; old note:
# highest passing level (same logic as the formal climb: pass -> +4, fail -> -4, stop).
UP=${LADDER_UP:-${LADDER:-18 22 26}}; DOWN=${LADDER_DOWN:-}
run_level() {  # $1 = N ; returns 0 if formal-est pass
  local N=$1
  local out=$RUN_DIR/N$N; mkdir -p $out; local extra=""; [ $first = 1 ] || extra="--skip-warmup"; first=0
  python3 $AX/verify_kit/metrics_sampler.py $out/metrics.jsonl 10 & local msp=$!
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > $out/gpu_util.csv 2>/dev/null & local gsp=$!
  ( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 run_dev.py --base-url http://127.0.0.1:$PORT --set dev-combined-v1 \
      --root $S1/data/dev-combined-v1 --cohort $S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json \
      --tok-dir /mnt/models --out $out --n $N $extra ) > $out/run_dev.log 2>&1
  local rdrc=$?
  kill $msp $gsp 2>/dev/null
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "LEVEL N=$N status=ENGINE_DEAD"; return 3; }
  # Verdict (T54): complete data (every dev request exactly once) scored by scripts/score_formal.py = harness s1_score
  # + task.md statistical allowance + tpot_p95 gate. Exit 0 pass / 1 fail / 2 INVALID measurement.
  python3 $AX/verify_kit/level_verdict.py $out $N --harness-dir $S1/harness --data-root $S1/data/dev-combined-v1 --rundev-rc $rdrc
  local vrc=$?
  awk -F", " '{u[$2]+=$3; n[$2]++} END {for (g in u) printf "  gpu%s util avg %.0f%%\n", g, u[g]/n[g]}' $out/gpu_util.csv 2>/dev/null | head -1 | sed "s/^/INFO N=$N /"
  return $vrc
}
lvl=0
for N in $UP; do
  lvl=$((lvl+1))
  run_level $N; rc=$?
  [ $rc -eq 0 ] && continue
  [ $rc -ge 2 ] && exit $rc          # 2 = invalid measurement, 3 = engine dead: report and stop
  if [ $lvl -eq 1 ]; then
    [ -z "$DOWN" ] && { echo "LADDER first level N=$N failed -> stop (no descent; diagnose and fix)"; exit 1; }
    echo "LADDER first level N=$N failed -> descending: $DOWN"
    for M in $DOWN; do run_level $M && { echo "LADDER highest pass = N=$M"; break; }; r=$?; [ $r -ge 3 ] && exit $r; done
  else echo "LADDER stop at N=$N (formal-est FAIL); highest pass = previous level"; fi
  break
done
