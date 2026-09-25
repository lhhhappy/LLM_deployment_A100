# Dev self-test on one engine: run the levels in LADDER_UP (e.g. "22" or "22 26"); each level is scored by
# verify_kit/level_verdict.py (complete data + harness scorer + task.md rules); stop at the first failure. Wrapper sets:
#   G_COMMIT (engine commit id, exported by scripts/engine/export.sh), G_ARGS, G_ENV, G_EXPECT, optional LADDER.
#   G_EXPECT lists the effective mechanism states the job relies on, e.g. "120=on 122=off 180=on"; the engine's
#   "[ax] mechanisms:" line must match every entry ("off" also matches "off:<reason>") or the job measures nothing.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_COMMIT" || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
ensure_engine "$G_COMMIT" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
[ -n "${G_EXPECT:-}" ] || { echo "MECHANISMS INVALID: set G_EXPECT"; exit 2; }
mech=$(grep -h "\[ax\] mechanisms:" "$RUN_DIR/server.log" | tail -1); mech=" ${mech#*mechanisms: }"
[ "$mech" != " " ] || { echo "MECHANISMS INVALID: engine printed no [ax] mechanisms line"; exit 2; }
for want in $G_EXPECT; do
  [[ "$want" =~ ^[A-Za-z0-9_]+=[A-Za-z0-9_.-]+$ ]] || { echo "MECHANISMS INVALID expectation=$want"; exit 2; }
  k=${want%%=*}; v=${want#*=}; got=$(grep -o " $k=[^ ]*" <<<"$mech" | head -1 | cut -d= -f2)
  case "$got" in "$v"|"$v":*) ;; *) echo "MECHANISMS MISMATCH $k expected=$v got=${got:-missing}"; exit 2 ;; esac
done
echo "MECHANISMS OK expected=[$G_EXPECT] engine=[${mech# }]"
grep -h "KV Cache is allocated\|max_total_num_tokens" $AX/engine_current.log | tail -2 | cut -c1-200
# Print the effective setup (informational; accuracy of the verdict does not depend on it).
kv=$(grep -oh "max_total_num_tokens=[0-9]*" $AX/engine_current.log | tail -1 | cut -d= -f2)
echo "PRECHECK (info only) kv_tokens=${kv:-?} commit=$G_COMMIT args=[$G_ARGS] env=[${G_ENV:-}]"
# Correctness gates (lenient: only drop clearly broken outputs, e.g. 0/12; local accuracy is stricter than online) BEFORE spending a 35-min level (09-23: a graph mode gave fast but WRONG outputs, 0/12).
if [ "${SMOKE_GATE:-1}" = 1 ]; then
  RUN_DIR=$RUN_DIR PORT=$PORT bash $AX/verify_kit/cap_smoke_body.sh | tee $RUN_DIR/smoke.log | grep CAP_SMOKE
  c=$(grep -o "correct=[0-9]*" $RUN_DIR/smoke.log | cut -d= -f2)
  [ -n "$c" ] && [ "$c" -ge "${SMOKE_MIN:-6}" ] || { echo "GATE FAIL: capability smoke ${c:-?}/12"; exit 6; }
fi
S1=$AX/s1/s1-dev; first=1
# Independent frozen datasets must use their own root/set/cohort consistently.
# These variables only affect newly scheduled jobs; existing pod jobs are not updated.
DATA_ROOT=${G_DATA_ROOT:-$S1/data/dev-combined-v1}
DATA_SET=${G_DATA_SET:-dev-combined-v1}
COHORT=${G_COHORT:-$S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json}
if [ -n "${G_DATA_ROOT:-}" ] && { [ -z "${G_DATA_SET:-}" ] || [ -z "${G_COHORT:-}" ]; }; then
  echo "DATA INVALID: custom G_DATA_ROOT also requires G_DATA_SET and G_COHORT"; exit 2
fi
[ -f "$DATA_ROOT/requests.jsonl" ] && [ -f "$COHORT" ] || { echo "DATA INVALID: missing root/cohort"; exit 2; }
# Start high (target is N22/26): climb LADDER_UP; stop at the first failure. No descending by default (user 09-23:
# an SLO failure is reported as-is; separate fixed-N jobs may still study higher N).
# LADDER_DOWN only if explicitly set; old note:
# highest passing level (same logic as the formal climb: pass -> +4, fail -> -4, stop).
UP=${LADDER_UP:-${LADDER:-18 22 26}}; DOWN=${LADDER_DOWN:-}
run_level() {  # $1 = N ; returns 0 if formal-est pass
  local N=$1
  local out=$RUN_DIR/N$N; mkdir -p $out; local extra=""; [ $first = 1 ] || extra="--skip-warmup"; first=0
  python3 $AX/verify_kit/metrics_sampler.py $out/metrics.jsonl 10 & local msp=$!
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > $out/gpu_util.csv 2>/dev/null & local gsp=$!
  local runner=("$AX/verify_kit/run_dev_checked.py")
  if [ "${G_WARMUP_PROFILE:-original}" != original ]; then
    # Each short-warmup level currently requires its own verified receipt.
    # Original warmup still supports the existing cross-N reuse path.
    extra=""
    runner+=(--warmup-profile "$G_WARMUP_PROFILE")
  fi
  if [ -n "${G_MEASURE_SECONDS:-}" ]; then
    runner=("$AX/verify_kit/timed_run.py" --seconds "$G_MEASURE_SECONDS" --warmup-profile "${G_WARMUP_PROFILE:-original}")
    echo "TIMED_DIAGNOSTIC N=$N admission_seconds=$G_MEASURE_SECONDS drain_all_admitted=true"
  fi
  ( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 -B "${runner[@]}" --runner "$S1/run_dev.py" -- --base-url http://127.0.0.1:$PORT --set "$DATA_SET" \
      --root "$DATA_ROOT" --cohort "$COHORT" \
      --tok-dir /mnt/models --out $out --n $N $extra ) > $out/run_dev.log 2>&1
  local rdrc=$?
  printf "%s\n" "$rdrc" > "$out/rundev_exit_code"
  kill $msp $gsp 2>/dev/null
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "LEVEL N=$N status=ENGINE_DEAD"; return 3; }
  if [ -n "${G_MEASURE_SECONDS:-}" ]; then
    python3 "$AX/verify_kit/timed_score.py" "$out" --harness-dir "$S1/harness" --data-root "$DATA_ROOT"
    local trc=$?
    printf "%s\n" "$trc" > "$out/verdict_exit_code"
    return "$trc"  # 0 = drained diagnostic; never a complete-cohort PASS.
  fi
  # Verdict (T54): complete data (every dev request exactly once) scored by scripts/score_formal.py = harness s1_score
  # + task.md statistical allowance + tpot_p95 gate. Exit 0 pass / 1 fail / 2 INVALID measurement.
  python3 $AX/verify_kit/level_verdict.py $out $N --harness-dir $S1/harness --data-root "$DATA_ROOT" --rundev-rc $rdrc
  local vrc=$?
  printf "%s\n" "$vrc" > "$out/verdict_exit_code"
  awk -F", " '{u[$2]+=$3; n[$2]++} END {for (g in u) printf "  gpu%s util avg %.0f%%\n", g, u[g]/n[g]}' $out/gpu_util.csv 2>/dev/null | head -1 | sed "s/^/INFO N=$N /"
  return $vrc
}
lvl=0
for N in $UP; do
  lvl=$((lvl+1))
  rc=0; run_level "$N" || rc=$?
  [ "$rc" -eq 0 ] && continue
  [ "$rc" -ge 2 ] && exit "$rc"  # INVALID / engine failure must never be swallowed.
  if [ "$lvl" -eq 1 ] && [ -n "$DOWN" ]; then
    echo "LADDER first level N=$N failed -> descending: $DOWN"
    for M in $DOWN; do
      rc=0; run_level "$M" || rc=$?
      if [ "$rc" -eq 0 ]; then echo "LADDER highest pass = N=$M"; exit 0; fi
      [ "$rc" -ge 2 ] && exit "$rc"
    done
    echo "LADDER no passing level"; exit 1
  fi
  echo "LADDER stop at N=$N (estimated FAIL); see individual levels for any earlier pass"
  exit 1
done
exit 0
