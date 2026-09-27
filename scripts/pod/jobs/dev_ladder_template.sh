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
# Optional full capability check (public AIME 2026 + GPQA-Diamond through the served engine; see verify_kit/cap_full_body.sh).
# G_CAP_FULL=1 runs it after the smoke; LADDER_UP=none then ends the job without a load level.
if [ "${G_CAP_FULL:-0}" = 1 ]; then
  RUN_DIR=$RUN_DIR PORT=$PORT CAP_DIR=${G_CAP_DIR:-$AX/verify_kit} CAP_CONCURRENCY=${G_CAP_CONCURRENCY:-24} \
    bash $AX/verify_kit/cap_full_body.sh | tee $RUN_DIR/cap_full.log | grep CAP_FULL
fi
if [ "${LADDER_UP:-}" = none ]; then echo "LADDER none: capability-only job"; exit 0; fi
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
# Opening profile under load (diagnostic only). With G_OPEN_PROFILE_LEN_S set, a watcher waits until the engine reports
# at least G_OPEN_PROFILE_QUEUE_MIN (default 20) waiting requests (the opening burst of a level), sleeps
# G_OPEN_PROFILE_START_S (default 8) seconds, then runs the torch profiler on every rank for G_OPEN_PROFILE_LEN_S seconds
# through /start_profile and /stop_profile. Trace export stalls the engine, so the level's gate numbers are not comparable
# with unprofiled runs; the value is the rank0/rank1 execution-time ledger, printed as OPEN_PROFILE_LEDGER lines.
open_profile_watch() {
  local out=$1 t=0 q=""
  while [ $t -lt 1200 ]; do
    q=$(curl -s -m 3 http://127.0.0.1:$PORT/metrics | awk '/^sglang:num_queue_reqs/ {print $NF; exit}'); q=${q%.*}
    if [ -n "$q" ] && [ "$q" -ge "${G_OPEN_PROFILE_QUEUE_MIN:-20}" ] 2>/dev/null; then break; fi
    sleep 1; t=$((t+1))
  done
  if [ $t -ge 1200 ]; then echo "OPEN_PROFILE trigger not seen (queue never reached ${G_OPEN_PROFILE_QUEUE_MIN:-20})"; return 0; fi
  sleep "${G_OPEN_PROFILE_START_S:-8}"
  mkdir -p "$out/traces"
  echo "OPEN_PROFILE start $(date -u +%FT%TZ) queue_at_trigger=$q len_s=$G_OPEN_PROFILE_LEN_S"
  curl -s -m 30 -X POST http://127.0.0.1:$PORT/start_profile -H 'Content-Type: application/json' \
    -d "{\"output_dir\":\"$out/traces\",\"profile_id\":\"open\",\"profile_prefix\":\"open\",\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false,\"merge_profiles\":false}" > "$out/profile_start.txt"
  sleep "$G_OPEN_PROFILE_LEN_S"
  curl -s -m 900 -X POST http://127.0.0.1:$PORT/stop_profile > "$out/profile_stop.txt"
  echo "OPEN_PROFILE stop $(date -u +%FT%TZ)"
}
open_profile_ledger() {
  local out=$1
  ls "$out"/traces/*TP-0*.trace.json.gz "$out"/traces/*TP-1*.trace.json.gz >/dev/null 2>&1 || { echo "OPEN_PROFILE no rank0/rank1 traces"; return 0; }
  python3 -B "$AX/verify_kit/prof_ledger.py" "$out"/traces/*TP-0*.trace.json.gz "$out"/traces/*TP-1*.trace.json.gz \
      --json "$out/ledger-rank0-rank1.json" > "$out/ledger-rank0-rank1.txt" 2> "$out/ledger.err" || echo "OPEN_PROFILE ledger failed (see ledger.err)"
  sed 's/^/OPEN_PROFILE_LEDGER /' "$out/ledger-rank0-rank1.txt"
}
run_level() {  # $1 = N ; returns 0 if formal-est pass
  local N=$1
  local out=$RUN_DIR/N$N; mkdir -p $out; local extra=""; [ $first = 1 ] || extra="--skip-warmup"; first=0
  python3 $AX/verify_kit/metrics_sampler.py $out/metrics.jsonl 10 & local msp=$!
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > $out/gpu_util.csv 2>/dev/null & local gsp=$!
  local wp=""; if [ -n "${G_OPEN_PROFILE_LEN_S:-}" ]; then open_profile_watch "$out" & wp=$!; fi
  local runner=("$AX/verify_kit/run_dev_checked.py")
  if [ "${G_WARMUP_PROFILE:-original}" != original ]; then
    # Each short-warmup level currently requires its own verified receipt.
    # Original warmup still supports the existing cross-N reuse path.
    extra=""
    runner+=(--warmup-profile "$G_WARMUP_PROFILE")
  fi
  if [ -n "${G_MEASURE_SECONDS:-}" ]; then
    runner=("$AX/verify_kit/timed_run.py" --seconds "$G_MEASURE_SECONDS" --warmup-profile "${G_WARMUP_PROFILE:-original}"
            --chain-start-interval-s "${G_CHAIN_START_INTERVAL_S:-0}")
    echo "TIMED_DIAGNOSTIC N=$N admission_seconds=$G_MEASURE_SECONDS chain_start_interval_s=${G_CHAIN_START_INTERVAL_S:-0} drain_all_admitted=true"
  elif [ -n "${G_CHAIN_START_INTERVAL_S:-}" ]; then
    echo "DATA INVALID: chain start interval requires a timed diagnostic"; return 2
  fi
  ( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 -B "${runner[@]}" --runner "$S1/run_dev.py" -- --base-url http://127.0.0.1:$PORT --set "$DATA_SET" \
      --root "$DATA_ROOT" --cohort "$COHORT" \
      --tok-dir /mnt/models --out $out --n $N $extra ) > $out/run_dev.log 2>&1
  local rdrc=$?
  printf "%s\n" "$rdrc" > "$out/rundev_exit_code"
  kill $msp $gsp 2>/dev/null
  if [ -n "$wp" ]; then wait $wp 2>/dev/null; open_profile_ledger "$out"; fi
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
