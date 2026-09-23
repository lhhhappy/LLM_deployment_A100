# Self-test ladder like the formal climb: one engine, N = $LADDER (default 10 14 18 22 26); after each level score
# with analyze_run.py; climb while the formal-rule estimate passes, stop at the first failure. Wrapper sets:
#   G_NAME, G_PATCHES, G_ARGS, G_ENV (same as dev_generic_template.sh), optional LADDER.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_NAME" $G_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
grep -h "KV Cache is allocated\|max_total_num_tokens" $AX/engine_current.log | tail -2 | cut -c1-200
# Fail fast on wrong preconditions (a mis-sized KV pool silently wasted a whole level once).
kv=$(grep -oh "max_total_num_tokens=[0-9]*" $AX/engine_current.log | tail -1 | cut -d= -f2)
echo "PRECHECK kv_tokens=${kv:-?} min_required=${LADDER_MIN_KV:-900000} patches=$(cat $AX/src/$G_NAME/PATCHES | tr ' ' ',') args=[$G_ARGS] env=[${G_ENV:-}]"
[ -n "$kv" ] && [ "$kv" -ge "${LADDER_MIN_KV:-900000}" ] || { echo "PRECHECK FAIL: KV tokens ${kv:-unknown} < ${LADDER_MIN_KV:-900000}"; exit 5; }
S1=$AX/s1/s1-dev; first=1
# Start high (target is N22/26): climb LADDER_UP; if the FIRST level already fails, descend LADDER_DOWN to find the
# highest passing level (same logic as the formal climb: pass -> +4, fail -> -4, stop).
UP=${LADDER_UP:-${LADDER:-18 22 26}}; DOWN=${LADDER_DOWN:-14 10}
run_level() {  # $1 = N ; returns 0 if formal-est pass
  local N=$1
  local out=$RUN_DIR/N$N; mkdir -p $out; local extra=""; [ $first = 1 ] || extra="--skip-warmup"; first=0
  ( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 run_dev.py --base-url http://127.0.0.1:$PORT --set dev-combined-v1 \
      --root $S1/data/dev-combined-v1 --cohort $S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json \
      --tok-dir /mnt/models --out $out --n $N $extra ) > $out/run_dev.log 2>&1
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "LADDER N=$N ENGINE_DEAD"; return 3; }
  local raw=$(ls -t $out/raw_*.jsonl 2>/dev/null | head -1); [ -n "$raw" ] || { echo "LADDER N=$N NO_RAW"; tail -5 $out/run_dev.log; return 4; }
  python3 $AX/verify_kit/analyze_run.py $S1/harness $raw $out/verdict.json > $out/analysis.txt 2>&1
  local v=$(python3 -c "import json;d=json.load(open('$out/verdict.json'));g=d['gates'];print('formal_est=%s harness=%s tpot=%.4f | '%(d['formal_est_all_pass'],d['harness_all_pass'],d['tpot_mean'])+' '.join('%s:%.2f(%d/%d)'%(k[:6],v['p95'],v['over'],v['allowed_over']) for k,v in g.items()))")
  echo "LADDER N=$N $v"
  python3 $AX/verify_kit/logstat.py $RUN_DIR/server.log 2>/dev/null | sed -n 2p
  python3 -c "import json,sys;sys.exit(0 if json.load(open('$out/verdict.json'))['formal_est_all_pass'] else 1)"
}
lvl=0
for N in $UP; do
  lvl=$((lvl+1))
  run_level $N; rc=$?
  [ $rc -eq 0 ] && continue
  [ $rc -ge 3 ] && exit $rc          # engine dead / no data: a crash is a failure of its own, report and stop
  if [ $lvl -eq 1 ]; then
    echo "LADDER first level N=$N failed -> descending: $DOWN"
    for M in $DOWN; do run_level $M && { echo "LADDER highest pass = N=$M"; break; }; r=$?; [ $r -ge 3 ] && exit $r; done
  else echo "LADDER stop at N=$N (formal-est FAIL); highest pass = previous level"; fi
  break
done
