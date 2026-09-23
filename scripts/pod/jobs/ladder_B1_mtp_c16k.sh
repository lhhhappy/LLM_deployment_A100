G_NAME=img0923a
G_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 114-indexer-row-shard.patch 140-kda-dual-snapshot.patch 120-sched-protect-chain.patch 130-async-tokenize.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--chunked-prefill-size 16384 --mem-fraction-static 0.74 --kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_COLD_CAP=16384 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1"
LADDER_UP=18
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
  ( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 run_dev.py --base-url http://127.0.0.1:$PORT --set dev-combined-v1 \
      --root $S1/data/dev-combined-v1 --cohort $S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json \
      --tok-dir /mnt/models --out $out --n $N $extra ) > $out/run_dev.log 2>&1
  kill $msp 2>/dev/null
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
    [ -z "$DOWN" ] && { echo "LADDER first level N=$N failed -> stop (no descent; diagnose and fix)"; exit 1; }
    echo "LADDER first level N=$N failed -> descending: $DOWN"
    for M in $DOWN; do run_level $M && { echo "LADDER highest pass = N=$M"; break; }; r=$?; [ $r -ge 3 ] && exit $r; done
  else echo "LADDER stop at N=$N (formal-est FAIL); highest pass = previous level"; fi
  break
done
