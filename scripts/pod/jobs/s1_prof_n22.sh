# Execution-time ledger of S1 under the real N22 mixed load (probe, not a gate run).
# Same engine as 036/042 (consolidated patch names, source-identical per evidence/T57). Two torch-profiler windows
# of 80 forward steps during the dev N22 replay; prof_ledger.py reports, per TP rank, GPU time in extend/decode spans,
# host gaps inside and between steps, and kernel time by category. The level verdict is printed but is perturbed by
# the profiler windows and must not be compared with 036/042.
G_NAME=s1
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 140-kda-dual-snapshot.patch"
G_ARGS="--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200"
G_ENV="SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1 SGLANG_AX_INDEXER_ROW_SHARD=1"
N=22; PROF_AT="480 840"; PROF_STEPS=80

source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_NAME" $G_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export $G_ENV
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
grep -h "max_total_num_tokens" $AX/engine_current.log | tail -1 | cut -c1-200
( cd $AX/patches && sha256sum $G_PATCHES ) | sed "s/^/PATCH_SHA /" | cut -c1-120
echo "PRECHECK (info only) patches=$(cat $AX/src/$G_NAME/PATCHES | tr ' ' ',') args=[$G_ARGS] env=[$G_ENV] prof_at=[$PROF_AT] steps=$PROF_STEPS"
RUN_DIR=$RUN_DIR PORT=$PORT bash $AX/verify_kit/cap_smoke_body.sh | tee $RUN_DIR/smoke.log | grep CAP_SMOKE
c=$(grep -o "correct=[0-9]*" $RUN_DIR/smoke.log | cut -d= -f2)
[ -n "$c" ] && [ "$c" -ge 6 ] || { echo "GATE FAIL: capability smoke ${c:-?}/12"; exit 6; }

S1=$AX/s1/s1-dev; out=$RUN_DIR/N$N; mkdir -p $out
( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 run_dev.py --base-url http://127.0.0.1:$PORT --set dev-combined-v1 \
    --root $S1/data/dev-combined-v1 --cohort $S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json \
    --tok-dir /mnt/models --out $out --n $N ) > $out/run_dev.log 2>&1 &
rd=$!; t0=$(date +%s); w=0
for at in $PROF_AT; do
  w=$((w+1)); while [ $(( $(date +%s) - t0 )) -lt $at ]; do kill -0 $rd 2>/dev/null || break 2; sleep 5; done
  mkdir -p $RUN_DIR/prof/w$w
  echo "PROF_START w$w at +$(( $(date +%s) - t0 ))s $(date -u +%H:%M:%S)"
  curl -s -m 900 -X POST http://127.0.0.1:$PORT/start_profile -H 'Content-Type: application/json' \
    -d "{\"output_dir\":\"$RUN_DIR/prof/w$w\",\"num_steps\":$PROF_STEPS,\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false,\"profile_id\":\"w$w\"}" \
    > $RUN_DIR/prof/w$w/start.resp 2>&1 &
done
wait $rd; rdrc=$?
sleep 30
python3 $AX/verify_kit/level_verdict.py $out $N --harness-dir $S1/harness --data-root $S1/data/dev-combined-v1 --rundev-rc $rdrc | sed 's/^LEVEL/LEVEL(perturbed by profiler)/'
grep -h "Profiling done\|Traces will be saved" $RUN_DIR/server.log | grep TP0 | cut -c1-200
for d in $RUN_DIR/prof/w*; do
  ls -la $d | head -20
  tr=$(ls $d/*TP-0*.json.gz $d/*TP-4*.json.gz 2>/dev/null)
  [ -n "$tr" ] || { echo "PROF $d: no TP-0/TP-4 trace"; continue; }
  python3 $AX/verify_kit/prof_ledger.py $tr --json $d/ledger.json
  for f in $d/*.json.gz; do case $f in *TP-0*|*TP-4*) ;; *) rm -f $f;; esac; done   # keep two ranks
done
