# DRAFT (048) — not queued. Queue owner (main Codex) reviews, syncs patches/122-tpot-paced-prefill.patch, and queues.
# Question: official A + 122 only (TPOT-paced prefill budget) at dev N22; compare gate by gate with 047 (official A as is, N22).
# Single change vs 047: patch 122 inserted after 121 + env SGLANG_AX_PACE_TPOT=0.085; launch args identical
# (the explicit --prefill-decode-interval 2 stays in the command and is ignored while 122 is on, as its .md states).
# Frozen 122 sha256: cefdb2688cbc291742fc3c3ad188e343420fad01407d172f164ca6746d712d3b
# Evidence beyond the verdict: "[ax-pace] on:" (effective values); 30 s "[ax-pace] ... guard=... mean_budget=..." lines;
# prefill batch sizes, KV usage and running/queue counts vs 047. 122 is a measurable candidate, not a TPOT guarantee.
# Refuted if: no TTFT gate improves vs 047, or tpot_p95 > 0.10, or mean_budget does not rise above 4096.
G_NAME=off_a_122
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 122-tpot-paced-prefill.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_PACE_TPOT=0.085"
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
  ( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 -B "$AX/verify_kit/run_dev_checked.py" --runner "$S1/run_dev.py" -- --base-url http://127.0.0.1:$PORT --set dev-combined-v1 \
      --root $S1/data/dev-combined-v1 --cohort $S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json \
      --tok-dir /mnt/models --out $out --n $N $extra ) > $out/run_dev.log 2>&1
  local rdrc=$?
  printf "%s\n" "$rdrc" > "$out/rundev_exit_code"
  kill $msp $gsp 2>/dev/null
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "LEVEL N=$N status=ENGINE_DEAD"; return 3; }
  # Verdict (T54): complete data (every dev request exactly once) scored by scripts/score_formal.py = harness s1_score
  # + task.md statistical allowance + tpot_p95 gate. Exit 0 pass / 1 fail / 2 INVALID measurement.
  python3 $AX/verify_kit/level_verdict.py $out $N --harness-dir $S1/harness --data-root $S1/data/dev-combined-v1 --rundev-rc $rdrc
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
