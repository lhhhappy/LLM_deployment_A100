N=10
AX_P120_VARIANT=on
# T41 proposal only: derived from dev_template.sh; not queued or run by W15.
# Pod worker supplies AX, RUN_DIR, N. Variants get distinct source names because
# lib.sh ensure_engine does not include scheduler env vars in its reuse signature.
: "${N:?Set concurrency N in the pod job wrapper}"
source "$AX/bin/scripts/pod/lib.sh"
case "${AX_P120_VARIANT:-on}" in
  off)     src_name=b126a; protect=0; cold_cap=2048 ;;
  on)      src_name=b126b; protect=1; cold_cap=2048 ;;
  cap4096) src_name=b126c; protect=1; cold_cap=4096 ;;
  *) echo "Unknown AX_P120_VARIANT" >&2; exit 2 ;;
esac
prepare_src "$src_name" 000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch \
  110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch \
  120-sched-protect-chain.patch || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_AX_SCHED_PROTECT=$protect
export SGLANG_AX_SCHED_COLD_CAP=$cold_cap
export SGLANG_AX_SCHED_SHORT_TOKENS=4096
ensure_engine "$src_name" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang || exit 1
S1=$AX/s1/s1-dev
cd "$S1" || exit 1
S1_HARNESS_DIR="$S1/harness" python3 run_dev.py --base-url "http://127.0.0.1:$PORT" --set dev-combined-v1 \
  --root "$S1/data/dev-combined-v1" --cohort "$S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json" \
  --tok-dir /mnt/models --out "$RUN_DIR/dev" --n "$N" ${DEV_EXTRA:-}
rc=$?
python3 -c 'import json,sys; print("SUMMARY", json.dumps(json.load(open(sys.argv[1])),ensure_ascii=False)[:1500])' "$RUN_DIR/dev/summary.json" 2>/dev/null
exit "$rc"
# Engine must still be alive after the run; otherwise the result is an infra failure, not a score.
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo ENGINE_DEAD_AFTER_RUN; grep -n "Scheduler hit an exception" -A30 $AX/engine_current.log 2>/dev/null | tail -12; exit 3; }
