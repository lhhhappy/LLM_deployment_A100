N=22
# Dev-set run at concurrency N (set by the generated wrapper) on the B+110+111 engine (reused if running).
# Official harness unchanged: preflight -> warmup -> flush KV -> measure -> score. Tokenizer from /mnt/models.
source $AX/bin/scripts/pod/lib.sh
prepare_src b111 000-interface-compliance.patch 101-d1v12-on-base.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0   # mHC large-batch path: tilelang instead of DeepGEMM (sm80)
ensure_engine b111 --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang || exit 1
S1=$AX/s1/s1-dev
cd $S1 && S1_HARNESS_DIR=harness python3 run_dev.py --base-url http://127.0.0.1:$PORT --set dev-combined-v1 \
  --root data/dev-combined-v1 --cohort harness/g0a/samples_v3/cohort_dev-combined-v1.json \
  --tok-dir /mnt/models --out $RUN_DIR/dev --n $N ${DEV_EXTRA:-}
rc=$?
python3 -c "import json;d=json.load(open('$RUN_DIR/dev/summary.json'));print('SUMMARY',json.dumps(d,ensure_ascii=False)[:1500])" 2>/dev/null
exit $rc
