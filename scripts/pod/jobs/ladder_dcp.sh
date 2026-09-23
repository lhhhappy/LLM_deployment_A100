G_NAME=b115dcp120
G_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 114-indexer-row-shard.patch 115-sm80-sparse-attn-many-heads.patch 120-sched-protect-chain.patch"
G_ARGS="--chunked-prefill-size 16384 --dcp-size 8 --cuda-graph-max-bs-decode 64"
G_ENV="SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192"
# Self-test ladder like the formal climb: one engine, N = $LADDER (default 10 14 18 22 26); after each level score
# with analyze_run.py; climb while the formal-rule estimate passes, stop at the first failure. Wrapper sets:
#   G_NAME, G_PATCHES, G_ARGS, G_ENV (same as dev_generic_template.sh), optional LADDER.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_NAME" $G_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
grep -h "KV Cache is allocated\|max_total_num_tokens" $AX/engine_current.log | tail -2 | cut -c1-200
S1=$AX/s1/s1-dev; first=1
for N in ${LADDER:-10 14 18 22 26}; do
  out=$RUN_DIR/N$N; mkdir -p $out; extra=""; [ $first = 1 ] || extra="--skip-warmup"; first=0
  ( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 run_dev.py --base-url http://127.0.0.1:$PORT --set dev-combined-v1 \
      --root $S1/data/dev-combined-v1 --cohort $S1/harness/g0a/samples_v3/cohort_dev-combined-v1.json \
      --tok-dir /mnt/models --out $out --n $N $extra ) > $out/run_dev.log 2>&1
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "LADDER N=$N ENGINE_DEAD"; exit 3; }
  raw=$(ls -t $out/raw_*.jsonl 2>/dev/null | head -1); [ -n "$raw" ] || { echo "LADDER N=$N NO_RAW"; tail -5 $out/run_dev.log; exit 4; }
  python3 $AX/verify_kit/analyze_run.py $S1/harness $raw $out/verdict.json > $out/analysis.txt 2>&1
  v=$(python3 -c "import json;d=json.load(open('$out/verdict.json'));g=d['gates'];print('formal_est=%s harness=%s tpot=%.4f | '%(d['formal_est_all_pass'],d['harness_all_pass'],d['tpot_mean'])+' '.join('%s:%.2f(%d/%d)'%(k[:6],v['p95'],v['over'],v['allowed_over']) for k,v in g.items()))")
  echo "LADDER N=$N $v"
  python3 $AX/verify_kit/logstat.py $RUN_DIR/server.log 2>/dev/null | sed -n 2p
  python3 -c "import json,sys;sys.exit(0 if json.load(open('$out/verdict.json'))['formal_est_all_pass'] else 1)" || { echo "LADDER stop at N=$N (formal-est FAIL)"; break; }
done
