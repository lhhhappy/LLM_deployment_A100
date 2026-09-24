# Continue 059 on the same live A+180 engine and frozen lite workload; only N14 -> N30.
# No N14 SLO pass prerequisite. Reuse the completed warmup, retain preflight, and perform a fresh checked KV flush.
# This job never restarts or replaces the engine; a signature mismatch requires explicit diagnosis.
# This run does NOT prove numeric correctness of host restore; that is hc180_numeric.py on the dev box (pending).
G_NAME=off_a_180_hicache_lite_n14
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch 180-hicache-glm-dsa.patch"
# 058 args + HiCache: 32 GB host tier per GPU (per-rank size), async write-through on insert. NUMA binding left out on
# purpose so the only change is the host tier; add --numa-node 0 0 0 0 1 1 1 1 as a separate follow-up.
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096"
LADDER_UP="30"
G_DATA_ROOT="$AX/data/s1-dev-longchain-lite"
G_DATA_SET=s1-dev-longchain-lite
G_COHORT="$G_DATA_ROOT/cohort.json"
unset LADDER_DOWN NUMREF SGLANG_AX_PACE_TPOT SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_KDA_FUSE_PROJ SGLANG_AX_MOE_FUSE_SWIGLU
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0
echo "EXPERIMENT formal_A_plus_180_hicache_lite_N30 chains=64 requests=1123 baseline=059 only_concurrency_changed reuse_engine skip_warmup no_profiler"
# Verify transferred bytes against the frozen manifest before loading the engine (same check as 058).
python3 - "$G_DATA_ROOT" <<'PY' || exit 2
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()
assert sha(root / 'manifest.json') == 'b8f8b668189d9a6ded593ff88462cda37285d8e7a68f7654cea9ab1624f9b999'
m = json.loads((root / 'manifest.json').read_text())
assert m['set'] == 's1-dev-longchain-lite'
for name, digest in m['artifacts'].items():
    assert sha(root / name) == digest, name
    print('DATA_SHA', digest, name, flush=True)
c = json.loads((root / 'cohort.json').read_text())
assert c['n_chains'] == 64 and c['n_requests'] == 1123
print('DATA_READY', c['set'], c['n_chains'], c['n_requests'], c['cohort_sha256'], flush=True)
PY
free -g | head -2; nvidia-smi topo -m | head -10
sha256sum "$AX/verify_kit/level_verdict.py" "$AX/verify_kit/run_dev_checked.py" \
  "$AX/verify_kit/score_formal.py" "$AX/verify_kit/metrics_sampler.py" \
  "$AX/s1/s1-dev/run_dev.py" "$AX/bin/scripts/pod/lib.sh" \
  "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh" || exit 2
( cd "$AX/patches" && sha256sum -c - <<'PATCH_SHA256'
e0d7924989ebb538e331110f4458ccd4e95a0bfa3cbd87dfb8324ac0328960f6  000-interface-compliance.patch
0f6ee401c8bafd6d3c62f18a33545e1a777140eb9df53774029b60ae0a3694e0  101-role-boundary-split.patch
a2c9f92d8a0abc96eaec515e6afa216a8ffa98037431748c1e0f846b66a03174  106-defer-chunk-on-no-kv.patch
6a592ffc62b655c91672d2c31d76d2ecf9bbc3526a6f86f95ff28907522c5360  110-sm80-dsa-indexer.patch
67e358bb9e52da343f0848b6fe96c778e6473efcd32f9df450213c6dcfb7d26a  111-sm80-fp8-moe-marlin.patch
70c078d01b17c3d1051e18f31253dd0beab0f58b642599978d4630fcd7352a9c  114-indexer-row-shard.patch
ef1744b3a51c2ee87fc0ceb858a04cca078c4720ad50b16ff7498a1b0ee7eb1a  120-sched-protect-chain.patch
dee943778854ba4d19c804fcee9070338e9f6eb51529b30d658b13e8c0f5faa0  121-sched-cap-while-decoding.patch
d48bf2a2d8a4c4daf68981aefdbc1646cf2d36b986ebf610f41e377a60ba2380  130-async-tokenize.patch
1fcb1ca8c6502b7c8a58f32c16bf34dbf3431d159cd649fdbf1b658ce7c2ff35  140-kda-dual-snapshot.patch
3d53476f8d94262e55241bb2295e3b9b96fb0ffc243469aaef9541d4169f20af  150-startup-warmup.patch
3547ff7d6d583b5ca19e0474a9ed0bd3fff3a355d5298868ee9f83ee96000a40  160-nextn-sm80.patch
48eeac6b956102d4394ace56e6fbbed343082b060b9aefd9548f73cc644320d5  170-glm-bcg-prefill.patch
3b63d9c88cb28520102455bc9081e5e8a7d9f9b0654bd1f1dde1f401747c8848  180-hicache-glm-dsa.patch
PATCH_SHA256
) || exit 2

source "$AX/bin/scripts/pod/lib.sh"
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export $G_ENV
srcsig=$(cat "$AX/src/$G_NAME/PATCHES_SIG")
[ "$srcsig" = 32989fc8418538ae ] || { echo "REUSE_FAILED: different source stack"; exit 2; }
envkey=$(env | grep -E '^(SGLANG_AX_|SGLANG_ARENA_|NCCL_|SGLANG_MAMBA|SGLANG_OPT_)' | sort | tr '\n' ' ')
expected_sig="$G_NAME $srcsig | --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS | $envkey"
[ "$(cat "$AX/engine.sig")" = "$expected_sig" ] || { echo "REUSE_FAILED: engine configuration changed"; exit 2; }
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "REUSE_FAILED: engine unavailable"; exit 2; }
# This verifies completed shape warmup only; it never inspects N14 SLO or generated-text equality.
prior="$AX/runs/059-official_a_180_hicache_lite_n14"
grep -q '^\[warmup\].*遗漏=0' "$prior/N14/warmup.log" || { echo "REUSE_FAILED: 059 warmup not complete"; exit 2; }
live_log=$(cat "$AX/engine_log_path")
[ -f "$live_log" ] || { echo "REUSE_FAILED: live log missing"; exit 2; }
ln -s "$live_log" "$RUN_DIR/server.log" || exit 2
printf 'ENGINE_REUSED source=%s WARMUP_REUSED_FROM=%s\n' "$srcsig" "$prior"
printf 'PRECHECK args=[%s] env=[%s]\n' "$G_ARGS" "$G_ENV"
S1="$AX/s1/s1-dev"
DATA_ROOT="$G_DATA_ROOT"
DATA_SET="$G_DATA_SET"
COHORT="$G_COHORT"
first=0

run_level() {  # $1 = N ; returns 0 if formal-est pass
  local N=$1
  local out=$RUN_DIR/N$N; mkdir -p $out; local extra=""; [ $first = 1 ] || extra="--skip-warmup"; first=0
  python3 $AX/verify_kit/metrics_sampler.py $out/metrics.jsonl 10 & local msp=$!
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > $out/gpu_util.csv 2>/dev/null & local gsp=$!
  ( cd $S1 && S1_HARNESS_DIR=$S1/harness python3 -B "$AX/verify_kit/run_dev_checked.py" --runner "$S1/run_dev.py" -- --base-url http://127.0.0.1:$PORT --set "$DATA_SET" \
      --root "$DATA_ROOT" --cohort "$COHORT" \
      --tok-dir /mnt/models --out $out --n $N $extra ) > $out/run_dev.log 2>&1
  local rdrc=$?
  printf "%s\n" "$rdrc" > "$out/rundev_exit_code"
  kill $msp $gsp 2>/dev/null
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "LEVEL N=$N status=ENGINE_DEAD"; return 3; }
  # Verdict (T54): complete data (every dev request exactly once) scored by scripts/score_formal.py = harness s1_score
  # + task.md statistical allowance + tpot_p95 gate. Exit 0 pass / 1 fail / 2 INVALID measurement.
  python3 $AX/verify_kit/level_verdict.py $out $N --harness-dir $S1/harness --data-root "$DATA_ROOT" --rundev-rc $rdrc
  local vrc=$?
  printf "%s\n" "$vrc" > "$out/verdict_exit_code"
  awk -F", " '{u[$2]+=$3; n[$2]++} END {for (g in u) printf "  gpu%s util avg %.0f%%\n", g, u[g]/n[g]}' $out/gpu_util.csv 2>/dev/null | head -1 | sed "s/^/INFO N=$N /"
  return $vrc
}
# A valid N14 SLO failure does not block this independent N30 measurement.
run_level 30
exit $?
