# Official A (attempt 45979 / 0923a), unchanged, on the FULL frozen long-chain set (311 chains / 5601 requests) at N30:
# the baseline for full-set N30 candidates. Identical to 060b except the data set. One complete replay; watch it with
# scripts/analysis/window_watch.sh (25-minute windows) and stop early with stopjob only if the windows clearly collapse.
# Needs data/s1-dev-longchain synced to $AX/data first (522 MB).
G_NAME=off_a_longchain_full_n30
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096"
LADDER_UP="30"
G_DATA_ROOT="$AX/data/s1-dev-longchain"
G_DATA_SET=s1-dev-longchain
G_COHORT="$G_DATA_ROOT/cohort.json"
unset LADDER_DOWN NUMREF SGLANG_AX_PACE_TPOT SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_KDA_FUSE_PROJ SGLANG_AX_MOE_FUSE_SWIGLU
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0
echo "EXPERIMENT formal_A_longchain_full_N30 chains=311 requests=5601 no_profiler"
# Verify transferred bytes against the frozen manifest before loading the engine.
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
assert sha(root / 'manifest.json') == '19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c'
m = json.loads((root / 'manifest.json').read_text())
assert m['set'] == 's1-dev-longchain'
for name, digest in m['artifacts'].items():
    assert sha(root / name) == digest, name
    print('DATA_SHA', digest, name, flush=True)
c = json.loads((root / 'cohort.json').read_text())
assert c['n_chains'] == 311 and c['n_requests'] == 5601
print('DATA_READY', c['set'], c['n_chains'], c['n_requests'], c['cohort_sha256'], flush=True)
PY
sha256sum "$AX/verify_kit/level_verdict.py" "$AX/verify_kit/run_dev_checked.py" \
  "$AX/verify_kit/score_formal.py" "$AX/verify_kit/metrics_sampler.py" \
  "$AX/s1/s1-dev/run_dev.py" "$AX/bin/scripts/pod/lib.sh" \
  "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh" || exit 2
# Freeze the complete 0923a-equivalent stack, including 121, on the pod as well.
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
PATCH_SHA256
) || exit 2
# Original preflight + warmup, checked flush, replay, and scoring all use this dataset.
source "$AX/bin/scripts/pod/jobs/dev_ladder_template.sh"
