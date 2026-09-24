# 053: resume the already-loaded 052r traced baseline; preserve its completed first request.
G_NAME=off_a_numtrace_r
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096"

source "$AX/bin/scripts/pod/lib.sh"
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export $G_ENV
unset SGLANG_AX_PACE_TPOT SGLANG_AX_KDA_FUSE_PROJ SGLANG_AX_MOE_FUSE_SWIGLU
export SGLANG_AX_NUMTRACE_DIR="$AX/runs/052r-official_a_numtrace/trace"
echo "58e9ff2066174488a91c1b957bb25f7d4da0363528067811b82d1eab9f4ad540  $AX/verify_kit/numcheck_baseline_diag.py" | sha256sum -c - || exit 2
echo "d4d0ce7865020d8e923e109f3ee1fa6af728da5f89dd345c3befe354206a5d4a  $AX/src/$G_NAME/sglang/srt/models/ax_numtrace.py" | sha256sum -c - || exit 2

# 052r completed one cold37 request before its next flush met remaining startup
# warmup. Preserve all eight complete traces in this run before a fresh series.
python3 - "$SGLANG_AX_NUMTRACE_DIR" "$RUN_DIR/prior-trace" <<'PYARCHIVE'
import json,sys,shutil
from pathlib import Path
src,dst=map(Path,sys.argv[1:]); files=sorted(src.glob('rank-*.jsonl'))
if {p.name for p in files}!={f'rank-{r}.jsonl' for r in range(8)}:raise RuntimeError('unexpected prior trace rank set')
for path in files:
 rows=[json.loads(x) for x in path.read_text().splitlines()]
 begins=[r for r in rows if r['kind']=='begin']; ends=[r for r in rows if r['kind']=='end']
 if len(begins)!=1 or len(ends)!=1 or begins[0]['tokens']!=37 or ends[0]['missing_layers']:raise RuntimeError('prior trace differs from reviewed single cold37 request')
if dst.exists():raise RuntimeError('archive destination already exists')
dst.mkdir()
for path in files:shutil.move(str(path),str(dst/path.name))
print('PRESERVED_PRIOR_TRACE 8 ranks, one cold37 request')
PYARCHIVE
[ "$?" = 0 ] || exit 2
prepare_src "$G_NAME" $G_PATCHES || exit 2
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 3
curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$RUN_DIR/server_info.json" || exit 3
python3 "$AX/verify_kit/numcheck_baseline_diag.py" "$RUN_DIR/diag" --mode first > "$RUN_DIR/diag.log" 2>&1 || { tail -30 "$RUN_DIR/diag.log"; exit 4; }
cat "$RUN_DIR/diag.log"
python3 "$AX/verify_kit/compare_numtrace.py" "$SGLANG_AX_NUMTRACE_DIR" --output "$RUN_DIR/trace-comparison.json" > "$RUN_DIR/trace-comparison.log" 2>&1 || { cat "$RUN_DIR/trace-comparison.log"; exit 5; }
cat "$RUN_DIR/trace-comparison.log"
echo 'BASELINE_TRACE_COMPLETE: all TP8 records present; diagnostic only'
