#!/usr/bin/env bash
# Exclusive queue job; one reviewed engine, 240 s request cap plus startup.
set -o pipefail
G_COMMIT=c4d01e5661c2d35e76fcf2e209d61624a9f04f58
G_EXPECT="124m=on 132=on 131_sync=rank0 131_chunk=auto 117=on 118=off 119=off 120=on 122=off 123=off 124=on 125=on 126=off 140=off 180=on spec=- dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1 128p=on 131=on 128=off dcp_local=off dcp_local_max=0 dcp_local_large=0"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --max-running-requests 48 --cuda-graph-max-bs 48 --prefill-decode-interval 2 --chunked-prefill-size 16384 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_DEADLINE_COLD_S=8 SGLANG_AX_MULTI_ROUND_PARK=1 SGLANG_AX_DEADLINE_MAX_WAIT_S=600 SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S=120 IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD=4096 SGLANG_AX_CHAIN_RISK_CHUNK=0 SGLANG_AX_SCHED_COLD_CAP_MAX=0 SGLANG_AX_DEADLINE_CHAIN_FIRST=1 SGLANG_AX_DEADLINE_LOAD=1.05 SGLANG_AX_CHAIN_RISK_INTERVAL=1 SGLANG_AX_PREFIX_PRODUCER=1 SGLANG_AX_DEADLINE_FAMILY=0 SGLANG_AX_PREFIX_TRACE_S=120 SGLANG_AX_PREFIX_TRACE_ROUNDS=2048 SGLANG_AX_DEADLINE_FREEZE_CLASS=1 SGLANG_AX_DSA_SPARSE_TRITON=0 SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=2048 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=16384 SGLANG_AX_DEADLINE_TIERS=1 SGLANG_AX_DEADLINE_WARM_S=15 SGLANG_AX_BACKLOG_RELIEF=1 SGLANG_AX_BACKLOG_COLD_CAP=16384 SGLANG_AX_BACKLOG_INTERVAL=0 SGLANG_AX_BACKLOG_HIGH_S=15 SGLANG_AX_BACKLOG_LOW_S=5 SGLANG_AX_BACKLOG_MAX_SLOW=80 SGLANG_AX_BACKLOG_GATE=0.10 SGLANG_AX_PACE_TPOT=0 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1"

source "$AX/bin/scripts/pod/lib.sh"
prepare_src "$G_COMMIT" || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
unset SGLANG_AX_DCP_LOCAL_EXTEND_LARGE_MAX SGLANG_AX_DCP_LOCAL_EXTEND_MAX_TOKENS SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
export $G_ENV
ensure_engine "$G_COMMIT" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
# Same receipt rule as dev_ladder_template; this job sends no harness requests.
mech=$(grep -h "\[ax\] mechanisms:" "$RUN_DIR/server.log" | tail -1); mech=" ${mech#*mechanisms: }"
for want in $G_EXPECT; do
  k=${want%%=*}; v=${want#*=}; got=$(grep -o " $k=[^ ]*" <<<"$mech" | head -1 | cut -d= -f2)
  case "$got" in "$v"|"$v":*) ;; *) echo "MECHANISMS MISMATCH $k expected=$v got=${got:-missing}"; exit 2 ;; esac
done
echo "MECHANISMS OK expected=[$G_EXPECT] engine=[${mech# }]"
echo "STATE_ONLY cold_budget=8 request_budget=240 startup_seconds=$SECONDS args=[$G_ARGS] env=[$G_ENV]"
# Synthetic input IDs test scheduler/KV lifecycle, not accuracy or SLO.
# The N30 ON/OFF jobs keep their normal 30 s cold budget and capability gate.
python3 -B "$AX/verify_kit/multiround_park_probe.py" run --mode on \
  --base-url "http://127.0.0.1:$PORT" --server-log "$RUN_DIR/server.log" --out "$RUN_DIR/probe"
