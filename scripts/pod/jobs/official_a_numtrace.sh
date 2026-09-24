# 052: locate first divergent target-prefill stage on formal A; synchronous traces are diagnostic only.
# Starts a copied baseline with diagnostic hooks; serialized one-token repetitions.
# No SLO/ability/numerical-equivalence verdict is inferred.
G_NAME=off_a_numtrace
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096"

source "$AX/bin/scripts/pod/lib.sh"
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export $G_ENV
unset SGLANG_AX_PACE_TPOT SGLANG_AX_KDA_FUSE_PROJ SGLANG_AX_MOE_FUSE_SWIGLU
printf '%s\n' 'DIAGNOSTIC_ONLY formal_A stage trace; synchronization affects execution timing'
echo "d46832a6decafaa2bc62562845c09bc1b1b10d9dd7034a6535dd60eb356898ed  $AX/verify_kit/numcheck_baseline_diag.py" | sha256sum -c - || exit 2
echo "efb9f59cd9fcc7af6f9060b8064c5be7e45f4e0ba27f3ab317bd7546cf9c3b12  $AX/verify_kit/install_numtrace.py" | sha256sum -c - || exit 2
echo "53f8f7e33d1b7d3cb323270383ab1de0158ee67bde21e71a42dbd39b46f1b9d3  $AX/verify_kit/numtrace_helper.py" | sha256sum -c - || exit 2
# CPU-only edge checks against the actual pod torch build before instrumenting.
CUDA_VISIBLE_DEVICES='' python3 - "$AX/verify_kit/numtrace_helper.py" <<'PYCPU'
import importlib.util,json,sys,torch
spec=importlib.util.spec_from_file_location('trace',sys.argv[1]); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=torch.tensor(1.0,dtype=torch.bfloat16)
assert m._fingerprint(x)['sha256']==m._fingerprint(x.clone())['sha256']
assert m._fingerprint(x)['sha256']!=m._fingerprint(x+1)['sha256']
r=m._fingerprint(torch.tensor([float('nan'),float('inf')]))
json.dumps(r,allow_nan=False)
net=torch.nn.Module();net.register_parameter('scalar',torch.nn.Parameter(torch.tensor(1.0)));net.register_parameter('empty',torch.nn.Parameter(torch.empty(0)))
assert m._parameter_digest(net)['parameter_tensors']==2
print('NUMTRACE_CPU_EDGES_PASS')
PYCPU
[ "$?" = 0 ] || exit 2
prepare_src "$G_NAME" $G_PATCHES || exit 2
python3 "$AX/verify_kit/install_numtrace.py" "$AX/src/$G_NAME/sglang" || exit 2
export SGLANG_AX_NUMTRACE_DIR="$RUN_DIR/trace"
( cd "$AX/patches" && sha256sum $G_PATCHES ) | sed 's/^/PATCH_SHA /'
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 3
curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$RUN_DIR/server_info.json" || exit 3
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$RUN_DIR/gpu_memory.csv" || exit 3
mkdir -p "$RUN_DIR/trace" || exit 3
touch "$RUN_DIR/trace/ARMED" || exit 3
python3 "$AX/verify_kit/numcheck_baseline_diag.py" "$RUN_DIR/diag" --mode first > "$RUN_DIR/diag.log" 2>&1 || { tail -30 "$RUN_DIR/diag.log"; exit 4; }
cat "$RUN_DIR/diag.log"
echo 'BASELINE_DIAGNOSTIC_COMPLETE: inspect repeat differences; this is not an acceptance gate'
