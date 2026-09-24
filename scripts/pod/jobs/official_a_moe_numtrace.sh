# 054: hash full first-MoE weights and all internal stages on formal A; diagnostic only.
# Starts a copied baseline with diagnostic hooks; serialized one-token repetitions.
# No SLO/ability/numerical-equivalence verdict is inferred.
G_NAME=off_a_moetrace
G_PATCHES="000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch 120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch 150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096"

source "$AX/bin/scripts/pod/lib.sh"
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export $G_ENV
unset SGLANG_AX_PACE_TPOT SGLANG_AX_KDA_FUSE_PROJ SGLANG_AX_MOE_FUSE_SWIGLU
printf '%s\n' 'DIAGNOSTIC_ONLY formal_A stage trace; synchronization affects execution timing'
echo "58e9ff2066174488a91c1b957bb25f7d4da0363528067811b82d1eab9f4ad540  $AX/verify_kit/numcheck_baseline_diag.py" | sha256sum -c - || exit 2
echo "efb9f59cd9fcc7af6f9060b8064c5be7e45f4e0ba27f3ab317bd7546cf9c3b12  $AX/verify_kit/install_numtrace.py" | sha256sum -c - || exit 2
echo "c9c00716afa624f2a7f0c3abc4d2f8571e535cb0bdcb6ff84fa3e302bf29776b  $AX/verify_kit/install_moe_numtrace.py" | sha256sum -c - || exit 2
echo "2c356f9d839555434716d36f187a2dcd63d450c9d0225acd53e1ec65bdbc278d  $AX/verify_kit/numtrace_helper.py" | sha256sum -c - || exit 2
echo "59797f2935630cef2349c249a105ab745caadff042f58f4da0ea77effaea7234  $AX/verify_kit/compare_numtrace.py" | sha256sum -c - || exit 2
# CPU-only edge checks against the actual pod torch build before instrumenting.
CUDA_VISIBLE_DEVICES='' python3 - "$AX/verify_kit/numtrace_helper.py" <<'PYCPU'
import importlib.util,json,sys,torch,os,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('trace',sys.argv[1]); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=torch.tensor(1.0,dtype=torch.bfloat16)
assert m._fingerprint(x)['sha256']==m._fingerprint(x.clone())['sha256']
assert m._fingerprint(x)['sha256']!=m._fingerprint(x+1)['sha256']
r=m._fingerprint(torch.tensor([float('nan'),float('inf')]))
json.dumps(r,allow_nan=False)
net=torch.nn.Module();net.register_parameter('scalar',torch.nn.Parameter(torch.tensor(1.0)));net.register_parameter('empty',torch.nn.Parameter(torch.empty(0)))
assert m._parameter_digest(net)['parameter_tensors']==2
net.start_layer=0;net.end_layer=0
with tempfile.TemporaryDirectory(dir=os.environ['RUN_DIR']) as directory:
    Path(directory,'ARMED').touch()
    ids=torch.arange(37)
    batch=SimpleNamespace(input_ids=ids,forward_mode=SimpleNamespace(name='EXTEND'))
    with patch.dict(os.environ,{'SGLANG_AX_NUMTRACE_DIR':directory}), patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
        m.begin(net,None,ids,batch,torch.zeros(37,4))
        m.finish(torch.zeros(37,4),None)
    records=[json.loads(x) for x in Path(directory,'rank-0.jsonl').read_text().splitlines()]
    assert records[0]['input_id_source']=='forward_batch' and records[-1]['kind']=='end'
print('NUMTRACE_CPU_EDGES_PASS')
PYCPU
[ "$?" = 0 ] || exit 2
prepare_src "$G_NAME" $G_PATCHES || exit 2
python3 "$AX/verify_kit/install_moe_numtrace.py" "$AX/src/$G_NAME/sglang" || exit 2
export SGLANG_AX_NUMTRACE_DIR="$RUN_DIR/trace"
export SGLANG_AX_NUMTRACE_DUMP_LAYER=3
( cd "$AX/patches" && sha256sum $G_PATCHES ) | sed 's/^/PATCH_SHA /'
ensure_engine "$G_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 3
# The HTTP endpoint is available before native warmup finishes. Arm only after
# the native completion receipt, so warmup cannot be mistaken for a repeat.
python3 - "$RUN_DIR/server.log" <<'PYREADY'
import pathlib,sys,time
path=pathlib.Path(sys.argv[1]);deadline=time.monotonic()+300
while time.monotonic()<deadline:
    if 'The server is fired up and ready to roll!' in path.read_text():break
    time.sleep(1)
else:raise RuntimeError('native startup warmup did not finish')
print('NATIVE_WARMUP_COMPLETE')
PYREADY
[ "$?" = 0 ] || exit 3
curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$RUN_DIR/server_info.json" || exit 3
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$RUN_DIR/gpu_memory.csv" || exit 3
mkdir -p "$RUN_DIR/trace" || exit 3
touch "$RUN_DIR/trace/ARMED" || exit 3
python3 "$AX/verify_kit/numcheck_baseline_diag.py" "$RUN_DIR/diag" --mode first > "$RUN_DIR/diag.log" 2>&1 || { tail -30 "$RUN_DIR/diag.log"; exit 4; }
cat "$RUN_DIR/diag.log"
python3 "$AX/verify_kit/compare_numtrace.py" "$RUN_DIR/trace" --require-moe --output "$RUN_DIR/trace-comparison.json" > "$RUN_DIR/trace-comparison.log" 2>&1 || { cat "$RUN_DIR/trace-comparison.log"; exit 5; }
cat "$RUN_DIR/trace-comparison.log"
echo 'BASELINE_TRACE_COMPLETE: all TP8 records present; diagnostic only'
