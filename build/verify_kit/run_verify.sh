#!/usr/bin/env bash
# Re-run the dev-box kernel/numerics conclusions INSIDE the L2 pod, against the exact engine commit tree
# that the engine will serve (memory: rerun-on-real-machine). Single GPU, ~10–20 min, needs the GPUs free
# (run it before the engine starts). Usage (pod):  bash run_verify.sh <engine commit> [gpu]
# Output: $AX/verify/<ts>/{env.txt,<check>.log,summary.txt}; summary lines "PASS|FAIL <check> <seconds>".
set -u
AX=${AX:-/tmp/ax}; SRC=${1:?engine commit (prepare_src <commit> first)}; GPU=${2:-0}
KIT=$AX/verify_kit; TREE=$AX/src/$SRC/sglang
OUT=$AX/verify/$(date -u +%Y%m%dT%H%M%SZ)_$SRC; mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$AX/src/$SRC${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}
[ -d "$TREE" ] || { echo "no tree $TREE (run prepare_src first)"; exit 2; }
{ nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader
  python3 - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
for m in ("triton", "tilelang", "sgl_kernel", "flashinfer", "transformers"):
    try: print(m, importlib.import_module(m).__version__)
    except Exception as e: print(m, "n/a", type(e).__name__)
PY
  echo "ENGINE_COMMIT $(cat $AX/src/$SRC/COMMIT)"; } > $OUT/env.txt 2>&1
W=$OUT/work; mkdir -p $W; cp $KIT/*.py $W/
# 112/113 tests load colocated kernels copied from the served tree.
cp $TREE/srt/layers/attention/dsa/sm80_indexer_kernels.py $W/sm80_indexer_112.py
cp $TREE/srt/layers/attention/dsa/sm80_indexer_kernels.py $W/sm80_indexer_113.py
cp $KIT/oracle110_sm80_deep_gemm.py $W/sm80_deep_gemm.py
check() { local name=$1 t=$2; shift 2; local s=$(date +%s)
  ( cd $W && timeout $t "$@" ) > $OUT/$name.log 2>&1; local rc=$? d=$(( $(date +%s) - s ))
  if [ $rc -eq 0 ] && ! grep -qE "^FAIL|FAILED|Traceback|AssertionError" $OUT/$name.log; then echo "PASS $name ${d}s"; else echo "FAIL $name ${d}s rc=$rc"; fi | tee -a $OUT/summary.txt; }
check f57_tilelang_sparse   900  python3 test_tilelang_sparse_sm80.py $AX/src/$SRC
check f58_marlin_moe_fp8    900  python3 test_fp8_moe_marlin_sm80.py $AX/src/$SRC
check t43_indexer112_all   1800  python3 test_sm80_indexer_112.py --mode all
check t44_indexer113_all   1800  python3 test_sm80_indexer_113.py --mode all
check t47_no_recompile     1200  python3 test_sm80_indexer_cache_47.py
[ -f $TREE/srt/mem_cache/kda_dual_snapshot.py ] && check t45_kda_snapshot140 1800 python3 test_kda_snapshot_140.py --source $TREE
for g in kda kpool dsa indexer; do check t48_mtp160_$g 1200 python3 test_mtp_sm80_160.py --source $TREE --group $g; done
echo "== $OUT"; cat $OUT/summary.txt
