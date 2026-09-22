#!/usr/bin/env bash
# Development A100 operator tests only. Run from T48's dedicated directory.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
cd /sjtu/linhang/arena/code/T48
export CUDA_HOME=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:/sjtu/linhang/arena/env/sgl/bin:$PATH
export SGLANG_CACHE_DIR=/sjtu/linhang/arena/cache/T48/sglang
export SGLANG_JIT_CACHE_DIR=/sjtu/linhang/arena/cache/T48/sglang/jit
export CUDA_VISIBLE_DEVICES=${T48_GPU:-0}
export PYTHONPATH=/sjtu/linhang/arena/cache/T48/deps${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:/sjtu/linhang/arena/cache/T48/deps/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TRITON_CACHE_DIR=/sjtu/linhang/arena/cache/T48/triton
export TILELANG_CACHE_DIR=/sjtu/linhang/arena/cache/T48/tilelang
nvidia-smi
if [[ ${1:-} == --resolve ]]; then
  exec /sjtu/linhang/arena/env/m0/bin/python -u test_mtp_resolve_160.py candidate model
elif [[ ${1:-} == --marlin ]]; then
  exec /sjtu/linhang/arena/env/m0/bin/python -u test_mtp_marlin_160.py candidate
fi
exec /sjtu/linhang/arena/env/m0/bin/python -u test_mtp_sm80_160.py --source candidate/sglang "$@"
