#!/usr/bin/env bash
# T47 operator validation only, existing development environment and task-local caches.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
cd /sjtu/linhang/arena/code/T47
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/sjtu/linhang/arena/cache/T47/${T47_CACHE:-validation}
exec /sjtu/linhang/arena/env/m0/bin/python -u "$@"
