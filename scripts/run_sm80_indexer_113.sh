#!/usr/bin/env bash
# T44: authorized GPU development directory, existing environment, no installation.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
cd /sjtu/linhang/arena/code/T44
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${T44_GPU:-0}
exec /sjtu/linhang/arena/env/m0/bin/python -u "$@"
