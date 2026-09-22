#!/usr/bin/env bash
# Authorized development-machine operator tests only, no server or model load.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
cd /sjtu/linhang/arena/code/T45
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${T45_GPU:-1}
nvidia-smi
exec /sjtu/linhang/arena/env/m0/bin/python -u test_kda_snapshot_140.py --source candidate/sglang "$@"
