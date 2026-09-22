#!/usr/bin/env bash
# Run only operator tests on the authorized 2-A100 development machine.
# Install no packages; all code, caches and evidence remain below arena.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
cd /sjtu/linhang/arena/code/T43
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${T43_GPU:-0}
nvidia-smi
exec /sjtu/linhang/arena/env/m0/bin/python -u test_sm80_indexer_112.py "$@"
