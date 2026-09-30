# Source this file from the A100 project directory.
# Node (playground CLI) ignores HTTP(S)_PROXY unless this is set
export NODE_USE_ENV_PROXY=1
export NODE_NO_WARNINGS=1
export ARENA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export CHALLENGE_ID=llm-challenge-arena-v1
# GPU dev box: `ssh GPU` (via ~/.ssh/config, tunneled through local proxy 127.0.0.1:7890)
# GPU box work root (user-designated): /sjtu/linhang/arena/{env,models,code,runs,cache}
export GPU_ROOT=/sjtu/linhang/arena
# On the GPU box: `source /sjtu/linhang/arena/env.sh` (pip mirror without proxy, caches on /sjtu, HF via hf-mirror)
