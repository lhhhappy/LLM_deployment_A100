#!/usr/bin/env bash
# vLLM route on the GPU dev box (2xA100). Everything lives under /sjtu/linhang/arena/vllm:
#   env/      venv with the official wheel's dependency set: run tools/vllm/setup_env.sh there,
#             then `install` below (setup_env reinstalls the wheel over an editable install)
#   wheels/   the official prebuilt wheel of the base commit
#   src/      copy of engine/vllm, installed editable on top of the wheel's binaries
#   tools/    copy of scripts/vllm and the harness pieces the tools import
#   runs/     probe outputs
#
#   scripts/vllm/devbox.sh sync            copy engine/vllm and tools to the dev box
#   scripts/vllm/devbox.sh install         editable install of src/ reusing the wheel's compiled parts
#   scripts/vllm/devbox.sh test [args]     pytest in the vLLM env on the dev box, GPUs hidden
#   scripts/vllm/devbox.sh gtest [args]    pytest with GPUs ${GPUS:-0} visible
#   scripts/vllm/devbox.sh serve <run> K=V  start tools/vllm/serve.sh detached (runs/<run>/server.log)
#   scripts/vllm/devbox.sh stop <run>      SIGTERM the server of <run> and show leftover GPU processes
#   scripts/vllm/devbox.sh sh '<cmd>'      run a command with the vLLM env exported
set -euo pipefail
cd "$(dirname "$0")/../.."
V=/sjtu/linhang/arena/vllm
WHL=vllm-0.30.1rc1.dev114+ga811738a6-cp38-abi3-manylinux_2_28_x86_64.whl
G=scripts/gssh

# Environment for every remote command: the venv, CUDA forward-compat driver libs (the dev box
# driver is 535 / CUDA 12.2, the wheel is built for CUDA 13.0), CUDA_HOME = the venv's CUDA 13.0
# toolkit arranged by setup_env.sh (FlashInfer JIT-builds its sampling kernels), caches under
# the arena tree (FlashInfer would otherwise write to the nearly full root filesystem).
remote_env="source /sjtu/linhang/arena/env.sh; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY PIP_PROXY;
export V=$V PIP_CONFIG_FILE=/dev/null VIRTUAL_ENV=$V/env PATH=$V/env/bin:\$PATH
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export CUDA_HOME=$V/cuda-home PATH=$V/cuda-home/bin:\$PATH
export TMPDIR=$V/cache/tmp UV_CACHE_DIR=$V/cache/uv VLLM_CACHE_ROOT=$V/cache/vllm TRITON_CACHE_DIR=$V/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$V/cache/inductor TILELANG_CACHE_DIR=$V/cache/tilelang HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=$V/cache/flashinfer
cd $V"

case "${1:-}" in
  sync)
    # Copy exactly the files git knows under engine/vllm (tracked + new, not ignored). Remote
    # deletions are limited to files a previous sync copied (src/.synced-files), so whatever the
    # precompiled install extracted into the tree (*.so, vllm-rs, vendored kernels) is never touched.
    GSSH_TIMEOUT=60 "$G" "mkdir -p $V/src $V/tools/vllm $V/tools/harness $V/models/glm53-config"
    list=$(mktemp)
    git ls-files -co --exclude-standard -- engine/vllm | sed 's#^engine/vllm/##' > "$list"
    rsync -a --files-from="$list" engine/vllm/ GPU:$V/src/
    rsync -a "$list" GPU:$V/src/.synced-files.new
    rm -f "$list"
    GSSH_TIMEOUT=120 "$G" "cd $V/src && python3 -c '
import os
new = set(open(\".synced-files.new\").read().split(\"\\n\")) - {\"\"}
old = set(open(\".synced-files\").read().split(\"\\n\")) - {\"\"} if os.path.exists(\".synced-files\") else set()
gone = sorted(p for p in old - new if os.path.isfile(p))
for p in gone:
    os.remove(p)
os.replace(\".synced-files.new\", \".synced-files\")
print(\"removed stale files:\", len(gone))
'"
    rsync -a --delete --exclude=__pycache__ scripts/vllm/ GPU:$V/tools/vllm/
    rsync -a --delete --exclude=__pycache__ s1-dev/harness/ GPU:$V/tools/harness/
    rsync -a s1-dev/glm_tok/ GPU:$V/models/glm53-config/
    echo "synced engine/vllm @ $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- engine/vllm || echo '+dirty')"
    ;;
  install)
    # The synced tree has no .git, so the version comes from VLLM_VERSION_OVERRIDE: the base
    # wheel's version with an `.ax` local label marking our source.
    GSSH_TIMEOUT=1800 "$G" "$remote_env
      uv pip install --python $V/env/bin/python --index-url https://mirrors.aliyun.com/pypi/simple \
        'cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' \
        'setuptools-rust>=1.9.0' wheel 'jinja2>=3.1.6' pytest pytest-asyncio pytest-timeout pytest-forked httpx tblib \
        2>&1 | tail -2
      VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=$V/wheels/$WHL \
      VLLM_VERSION_OVERRIDE=0.30.1rc1.dev114+ga811738a6.ax \
        uv pip install --python $V/env/bin/python --no-deps --no-build-isolation -e $V/src 2>&1 | tail -5
      cd /tmp && python -c 'import vllm, vllm._C_stable_libtorch; print(vllm.__version__, vllm.__file__)'"
    ;;
  test)
    shift
    GSSH_TIMEOUT=1800 "$G" "$remote_env; cd $V/src; CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider $*"
    ;;
  gtest)
    shift
    GSSH_TIMEOUT=3600 "$G" "$remote_env; cd $V/src; CUDA_VISIBLE_DEVICES=${GPUS:-0} python -m pytest -q -p no:cacheprovider $*"
    ;;
  serve)
    # scripts/vllm/devbox.sh serve <run> KEY=VAL...  -> tools/vllm/serve.sh detached in tmux,
    # log runs/<run>/server.log, pid runs/<run>/serve.pid (serve.sh execs vllm).
    run=$2; shift 2
    GSSH_TIMEOUT=60 "$G" "mkdir -p $V/runs/$run"
    scripts/gjob run "vllm-$run" "$remote_env; cd $V/runs/$run; env $* bash -c 'echo \$\$ > serve.pid; exec bash $V/tools/vllm/serve.sh' > server.log 2>&1"
    ;;
  stop)
    # SIGTERM lets vllm shut its workers down; report if GPU processes remain.
    run=$2
    GSSH_TIMEOUT=300 "$G" "cd $V/runs/$run && pid=\$(cat serve.pid) && kill -TERM \$pid 2>/dev/null;
      for i in \$(seq 60); do kill -0 \$pid 2>/dev/null || break; sleep 2; done
      kill -0 \$pid 2>/dev/null && echo 'still running' || echo stopped
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
    ;;
  sh)
    shift
    GSSH_TIMEOUT=${GSSH_TIMEOUT:-1800} "$G" "$remote_env; $*"
    ;;
  *)
    sed -n 2,13p "$0"; exit 2 ;;
esac
