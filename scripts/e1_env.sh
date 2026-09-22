#!/usr/bin/env bash
# E1-only environment on the GPU development host; never changes system drivers.
# Source the shared environment first, then override process-local cache/network settings.
source /sjtu/linhang/arena/env.sh
export PIP_CONFIG_FILE=/dev/null PIP_NO_INPUT=1
# Exact pinned packages cuda-toolkit==13.0.3 and sglang-kernel==0.4.7 were
# missing from the mirror. Keep it as primary and allow official PyPI fallback.
# volces mirror lacks sglang-kernel and pypi.org direct is ~10 kB/s; aliyun measured ~18 MB/s (2026-09-23)
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ PIP_TRUSTED_HOST=mirrors.aliyun.com
export TMPDIR=/sjtu/linhang/arena/cache/tmp
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
export XDG_CACHE_HOME=/sjtu/linhang/arena/cache/xdg
export CUDA_CACHE_PATH=/sjtu/linhang/arena/cache/cuda-jit
export TORCH_EXTENSIONS_DIR=/sjtu/linhang/arena/cache/torch-extensions
export FLASHINFER_WORKSPACE_BASE=/sjtu/linhang/arena/cache/flashinfer
export SGLANG_CACHE_DIR=/sjtu/linhang/arena/cache/sglang
# Unlike the other caches, v0.5.20 JIT does not inherit SGLANG_CACHE_DIR.
export SGLANG_JIT_CACHE_DIR=/sjtu/linhang/arena/cache/sglang/jit
export PYTHONPYCACHEPREFIX=/sjtu/linhang/arena/cache/pycache
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/sjtu/linhang/arena/cache/uv
export UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=4 UV_LINK_MODE=copy
export PATH=/sjtu/linhang/arena/env/sgl/bin:$PATH
# CUDA 13 pip wheels place nvcc, headers and libraries under one cu13 tree.
export CUDA_HOME=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
# Extracted user-mode forward-compatibility libraries; no apt install/driver changes.
if [[ -d /sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat ]]; then
    export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
