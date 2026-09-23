#!/usr/bin/env bash
# Dry-run the pod verification suite on the 2xA100 dev box with a pod-like layout (AX dir, src/<name>, verify_kit).
# Proves the suite itself works; the pod run is still required (different image stack).
set -euo pipefail
R=/sjtu/linhang/arena/repo; AX=/sjtu/linhang/arena/runs/verify_sim; rm -rf $AX; mkdir -p $AX/src/bverify
cp -a $R/build/base_exact/sglang $AX/src/bverify/
for p in 000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin \
         112-sm80-indexer-kernels 113-sm80-prefill-indexer 140-kda-dual-snapshot 120-sched-protect-chain 130-async-tokenize 150-startup-warmup 160-nextn-sm80; do
  patch -p3 -d $AX/src/bverify/sglang --fuzz=0 -s < $R/patches/$p.patch; echo "$p" >> $AX/src/bverify/PATCHES; done
cp -a $R/build/verify_kit $AX/verify_kit
source /sjtu/linhang/arena/env.sh >/dev/null 2>&1
export CUDA_HOME=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:/sjtu/linhang/arena/env/sgl/bin:/sjtu/linhang/arena/env/m0/bin:$PATH
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:/sjtu/linhang/arena/cache/T48/deps/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH_EXTRA=/sjtu/linhang/arena/cache/T48/deps
export TRITON_CACHE_DIR=$AX/cache/triton TILELANG_CACHE_DIR=$AX/cache/tilelang SGLANG_JIT_CACHE_DIR=$AX/cache/jit SGLANG_CACHE_DIR=$AX/cache/sgl
AX=$AX bash $AX/verify_kit/run_verify.sh bverify ${1:-0}
