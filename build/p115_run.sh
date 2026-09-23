set -e
R=/sjtu/linhang/arena/repo; T=/sjtu/linhang/arena/runs/p115; rm -rf $T; mkdir -p $T/old $T/new
for side in old new; do cp -a $R/build/base_exact/sglang $T/$side/; done
for p in 000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 114-indexer-row-shard; do
  patch -p3 -d $T/old/sglang --fuzz=0 -s < $R/patches/$p.patch; patch -p3 -d $T/new/sglang --fuzz=0 -s < $R/patches/$p.patch; done
patch -p3 -d $T/new/sglang --fuzz=0 -s < $R/patches/115-sm80-sparse-attn-many-heads.patch
source /sjtu/linhang/arena/env.sh >/dev/null 2>&1
export CUDA_HOME=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13 PATH=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13/bin:/sjtu/linhang/arena/env/sgl/bin:/sjtu/linhang/arena/env/m0/bin:$PATH
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:/sjtu/linhang/arena/cache/T48/deps/z3/lib CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/sjtu/linhang/arena/cache/T48/deps TILELANG_CACHE_DIR=$T/tlcache
/sjtu/linhang/arena/env/m0/bin/python -u $R/build/p115_test.py $T/old $T/new 2>&1 | grep -E "^H=|old H|PASS|FAIL"
