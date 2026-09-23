set -e
R=/sjtu/linhang/arena/repo; T=/sjtu/linhang/arena/runs/p114; rm -rf $T; mkdir -p $T/src/b114
cp -a $R/build/base_exact/sglang $T/src/b114/
for p in 000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 114-indexer-row-shard; do patch -p3 -d $T/src/b114/sglang --fuzz=0 -s < $R/patches/$p.patch; done
source /sjtu/linhang/arena/env.sh >/dev/null 2>&1
export CUDA_HOME=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:/sjtu/linhang/arena/env/sgl/bin:/sjtu/linhang/arena/env/m0/bin:$PATH
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:/sjtu/linhang/arena/cache/T48/deps/z3/lib
export NCCL_CUMEM_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$T/src/b114:/sjtu/linhang/arena/cache/T48/deps SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export TRITON_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/triton SGLANG_JIT_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/jit SGLANG_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/sgl TILELANG_CACHE_DIR=/sjtu/linhang/arena/runs/rankprof/cache/tl
PY=/sjtu/linhang/arena/env/m0/bin/python
cd $R
for flag in 0 1; do
  SGLANG_AX_INDEXER_ROW_SHARD=$flag AX_CHECK_OUT=$T/logits_$flag.pt timeout 1500 $PY scripts/analysis/extend_check.py \
    --model-path /sjtu/linhang/arena/runs/rankprof/p8192/model --load-format dummy --tp-size 2 --page-size 64 \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --mamba-radix-cache-strategy extra_buffer \
    --batch-size 1 --input-len ${AX_IN:-16384} --output-len 1 --chunked-prefill-size ${AX_IN:-16384} --mem-fraction-static 0.7 \
    --cuda-graph-backend-decode disabled --disable-custom-all-reduce > $T/run_$flag.log 2>&1 || true
  grep "AX_CHECK\|row-shard active\|Error" $T/run_$flag.log | tail -3
done
$PY -c "import torch;a=torch.load('$T/logits_0.pt');b=torch.load('$T/logits_1.pt');print('shape',tuple(a.shape),'bit_exact',torch.equal(a,b),'max_abs',(a-b).abs().max().item())"
