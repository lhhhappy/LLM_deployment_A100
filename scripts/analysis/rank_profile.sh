#!/usr/bin/env bash
# Per-component GPU time for ONE TP8 rank of GLM-5.3-Flash, using SGLang's real model code (random weights).
# Works on the dev box and inside the L2 pod (same script => rerun on the real image). Needs a free GPU.
# Usage: rank_profile.sh <sglang_src_parent (has sglang/)> <full_config.json> <tokenizer_dir> <out_dir> [input_len] [batch]
set -euo pipefail
SRC=$1; CFG=$2; TOK=$3; OUT=$4; IN=${5:-8192}; BS=${6:-1}
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p $OUT
${PY:-python3} $HERE/make_rank_model.py $CFG $TOK $OUT/model
export PYTHONPATH=$SRC${PYTHONPATH:+:$PYTHONPATH} SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_TORCH_PROFILER_DIR=$OUT/trace
${PY:-python3} -m sglang.benchmark.one_batch --model-path $OUT/model --load-format dummy --tp-size 1 --page-size 64 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --mamba-radix-cache-strategy extra_buffer \
  --batch-size $BS --input-len $IN --output-len 8 --mem-fraction-static 0.85 --chunked-prefill-size $IN \
  --profile --profile-activities CPU GPU ${EXTRA_ARGS:-} 2>&1 | tee $OUT/one_batch.log
${PY:-python3} $HERE/component_table.py $OUT/trace | tee $OUT/components.txt
