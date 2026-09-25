#!/usr/bin/env bash
# Start `vllm serve` with the contest interface and our standard flags.
# Environment (defaults in brackets):
#   MODEL (required)  model dir          TP [8]            PORT [8000]        HOST [0.0.0.0]
#   MAX_MODEL_LEN [524288]               GPU_MEM_UTIL [0.92]
#   MAX_NUM_SEQS [16]                    MAX_NUM_BATCHED_TOKENS [8192]
#   LOAD_FORMAT [auto]                   COMPILATION_CONFIG [{"cudagraph_mode":"FULL_AND_PIECEWISE"}]
#   EXTRA_ARGS                           appended verbatim (word-split)
# The flags follow task.md's example submission for this engine family: MTP with 3 draft tokens,
# prefix caching, custom all-reduce off, NCCL Ring/Simple so full-graph replay reissues the
# captured collectives, GLM tool and reasoning parsers. /generate and /flush_cache come from the
# generate_compat endpoint plugin, which only loads when named in VLLM_PLUGINS.
set -euo pipefail
: "${MODEL:?MODEL must name the model dir}"
export VLLM_PLUGINS=generate_compat,lora_filesystem_resolver,lora_hf_hub_resolver
export NCCL_ALGO=Ring NCCL_PROTO=Simple

args=(
  serve --model "$MODEL"
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --served-model-name glm-5-3-flash
  --tensor-parallel-size "${TP:-8}"
  --max-model-len "${MAX_MODEL_LEN:-524288}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}"
  --max-num-seqs "${MAX_NUM_SEQS:-16}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
  --enable-prefix-caching --enable-prompt-tokens-details --disable-custom-all-reduce
  --compilation-config "${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}}"
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
  --load-format "${LOAD_FORMAT:-auto}"
)
# shellcheck disable=SC2206
extra=(${EXTRA_ARGS:-})
echo "[serve] $(date -u +%FT%TZ) vllm ${args[*]} ${extra[*]}"
python -c 'import vllm; print("[serve] vllm", vllm.__version__, vllm.__file__)'
exec vllm "${args[@]}" "${extra[@]}"
