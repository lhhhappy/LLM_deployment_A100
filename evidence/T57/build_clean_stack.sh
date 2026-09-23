#!/usr/bin/env bash
# T57 (Claude, 2026-09-24): derive the one-version-per-mechanism patch stack from the old stack and prove that the final
# source tree of every recorded configuration is byte-identical. CPU only. Run from the repo root at the commit BEFORE
# the T57 file moves (needs the old patch files). Output patches go to $W/new/.
set -euo pipefail
W=${W:-/tmp/t57}; PS="python3 scripts/patch_stack.py"; mkdir -p $W/new
OLD_S0="000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 140-kda-dual-snapshot drafts/120-sched-protect-chain-v2"
OLD_A="000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 114-indexer-row-shard 140-kda-dual-snapshot 120-sched-protect-chain 130-async-tokenize 150-startup-warmup 160-nextn-sm80 170-glm-bcg-prefill"
OLD_DCP="000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 114-indexer-row-shard 115-sm80-sparse-attn-many-heads 116-dcp-dsa-address"
# reference trees (old stack)
$PS apply $W/old_s0 $OLD_S0; $PS apply $W/old_a $OLD_A; $PS apply $W/old_dcp $OLD_DCP
# intermediate trees for folding
$PS apply $W/t_000 000-interface-compliance
$PS apply $W/t_101 000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial
$PS apply $W/t_pre110 000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv
$PS apply $W/t_post113 000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer
$PS apply $W/t_pre115 000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 114-indexer-row-shard
$PS apply $W/s0_v3 000-interface-compliance 101-d1v12-on-base 105-role-split-single-partial 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 112-sm80-indexer-kernels 113-sm80-prefill-indexer 140-kda-dual-snapshot 120-sched-protect-chain
# folded patches
$PS make $W/t_000 $W/t_101 $W/new/101-role-boundary-split.patch srt/managers/schedule_policy.py
$PS make $W/t_pre110 $W/t_post113 $W/new/110-sm80-dsa-indexer.patch \
  srt/layers/attention/dsa_backend.py srt/layers/attention/dsa/dsa_indexer.py srt/layers/attention/dsa/dsa_indexer_kpool.py \
  srt/layers/attention/dsa/kpool_fp8_index.py kernels/ops/attention/dsa/triton_kernel.py srt/layers/attention/dsa/sm80_deep_gemm.py \
  kernels/ops/attention/dsa/ax_soft_fp8.py srt/layers/attention/dsa/sm80_indexer_kernels.py
$PS make $W/t_pre115 $W/old_dcp $W/new/115-dcp-sm80.patch \
  kernels/ops/attention/dsa/tilelang_kernel.py kernels/ops/kvcache/mla_buffer.py srt/layers/attention/dsa_backend.py \
  srt/mem_cache/kv_cache_configurator.py srt/model_executor/pool_configurator.py
cp patches/drafts/120-sched-protect-chain-v2.patch $W/new/120-sched-protect-chain.patch
$PS make $W/old_s0 $W/s0_v3 $W/new/121-sched-cap-while-decoding.patch srt/managers/schedule_policy.py srt/managers/scheduler.py
