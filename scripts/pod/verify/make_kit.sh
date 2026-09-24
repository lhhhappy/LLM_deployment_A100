#!/usr/bin/env bash
# Collect the pod verification kit into build/verify_kit (pushed to $AX/verify_kit by scripts/pod/bootstrap).
set -euo pipefail; cd "$(dirname "$0")/../../.."
K=build/verify_kit; rm -rf $K; mkdir -p $K
cp scripts/pod/verify/run_verify.sh tests/gpu/test_sm80_indexer_112.py tests/gpu/test_sm80_indexer_113.py \
   tests/gpu/test_sm80_indexer_cache_47.py tests/gpu/test_kda_snapshot_140.py tests/gpu/test_mtp_sm80_160.py \
   build/p110/test_tilelang_sparse_sm80.py build/p111/test_fp8_moe_marlin_sm80.py $K/
cp build/p110/sm80_deep_gemm.py $K/oracle110_sm80_deep_gemm.py     # 110 torch shim = numeric oracle for 112/113
cp scripts/pod/verify/numcheck.py scripts/pod/verify/numcheck_cmp.py \
   scripts/pod/verify/numcheck_baseline_diag.py \
   scripts/pod/verify/install_numtrace.py scripts/pod/verify/numtrace_helper.py \
   scripts/pod/verify/cap_smoke_body.sh scripts/pod/verify/metrics_sampler.py scripts/pod/verify/prof_ledger.py $K/
cp scripts/pod/verify/level_verdict.py scripts/score_formal.py $K/   # verdict path (T54): must exist, no silent skip
if command -v sha256sum >/dev/null 2>&1; then sha256sum $K/* > $K/SHA256SUMS
else shasum -a 256 $K/* > $K/SHA256SUMS; fi
ls $K
