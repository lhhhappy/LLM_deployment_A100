#!/usr/bin/env bash
# Collect the pod verification kit into build/verify_kit (pushed to $AX/verify_kit by scripts/pod/bootstrap).
set -euo pipefail; cd "$(dirname "$0")/../../.."
K=build/verify_kit; rm -rf $K; mkdir -p $K
cp scripts/pod/verify/run_verify.sh scripts/test_sm80_indexer_112.py scripts/test_sm80_indexer_113.py \
   scripts/test_sm80_indexer_cache_47.py scripts/test_kda_snapshot_140.py scripts/test_mtp_sm80_160.py \
   build/p110/test_tilelang_sparse_sm80.py build/p111/test_fp8_moe_marlin_sm80.py $K/
cp build/p110/sm80_deep_gemm.py $K/oracle110_sm80_deep_gemm.py     # 110 torch shim = numeric oracle for 112/113
[ -f scripts/pod/verify/bench_moe_int8.py ] && cp scripts/pod/verify/bench_moe_int8.py $K/
cp scripts/pod/verify/component_table.py scripts/pod/verify/coldprobe.py scripts/pod/verify/logstat.py scripts/pod/verify/analyze_run.py scripts/pod/verify/interference.py $K/ 2>/dev/null || true
sha256sum $K/* > $K/SHA256SUMS; ls $K
