#!/usr/bin/env bash
# Isolated non-score profile after 130ezo02; reuse the live exact 47266 engine.
# Frozen diagnostic; independent of the following latency measurement.
set -u
G_COMMIT=bf6b66faf3ffbe10e593e858d3bbedd1c48cefb2
G_ARGS='--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --max-running-requests 48 --cuda-graph-max-bs 48 --max-mamba-cache-size 400 --enable-attn-tp-input-scattered --prefill-decode-interval 2 --chunked-prefill-size 16384 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87'
G_EXPECT="132=on 131_sync=rank0 131_chunk=auto 117=on 118=on 119=off 120=on 122=off 123=off 124=on 125=on 126=off 140=off 180=on spec=- dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_SM80_FP8_MOE_HUMMING=1 128p=on 131=on 128=off dcp_local=off dcp_local_max=0 dcp_local_large=0"
source "$AX/bin/scripts/pod/storage_env.sh" || exit 2
signature=$(cat "$AX/engine.sig") || exit 2
expected_prefix="$G_COMMIT | --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS | "
[[ "$signature" == "$expected_prefix"* ]] || { echo 'PROFILE INVALID: live engine is not frozen 47266'; exit 2; }
log_path=$(cat "$AX/engine_log_path") || exit 2
[ -f "$log_path" ] || { echo 'PROFILE INVALID: live server log missing'; exit 2; }
profile_out="$AX/codex/profiles/130ezo03-$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$profile_out" > "$RUN_DIR/profile-output.txt"
python3 -B "$AX/verify_kit/tp8_decode_profile_0930.py" \
  --root "$AX/codex/data/s1-dev-longchain-v5g-tail-rot150" \
  --tok-dir /mnt/models --harness-dir "$AX/s1/s1-dev/harness" \
  --base-url "http://127.0.0.1:${PORT:-30000}" \
  --out-dir "$profile_out" --server-log "$log_path" \
  --tag 130ezo03 --batch 32 --outputs 128 --steps 5 --trace-budget-mib 200
