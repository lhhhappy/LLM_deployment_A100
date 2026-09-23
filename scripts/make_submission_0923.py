# Generate build/submit_0923_{A,B}/submission.json for image lh-img:0923a (15 patches, see build/image/0923a.patches.txt).
# A = ladder 028 config (MTP+114+v3 cap4096 interval2, eager prefill). B = 026-style big chunks (no cap, no interval) + MTP+114.
# Usage: python3 scripts/make_submission_0923.py <imageUrl@sha256:...> <B_chunk: 16384|8192>
import json, os, sys
img, bchunk = sys.argv[1], sys.argv[2]
BASE = ("python3 -m sglang.launch_server --model-path /mnt/models --host 0.0.0.0 --port 8000 --tp-size 8 --served-model-name default "
        "--enable-metrics --incremental-streaming-output --page-size 64 --mamba-radix-cache-strategy extra_buffer "
        "--reasoning-parser glm45 --tool-call-parser glm47 --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang")
MTP = ("--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN "
       "--speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 "
       "--max-running-requests 32 --cuda-graph-max-bs 32")
ENV = {"SGLANG_OPT_USE_TOPK_V2": "0", "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0", "SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS": "154827,154829",
       "SGLANG_AX_KDA_DUAL_SNAPSHOT": "0", "SGLANG_AX_SCHED_PROTECT": "1", "SGLANG_AX_SCHED_SHORT_TOKENS": "8192",
       "SGLANG_AX_ASYNC_TOKENIZE": "0", "SGLANG_MAMBA_SSM_DTYPE": "float32", "SGLANG_OPT_FUSED_KDA_VERIFY": "0",
       "SGLANG_AX_INDEXER_ROW_SHARD": "1"}
arms = {
  "A": (f"{BASE} {MTP} --prefill-decode-interval 2", {**ENV, "SGLANG_AX_SCHED_COLD_CAP": "4096"}),
  "B": (f"{BASE} {MTP} --chunked-prefill-size {bchunk}" + (" --mem-fraction-static 0.74" if bchunk == "16384" else ""),
        {**ENV, "SGLANG_AX_SCHED_COLD_CAP": "16384"}),
}
for arm, (cmd, env) in arms.items():
    d = f"build/submit_0923_{arm}"; os.makedirs(d, exist_ok=True)
    sub = {"image": img, "command": cmd, "env": env, "model_name": "default"}
    json.dump(sub, open(f"{d}/submission.json", "w")); json.dump(sub, open(f"submission/official-0923-{arm}.json", "w"), indent=1)
    print(arm, cmd[-140:], env["SGLANG_AX_SCHED_COLD_CAP"])
