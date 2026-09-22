#!/usr/bin/env python3
"""CPU-only audit of the frozen competition config and public workload."""
import argparse
import collections
import hashlib
import json
import math
import subprocess
from pathlib import Path


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "max": values[-1],
            "mean": sum(values) / len(values),
            **{f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
               for p in (50, 90, 95, 99)}}


def audit(root):
    config_path = root / "s1-dev/glm_tok/config.json"
    cfg = json.loads(config_path.read_text())
    c = cfg["text_config"]
    counts = collections.Counter(c["layer_types"])
    linear = c["linear_attn_config"]
    layers = counts["deepseek_sparse_attention"]
    # Pinned SGLang DSATokenToKVPool allocates a full-sized index buffer.
    # KPool changes valid entries, not this allocation's token dimension.
    latent_bf16 = layers * (c["kv_lora_rank"] + c["qk_rope_head_dim"]) * 2
    index_alloc = layers * (c["index_head_dim"] + c["index_head_dim"] // 128 * 4)
    full_ssm = counts["linear_attention"] * linear["num_heads"] * linear["head_dim"] ** 2 * 4
    full_conv = counts["linear_attention"] * 3 * linear["num_heads"] * linear["head_dim"] * (linear["short_conv_kernel_size"] - 1) * 2
    requests = [json.loads(line) for line in (root / "s1-dev/data/dev-combined-v1/requests.jsonl").read_text().splitlines() if line.strip()]
    index = json.loads((root / "s1-dev/glm_tok/model.safetensors.index.json").read_text())
    try:
        commit = subprocess.check_output(["git", "-C", str(root / "src/sglang"), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    by_phase = {}
    for phase in sorted({r["phase"] for r in requests}):
        group = [r for r in requests if r["phase"] == phase]
        by_phase[phase] = {key: stats([r.get(key) for r in group]) for key in ("glm_tokens", "uncached_expected", "max_output_i", "replay_gap_ms")}
    family_counts = collections.Counter(r["prefix_family_id"] for r in requests)
    routes = {}
    for dp in (2, 4, 8):
        for key in ("session_id", "prefix_family_id"):
            load = [0] * dp
            for r in requests:
                rank = int.from_bytes(hashlib.blake2b(r[key].encode(), digest_size=8).digest(), "big") % dp
                load[rank] += r["uncached_expected"]
            routes[f"{key}_dp{dp}"] = {"uncached_tokens_per_rank": load, "max_over_mean": max(load) / (sum(load) / dp)}
    return {
        "evidence_kind": "config_and_source_estimate_not_gpu_measurement",
        "sglang_commit": commit,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "architecture": cfg["architectures"], "layer_counts": counts,
        "model_index_total_bytes": index.get("metadata", {}).get("total_size"),
        "config": {k: c[k] for k in ("num_hidden_layers", "hidden_size", "kv_lora_rank", "qk_rope_head_dim", "mla_use_nope", "index_head_dim", "index_topk", "index_kpool", "index_kpool_compress", "n_routed_experts", "num_experts_per_tok")},
        "bf16_dsa_allocation_bytes_per_token_per_attn_rank": latent_bf16 + index_alloc,
        "bf16_latent_only_bytes_per_token": latent_bf16,
        "index_allocation_bytes_per_token": index_alloc,
        "index_ideal_pooled_bytes_per_token_excluding_tail": index_alloc / c["index_kpool"],
        "kda_state_assumptions": "FP32 recurrent state, BF16 conv; one slot, no extra buffers or MTP",
        "tp8_dp_variants": [{"dp": dp, "attention_tp": 8 // dp,
            "kda_bytes_per_slot_per_rank": (full_ssm + full_conv) // (8 // dp),
            "dsa_logical_capacity_multiplier_if_equal_free_memory": dp}
            for dp in (1, 2, 4, 8)],
        "dsa_128k_gib_per_attn_rank": 131072 * (latent_bf16 + index_alloc) / 2**30,
        "workload": {"requests": len(requests), "chains": len({r["chain_id"] for r in requests}),
            "prefix_families": len(family_counts), "largest_prefix_family_requests": max(family_counts.values()),
            "prompt_tokens": stats([r["glm_tokens"] for r in requests]),
            "uncached_expected": stats([r["uncached_expected"] for r in requests]),
            "output_tokens": stats([r["max_output_i"] for r in requests]),
            "ideal_lcp_fraction": sum(r["glm_lcp_with_prev"] for r in requests) / sum(r["glm_tokens"] for r in requests),
            "by_phase": by_phase, "offline_hash_load_diagnostic_not_runtime_balance": routes},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = json.dumps(audit(args.root), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    else:
        print(result, end="")
