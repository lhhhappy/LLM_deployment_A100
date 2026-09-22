#!/usr/bin/env python3
"""Build deterministic random Kimi-Linear weights for E1 cache tests, not quality eval.

Uses the pinned SGLang KimiLinearForCausalLM checkpoint names. No original model
weights are downloaded; tokenizer assets are copied byte-for-byte from dev.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=731)
    p.add_argument("--q-lora-rank", type=int, default=0,
                   help="0: full Q projection; avoids pinned Kimi wrapper's missing AttentionInputs setup")
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing model directory: {args.output}")
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    vocab = max(tok.get_vocab().values()) + 1
    # Keep the original vocabulary padding too, if present in the model config.
    original = json.loads((args.tokenizer / "config.json").read_text())
    original = original.get("text_config", original)
    vocab = max(vocab, original.get("vocab_size", vocab))
    hidden, heads, dim, intermediate, layers = 512, 4, 128, 1536, 4
    if args.q_lora_rank < 0:
        p.error("q-lora-rank must be nonnegative")
    kv_rank, q_rank, rope_dim = 512, (args.q_lora_rank or None), 64
    cfg = {
        "architectures": ["KimiLinearForCausalLM"], "model_type": "kimi_linear",
        "vocab_size": vocab, "hidden_size": hidden, "head_dim": dim,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": heads, "intermediate_size": intermediate,
        "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144, "tie_word_embeddings": False,
        "dtype": "bfloat16", "torch_dtype": "bfloat16", "use_cache": True,
        "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id, "num_experts": None,
        "q_lora_rank": q_rank, "kv_lora_rank": kv_rank,
        "qk_nope_head_dim": dim, "qk_rope_head_dim": rope_dim,
        "v_head_dim": dim, "num_nextn_predict_layers": 0,
        "linear_attn_config": {
            "kda_layers": [1, 2, 3], "full_attn_layers": [4],
            "num_heads": heads, "head_dim": dim, "short_conv_kernel_size": 4,
        },
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    weights = {}

    def random(name, shape, dtype=torch.bfloat16, std=0.01):
        weights[name] = (torch.randn(shape, generator=generator) * std).to(dtype)

    def norm(name, size):
        weights[name] = torch.ones(size, dtype=torch.bfloat16)

    random("model.embed_tokens.weight", (vocab, hidden))
    random("lm_head.weight", (vocab, hidden))
    norm("model.norm.weight", hidden)
    projection = heads * dim
    for i in range(layers):
        prefix = f"model.layers.{i}"
        norm(f"{prefix}.input_layernorm.weight", hidden)
        norm(f"{prefix}.post_attention_layernorm.weight", hidden)
        random(f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden))
        random(f"{prefix}.mlp.up_proj.weight", (intermediate, hidden))
        random(f"{prefix}.mlp.down_proj.weight", (hidden, intermediate))
        attn = f"{prefix}.self_attn"
        if i < 3:
            for name in ("q", "k", "v"):
                random(f"{attn}.{name}_proj.weight", (projection, hidden))
                random(f"{attn}.{name}_conv1d.weight", (projection, 1, 4), torch.float32)
            random(f"{attn}.b_proj.weight", (heads, hidden))
            for name in ("f", "g"):
                random(f"{attn}.{name}_a_proj.weight", (dim, hidden))
                random(f"{attn}.{name}_b_proj.weight", (projection, dim))
            weights[f"{attn}.dt_bias"] = torch.full((projection,), -3.0)
            weights[f"{attn}.A_log"] = torch.zeros(1, 1, heads, 1)
            norm(f"{attn}.o_norm.weight", dim)
            random(f"{attn}.o_proj.weight", (hidden, projection))
        else:
            # SGLang's shared DeepseekV2AttentionMLA stores the low-rank A
            # projections fused; KimiLinear's loader accepts this exact key.
            if q_rank is not None:
                random(f"{attn}.fused_qkv_a_proj_with_mqa.weight",
                       (q_rank + kv_rank + rope_dim, hidden))
                norm(f"{attn}.q_a_layernorm.weight", q_rank)
                random(f"{attn}.q_b_proj.weight", (heads * (dim + rope_dim), q_rank))
            else:
                random(f"{attn}.q_proj.weight", (heads * (dim + rope_dim), hidden))
                random(f"{attn}.kv_a_proj_with_mqa.weight", (kv_rank + rope_dim, hidden))
            norm(f"{attn}.kv_a_layernorm.weight", kv_rank)
            random(f"{attn}.kv_b_proj.weight", (heads * (dim + dim), kv_rank))
            random(f"{attn}.o_proj.weight", (hidden, heads * dim))

    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    assets = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    for asset in assets:
        shutil.copy2(args.tokenizer / asset, args.output / asset)
    save_file(weights, str(args.output / "model.safetensors"), metadata={"format": "pt"})
    manifest = {
        "purpose": "E1 random-weight cache/scheduler stand-in, NOT quality/performance evaluation",
        "sglang_commit": "94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
        "seed": args.seed, "torch_version": torch.__version__,
        "kda_layers_zero_based": [0, 1, 2], "mla_layers_zero_based": [3],
        "dsa": False, "mtp": False, "moe": False,
        "q_lora_rank": q_rank, "kv_lora_rank": kv_rank,
        "parameter_count": sum(t.numel() for t in weights.values()),
        "tensor_shapes": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                          for k, v in weights.items()},
        "sha256": {f: sha256(args.output / f)
                   for f in (*assets, "config.json", "model.safetensors")},
    }
    (args.output / "E1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tensor_shapes"}, indent=2))


if __name__ == "__main__":
    main()
