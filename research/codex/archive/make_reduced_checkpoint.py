#!/usr/bin/env python3
"""Copy a prefix of GLM-5.3-Flash layers into a separate DEVELOPMENT fixture.

Default five layers preserve dense KDA + sparse DSA + sparse KDA and all original
head dimensions. Untrained/truncated behavior is not a quality/performance score.
No downloads, source edits, vocabulary remaps, or changes to the original model.
--config-only produces a config and selection manifest, not a loadable checkpoint.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

LAYER = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.")


def select_key(key, layers):
    match = LAYER.match(key)
    return match is None or int(match.group(1)) < layers


def reduce_config(config, layers):
    config = copy.deepcopy(config)
    text = config["text_config"]
    original = text["num_hidden_layers"]
    if not 5 <= layers < original:
        raise ValueError("Use 5 <= layers < original layers, retaining both attention types and MoE")
    text["num_hidden_layers"] = layers
    text["num_nextn_predict_layers"] = 0
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        if key in text:
            text[key] = text[key][:layers]
    for key in ("kda_layers", "full_attn_layers"):
        if key in text["linear_attn_config"]:
            text["linear_attn_config"][key] = [i for i in text["linear_attn_config"][key] if i < layers]
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--config-only", action="store_true")
    args = p.parse_args()
    # Refuse overwrite to keep source and preexisting fixtures recoverable.
    if args.out.exists():
        p.error("output path must not already exist")
    source_config = args.source / "config.json"
    config = reduce_config(json.loads(source_config.read_text()), args.layers)
    source_index = json.loads((args.source / "model.safetensors.index.json").read_text())
    selected = {key: file for key, file in source_index["weight_map"].items() if select_key(key, args.layers)}
    groups = collections.defaultdict(list)
    for key, file in selected.items():
        if Path(file).name != file:
            raise ValueError("Expected flat safetensors shard names")
        groups[file].append(key)
    if not args.config_only:
        missing = [name for name in groups if not (args.source / name).is_file()]
        if missing:
            p.error(f"missing {len(missing)} source shards; use --config-only for planning")
        from safetensors import safe_open
        from safetensors.torch import save_file
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    receipt = {"kind": "development_only_reduced_checkpoint", "config_only": args.config_only,
        "layers": args.layers, "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "selected_tensors": len(selected), "source_shards_needed": sorted(groups),
        "complete": False, "note": "Original attention dimensions, experts, tokenizer and vision retained; MTP removed."}
    manifest = args.out / "fixture_manifest.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.config_only:
        print(json.dumps(receipt, indent=2))
        return
    out_map, total_size = {}, 0
    for i, (source_name, keys) in enumerate(sorted(groups.items()), 1):
        target_name = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        # At most one source shard's selected tensors are held in CPU RAM.
        with safe_open(str(args.source / source_name), framework="pt", device="cpu") as shard:
            tensors = {key: shard.get_tensor(key).contiguous() for key in keys}
            total_size += sum(t.numel() * t.element_size() for t in tensors.values())
            save_file(tensors, str(args.out / target_name), metadata={"format": "pt"})
        del tensors
        out_map.update({key: target_name for key in keys})
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "processor_config.json",
                 "preprocessor_config.json", "chat_template.jinja", "special_tokens_map.json", "tokenizer.model"):
        if (args.source / name).is_file():
            shutil.copy2(args.source / name, args.out / name)
    (args.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": out_map}, indent=2) + "\n")
    receipt.update(complete=True, total_weight_bytes=total_size)
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
