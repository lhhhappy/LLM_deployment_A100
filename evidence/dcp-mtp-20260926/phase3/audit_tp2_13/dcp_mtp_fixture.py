#!/usr/bin/env python3
"""Make a NextN config from the scaled target, retaining checkpoint precision.

Only writes a diagnostic config and its provenance. Tokenizer/processor assets
are copied separately from the existing scaled model; no weights are created.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaled-config", type=Path, required=True)
    parser.add_argument("--original-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.scaled_config.read_text())
    original = json.loads(args.original_config.read_text())
    text = cfg.get("text_config", cfg)
    original_text = original.get("text_config", original)
    old_layer = original_text["num_hidden_layers"]
    new_layer = text["num_hidden_layers"]
    assert new_layer < old_layer
    assert original_text["num_nextn_predict_layers"] == 1
    text.update(num_nextn_predict_layers=1, pad_token_id=0, eos_token_id=[1])
    cfg["num_nextn_predict_layers"] = 1
    # A shortened target moves NextN from layer 45 to layer 8. Keeping the
    # original exclusion names silently quantizes draft kv_b_proj in the dummy
    # model, collapsing the sensitive attention signal. Do not change the
    # original checkpoint's choice of FP8 versus BF16 modules.
    rules = original["quantization_config"]["modules_to_not_convert"]
    mapped, nextn_rules = [], 0
    for rule in rules:
        match = re.search(r"model\.layers\.(\d+)\.", rule)
        if match:
            layer = int(match.group(1))
            if layer == old_layer:
                rule = rule[:match.start(1)] + str(new_layer) + rule[match.end(1):]
                nextn_rules += 1
            elif layer >= new_layer:
                continue
        mapped.append(rule)
    assert nextn_rules > 0
    cfg["quantization_config"]["modules_to_not_convert"] = mapped
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, indent=2) + "\n")
    provenance = {
        "scaled_config_sha256": hashlib.sha256(args.scaled_config.read_bytes()).hexdigest(),
        "original_config_sha256": hashlib.sha256(args.original_config.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "nextn_layer_remap": [old_layer, new_layer], "remapped_rules": nextn_rules,
        "scope": "Scaled dummy diagnostic; preserve NextN quantization exclusions.",
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
