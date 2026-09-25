#!/usr/bin/env python3
"""Write a GLM-5.3-Flash model dir with only the first N decoder layers.

For dev-box probes with ``--load-format dummy``: the real config.json and
tokenizer files are copied, ``text_config.num_hidden_layers`` becomes N and the
per-layer lists (layer, indexer and MLP types) are cut to N entries. Everything
else (MTP layer, vision tower, quantization, dims) stays as in the checkpoint.
Output is a probe stand-in, not the served model.

  python3 scripts/vllm/truncate_config.py s1-dev/glm_tok /path/out --layers 8
"""

import argparse
import json
import shutil
from pathlib import Path

PER_LAYER_LISTS = ("layer_types", "indexer_types", "mlp_layer_types")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="dir with config.json and tokenizer")
    ap.add_argument("dst", type=Path)
    ap.add_argument("--layers", type=int, required=True)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    for f in args.src.iterdir():
        if f.is_file() and f.name not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            shutil.copy2(f, args.dst / f.name)

    config = json.loads((args.src / "config.json").read_text())
    text = config["text_config"]
    full = text["num_hidden_layers"]
    assert 0 < args.layers <= full, (args.layers, full)
    for key in PER_LAYER_LISTS:
        assert len(text[key]) == full, (key, len(text[key]))
        text[key] = text[key][: args.layers]
    text["num_hidden_layers"] = args.layers
    (args.dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    kinds = {k: text["layer_types"].count(k) for k in set(text["layer_types"])}
    print(f"{args.dst}: {args.layers}/{full} layers {kinds}")


if __name__ == "__main__":
    main()
