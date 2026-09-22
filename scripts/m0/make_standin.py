#!/usr/bin/env python3
"""M0 stand-in: GLM-5.3-Flash architecture cut to 4 layers (KDA, KDA, KDA, DSA) with the real
attention/indexer dimensions (index_kpool=4, index_topk=2048), 16 routed experts, real FP8
quantization config, random weights via `--load-format dummy`. Purpose: reproduce on 2xA100 the
exact DSA/KDA kernel paths the submitted image takes (research/claude/base/00-summary-mainline.md §1.1).

Usage: make_standin.py <real config.json> <tokenizer dir> <out dir>
"""
import json
import shutil
import sys
from pathlib import Path

src, tok, out = map(Path, sys.argv[1:4])
c = json.loads(src.read_text())
t = c["text_config"]
L = 4
t["num_hidden_layers"] = L
t["layer_types"] = t["layer_types"][:L]            # linear x3, deepseek_sparse_attention
t["mlp_layer_types"] = t["mlp_layer_types"][:L]    # dense x3, sparse
t["indexer_types"] = t["indexer_types"][:L]
lac = t["linear_attn_config"]
lac["kda_layers"] = [i for i in lac["kda_layers"] if i < L]
lac["full_attn_layers"] = [i for i in lac["full_attn_layers"] if i < L]
t["n_routed_experts"] = 16
t["num_nextn_predict_layers"] = 0
c["vision_config"]["depth"] = 2
q = c.get("quantization_config")
if q and "modules_to_not_convert" in q:
    keep = []
    for m in q["modules_to_not_convert"]:
        parts = m.split(".")
        if len(parts) > 2 and parts[1] == "layers" and parts[2].isdigit() and int(parts[2]) >= L:
            continue
        keep.append(m)
    q["modules_to_not_convert"] = keep
out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(c, indent=2) + "\n")
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json"):
    if (tok / f).exists():
        shutil.copy(tok / f, out / f)
print("wrote", out, "layers", t["layer_types"], "kpool", t.get("index_kpool"), "topk", t.get("index_topk"))
