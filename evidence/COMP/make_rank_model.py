# Build a "one TP8 rank" GLM-5.3-Flash surrogate config for single-GPU component profiling with random weights.
# Per-rank shapes: attention/KDA heads 64->8, MoE intermediate 2048->256 (288 experts kept), dense 12288->1536,
# vocab /8; replicated parts kept full (indexer 32 heads, q/kv low-rank, mHC hc_mult=4). Layers: [KDA,KDA,KDA,DSA]x2,
# all MoE. Usage: make_rank_model.py <full_config.json> <tokenizer_dir> <out_dir>
import json, shutil, sys, os
src, tok, out = sys.argv[1:4]
c = json.load(open(src)); t = c["text_config"]
L = 8; lt = ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"] * 2
t.update(num_hidden_layers=L, layer_types=lt, mlp_layer_types=["sparse"] * L, first_k_dense_replace=0,
         num_attention_heads=8, num_key_value_heads=8, moe_intermediate_size=256, intermediate_size=1536,
         vocab_size=t["vocab_size"] // 8, num_nextn_predict_layers=0)
t["linear_attn_config"].update(num_heads=8, kda_layers=[i for i, x in enumerate(lt) if x == "linear_attention"],
                               full_attn_layers=[i for i, x in enumerate(lt) if x != "linear_attention"])
if "indexer_types" in t: t["indexer_types"] = ["full"] * L
os.makedirs(out, exist_ok=True)
json.dump(c, open(f"{out}/config.json", "w"), indent=1)
for f in os.listdir(tok):
    if f != "config.json" and not f.endswith((".safetensors", ".bin")) and os.path.isfile(f"{tok}/{f}"):
        shutil.copy(f"{tok}/{f}", out)
print("wrote", out, "layers", lt)
