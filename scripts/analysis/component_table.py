# Group GPU kernel time from torch-profiler traces into GLM-5.3 components (prefill vs decode traces separately).
import gzip, json, glob, re, sys, collections
RULES = [  # (component, regex on kernel name) — first match wins
 ("moe", r"marlin_moe|moe_align|moe_sum|_router|fused_moe|topk_sigmoid|gate_topk|moe_"),
 ("dsa_indexer", r"_ragged|_paged\b|_paged\(|_prefill\b|_unpack_prefill|act_quant|kpool|topk_transform|mqa|_e4m3|indexer"),
 ("mhc", r"mhc|hc_pre|hc_post|sinkhorn"),
 ("dsa_sparse_attn", r"sparse|main_kernel|flash|attn"),
 ("kda", r"kda|chunk_|recompute_w_u|delta_h|fused_recurrent|causal_conv1d|layer_norm_gated|sigmoid_gating|solve|wy_|gated_delta"),
 ("dense_gemm", r"marlin::Marlin|gemm|cutlass|cublas|Kernel2|splitK"),
 ("allreduce", r"all_reduce|allreduce|nccl"),
 ("norm_elementwise", r"norm|elementwise|reduce_kernel|copy|cat|index|gather|scatter|fill|memcpy|memset"),
]
def comp(name):
    for c, rx in RULES:
        if re.search(rx, name, re.I): return c
    return "other"
root = sys.argv[1]
for f in sorted(glob.glob(f"{root}/**/*.trace.json*", recursive=True)):
    t = json.load(gzip.open(f, "rt") if f.endswith(".gz") else open(f))
    ev = t["traceEvents"] if isinstance(t, dict) else t
    tot = collections.Counter(); top = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            n = e["name"]; d = e.get("dur", 0); tot[comp(n)] += d; top[(comp(n), re.sub(r"<.*", "", n)[:70])] += d
    s = sum(tot.values()) or 1
    print(f"== {f.split('/')[-1]}  total GPU {s/1000:.1f} ms")
    for c, v in tot.most_common(): print(f"  {c:18s} {v/1000:9.2f} ms {100*v/s:5.1f}%")
    print("  top kernels:")
    for (c, n), v in top.most_common(12): print(f"    {v/1000:8.2f} ms {c:16s} {n}")
