# EP8 vs TP8 MoE layer time, dev box (2026-09-25)

Measured on one A100-SXM4-80GB of the GPU dev box (torch 2.13+cu130, source 37e90023 exported to
/sjtu/linhang/arena/runs/ep8/src), script `scripts/pod/verify/bench_moe_ep8.py` (md5 3c3279f4), raw lines in
`bench1.jsonl`. GLM-5.3-Flash shapes: 288 routed + 1 shared expert, hidden 4096, expert intermediate 2048,
top-8, FP8 128x128 block scales, Marlin W8A16 (111). TP8 = rank 0's slice (N=256, shared fused, top-k 9);
EP8 = each of the 8 ranks' 36 whole experts with the dispatcher's -1 ids and is_ep, layer time = slowest rank
+ the shared expert on its TP slice. Communication is not timed (the same all-reduce in both).

Result: EP8 is slower than TP8 at every size (8192 tokens: 6.35 vs 5.53 ms uniform, 7.12 skewed; 16384:
12.9 vs 10.9). A rank's Marlin time is about the same in both layouts (equal FLOPs per rank; EP8 71 TF vs TP8
84 TF at 8192), so the narrow TP8 shape is not what limits Marlin; EP8 adds the unfused shared expert and
load imbalance. Numerics: EP8 rel. error 5.4e-3, TP8 5.7e-3 vs an fp32 reference of the dequantized weights;
the shifted-id negative control gives 0.58.

Caveats: the source predates the block-size fix in 3df1b172 (small-batch EP numbers are pessimistic; large
batches choose the same 64-row blocks either way); routing is synthetic (uniform and a 1/rank^0.6 skew);
dev-box timings are provisional until TP8 on the pod.
