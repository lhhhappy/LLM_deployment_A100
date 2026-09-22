你是 Codex worker W17，任务 T43：为 A100（sm80）写 DSA indexer 的融合 kernel（补丁 112），替换补丁 110 里的 torch 实现。这是目前最大的性能瓶颈。
仓库 /workspace/Agentic_science_challenge。先读：AGENTS.md、rule.md §2（GPU 开发机）与 §4（红线）、notes/findings.md 的 F56–F62、patches/110 的生成器 scripts/make_110.py 与源 build/p110/（sm80_deep_gemm.py、ax_soft_fp8.py、三个单测）、evidence/T43/profile_decode_b112_n6_TP0.txt。
基线代码 = build/base_exact/ + patches 000→101→105→110→111（`patch -p3 --fuzz=0`，照 base_exact，不照 src/sglang）。

证据（真实 8×A100，dev N6，12 步 profile）：GPU 时间约 60% 是 110 torch shim 的 elementwise/reduce/gather（fp8→bf16 物化整段上下文、按头 relu·w 求和、mask），decode 每请求每步约 10ms；预填充 fp8_mqa_logits 按 32 头循环 mm，预填充吞吐仅数百到数千 tok/s。
模型：index_n_heads=32、index_head_dim=128、index_topk=2048、index_kpool=4、KV 页 64 token，kv_cache 每块 [64, 1, 128+4] 字节（128 个 fp8 e4m3 值 + 1 个 f32 scale，布局见 sm80_deep_gemm.py）。

目标：补丁 `patches/112-sm80-indexer-kernels.patch`（+ .md + 可复现生成器 scripts/make_112.py），在 110 之上替换两个入口的实现（接口/语义与 110 torch 版逐项一致，含 clean_logits、context_lens 形状 [B] 或 [B,N]、N>1 的 MTP 情形、超出上下文位置置 0、CUDA graph 可捕获、无 host 同步）：
1. fp8_paged_mqa_logits（decode）：每个 (请求, 64-token 块) 一个 program；在 kernel 内从 uint8 解码 e4m3→bf16/fp32（sm80 无 fp8 mma；Triton 在 sm80 拒绝 fp8e4nv 类型，所以按 uint8 位运算解码，或用 tilelang/CUDA 的软件转换），对 32 头做 [64×128]·[128×32] 点积（tl.dot bf16），relu、乘头权重、跨头求和、乘 k scale，写出 logits；只遍历 ≤ context_len 的块（按 block_table），不物化中间大张量。
2. fp8_mqa_logits（prefill，ragged：q [nq,H,D] fp8，kv [nk,D] fp8 + scale，ks/ke 每个 query 的有效 key 区间）：分块 tiled kernel（BLOCK_Q×BLOCK_K），同样在 kernel 内解码 fp8，只算 [ks,ke) 内的块，不在区间内的位置按 110 语义处理。
3. 若 base_exact 里现有的 tilelang `fp8_paged_mqa_logits_kernel`（kernels/ops/attention/dsa/tilelang_kernel.py:1423 附近）能改成 sm80 可用版本，也可以采用，但须说明理由并给出同样的对照数据。
验证（GPU 开发机 `ssh GPU`，只在 /sjtu/linhang/arena/ 下工作，先 `source /sjtu/linhang/arena/env.sh`，用前 nvidia-smi 看占用；pod 与 8 卡由 Claude 负责，你不要碰 bohr/Trisol/pod）：
- 数值：与 110 torch 版对照（随机数据 + 真实形状：上下文 1k/32k/190k，batch 1/6/32，N=1 与 N=2），logits 相对误差 < 1e-2，且 topk(2048) 选中集合重合率 ≥ 99.5%；边界：非整块尾部、context_len=0/1、block_table 含 -1。
- 性能：同形状下新旧耗时对比（decode：B=6、上下文 3 万–19 万；prefill：8192 query × 3 万–19 万 key），给出 ms 与加速倍数；CUDA graph 捕获与重放一致。
- 补丁叠加：000→101→105→110→111→112→120→130 全部 fuzz=0 可打；py_compile 通过。
落盘：notes/dispatch.md T43 行 accepted→in-progress→done/blocked（落盘即通知，阻塞立即写）；证据 evidence/T43/；最后追加「T43 W17 → Claude：交付」节（VERIFIED/INFERRED、对照数据表、开放问题）。每一步可回溯：生成器 + 测试脚本 + 原始日志。
