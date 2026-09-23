# 简报（T49，2026-09-23）：8 卡真实数据出来后的核心问题 — 请进代码分析
仓库 /workspace/Agentic_science_challenge；底包源码 build/base_exact/sglang（= 比赛镜像内的 SGLang）；我们的补丁 patches/（000→101→105→110→111→112→113→114→115→120→130→140→150→160，各有 .md）。
必读：HANDOFF.md、research/claude/R8_next_directions.md（§6/§7）、notes/findings.md 的 F57–F76（尤其 F71 成本表、F75 实时日志、F76 N6 基线）、evidence/N6_b113/analysis.txt、research/claude/base/00–04（注意：其中部分结论已过时，这正是要你帮忙找出的）。
只读：不改代码、不碰 8 卡/bohr/Trisol、不打镜像不提交。开发机 2×A100 可用于小实验（/sjtu/linhang/arena/，先 source env.sh，nvidia-smi 看占用；gjob 见 scripts/gjob）。

## 8 卡真实数据（b113 = 000+101+105+110+111+112+113，tilelang DSA，TP8，dev N6）
- 门：fast_intra p95 7.51s（超标 55，允许约 23）FAIL；overall_intra 7.52s FAIL；turn_start 4.01s PASS；chain_start 19.86s PASS（余量大）。TPOT 均值 0.0304。
- intra 排队 p95 6.4s，exec→首 token p95 1.0–2.8s；最差请求排队 8–15s 而未命中只有数百到数千 token。
- 缓存：总命中 ≥ 冻结期望（×1.107），但 19 个 intra 请求相对同链上一请求丢 >4096 token：(a) 空闲 244–300s 后几乎全丢（KV 峰值仅 50%）；(b) 短间隔丢 5–8k（状态停在上一轮角色边界而非末尾）。另有极端例：25 万 prompt 期望未命中 1281、实际 cached=64。
- 容量：N6 时 KV 峰值 50%（每卡 94.3 万 token，MLA+indexer 每卡全量复制），KDA 状态 584 槽用 4%，decode CUDA graph 6.1GB。
- 冷预填充：约 1 万 tok/s（2 万 1.98s / 6 万 6.37s / 19 万 19.3s）；8 卡 profile：MoE 31.8%、稀疏注意力 17.0%、稠密 GEMM 13.2%、mHC 11.8%、allreduce 11.1%、逐元素 6.1%、indexer 4.3%、KDA 4.1%。
- decode：bs1 10.8ms/步、bs6 16.4ms/步。
- 探针：`--enable-prefill-cp` 需要 `--cp-strategy`（已重排）；`--dcp-size 8` 原版因 tilelang 稀疏注意力 64 头需 256KB 共享内存失败 → 补丁 115 已修（开发机验证），待 8 卡重测。
- 正在 8 卡跑：补丁 120（调度保护链中间请求）的 N6 A/B；随后 114 探针、CP 探针、DCP+115 探针、140 A/B、N10、N14。

## 请回答（中文，给出 文件:行号，区分 VERIFIED/INFERRED）
