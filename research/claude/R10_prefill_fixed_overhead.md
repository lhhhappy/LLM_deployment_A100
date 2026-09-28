# R10 — 分块 prefill 的"每块固定开销"从哪来（T51）

> 口径更正：45 层替身的 CPU/GPU 时间分解是开发机 TP1 **实测**；下文“8 卡真机约 150ms 固定截距”是从 F85 墙钟曲线**反推**，不是 8 卡逐块 profiler 直测。后来的 026g 直测在 P=0 给出约 107ms + 60µs/token（F87），说明不应把 150ms 用作所有上下文/块长的固定常数。#38522 已移植为候选 170；开发机数值修复见 F87/F88/F90，8 卡完整档收益仍需按实验记录判定。

> 2026-09-23 · 开发机 GPU0（A100-80G，TP1）· 底包 base_exact + 000/101/105/106/110/111/112/113/140/120（`evidence/T51/src/PATCHES.md5`）
> 模型：单 rank（TP8 中的一份）替身，`--load-format dummy`。两种层数：8 层 `[KDA×3,DSA]×2`（`rankprof/p8192/model`）和 45 层 `[KDA×3,DSA]×11+KDA`，即 34 KDA + 11 DSA（`evidence/T51/model45/config.json`）。
> 证据目录 `evidence/T51/`。大 trace 留在开发机 `/sjtu/linhang/arena/runs/T51/prof{,45}/`，其中带 stack 的每份约 19 MB。

## 0. 结论（先看这里）

1. **开发机 TP1 实测：45 层替身在 P=98k、c=1024 时有约 150ms 的 GPU 空闲，几乎全是 host 端（CPU/Python）发射间隙。它不是 O(P) 的 GPU 工作，也不是 launch API 本身的耗时。**
   - 45 层替身，P=98k，c=1024，一次 extend：GPU kernel 合计 **100.0ms**；step 墙钟 **251.0ms**（profiler 下测得）；GPU 空闲 **150.8ms**。其中 **149.8ms（99.3%）属于 "CPU-starved"**：下一个 kernel 的 launch 调用发出时，前一个 kernel 已经跑完。
   - host 在模型 forward 里花了 **244.4ms**，共发出 **2358 个 kernel**，平均每个 kernel 约 **104µs** 的 Python/dispatch 时间。CUDA launch API 本身合计只有 21.6ms。
   - 同步点有 56 次 `cudaStreamSynchronize` 和 110 次 memcpy，但一共只等了约 2ms：GPU 一直在等 CPU，反过来的情况几乎没有。
2. **开发机实测的 45 层替身与早期 8 卡墙钟推断处在相近量级；8 卡行是推断值。**

   | 块长 c（P≈98k） | 1024 | 2048 | 4096 | 16384 |
   |---|---|---|---|---|
   | 替身 | 215ms | 270ms | 420ms | 1388ms |
   | 真机 | ~220ms | ~294ms | ~460ms | ~1640ms |

   线性拟合的截距：替身 135ms，早期 8 卡墙钟曲线反推约 150ms。替身证明大量开销在没有 TP8 通信时也存在；通信的大块斜率贡献仍是推断。后来的 026g 直测 P=0 约 107ms + 60µs/token（F87），不能把 150ms 当作普适常数。
3. **Q1（截距随 P 怎么变）实测：绝大部分是常数，随 P 只小幅增长。**
   - 45 层的拟合截距：P=0/98k/180k 分别为 118.7/135.2/146.7ms。
   - c=256 的地板：181/201/218ms。
   - 也就是说，约 **85% 与 P 无关**（host 开销）。与 P 有关的部分主要是 DSA indexer 的 `_prefill` logits kernel，它是 O(c·P)：P=98k 时 11 层合计 17.7ms，P=0 时约 0。
4. **修法排序（推断，按收益从大到小）：**
   1. 让 prefill 走 CUDA graph（breakable/piecewise，PR #38522）。c≤2048 的块预计从 ~215ms 降到 ~100ms（P=98k）/ ~76ms（P=0），前提是能兼容 KDA 并去掉 forward 内的同步。
   2. 若 graph 走不通，就逐模块削减 Python 开销：MoE、KDA backend、mHC 的 tilelang eager dispatch、DSA indexer 准备。
   3. 优化 indexer `_prefill`（O(c·P)）。在 CPU-bound 时它被 host 开销盖住了，graph 上线后就会暴露出来，大块时也占主导。
   4. 调度侧的 CPU（约 15ms/块）。

## 1. 方法

- **历史启动配置：** TP1、`--page-size 64 --mamba-radix-cache-strategy extra_buffer --chunked-prefill-size 16384`、tilelang DSA、`--enable-metrics`（当时的成本探针读取 `meta_info` 时间戳需要它）、`--skip-server-warmup`。当时的一次性服务脚本已清理。
  - 替身词表是 vocab/8 = 19360，所以探针 token id 取 < 19000（`VOCAB_MAX`）。服务自带的 warmup 用了真实 token id，会越界，所以跳过。
  - 8 层版把 KV 池限制为 1M token。不限制时池子有 13M token，首个 forward 就在 cuBLAS 报错，原因未深究。
- **成本探针：** 当时的一次性探针量的是 `prefill_finished_time − request_received_ts`，包含调度开销；每个 (P,c) 取 2 次中较小的一次。探针已清理，原始结果保留在 `evidence/T51/chunkcost8/` 与 `chunkcost45/`。
- **Profiler：** 当时的采集脚本使用 `/start_profile` 和 `/stop_profile`（CPU+GPU）各做两种；脚本已清理，原始结果保留在 `evidence/T51/prof8/` 与 `prof45/`：
  - `nostack`：计时准确，是本文的主数据；
  - `stack`：带 Python 调用栈，用于归因。它把 host 时间拉长约 1.6 倍，只用来看比例。
- **统计：** 当时的 trace 分析只统计 `step[EXTEND…]` 这个 GPU annotation 窗口，从而排除 overlap 调度顺带跑的一个 decode step；一次性分析器已清理。
  - "CPU-starved gap" 的定义：下一个 kernel 的 launch 调用结束时刻 ≥ 前一个 kernel 的结束时刻。
  - profiler 本身有开销：45 层时 profiler 下的 step 为 251ms，不开 profiler 时 e2e 为 212ms（`prof45/profile_summary.json`）。因此这两种采集方式不能直接比绝对值；约 0.8 只是该样本的比值，没有证明所有阶段等比例变化，也不能声称比例不受影响。

## 2. 成本探针（Q1）

extend 耗时，单位 ms。数据来自 `evidence/T51/chunkcost45/chunkcost.json` 和 `logs/t51_probe45.log`。

| P \ c | 256 | 1024 | 2048 | 4096 | 8192 | 16384 | 拟合截距 | 斜率 µs/tok |
|---|---|---|---|---|---|---|---|---|
| 0 | 181.3 | 180.1 | 213.6 | 326.1 | 553.0 | 1066.7 | 118.7 | 56.5 |
| 98304 | 201.2 | 214.7 | 269.5 | 419.9 | 722.4 | 1387.5 | 135.2 | 75.1 |
| 180224 | 217.6 | 242.4 | 314.4 | 495.5 | 866.9 | 1662.7 | 146.7 | 91.2 |

8 层替身的对应数字（`chunkcost8/chunkcost.json`）：
- 截距：30.6 / 33.6 / 40.5 / 50.2ms（P = 0 / 32k / 98k / 180k）；
- c=256 的地板：40.5ms（P=0）、52.6ms（P=98k）、63.4ms（P=180k）。

解读：
- 这条曲线**不是线性的**。c≤1024 时基本是平的（地板约 180ms，全部是 host 时间）；c≥4096 后才转为 GPU-bound。线性拟合的"截距"其实是这两段的折中，比真实地板要低。
- 斜率随 P 上升（56→91 µs/tok），这是 indexer 的 O(c·P) 项。

decode（CUDA graph，45 层）：bs=1 为 10.3ms，bs=12 为 11.9ms。真机 bs=12 为 17ms，差值包含 allreduce。可以对比：同样 45 层，一次 eager 的 c=256 extend 要 181ms。

## 3. 一次 extend 的剖面（c=1024）

| 指标 | 45 层 P=0 | 45 层 P=98k | 8 层 P=0 | 8 层 P=98k |
|---|---|---|---|---|
| step[EXTEND] 墙钟（profiler 下） | 245.0 | 251.0 | 48.5 | 49.9 |
| 模型 forward 的 CPU 时间（`language_model_prefill`） | 238.7 | 244.4 | 42.4 | 43.6 |
| GPU kernel 合计 | 76.0 | 100.0 | 13.95 | 18.4 |
| GPU 空闲 / 其中 CPU-starved | 168.9 / 168.4 | 150.8 / 149.8 | 34.5 / 34.4 | 31.5 / 31.4 |
| kernel 数 / memcpy 数 | 2259 / 110 | 2358 / 110 | 479 / 45 | 504 / 45 |
| `cudaStreamSynchronize` 次数（耗时） | 57（0.55ms） | 56（0.55ms） | 29 | 28 |
| 全部 launch API 耗时 | 21.1 | 21.6 | 4.3 | 4.5 |
| 不开 profiler 的 e2e | 211–214 | 212–214 | 47 | 52–55 |

来源：`prof45/*/stats.json`、`prof8/*/stats.json`、`*/profile_summary.json`。

GPU 空闲间隙的分布（45 层，P=98k）：
- 0.1–1ms 的间隙合计 106ms；
- 20–100µs 的间隙合计 42ms；
- ≥1ms 的间隙为 0。

也就是说，空闲是被无数个小间隙"均匀摊薄"出来的，没有一个单点大停顿。

**Top kernels**（45 层，P=98k，`prof45/P98304_c1024_nostack/stats.json`）：

| kernel | 耗时 | 次数 |
|---|---|---|
| marlin_moe | 23.3ms | 90 |
| **indexer `_prefill`** | **17.7ms** | 11 |
| DSA sparse attn `main_kernel` | 14.2ms | 11 |
| mhc_pre_gemm_sqrsum | 4.0ms | 90 |
| mhc_post | 3.6ms | 90 |
| bf16 gemm 256x128 64x3 | 3.6ms | 34 |
| **kpool_topk_transform** | **3.2ms** | 11 |
| mhc_pre_big_fuse | 3.2ms | 90 |
| bf16 gemm 32x3 | 3.0ms | 45 |
| marlin dense | 3.0ms | 33 |
| vectorized_elementwise | 2.7ms | 326 |
| chunk_kda_inter_solve_fused | 2.4ms | 34 |
| moe_sum_reduce | 2.4ms | 45 |
| elementwise | 2.0ms | 194 |
| bf16 gemm 128x128 | 1.2ms | 90 |

按组件汇总（P=98k，括号内为 P=0）：

| 组件 | 耗时（ms） |
|---|---|
| moe | 26.4（26.4） |
| dsa_indexer | **21.3（0.17）** |
| dsa_sparse_attn | 14.4（13.5） |
| dense_gemm | 14.0（12.7） |
| mhc | 10.8（10.8） |
| norm/elementwise | 7.5（7.2） |
| kda | 5.0（5.0） |

**与 P 相关的 GPU 工作**：P 从 0 到 98k，GPU 合计多出 24ms，其中 dsa_indexer 占 +21.1ms（`_prefill` 17.7 + kpool_topk 3.2）。c=1024 时，这 24ms 大部分被 host 开销盖住：不开 profiler 时，P=0 和 P=98k 的 e2e 都是约 212ms。

**forward 之外、调度侧的 CPU**（45 层，nostack 时间线，`prof45/annotations_timeline_nostack.txt`）：

| 阶段 | P=98k | P=0 |
|---|---|---|
| `get_next_batch_to_run`（含 match_prefix 和 prepare_for_extend） | 5.8ms | 3.7ms |
| step 内、模型之前（init_forward_metadata 等） | 6.0ms | 6.0ms |
| `process_batch_result`（含 cache_finished_req 和 metrics） | 4.5ms | 2.2ms |

stack 模式下的细分（`prof8/pytree_sched_P98304_P0.txt`；数值被放大约 1.6 倍）：
- match_prefix：1.5ms（P=98k），P=0 时可忽略；
- cache_finished_req 里的 insert：1.8ms，P=0 时 0.4ms；
- kpool plan 上 GPU（`_kpool_plan_to_gpu`）：1.4ms；
- metrics `report_prefill_stats`：1.8ms，由 `--enable-metrics` 引入。

这些是该 TP1 替身样本的阶段时间，不能直接作为当前 TP8 每块固定开销；是否在关键路径、能否与其他工作重叠，需要看对应运行的流和依赖。

## 4. host 时间花在哪里

数据来自 8 层 stack trace（P=98k，`prof8/pytree_hotspots_inclusive.txt`、`prof8/*_stack/pyself_model.txt`）。每层的 nostack 估计值 = stack 值 × 0.62，其中 0.62 = 43.6/70.1。

| 模块（inclusive） | stack 下 8 层合计 | 每层（nostack 估计） | 45 层外推 |
|---|---|---|---|
| MoE `forward_normal` | 16.0ms | 1.24ms × 45 层 | 56ms |
| KDA 层（`glm5_next:562`；其中 backend `forward_extend` 为 14.9） | 21.7ms / 6 层 | 2.24ms × 34 层 | 76ms |
| mHC（`hc_attn_pre` 5.0 + `attn_to_mlp` 7.2 + `mlp_combine` 2.5） | 14.6ms | 1.13ms × 45 层 | 51ms |
| DSA MLA 注意力（prep 10.6，其中 IndexerKPool 约 3.9；core 3.7） | 14.6ms / 2 层 | 4.5ms × 11 层 | 50ms |
| **外推合计** | | | **233ms** |

外推合计 233ms，45 层实测为 244ms（nostack），两者吻合。

按框架分组的 Python self time（stack 模式，占模型 forward 的比例）：

| 分组 | 占比 |
|---|---|
| builtin C 调用（torch op 的 dispatch、empty/view/linear、pybind） | 37% |
| sglang 自身 Python | 30% |
| **tilelang/tvm eager dispatch** | **12.5%** |
| triton launcher | 8.5% |
| torch nn.Module | 5% |
| dynamo 包装 | 2.5% |

tilelang/tvm eager dispatch 被单独列出，是因为它每次调用都要走 `_parse_phase2_key`、`script_printer._script`、`_bind_fast`：8 层共 50 次调用，合计 8.6ms。triton 的 `jit.run` 在 8 层中有 78 次调用，合计 8.7ms。

结论（实测）：**没有单一热点**。开销是约 100µs/kernel 的通用 Python dispatch，乘以约 52 kernel/层，再乘以 45 层。

## 5. 约 150ms/块的解释：TP1 实测与 TP8 推断，不能当 TP8 归因表

| 类别 | 量级 | 依据 |
|---|---|---|
| host Python/dispatch（模型内） | 约 190–240ms，GPU 只能用上其中约 100ms | 实测：45 层 244ms（profiler 下），不开 profiler 时约 0.8 倍 |
| 调度侧 CPU | 约 15ms | 实测，见 §3 |
| O(P) GPU kernel | P=98k 时 +24ms，P=180k 时约 +45ms（推断），c=1024 时大部分被盖住 | 实测：indexer `_prefill` 17.7ms + kpool_topk 3.2ms |
| O(c) GPU kernel | 76ms（c=1024，P=0） | 实测 |
| 纯 launch API | 21.6ms（已包含在 host 时间里） | 实测 |

在拟合"截距"约 150ms 中，host 开销减去被 GPU 盖住的部分占约 85%，O(P) 的 indexer 占约 15%（推断，依据是截距随 P 的增量）。

**替身没覆盖的部分：**
- TP8 每层 2 次 allreduce（约 90 次 NCCL）。各 rank 的 CPU 抖动会在每次 collective 处取最大值，这会让 host-bound 更严重；真机大块时比替身慢约 18%，与此一致（推断）。
- 真机的 CPU 型号和频率不同。
- dummy 权重下 MoE 路由分布不同。
- 替身全部是 MoE 层，没有前几层 dense。
- DSA 注意力的每 rank 头数与真机一致（64/8=8），不属于缺口。

## 6. 修法排序（推断）

1. **prefill CUDA graph（小块固定成本候选）。** 这是 T51 时底包的 KDA 限制；后续已有 170 v2 的 TP2 修复证据，真实 TP8 数值与性能仍待复验，不能继续按“尚无实现”估工。
   - 需要 KDA extend 与状态追踪满足图执行契约。T51 的 sync/memcpy 次数是线索，不代表每次调用都必须删除；应区分图外元数据、真正同步与可捕获拷贝，并核对 170 已覆盖的部分。
   - 预期：c≤2048 的块从约 215ms 降到约 GPU 时间：100ms（P=98k）/ 76ms（P=0），再加真机 allreduce。c=256 的地板会从 181ms 大幅下降（graph 下 GPU 耗时未测）。c=16384 的块收益很小（GPU-bound）。
   - 意义：小块变便宜，才可以用小块来保护 TPOT。
2. **不用 graph 时的逐模块减 host。** 可行方向：
   - mHC 的 tilelang kernel 缓存编译句柄，绕过每次调用的 eager key 解析，约 51ms 中可省一部分；
   - triton kernel 缓存 launcher；
   - 合并 MoE/KDA 的小 op（empty/view/zeros 类，8 层替身里就有 316 次 empty）；
   - 对 decoder layer 做 torch.compile。

   预计能拿到 host 时间的 30–50%，未验证。
3. **indexer `_prefill`（O(c·P)）加速。** 候选包括 114-indexer-row-shard（不在本次补丁栈中）和更快的 logits kernel。现在收益被 host 开销盖住，但在 graph 上线后以及大块时会成为主项：斜率 +35µs/tok（P 从 0 到 180k）大部分来自这里。
4. **调度侧（≤15ms/块）。**
   - `report_prefill_stats` 的 metrics 可以关掉或移到异步；
   - match_prefix 和 cache insert 是 O(P/page)，但量级只有毫秒；
   - kpool plan 的 H2D 拷贝可以合并。

## 7. 历史实验与证据

当时在开发机 GPU0 上依次运行 8 层、45 层 TP1 替身，采集不同前缀长度与块长的成本曲线、无栈与带栈 profiler，再分析 `step[EXTEND…]` 窗口。原服务、采集与 trace 汇总脚本都是一次性实验工具，仓库清理后已移除；旧命令不能在当前仓库直接复跑。保留的输入配置和统计口径见 §1，数值及 profiler 输出见 [T51 原始证据](../../evidence/T51/)；新的复验需要依当前底包、环境和 pod 规则另建实验。
