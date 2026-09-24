# 底包与当前技术认识

赛题规则以 [task.md](../../../llm-challenge-arena-v1/task.md) 为准；正在跑的作业与最新结果见仓库根部 [README](../../../README.md) 和实验记录。本文件只整理较稳定的源码事实、已核实实验和解释边界。源码地图分别见[请求入口](01-request-path.md)、[调度](02-scheduler.md)、[混合缓存](03-hybrid-cache.md)、[模型与算子](04-model-kernels.md)。

## 代码和基线

- `build/base_exact/sglang` 是 L3 底包的逐字节副本，对应公开提交 `fe236ea6c3` 加两处多模态修复（F53/F54）。`src/sglang` 是 v0.5.20，不能用来设计或验证当前补丁。
- [补丁索引](../../../engine/README.md) 定义 S0：`000 101 106 110 111 120 140`，具体启动参数与开关也列在其中。`evidence/T57/equivalence.log` 已核实当前 120 补丁的 SHA256 为 `ef1744b3…`，重建的 S0 与 026/035 代码树 4686 个文件一致。旧材料所说“当前 120 与 S0 版本不同”已经失效。
- `000` 修 `/flush_cache` 的 JSON 与全 worker 确认，并补接收时间戳；`101` 和 `140` 保存角色边界及尾部 KDA 状态；`106` 处理 KV 用尽时的分块续算；`110/111` 让 A100 上 DSA indexer 与 FP8 MoE 能运行；`120` 在长预填充间保护解码及短命中请求。候选 `114/115/121/130/150/160/170` 的验证边界见各补丁说明。

## 实验说明了什么

- **开发集只用于同一负载的相对 A/B。** 它是链前缀抽样：722 请求里本轮链首 311 条；正式负载的结构不同，开发集档位不能直接当正式 `n_at_slo`（task.md:354，R19）。
- 026/S0 在开发集 N18 的四道 TTFT 门按正式统计口径通过，但 `tpot_p95=0.219` 未过 0.10 硬门。035/S0 在开发集 N22 为 11 门过 10，`tpot_p95=0.296193`、`tpot_mean=0.103956`（[重评分](../../../evidence/T58/README.md)）。这不是正式排名成绩。
- 036/S0+114 在 N22 相对 035 改善了 fast_intra 超标条数（17→7）与 `tpot_p95`（0.296→0.253），但仍未过 TPOT 门；精确结果以作业原始日志及实验台账为准。
- N6 的 intra 排队 p95 曾为 6.4 秒，调度 120 的早期变体降至 0.36 秒（F76/F77）。如今主要待解决的是大块 prefill 期间 decode 出字停顿；035 的 205 条 TPOT 超标里，许多请求自身缓存命中很高、首 token 很快，却在密集 prefill 时生成缓慢。[R21](../../codex/R21_N22_tpot_failures.md)给出逐条证据。秒级日志不能分解每一秒的 GPU 阻塞、CPU 调度和通信占比。
- 冻结 `uncached_expected` 不是实际可复用长度。026 的 411 条本轮后续请求，按相邻 prompt 的真实 token LCP 计算，64-token 网格下可修的正缺口上限为 850,432 token，占全档实际 prefill 的 8.0%；这是上限，不是补丁保证的收益（[R20](../../codex/R20_true_lcp_attribution.md)）。
- 缓存是 `UnifiedRadixCache`，KDA state 的存在限制可服务的 KV 前缀。状态、KV、CUDA graph 是不同的显存池；“日志里的 KV 50%”和“在飞 prompt token 总和”都不能直接当物理余量。[R18](../../codex/R18_cache_loss_and_capacity.md)记录实测内存账与未确定的淘汰触发条件。

## 已撤回的判断

早期报告中的以下说法不能用于下一轮决策：把 v0.5.20 当底包；A100 上选 fa3 或原始 DeepGEMM FP8 路径；“TPOT 不构成门”；冻结代理量等于同链缓存浪费 5.3M token；按冻结标签得到 43 条 fast 超时；8.1–8.5k token/s 等于机器纯 prefill 能力；在飞 prompt 长度等于 KV 驻留；`1−p10/mean` 等于实测 decode 停顿比例；“闭环负载必使 MTP 降 N”；从公开榜单推断对手的具体实现。更正与反例见 [R19](../../codex/R19_progress_and_cache_review.md)。

## 评测与工程边界

单档是否通过，要用 harness 原评分器和 task.md 的 11 道门核对完整 cohort。TTFT 四门带统计余量，`tpot_p95≤0.10` 没有余量。空 raw、缺请求、缺指标、flush 未确认都不能判通过。`meta_info` 时间戳和 token 计数必须如实，`/flush_cache` 必须真清缓存。

代码实验按一个改动一次 A/B 进行；若结果改写了上面的判断，同时更新本文件、补丁说明和实验记录。数值、能力与正式榜单要分别报告，不能以 12 题冒烟推断两科能力已过线。
