# R34 — 显存容量墙：我们这套栈上能做什么（2026-09-26，Sonnet 子代理只读调研，fable 整理）

背景：本地 N30（111）四门全挂的原因是准入前等显存（KV 池 99.3%、同时活跃 19 路 × 85k > 1.4M 池）；线上 N22 只约 8 路活跃不到墙，但 N34 按第一名的形态（16 路同时解码）会撞上同一堵墙。本篇回答"每卡显存还能挤出多少给 KV、FP8 KV / DCP / 独占式主机层各要多少工程"。标注：【实测】evidence 日志；【代码】engine/sglang 行号；【上游】公开资料；【推断】算术或判断。

## 1. 每卡显存账（L109 server.log 实测）

| 组成 | 每卡 |
|---|---:|
| 权重 FP8 | 39.15 GiB |
| MTP 草稿权重 | 0.85 |
| KDA 状态池 418 槽（conv 0.25 + ssm 7.16 + intermediate 2.26+0.08） | 9.75 |
| 目标 KV 11 层 DSA：1,397,760 token × 12,716 B | 16.55 |
| 草稿 KV 1 层 | 1.50 |
| CUDA graph（verify + draft） | 2.17 |
| 启动完成后剩余（激活/工作区） | 7.73 |

公式（【代码】`kv_cache_configurator.py:2052-2098`）：`slack = 载权重前可用 × (1 − mem_fraction_static)` 永远不进池子；`rest = 载权重后可用 − slack − 0.10`；再由 `mamba_full_memory_ratio=0.9`（状态池:KV ≈ 0.9:1，即状态池拿 47%）联立解出整数槽数 K（`:2392-2395`），每在飞请求占 5 槽（基础 3 + overlap/extra_buffer 2，`runtime_context.py:1672-1690`），每槽 17.6 MiB → K=418 与日志逐位吻合。用实测数反推 slack ≈ 7.73+2.17+0.10 = 10.0 GiB ≈ 77.8×0.13，公式闭合。

**隐藏耦合**：`resolve_max_num_reqs`（`:2167-2200`）在 `max_mamba_cache_size // 5 < max_running_requests` 时**静默压低并发**：槽数 200 → 并发 40，槽数 256 → 51。任何砍状态池的实验若目标并发 48，必须用 ≥ 240 槽，否则单变量被污染。

三份日志（081/082/111）`mamba usage` 峰值 0.29–0.35：在飞最多约 146 槽，其余约 270 槽是 radix 树里的 KDA 检查点（`mamba_track_interval=256`）。砍槽 = 少留老会话的记忆点，回来时可能整段重算；这笔代价目前没有计数器，Fable 顾问与 fable 均决定先加"检查点未命中→重算"计数器再砍。

### 措施表（每卡；KV token 按含草稿层的边际 13,872 B/token 折算）

| 措施 | 省下 | 折合 KV token | 风险 | 验证 |
|---|---:|---:|---|---|
| mem 0.87→0.90，**同时 `--max-mamba-cache-size 418`** | +2.34 GiB | +181k（+13%） | 唯一的激活峰值证据 L081p：常驻 76.4 GiB，16k 块峰值 +1.82 GiB → 78.4/80；0.90 后余量约 1.6 GiB 以下 | 60 分钟单变量，看长块 + 满并发是否 OOM |
| mem 0.92 + 钉住 418 | +3.89 | +301k（+21%） | 余量更薄，需把 `max_prefill_tokens` 16384→8192 兜底 | 先最坏形态探针 |
| `--max-mamba-cache-size` 418→256 | +2.78 | +215k（+15%） | 检查点变少（见上），并发上限 51 | 先加计数器摸底 |
| 418→200 | +3.75 | +290k | 同上，且并发被压到 40 | 不要用 200 |
| `SGLANG_MAMBA_SSM_DTYPE` float32→bfloat16 | ≈4.7 | +363k（+26%） | KDA 递推状态 bf16 累积，长上下文漂移未验证 | numcheck + 能力题，再 8 卡 |
| 关 MTP 且钉住状态槽 | 5.5–5.8 | +420–450k（+30%） | TPOT 变差（接受长度 2.7–3.0 消失）；不钉槽则非投机分支解出约 750 槽、KV 不涨 | 看 TPOT p95 与 chain |
| 并发/graph 32→48 | 图 +1.1 GiB、中间态 +1.1 | KV 净变化≈0 | 与其他项联立求解，不能相加 | 组合一起测 |

## 2. FP8 KV：存储层已在代码里，计算层没有

- 【代码】`memory_pool.py:4293-4297/4416-4457`：`dsa_kv_cache_store_fp8` 时写入 512 B fp8 + 16 B scale = 528 B/层（`calculate_mla_kv_cache_dim`，`kv_cache_configurator.py:2441-2495`），11 层 + indexer 132 B = 7,260 B/token → ×1.75，与 R24 推算、上游 #36830 H20 实测一致。
- 【代码】消费这条布局的 kernel 只有 `flashmla_sparse_q8`（`dsa_backend.py:648-664` 显式 SM90-only）和 decode 的 `flashmla_kv`；只在 `dsa_prefill_impl == "flashmla_auto"` 时启用。我们部署的 tilelang 主 kernel（`kernels/ops/attention/dsa/tilelang_kernel.py:305` 附近）dtype 写死、没有 bf16-Q × fp8-KV 分支；110/114/170 改过的正是这条路径。
- 【上游】vLLM #57144：Triton 在 sm80 无法表示 fp8e4nv，但 TileLang 可以在 SM80 做 e4m3 转换——与我们 indexer 已在 sm80 用 tilelang 存 fp8 一致，tilelang 是对的基座。
- 工程量：A 试 `--dsa-prefill-backend flashmla_auto`（0 行代码，大概率 sm90-only，0.5–1 天止损）；B 自研 tilelang 反量化分支 + 联动 110/114/170 地址算式（5–10 天，数值风险最高）。草稿层需同步 fp8（`--speculative-draft-kv-cache-dtype`），HiCache 主机行宽自动跟随（未 GPU 验证）。

## 3. DCP（序列分片 KV）

- 【代码核查，2026-09-26更正】025是虚拟loc/物理池协议问题，历史115/116与r7覆盖该修复及普通decode graph。041的33/40是另一处真实/补齐行数错误，基线`aebaff56`仍能复现；不能说只差8卡。新修复与实际函数探针见[DCP计划](../../notes/plan-dcp-8card.md)。
- 【代码】当前topk=1链式MTP不调用接受KV压实；`move_kv_cache`的跨rank修复是工程完整性项，不能解释为现有链式MTP唯一卡点。真实缺口包括草稿分配与写入布局不符、DRAFT_EXTEND_V2未完整Q gather/LSE合并、HiCache indexer主机池覆盖不足及组合guard。调用点审查见[记录](../../notes/reports/dcp-code-review-20260926.md)。
- 【上游】#39330（Hopper 专用）：H200×8 DCP8 每副本 2.34M 可调度 token vs TP8 505k（4.6×）；评论区吞吐 +19–48%、TTFT p99 → 0.46×，但 ITL（≈TPOT）+25%。vLLM 与 SGLang roadmap 均不覆盖混合线性注意力模型——这条路没有上游先例。
- 【算术】DCP8每DSA层每逻辑token每卡`1024/8+132=260 B`。草稿也按真实写入协议分片时，11目标层+1草稿层的KV容量预算比为4.45；草稿复制的替代方案为3.45。KDA、图池、collective临时量另计，不能把容量比换成并发档位保证。

## 4. 独占式主机层：底包已有 `write_back`

- 【上游/代码】`--hicache-write-policy write_back`（只在设备淘汰时写回主机）与 `write_through_selective` 是现成选项（`server_args.py:2740-2747`）；180 的测试只覆盖了 write_through。
- 【代码】write_back 的 D2H 备份在设备淘汰临界路径上同步发生，成功才释放设备槽；主机满且未加锁则整段丢弃；有 `_reclaim_full_host_duplicates` 处理包含式冗余。设备与主机各自 LRU，主机满后两边顺序分叉，包含式保证已打折。
- 有效容量【推断】：64 GB 主机独占后多覆盖约 +1.4M token 冷内容；搬回延迟不受写策略影响（60k token ≈ 33–66 ms vs 重算约 4.9 s）。
- 零代码实验：write_through→write_back 60 分钟单变量；再复测 write_through_selective。真正的独占+去重是新状态机，3–5 天，无先例。

## 5. 准入侧

- 【代码】等待队列不占 KV（分配在 `prepare_for_extend`）；例外：前缀匹配时 KDA 状态 COW 可能先占 1 槽（`mamba_component.py:188-212`）。
- 已完成请求的 KV 是普通可淘汰节点，无会话级 pin；140 只是软优先级。
- 【实测】081/082 `num_retracted_reqs` 在 647/654 个采样点全为 0：N34 的坏化不是 decode 中途被抢，是准入排队（R31：新增 fast 坏例待算量中位 946 token，接收→首次执行 +5.49 s）。

## 6. 按"容量倍数 ÷ 工程量"的清单

| 序 | 措施 | 人天 | 增益 | 状态 |
|---|---|---:|---|---|
| 0 | 修排队实验：mamba 200→256；mem089 与 nomtp 钉住 `--max-mamba-cache-size 418` | 0.1 | — | 已改（139/140/142） |
| 1 | mem 0.90 + 钉 418 | 0.5 | +13% | 待排 |
| 2 | DCP保留链式MTP：先修补齐、草稿布局与HiCache，再TP8验证 | 以验收为准 | W8 KV预算×4.45，净收益待测 | 见现行DCP计划 |
| 3 | `write_back`（零代码） | 0.5 | 主机 +1.4M | 已排（143） |
| 3b | 复测 `write_through_selective` | 0.5 | 未知 | 待排 |
| 4 | 试 `flashmla_auto`（开发机冒烟） | 0.5–1 | ×1.75 若可行 | 待做 |
| 5 | mamba 418→256 + 检查点未命中计数器 | 2 | +15% | 计数器待写；139 先量 11 门与缓存命中 |
| 6 | `SGLANG_MAMBA_SSM_DTYPE=bfloat16` + numcheck | 0.2+ | +26% | 已排（144，能力冒烟开） |
| 7 | 关 MTP + 钉槽 | 0.2 | +30%，TPOT 代价 | 已排（142） |
| 8 | FP8 KV tilelang 反量化 kernel | 5–10 | ×1.75 | 执行层立项 |
| 9 | DCP跨owner KV搬运与错位用例 | 以验收为准 | 工程完整性；topk=1不依赖该压实 | 属DCP计划；压缩树式搬运另验 |
| 10 | 真正独占式 + 去重 | 3–5 | <2× | 看 3 的结果再定 |

未核实：COW 槽在候选未接纳时是否释放；`move_kv_cache`跨owner原语只有两卡合成池证据，压缩树式MTP与TP8仍未覆盖；fp8 反量化在 tilelang/sm80 的最小可行性；write_back 与 256 组所有权/140 的组合只做了代码走读；250k+ 上下文 indexer 临时缓冲的激活峰值；DCP 在 A100 的实际 TPOT 代价。
