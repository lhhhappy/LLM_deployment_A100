# 底包源码地图汇总 · 新主线 · 本地探索复盘（2026-09-22）

依据：`build/base_exact/`（L3 实际运行的 SGLang，证据链见 F54）与四份分图：
[01 请求入口](01-request-path.md) · [02 调度器](02-scheduler.md) · [03 混合缓存](03-hybrid-cache.md) · [04 模型与算子](04-model-kernels.md)。
标 ✔ 的结论由 Claude 对照代码复核过；其余出自分图，带 `文件:行号`，未复核。

## 一、源码告诉我们的事（按对 N@SLO 的影响排序）

### 1. 正确性风险：A100 上 DSA 预填充后端（✔ 代码成立，运行时未证）
默认参数下，sm80 的 DSA 预填充后端 = `flashmla_sparse`（`arg_groups/overrides.py:740-744`）。
GLM `index_kpool=4`，`_resolve_kpool_tail_backend` 只在 sm9x/sm10+ 把它换成 fa3/trtllm（`layers/attention/dsa_backend.py:858-873`）；
sm80 保持原样，随后 `_check_kpool_tail_backend` 在存在 top-k 索引时抛 `NotImplementedError`（`:840-856`，调用 `:2946-2947`）。
主办方说明里要求 A100 设 `SGLANG_OPT_USE_TOPK_V2=0`，说明他们在 A100 上跑过这个底包，所以长 prompt 可能走了别的路径（如 dense MHA）而不触发。
**A/B（45734/45735）都没显式指定 DSA 后端**：若运行时触发，两个提交都会失败。
→ 必须实测。最快途径：GPU 机 2×A100 装 base_exact，替身模型带一层 DSA（index_kpool=4），发一个 >2048 token 的请求（见第三节 L1-1）。

### 2. 调度：长冷预填充独占 GPU（✔ 已复核）
- 有预填充就不跑 decode（`managers/scheduler.py:3476-3484`）；`prefill_decode_interval=0`，mixed chunk 默认关。
- 正在分块的请求先入批并吃掉整轮分块预算，准入循环随即退出（`scheduler.py:3657-3680`，`schedule_policy.py:810`）。
- 结果：一个 10 万 token 冷启动 ≈ 13 轮（每轮 8192）连续预填充，期间所有在跑请求停止出字、链中间短请求全部排队。
  这直接打 fast_intra/overall_intra 和 tpot_p95；与领先者画像一致：intra 很快（1.79s/3.00s）、chain_start 放到 61s（统计余量内）。
- 其他：lpm 按绝对命中长度排序、等待 >128 退化 fcfs（`schedule_policy.py:293,384`）；优先级调度不能与 lpm 同用（`validation_hook.py:142`）；
  第一个放不下的请求会 `break` 并让 `batch_is_full` 粘住（`scheduler.py:3581,3747,3755`）。

### 3. KDA 状态复用（缓存是 UnifiedRadixCache，✔ 已复核 `mem_cache/registry.py:89,143,165`）
- 每次 extend 只记一个状态；分叉点优先于末尾（`schedule_batch.py:2836-2853`）；decode 每 256 token 记一次，只有最后一个在结束时入树，其后 KV 全部释放（`unified_radix_cache.py:882-905`）。
- KV 命中被截到最深的带状态节点（`unified_tree_core.py:844`）；分叉点状态记的是**上一轮**的分叉位置 → 下一轮多重算一整轮新增 token（推断）。
- 精度：中途状态取自 bf16 的 h（`chunk_delta_h.py:349`），末尾/分块末状态是 fp32。101（拆分）拿到的是 fp32 精确状态。
- 容量（估算，需开机日志核实）：每个 KDA 状态约 18.5 MB/卡，约等于 1450 个 KV token；状态约占缓存预算 47%；先耗尽的是 KV 不是状态槽。

### 4. 请求入口
- 原版 `/flush_cache` 返回纯文本、只看 `[0]`；**000 已在底包上修好**（✔ 已在 base_exact 上打补丁核对：JSON、全 worker、接收时间戳）。
- `X-S1-*` 请求头无处读取；`routing_key` 可经 JSON body 传入，只被 routing-key 策略和指标用。
- 分词在单个事件循环里同步做、无缓存（`tokenizer_manager.py:926-938`）；25 万 token 可能 0.4–2s（未测）。

### 5. 模型与算子（A100）
- KDA 只有 Triton 路径可用；预填充 `chunk_kda`（chunk 64），decode 不走更快的 packed 版（`kda_backend.py:654`）。
- MoE 走 Triton FP8 block kernel，镜像里没有 A100 的 288 专家调优配置（只有 A800）。
- 预填充不用 CUDA graph（`cuda_graph_hook.py:277`）。
- MTP 可用（`--speculative-algorithm NEXTN` + 必须给 `--speculative-draft-model-path /mnt/models`）；开启后 `max_running_requests` 被设为 48（除非显式给）。
- DP8 attention 不适合：分块预算 /8（8192→1024）、每个 KDA 状态 ×8（~148MB/卡）、各分片独立缓存。

## 二、新主线（结合 vLLM #56960 与 SGLang #31170）

| 顺序 | 主线项 | 对应瓶颈 | 做法 | 两个 PR 的作用 |
|---|---|---|---|---|
| M0 | **先保证能跑** | DSA sm80 风险 | L1 实测 base_exact + 带 DSA 的替身；必要时提交命令显式指定 DSA 后端（候选 `--dsa-prefill-backend fa3`，sm80 是否可用须实测） | — |
| M1 | **调度：保护链中间请求** | 冷启动独占 GPU（第一节 2） | 冷启动分块期间穿插 decode；缓存命中的短请求优先于冷启动分块续算；限制冷启动每轮分块；按门限截止时间排序 | — （源码里新发现的主瓶颈） |
| M2 | **KDA 状态复用（D1 的正确实现）** | reminder 分叉 + 状态复用滞后一轮 | 一次预填充同时导出"角色边界"和"末尾"两个状态、都用 fp32（在 `chunk_delta_h` kernel 里对指定 chunk 输出 fp32 快照，再像 #56960 的 exporter 一样连同卷积历史写入状态槽）；再加：淘汰时先丢 reminder 之后的无用状态 | **#56960 直接给出做法**：kernel 内导出中途快照 + 卷积历史 + 验证方法 |
| M3 | 入口：分词与路由键 | 分词阻塞事件循环；路由键没接 | 把分词移出事件循环（或按会话缓存前缀分词）；`X-S1-Routing-Key`/`Session-ID` 接到请求字段，供 M1 调度用 | **#31170 的"亲和+过载退让"思路**用在单调度器内的会话感知排序，而不是 DP 分发 |
| M4 | TPOT | tpot_p95 / 同分排名 | MTP（NEXTN），配合显式 `max_running_requests` | — |
| 放弃/暂缓 | DP8 + 亲和路由 | — | DP8 与本负载不匹配（第一节 5） | #31170 只在将来做小 DP（如 DP2）时再用 |

说明：M1 是本次读源码最大的新发现，不依赖 PR，改动集中在 `scheduler.py` / `schedule_policy.py`；M2 是 D1 的"正确版本"，101 是它的过渡版（精确但多一轮、限准入）。

## 三、本地探索（L1）复盘

**做过什么**：GPU 机 2×A100 上用 SGLang v0.5.20 + 随机权重替身（3 KDA + 1 MLA、无 DSA）跑 E1/E2/E2b，验证 D1 v1.1/v1.2 的缓存行为；另有离线模拟器、SPF(002)/004 补丁线。

**仍然成立的**：原版确实丢失可复用状态（E1：72 请求中 70 个与预测吻合）；D1 让 reminder-heavy 的未命中 p95 从 8135 降到 2439、0 退步——**机制方向对**。

**作废或不可迁移的**：
- 代码不同：v0.5.20 用 MambaRadixCache、有 fp32 快照缓冲；底包用 UnifiedRadixCache、中途状态是 bf16——L1 的精度与缓存细节不代表底包。
- 模型不同：替身没有 DSA，所以**第一节 1 的风险 L1 完全测不到**。
- SPF(002)、004、HRRN、模拟器的所有定量结论。

**花得不值的**：先假设底包 = v0.5.20 就写补丁（001/002/004 全打不上）；模拟器投入；40 项调参队列；取源码时绕了 6 次诊断构建，而底包提交本就在 GitHub 公开。
**根因**：没有先把"底包到底是什么"这件最基本的事查清，就在假设上堆工作。

**今后 L1 怎么做**：
1. L1-1 在 GPU 机装 base_exact（= 提交用的代码，打同样的补丁），替身模型加一层 DSA（index_kpool=4），先测第一节 1 的风险。
2. L1 只回答"对不对"：接口合规、缓存命中、数值一致（fp32 vs bf16 状态续算的 logits 差）、不崩；性能一律交给 L2。
3. 每个补丁先在 `build/l3_<tag>` 生成提交时的完整代码，再上 L1，保证测的就是要交的。
