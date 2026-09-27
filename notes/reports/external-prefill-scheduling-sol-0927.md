# 外部 prefill 调度实践：在单个 TP8 副本里先消除长冷请求的独占

2026-09-27，Codex 子代理。范围：只读本地源码、既有 raw 审计和外部一手论文、官方源码/文档；没有修改引擎、运行 GPU 或推断正式失败档的逐请求原因。外部 `main` 链接按 2026-09-27 查阅，后续会变；论文数字均为其自身模型、硬件、负载上的报告，不是本赛预测。

## 决策前的容量账与当前缺口

目标先是完整回放的 `n_at_slo`，再是同档 `tpot_mean`。正式 46677/S2 为 N26、chain p95 39.13 s、TPOT 25.10/47.54 ms（均值/p95）；46758/S4 也是 N26，chain 41.11 s、turn 9.43 s。本地共同请求中，S4 路径再加 128p 的开场 chain 超时 **11→11**，修复/新增互抵；N34→N38 的同 30 分钟窗口又出现 **11→14**。这些不是正式失败档归因。[正式复盘](official-results-retrospective-0927.md)与[开场核验](chain-opening-source-audit-0927.md)已经表明一个 157k 冷请求 TTFT 约 96 s，但首次入批后约 14 s，提示它的主要损失发生在首次入批前；另一个 256k 请求首次入批后到首 token 约 33 s。后一个区间仍含分块间让路、调度与处理时间，不能当作纯 GPU 时间，更不能据此判它在 30 s 门内物理不可救。

当前 [scheduler.py](../../engine/sglang/srt/managers/scheduler.py) 的 `_get_new_batch_prefill_raw` 在 `chunked_req` 存在时优先调用 `PrefillAdder.add_chunked_req`；[schedule_policy.py](../../engine/sglang/srt/managers/schedule_policy.py) 以单个 `new_chunked_req` 和 `assert self.chunked_req is None` 保持全局仅一个 partial。124 的 `should_park` 只接受本轮预算和 KV 都能**完整**做完的 waiter；128p 的 READY 也只尝试完整 tail。故后到的 71k 冷请求不能替换 229k continuation，已有坏例等待 25.75 s 后还需 8.25 s。`stash_chunked_request` 在块间插入前缀树，但这不等于可安全同时挂起多个活的 KDA/DSA partial。125 正在由 Fable 负责，本报告不建议改它。

另外，开场 20 s 的既有 profile 是约 98.7% prefill、0% decode；那段时间没有可供 prefill 兑换的 decode 时间。现有 TP8 每卡 target 权重约 39.15 GiB、draft 0.85 GiB；TP4+TP4 两份 target 每卡粗算 78.3 GiB，几乎不留 KV/KDA/激活空间。因此以下以**单个 TP8 副本**为优先条件。链首、turn、overall、fast 的期限分别是 30/15/5/3 s；TPOT p95 ≤100 ms 是硬门，不能用当前均值 25 ms 推算“四倍空闲预算”。

### 真正的目标是少一个 miss，而非所有请求的 TTFT 都变小

用户提出的关键取舍正确：若请求的 TTFT 已注定超过 30 s，把它从 60 s 改成 40 s 通常不减少该桶的超时条数；把仍可能完成的 32 s 改成 28 s 则可能减少一条。**目标是在固定 N、完整执行全部请求且所有门都满足的条件下减少净 miss；先争取 N@SLO 晋档，再比较同档 TPOT 均值。**正式门是原 harness 的分位数、统计余量和其他十门的联立判定，不能把“可容忍若干失败”硬编码成超时配额。尤其不能把 341 条会话链当作 chain 桶固定样本数：该桶还含 context_reset；即便假设 n=341，所谓“最多 24 条”也只是在特定本地 CP 估计法下的算术估计，官方 CI 实现未公开。所有请求仍须完整执行，不取消、不改 token 合同；已经晚的请求可以暂时降级，但必须有有界进度。若它持有其他请求将命中的前缀，它的下一个块还可能降低多条后续请求的工作量，不能仅凭本请求 slack 判为低价值。

[JITServe（arXiv:2504.20068v3，2025-12；早期版本名 Tempo）](https://arxiv.org/pdf/2504.20068v3)是最贴近这点的外部实践：在 vLLM 上按保守的剩余工作与期限估计分配“刚好够达到 SLO”的带宽，逐步修正估计，把预期有效服务/工作量作为排序信号，并给长期等待者增加优先级。其 v3 算法见论文 §4.2 `GMAX`，后端实现说明在 §5；论文报告 1.4–6.3× *service goodput*，不是我们按请求计数的 N@SLO，更不是长上下文 GLM 的预计倍率。我们可借鉴的是**可行性分类和 aging**，不应照搬它的按 token 计算服务价值。对本赛一个自然的在线排序是：有保守完成时间上界且仍可达标的请求优先；已晚或保守估计无望者分配最小保障进度；两者内再按预计完成成本、前缀生产收益和等待年龄排。须把“上界预测判无望但其实可救”作为最高优先级的误判风险，保留探索或重估机会。

[QLM SoCC 2024 论文](https://haoran-qiu.com/pdf/socc24-qlm.pdf)的请求等待时间估计和排序提供另一个边界：它解决多模型、多 GPU、batch 与交互混部，报告自身负载下 SLO attainment 提高 40–90%；其多实例 request eviction/model swapping 不是本节点可直接迁移的机制。可取的是每次按**真实排队及剩余服务**重估，而不是只按入队先后或固定 prompt 长度推断；我们的闭环会话链使“某请求完成后释放 client slot 并产生下一轮”的未来成本尤其重要。

## 1. 最值得实验：多 partial 的有界暂停与恢复（vLLM）

[vLLM V1 scheduler 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)把每个请求的 `num_computed_tokens` 当进度，`schedule()` 在同一轮遍历 running 与 waiting；`long_prefill_token_threshold` 在有竞争者时限制单个长请求本轮 token，单独运行时解除限制，遇到无法安排的请求可 `continue` 寻找后续可运行者。[官方配置文档](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)还描述按排队加运行数量分摊 token 预算的 adaptive 下限。这里是**代码行为**，没有可直接外推的 vLLM 官方 GLM/A100 速度结果。旧 [vLLM 0.10.1 引擎参数](https://docs.vllm.ai/en/v0.10.1/configuration/engine_args.html)另明说 `max_long_partial_prefills < max_num_partial_prefills` 可让短 prompt 在部分场景越过长 prompt；版本不能与今日 V1 代码混为同一实现。

迁移思路是先让多个冷请求能保存独立 partial 进度，并安全暂停、恢复。这是一项**调度能力**，不意味着平均分块或公平轮转：开场同到达、同 deadline 时，若短作业优先已经能使更多请求及时完成，强行交替反会减少达标数。`active_partial` 数应小且有上界；没有可证明的跨门收益时让当前请求继续用高效块。一个 229k 已推进请求遇到 71k 冷请求时，先用在线可见的年龄、真实未命中工作、30 s slack、前缀生产收益与 KV 占用评估切换能否增加净达标请求，再决定是否让后者连续获得块；不能仅因后者更短就轮转。成本是更多在飞 KV、KDA 槽与树节点锁，更多小块固定开销；若把已入批长请求长期搁置，会让它自身 chain 尾部恶化。当前 124/128 的单-owner 不变量、TP8 rank 一致的计划广播、DSA indexer 对齐、KDA 快照与 MTP target/draft 状态都要扩展，不能只把 `chunked_req` 改成列表。每个 partial 还须保留请求槽、KV 页、KDA state/checkpoint、实际 prefix/device/host hit、cache lock、重入时的对齐位置；先验证“两个冷 partial 可安全暂停、恢复并精确接续”，再验证按期限选择块是否真正减少净 miss。

**净收益判据**：同一批共同 ID 中，chain 的修复数大于新增数，特别核对独立长请求和早到请求是否被挤坏；记录真实输入 token/GPU 时间、KV/KDA 峰值、重算与 HiCache 迁移。仅短请求变快而 229k 请求改为超时，不算已证明 N@SLO 增益。

## 2. 次优先：可行性、期限与缓存收益一并计价，设置饥饿界限（JITServe / FastServe / 我们的 124、128）

[FastServe NSDI 2026 论文页](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)提出按预计首轮成本跳入多级反馈队列、逐生成 token 抢占、提前卸载/回迁 KV；[官方 artifact](https://github.com/MachineLearningSystem/NSDI26-FastServe/blob/main/benchmarks/artifact-evaluation/README.md)给出 `sj-mlfq`、`proactive-offloading`、`use-skip-join` 开关与单 GPU OPT-13B 示例。论文声称对其 vLLM 基线吞吐最多 6.1×；这是该系统和负载的上界报告，未覆盖 100k–256k GLM 混合 KDA/DSA、TP8、DCP2，也不是本赛 N 档收益。

可迁移的是“接近期限且仍可完成的请求可越过长作业，但有 aging/最大等待”；现有 124 的 [ax_deadline.py](../../engine/sglang/srt/managers/ax_deadline.py) **已经**按 slack 区分可救与无望、给长期等待者 aging，并限制停车次数和时长，128p 也已考虑生产者依赖。具体缺口是这些选择主要施于未入批队列：活跃长 partial 只能让给单轮完整做完的 waiter。71k 请求即使可能在剩余 30 s 内完成，也无法在 229k 的中途连续获得多个块。下一步应把现有 124 的选择扩展到进行中的 partial 与每轮块预算，而非重新发明一套队列优先级。先给每轮的 `chosen`, `skipped(reason)`, `estimated finish`, `actual finish`, `producer benefit`, `displaced deadline` 做小额结构化日志，再用同 ID 找到 128p 的新增边缘坏例。切换的目标是增加预期及时完成的**请求数**与可复用的后续工作，并保障每条请求最终完成。成本是牺牲部分 LPM 连续性、可能增加 host cache 恢复与 128p 生产者抢占；若服务时间模型不准，deadline 排序会频繁救错对象。论文式每 token 抢占主要针对 decode JCT，对开场 0% decode 不是直接解法；完整换出 100k–250k GLM 状态需要 DSA KV/indexer 与 KDA 状态一起保持一致，不能按普通 Transformer KV 直接照搬。

## 3. 条件实验：prefill/decode 预算随活跃 decode 期限变化（Sarathi-Serve、vLLM）

[Sarathi-Serve OSDI 2024 论文](https://apanwariisc.github.io/publications/osdi-2024-sarathi-serve/osdi24-sarathiserve.pdf)的算法先放进行中的 decode，再用一个 prefill chunk 填余量，尽量让 decode 随 prefill 一同前向。论文在 Mistral-7B 单 A100、Yi-34B 双 A100 等报告容量收益；模型是普通 Transformer，与我们的 34 层 KDA、11 层 DSA 及 MTP 不同。[vLLM 0.10.1 `_schedule_chunked_prefill`](https://docs.vllm.ai/en/v0.10.1/api/vllm/core/scheduler.html)也明确先安排 decode、再安排 partial 和新 prefill。两者说明这是实际运行的策略，不证明它适用于我们的混合 kernel。

我们已有 prefill/decode interval、mixed chunk 入口、125 backlog relief，以及现行 `scheduler.py` 中 `_ax_pace_limits` 的基于 decode slack 限块。更有价值的实验不是固定把 interval 调到 1 或把 TPOT 均值拖高，而是用块长×上下文×batch 的**实测**成本预测下一块是否会撞当前最紧的 decode 期限：有 decode 时选择最大安全对齐块；没有 decode 时全速 prefill。130ez7 的局部反例已见 interval 2→1 使 TPOT p95 约 98→112 ms 而 chain 不变。MTP 下 mixed batch 的 target/draft、KDA 状态更新和 DCP 通信必须在 TP8 真机检查。接受更高 TPOT 均值只能以 p95 保门、chain/turn/fast/overall 净收益为条件。它对开场长等待的直接帮助很小，优先级低于多 partial。

## 4. 当前最不值得做：同节点 P/D 分离（DistServe）

[DistServe OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)把 prefill 与 decode 放到**不同 GPU**，以减少干扰并分别选并行方式；其 13B、512 输入/64 输出、单 A100 的示例与本赛长冷提示差距很大。SGLang [官方 P/D 文档](https://docs.sglang.ai/backend/pd_disaggregation.html)与本地 `srt/disaggregation/prefill.py`、`decode.py` 确有 bootstrap、预分配、传送机制；本地代码的 prefill send 还要处理 Mamba 状态和 DCP 物理页边界，说明它绝非搬普通 KV 一个张量即可。

在仅 8×A100 上，TP4+TP4 双副本的 target 权重粗算每卡约 78.3 GiB，超出可用 KV/KDA/激活预算；TP8+TP8 需要另一整组 GPU。即使另有 GPU，长上下文每请求的 DSA KV/indexer、KDA state、MTP 状态与 HiCache 所有权、DCP2 布局都须正确迁移，且传送延迟占 TTFT。更关键的是，开场没有 decode 干扰可消除。因此它是未来扩容路线，不是当前 N30/N34 攻坚臂；不要以论文 7.4× goodput 作为本节点预期。

## 实验顺序与反例

1. **先补观测，不碰 125 逻辑。** 对当前 S4 与合并 128p 同 ID 记录每个请求的每次可运行、跳过原因、块工作量、锁/槽/KV 拒绝、cache hit 与实际 forward 时间；按正式 bucket 对上前 30 s、稳态和长冷插入。用原 harness、完整 raw 判分，窗口只作诊断。
2. **先做两路 cold partial 的正确性与成本账，再比较调度规则。** 对照一个全局 continuation、两个可安全暂停的 partial 但默认不切换、以及只在预测净达标数增加时切换的规则；固定其他开关与同一负载。核对 TP8 数值/状态、总 prefill 工作、重算、峰值 KV/KDA、TPOT p95 和四个 TTFT 桶。若切换只是让 71k 赢而 229k 输，或因驻留压力增加重算，则不算净收益。
3. **在 decode 活跃段评估自适应块预算。** 与固定 8k/16k 及现行 `_ax_pace` 配对；若 chain 净改善来自开场，应先怀疑 cohort 或别的机制，因为开场没有 decode 可让。
4. **须主动寻找反例。** (a) 两个独立 200k 冷请求同时到达，交替是否比串行多耗时并双超时；(b) 生产者快要完成时插入多个 3 s fast，是否丢掉大量 prefix reuse；(c) KV/KDA 槽接近满时第二 partial 是否导致拒绝或抖动；(d) DCP2+MTP 与 HiCache host hit 下恢复是否一致；(e) cohort 次序改变时净 chain 收益是否还在；(f) 成本模型把实际上 28 s 可完成的请求误判为“无望”，是否反而新增 miss；(g) 连续短请求让已晚长请求长期没有进度，或其持有 KV 阻止新请求准入；(h) 把已超时链首延后，是否因为闭环 client slot 未释放而推迟后续 turn/chain，导致全场更多 miss。最终只按完整同配置回放及正式可得结果决定档位，不把局部修复条数当晋档承诺。

一手来源的证据边界：上面引的 vLLM/SGLang `main` 是查阅日源码而非我们固定底包；Sarathi、FastServe、DistServe 的性能数字来自各自论文测试，没有 GLM-5.3-Flash+DCP2+TP8 的结果。本地文件与报告指向当前 worktree 的实现；其适用性须重新以选定引擎 commit 和运行收据锁定。
