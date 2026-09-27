# 高并发目标：用有限的计算救回更多达标请求

2026-09-27，Codex；结合两个 `gpt-6-sol` 子代理的独立调研，主代理复核赛规、当前调度代码及关键外部来源。本文是目标与设计判断，未修改引擎、运行 GPU 或改变实验队列；不承诺新的正式档位。

## 要优化的究竟是什么

按 [task.md 的排名规则](../../llm-challenge-arena-v1/task.md)，第一顺位是所有硬门通过的最大逻辑会话并发 `N@SLO`，第二顺位才是该档 `tpot_mean`。这里的 N 包含正在执行工具或等待下一轮的会话，不等于 GPU running 数、HTTP 连接数或每个 batch 的大小。TPM 是诊断，不参与排名。

因此在两个配置都完整过门的假设下，N30、TPOT 均值 50 ms 优于 N26、25 ms。这个例子只是解释排名，不是新的运行成绩。当前应优先争取更多请求在各自期限内完成首 token，并保住 TPOT p95 与其他硬门；不需要把每一条 chain 的 TTFT 都做到 30 s 以下。

TTFT 门看超标率及统计余量。已经超过 30 s 的 chain 从 60 s 缩到 40 s，单独看这道门仍是一条超标；将一条本来 32 s 的请求救到 28 s 才少一条。允许暂时降低前者的执行优先级，把有限的算力给后者。但所有请求仍须完整执行，保持真实时间戳、输入输出与缓存合同；不能靠取消、无限挂起或只算已完成请求减少分母。

“341 条链最多 24 条超时”不是可写进引擎的固定额度。[原 harness](/workspace/Agentic_science_challenge/s1-dev/harness/s1_common.py) 的 chain 桶还包括 `context_reset`，桶样本数不一定等于链数；统计余量随实际样本数变动。主办方只公开了单侧置信下界的规则，本地 [score_formal.py](../../scripts/score_formal.py) 的 Clopper–Pearson 计算明确是估计。优化依据应当是运行时工作量、等待时间和依赖，不依赖评测 ID、隐藏桶标签或预设超时名额。

## 当前代码已经做到了哪里

124 的源码调用关系是 `Scheduler._ax_admission_plan` → `tier_order` / `PrefillAdder.ax_complete_waiter_budget` / `should_park`。在 [ax_deadline.py](../../engine/sglang/srt/managers/ax_deadline.py) 中，124 已经将等待者按预计可救、预计无望分层，并为不在 LPM `held` 集合中的请求设等待提升阈值；不是完全先到先服务。`held` 的排序检查早于 aging，这个阈值不是对每条请求的最大时延保证。slack 使用剩余工作成本和已等待时间，忽略前方队列，因而属于乐观的单请求估计；deadline 类别也只是服务端缓存形态的启发式，不等于评测桶。

[schedule_policy.py](../../engine/sglang/srt/managers/schedule_policy.py) 的 `ax_complete_waiter_budget` 要求 waiter 在一轮预算内完成，否则返回 0；`should_park` 再检查完整完成的预算与 KV。当前一个全局 `chunked_req` 的状态约束使长 continuation 不能给另一个需要数轮的冷请求持续让路。128p 为 READY 兄弟提供的是可完整结束的短尾准入。

所以下一个具体缺口是：**把已经存在的“可救者优先”扩展到进行中的长 prefill，在安全块边界暂停旧请求，让更有机会达标的请求获得足够的连续计算。** 多 partial 是实现这种选择的状态管理能力，不是要求平均分配 token 或每轮轮转。开场同到达、同期限的独立请求若已按准确的最短剩余工作排序，平均轮转可能让达标数减少，不能仅因支持多 partial 就宣称开场获益。

## 我建议的机制与取舍

1. **比较本轮选择的净收益。** 以实测块长 × 上下文 × batch 成本估计剩余首 token 时间，包括不可避的 decode/通信与切换开销。比较继续当前请求和切换候选后，哪些请求会跨过各自期限；不能只看被救者。已超时与“预测无望”分开，后者每轮随实际速度、前缀 READY 和剩余工作重新评估。
2. **给已晚请求保留有界进度。** 不能把 aging 简化为一次性绝对抢占后独占到完成；采用有界块服务，并记录最长未服务时间。仍需测量额外 KV、KDA 槽、锁、恢复与重算成本。若长期停住的请求占满池或卡住闭环会话槽，救了眼前几条也可能增加全场 miss。
3. **按依赖修正单请求价值。** 已晚请求如果即将生产多条可救兄弟的共享前缀，其下一块仍有价值；临近完成能释放大量 KV 的请求也可能应该先做。只按自身 slack 排序会错过这些收益。
4. **先验证两个 partial 的正确接续，再验证规则。** 每个请求独立保存真实进度、DSA KV、KDA checkpoint、MTP 状态和缓存锁；rank 0 生成一致的计划。先在默认关闭的独立候选里验证两个 partial 交替与恢复，然后与现行单 partial 做同配置、同 ID 比较。状态契约正确不等于调度有收益。

边界例应包含：已晚的大冷请求遇到一个尚可救的中冷请求；两个同到达的大冷请求；临近 READY 的前缀生产者遇到短请求涌入；低 KV 余量下的暂停与恢复；成本估计误差；开场顺序改变；稳态晚到的 context reset。观察 chain 修复数与新增数，同时检查其他 TTFT 桶、全时段 TPOT p95、最长等待和排空完整性。窗口内改善只作诊断，完整回放决定是否过档。

## 外部实践与执行层配合

[JITServe](https://arxiv.org/pdf/2504.20068v3) 明确将满足 SLO 的有效服务作为调度目标，并处理不精确的工作量预测。其默认主要指标按有效 token 计，§3 另明确支持并验证请求级 goodput；这些都不是本赛 N@SLO。[vLLM V1 当前源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) 以请求独立进度和长 prefill 的每轮预算限制提供实现参考。这两者支持方向，不证明 GLM 混合 KDA/DSA、DCP2、MTP 上的正确性或收益。详细适配与反例见[调度调研](external-prefill-scheduling-sol-0927.md)。

调度只决定谁先得到现有产能；当开场几乎全在 prefill 时，不能靠减少 decode 创造大量新时间。执行侧优先找长上下文 kpool indexer 的真实关键路径，其次按形状验证已有 170 graph 能否降低小块固定成本。A100 上公开 indexer 加速示例是相对 Torch decode fallback，不能套成我们 ragged prefill 的加速倍数；prefill CP 也不能当成 DCP 的现成开关收益。细节、显存与最小区分性实验见[执行调研](external-prefill-execution-sol-0927.md)。

推荐的近期目标是：在不破坏其他门与完整执行的条件下，让“已经晚的长请求”少挡住“仍能达标的请求”，逐请求证明净 miss 下降；同时从执行层降低每个长块的真实成本。现有结果不足以保证 N34/N38 通过。
