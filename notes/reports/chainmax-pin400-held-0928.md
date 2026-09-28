# 两个可救链首为何让位于 222k 巨头

2026-09-28 UTC，Codex。范围是 ezn3（不钉池）与 ezn8（400 槽）的三个同 ID 开场请求。直接排序原因已由逐决策日志与源码对应；尚不能把更早的缓存状态分歧归因于钉池本身。本报告不判整档成绩。

## 同 ID 实测

三条都是 harness 的 `idx_in_chain=0`，即使业务 `phase` 为 intra/turn_start 也计入 chain。这里使用 raw 的服务端 `ttft_s` 与 `queue_time_s`，不混用客户端相对开场时间。

| 提示长度 | TTFT：不钉 → 400（秒） | 排队：不钉 → 400（秒） | 命中 token：不钉 → 400 |
|---|---:|---:|---:|
| 222,738 | 44.912 → 40.157 | 28.025 → 23.254 | 0 → 0 |
| 56,351 | 29.134 → 48.246 | 24.193 → 43.290 | 1280 → 256 |
| 35,399 | 21.869 → 43.632 | 17.850 → 39.453 | 1280 → 256 |

[逐请求 CSV](../../evidence/chainmax-fixes-0927/pin400-three-requests.csv) 保存完整 ID、raw 路径与行号。所读 ezn3 raw 有 1572 行，ezn8 是 1263 行的窗口快照；三个请求均已完成且各出现一次。因此此表可用于三个坏例，不能冒充闭合整轮配对。

## 直接原因：浅前缀 held 高于可救性

[400 臂决策](../../evidence/chainmax-fixes-0927/pin400-on-decisions.json) 的 epoch 2 / sequence 38，对应 Pod `130ezn8-tail_n26_chainmax16k_mamba400_40m/server.log` 第 1199 行：

- 没有在飞 continuation，冷块预算 16384。222k 巨头排第 0，获 `ordinary_admitted`。
- 56k、35k 分别排第 2、第 7；两者 `held_by` 都是这个巨头，`held_depth=413`，`effective_held=true`，结果均为 `not_attempted`。
- 三条均为 `ORDINARY`，`work=null`，没有 128p 家族折算或 READY rider 加成。
- 用该臂的 124 成本模型（load 1.05、每块 0.13 秒、每 token 68 微秒、arrival offset 0.4 秒）代入日志，巨头 slack 为 **−11.467 秒**，56k 为 **+1.778 秒**，35k 为 **+4.752 秒**。这是模型估计，正 slack 不保证实际过门，也不能把两条独立估计直接相加。

控制流是：`SchedulePolicy._compute_prefix_matches` → 原生模拟前缀树的 held → `ax_prefix_producer.Tracker.prepare` 排序 → rank0 plan → PrefillAdder 准入。`ax_prefix_producer.py` 的排序先把 effective held 放到 tier 3，随后才算 slack；无 held 的不可救请求是 tier 2。因此巨头仍然优先。124 原来的 `tier_order` 也有同样的 held 优先级。

日志没有显示这两条在这一轮因 KV 容量被拒绝：它们根本尚未尝试准入。也不是成本模型把巨头判断成可救。

## 巨头入批后的第二道阻塞

紧接着 sequence 39 / 第 1203 行，两个中等头都取得 256 token 的可用命中，held 已释放。但巨头变成 continuation，二者剩余 35,143 / 56,095 token，超过 16k 单轮预算。现有 `should_park` 要求候选头能在一个预算内完成，直接拒绝这种多轮救援。日志结果为 `not_reached:queue_stop_chunk_budget`。

所以单纯在下一轮重新排序救不了这两条：浅 held 先让巨头抢到通道，多轮停车缺口再延长阻塞。这两个机制都与“开场还是稳态”无关，稳态出现同样队形也会触发。

## 尚未证明的部分

[不钉池决策](../../evidence/chainmax-fixes-0927/pin400-off-decisions.json) 中，两条中等头在 sequence 13 / 第 1088 行已获得 device=1280 的真实命中并解除 held。400 臂早期却是 device=0、full=1280/2304：KV 前缀存在，但缺少可配套续算的状态检查点。后者一直等到巨头首块产出 256 的可用命中。

已有日志不足以说明更早的状态检查点为何不同；请求到达与代表执行顺序也不同。不能把“400 臂发生了这条阻塞链”写成“减到 400 必然淘汰了检查点”，也不能据此判定更大 KV 池必然损害 chain。

## 交付与下一步

按 chain 第一的最终决定，候选撤掉钉池，加入 cold600/warm120、held4096 与 running/graph48，镜像与引擎保持不变，见[FINAL 交付](../../evidence/submission-0927-chainmax-final/README.md)。这是用户选择的组合，性能收益尚待验证，不能拆成单旋钮结论。

FINAL 采纳现有 `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD=4096`：避免为 58/413 token 的浅共享把可救请求放到最后，保留 ≥4k 的原生深前缀 held 与 128p 自己的依赖发现。它会牺牲一部分浅共享、改变代表顺序，收益必须实测。旧 128g 的“可复用上界为零才释放”覆盖不了 413 深度在 256 网格上仍有非零收益的情况。

更完整的修正应按在线可观测的等待时间、剩余计算和共享收益决定 held 是否值得继续，并支持块边界上的多轮抢占与恢复。目标是整轮更多 chain 过门，不使用 cohort 的请求 ID、开场时间或隐藏相位制定特例；所有请求仍完整执行。当前 131 修复解决了 rank 一致性和成本口径，尚未实现这部分抢占。

## 复现

`scripts/analysis/trace_chainmax_pin400.py --log <server.log> --targets evidence/chainmax-fixes-0927/pin400-targets.json --output <decisions.json>` 提取含三个请求的前 65 秒全部决策；保留整个候选列表和 plan，不读取正文。原始 on 日志来自 Pod `/tmp/ax/runs/130ezn8-tail_n26_chainmax16k_mamba400_40m/server.log`；off 来自共享 checkout `evidence/L130ezn3-tail_n26_chainmax16k_40m/N26/server.log`。
