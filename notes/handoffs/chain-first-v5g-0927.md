# 给 Fable（Claude）：数据复核后启动 chain 优先实验

用户 2026-09-27 最新授权：先把数据做完，发给 Claude 独立 review，全部检查通过后开测；Codex 可随时只读监控并提建议。
用户目标：**今晚看到一版大幅缓解 chain 的候选，不限定方案。本地 TPOT / fast 略超可以接受，优先 chain 与并发。**
这允许探索失败的局部 SLO，不改变正式判分、不把诊断结果写成 N@SLO 通过。
**数据交付与 Fable 独立复核已完成**，结论为可用的敏感性臂。代码 `c9c14153`，部署与实际 gap 收据 `0d411679`。
实验由 Fable 按本交接编排；运行状态见共享[实验队列](/workspace/Agentic_science_challenge/notes/queue.md)，不在本文维护第二份状态。

## 数据已部署到两处，正文均复用

- 开发机基目录：`/sjtu/linhang/arena/codex/longchain-repair-0927/data/`
- Pod 基目录：`/dev/shm/arena-runtime/ax/codex/longchain-repair-0927/data/`
- 三个子目录分别为 `s1-dev-longchain-v5-review-0927`、`s1-dev-longchain-v5g-review-0927`、`s1-dev-longchain-v5g-tail-review-0927`。
- 主候选建议用 **v5g-tail**；v5 是单变量对照；原 v5g 保留为较强压力参照。
- `G_DATA_ROOT` 指向所选子目录，`G_DATA_SET` 与子目录名相同，`G_COHORT=$G_DATA_ROOT/cohort.json`。
- 每侧基目录 `publication.json` 提供请求文件摘要、cohort 标识、正文摘要与 gap 分布。新 job 的 DATA_READY 须用这份收据，不能沿用旧 requests 字节摘要。

三版均 311 链 /5,601 请求 /5,290 链中间隔，v4 的正文、token/LCP、phase、请求身份、链内顺序、cohort 顺序不变。
v5 /旧 v5g 的请求值分别与原冻结文件逐字段相同，修复了 manifest、provenance 与覆盖保护。

| 数据 | gap P50 /P90 | P95 /P99 | >60s | 链中总等待 |
|---|---|---|---:|---:|
| v5 | 2.653 /12.551 s | 22.110 /33.210 s | 18 | 29,874.685 s |
| 旧 v5g | 7.319 /50.904 s | 99.846 /310 s | 440 | 122,278.798 s |
| v5g-tail | 2.653 /12.551 s | 33.210 /245.467 s | 250 | 59,455.077 s |

原因：用户新查源业务 raw gap P50=2.396、P90=21.528、P95=91.668、P99=551.314 秒，>60s 仅 340 个。
这些 raw 间隔还没拆工具/思考，也没逐步和按链封顶，不能拿 raw p95 当 replay 目标。
旧 regap 不仅补尾部，还把普通等待放大；440 个 >60s 已多于源 raw 的 340 个，不适合称真实校准。

v5g-tail 规则：从 v5 出发，只接受“合成、非链首、原 gap >v5 全部链中 gap P90=12.551 秒、
旧 regap 提案 >max(原 gap,60 秒)”的延长；其余等待保留。各提案 ≤310 秒。
这保留普通等待，把部分回访拉远；**位置仍是合成的，是保守敏感性臂，不是恢复了源逐请求时间**。
源导出已删除，用户只能提供汇总。源 P95、累计 3.5 倍等都没有被硬凑为目标。

## 独立复核结果与测试边界

实现分支：`codex/longchain-repair-0927`，本机 worktree：
`/workspace/Agentic_science_challenge/build/worktrees/longchain-repair-0927`。
报告 `notes/reports/longchain-incremental-repair-0927.md`，新入口 `scripts/longchain/finalize_v5g.py`，发布契约 `longchain_metadata.py`。
12 项针对性 CPU 回归已通过；修复后的 rebudget /regap 真实 5,601 行分别与旧 v5 /v3g 完全一致。
主候选在 Pod 的全量结构验收已经完成：`STRUCTURAL_OK`，5,601 请求/正文，0 错误、7/7 文件摘要一致。
完整报告在 Pod `codex/longchain-repair-0927/validation/v5g-tail-structure.{json,txt}`；
本地 [pod-review-receipt.json](../../evidence/longchain-incremental-repair-0927/pod-review-receipt.json) 保存摘要与复核收据。
未重做全量 token/LCP 渲染；本次正确性依据是完整正文摘要、冻结标签不变和原构建记录，不能写成独立 `VALID`。
GPU 首次 checker 无完成报告，已由 Pod 补跑覆盖；两侧三套发布收据已核对一致。
**以实际收据为准：有错误、缺正文、重复 ID、错 body_ref，或部署收据不一致，先修，不开测。**

重点复核：

1. 三组只发生允许的元数据变化；原 722 个公开请求正文/预算不动，cohort 不重新抽样，gap 不改 chain 标签。
2. body shard 共享正确、清单完整；原 token/LCP/正文完全不变的增量证据成立。结构验收不能说成新做了全量 token 重渲染。
3. 原 `build_gap_plan` 已实际运行：v5 /tail 无链压缩，上表值即有效 gap；旧 v5g 有 1 条链压缩，
   有效链中总等待 122,275.427 秒，分位数和 >60 秒计数不变。请复核验收收据内的 `effective_gap`。
4. 接受它作为敏感性实验，不接受“已复原正式负载”的断言。v4 的逐链合成新增量仍有明显误差，详见报告。

上述各项已由 Fable 在 Pod 独立复核并 ACK，见[完整收据](../../evidence/longchain-incremental-repair-0927/fable-review-receipt.json)。
其 `chain_index` 表示源会话切段编号，不能当作链内位置；实际未改任何 cohort 链首。

补充开场口径：207 个公开切段链首本来保留非零 gap，原 harness 会等；最长 62.202 秒的头在 v4 第 26 位。
这解释 N34 前 60 秒常只有 33 个头。不要按源统计的 head=NULL 擅自清零。

## 编排建议：先找 chain 大收益，再做细归因

现有 118 OFF/ON 剖析按当前任务完成。之后统一使用审查通过的 v5g-tail，避免 S1、S6 各跑不同数据。

1. **S1 原配置 @N26 对照**：保留原定 40 分钟派发+排空，分开开场/后续 chain、链首/内部 reset。作为当前数据的正式 S1 锚点，不用另一个 cohort 的 41 秒替代。
2. **chain 优先组合 @N26**：S6（去 MTP +128p +131）优先与同数据 S1 比；用户接受组合先取得收益，单旋钮归因可排在后面。
3. **大块 prefill 候选**：用已有 16k/32k 候选逐级筛选，记录单块时长、链首完成数、short/warm 代价；已经过期且算力大者不应持续挤掉仍可在门内完成者。
4. **两条长请求保一条**：不要只改等待队列排序。现有单 `chunked_req` 与“只救一轮内可完成请求”的停车条件，无法表达“暂停 A，让多块 B 跑到完成，再恢复 A”。需要块边界可恢复暂停、KV/KDA/跨 rank 计划一致、剩余工作与可行完成次序判断。所有请求最终完成。
5. 开场短探针筛掉无收益方案后，必须观察长等待回访及稳态冷头；有明显 chain 净减少，再固定 N30/N34 诊断，最佳候选跑完整回放。

本地 **TPOT /fast 略超仍可继续诊断、排空并比较 chain**；用单档固定 N 的独立 job，避免 ladder 在上一档 SLO fail 后不再触达待研究档。
评分照常记录 FAIL，数值不改。候选比较同时看 chain 修复/新增坏例，不能只看 p95 或净数。
相同引擎/数据/时长/派发口径；不拿 30 分钟排空时间与仍在派发的 60 分钟臂宣称吞吐提速。

Codex 负责持续查看原始记录与 118/prefill 执行层证据，给出链首等待/入批后成本、局部提速和调度兑现的建议。
请回 ACK + review 结论 + 实际入队名字；用户要的是今晚的实测 chain 改善，不是停留在数据整理报告。
