# codex-分析：阻塞窗口复核与执行路线

2026-09-24。已审阅并更新 scripts/analysis/blocking.py，重算 042 与 037d 各 722 条请求。本轮为 CPU 分析和调研更新，没有新增 GPU 运行或正式提交。

## 1. 哪些结论成立

长且未命中量大的链首，是这两档最值得继续检查的干扰来源：大量超时请求的等待/生成窗口，与这些链首的 prefill 生命周期重合。这个发现支持同时研究降低 prefill 成本与改善交替执行。**它还不能给“活太多、活太贵、排得不好”分配因果百分比。**

| 实测/窗口统计 | 042 | 037d |
|---|---:|---:|
| fast 超过 3 秒的请求数 | 15 | 13 |
| chain 超过 30 秒的请求数 | 17 | 23 |
| 上述 chain 中首轮前向前耗时大于 TTFT 一半的请求数 | 17 | 22 |
| fast 超时请求的前向前时间，被其他 prefill 窗口覆盖的比例 | 98.0% | 98.3% |
| chain 超时请求的前向前时间，被其他 prefill 窗口覆盖的比例 | 99.3% | 99.0% |
| TPOT>0.10 请求数 | 175 | 203 |
| 这些请求生成窗口与其他 prefill 窗口重合比例的中位数 | 86.9% | 98.9% |
| 旧成本公式估计超过整个执行窗口的请求数 | 218/722 | 123/722 |

超标条数是超过单请求阈值的条数，不能单凭此表判整档；统计余量、完整 runner 与所有硬门仍由 level_verdict.py 和原 harness 判定。来源：[042 新结果](../evidence/L042/blocking.json)、[037d 新结果](../evidence/L037d/blocking.json)。

## 2. 旧分析的具体问题与改法

1. **延迟阶段被写成了原因。** recv→first forward 包含前置处理、准入和等待；算子慢、容量紧、调度策略都可使它增长。新版保留“前向前/执行窗口”实测时间，删除 primary cause 和 arrangement/cost/work 的因果标签。
2. **成本模型把残差伪装成 interleave。** 旧脚本用 min(估计成本,执行窗口) 截断估计，再把剩余时间叫别人插进来的活。042 有 218 条、037d 有 123 条估计越界；模型不是逐请求实测。新版保留原始估计和可为负的 model_residual_s，不再硬凑四块之和。
3. **固定 16k 不是真实块数。** 042 设置 cold cap=8192，日志确有大量 8k 块；一批可共享固定开销，角色边界又会拆块，成本还随上下文变化。新版把 fixed/token/chunk/page 暴露成显式估算参数，不称观察到了 forward 数。改变模型不会改变实测窗口或超标条数。
4. **生命周期重合不等于 GPU 忙于 prefill。** [exec_start,first_token] 可能包含 decode、其他请求和空档。037d 那条 211,336 未命中 token 的链首，执行窗口 19.609 秒，其内部日志出现 decode；“整段独占 GPU”应撤回。042 同请求 15.870 秒窗口没有采样到 Decode batch 日志，也不能由此证明完全无 decode。
5. **175 秒的单位错觉。** 042 该请求分摊到的 174.565 是多个超时请求累计的 request-seconds；其自身窗口仅 15.870 秒。不能当 GPU 时间，也不能理解为单请求等待 175 秒。新版完整标单位，逐请求 JSON/CSV 保留候选请求 ID 与分摊重合秒数。
6. **79%/67% 是条件比例。** 分母是已被其他 prefill 窗口覆盖的前向前时间，多候选同时存在时等分；不是所有排队时间的直接因果分解。新版同时输出覆盖率、未覆盖秒数和分母，汇总使用未四舍五入数据。
7. **完整性和 LCP 不能静默假定。** 旧版只查 722 条且 ID 不重复；现在按实际 data-root 的冻结 cohort 核对 ID、分桶、链位置、输入/输出长度，拒绝坏时间顺序、NaN/Inf 和非法缓存计数。LCP 文件需显式传入，核对同链前驱、位置、prompt 长度与 LCP 范围；缺失是 unknown，不能记为零。两档 411 条后续请求均有覆盖。调用者仍须保证 LCP 来自同一冻结 token 化文本；长度检查不是内容哈希。
8. **跨时钟统计须说明。** decode 的分母直接取客户端生成窗口，不再用舍入后的 TPOT×token 数反推。只有显式给出 server-minus-client-s 才计算跨时钟重合；本次同 pod 时间基准传 0。它声明已知偏移，不会自动校准时钟。

“缓存只占 fast TTFT 的 16–19%，所以收益次要”也不能据此下结论：那是模型分配的自身计算部分，没计算缓存修复给其他请求省下的等待，更不是边际 SLO 收益。

## 3. 通用脚本怎么用

直接使用现有 [blocking.py](../scripts/analysis/blocking.py)，没有新增临时分析脚本。从仓库根目录：

```bash
python3 -B scripts/analysis/blocking.py evidence/L042/raw_dev-combined-v1_N22_1790203410.jsonl \
  --pairs evidence/T56/pairs_attributed.json \
  --server-minus-client-s 0 \
  --csv evidence/L042/blocking.csv \
  --json evidence/L042/blocking.json
```

换运行只换 raw 和输出路径；不同 cohort 使用 --data-root，不再写死 722。不同模板/数据不能沿用这份 pairs；无 LCP 时省略 --pairs。时钟关系未知时省略偏移，TTFT 阶段统计仍可用，decode 重合留空。

输出 schema_version=2。旧 primary、own_cost_est、cache_gap_s、interleave、blockers 字段已替换为 measured window、prefill_cost_est_s、model_residual_s、cache_gap_tokens 和 overlap_candidates。原始精度保留在 JSON/CSV，文本只作摘要；模型参数及边界写进 JSON。--fixed-s、--per-token-us、--model-chunk、--page-size 是估算参数，不能通过调它们改变已测结果。

[8 个 CPU 回归用例](../tests/test_blocking.py)覆盖错 cohort、重复/错桶、坏时间与缓存计数、重合去重、估计越界不截断、模型不影响实测量、显式时钟及 LCP 不匹配。两档完整 raw 重算通过；旧派生输出保存在 [审计证据](../evidence/blocking-audit-20260924/)，原始 raw/server 日志不改。

## 4. 接下来测什么才能回答因果问题

要量“真正挡了多久”，还缺批级证据：batch/step ID、参与请求、每请求实际新 token 数、GPU forward 起止，以及期间可运行的 decode/短请求。已有 raw 适合圈候选，profiler 适合量 kernel 与 step；没有请求→step 关联时，两者不能精确拼成逐请求 GPU 成本。日志里的累计吞吐也不是每个请求的实际 kernel 时间。

旧 044/045 漏 121，校准结论作废；使用 044r/045r 重跑与 046r 混合 profile。核对 046r 是否保留请求到 step 的关联，缺失时明确只做组件账。随后单变量比较降低块成本、允许短请求进入、改变 decode 交替预算的效果，完整回放看全部门。按此步骤，才能区分“变便宜”和“重新分配等待”的实际收益。

广度调研已直接写回 [R9](../research/claude/R9_upstream_since_base.md)。首个替代引擎候选是题面主办方跑通的 vLLM sm80 backport；首个局部源码候选是 KDA BF16 投影融合；170 v2 继续数值复验，MoE EP/选择性 BF16 作为下一层筛选。没有证据承诺哪条一定到 N26。


## 5. Fable 评估与判分工具复核

Fable 原评估中“同 TPM 证明产能足够”“开发集测不出驻留”“所有 fast 超时都是缓存丢失造成”等结论已直接从正文撤回，替换为真实 LCP、工作量和等待窗口的有边界结论。[更新后的评估](fable-评估-2026-09-24.md)

评测工具已修清缓存失败、取证错档与失败码、复用日志；原 CP 评分不变，10档完整raw回归一致。037d的chain方法敏感不改变TPOT失败。修复尚未同步pod，旧任务和原始证据未改。[实现与验证](fable-审计-2026-09-24.md)
