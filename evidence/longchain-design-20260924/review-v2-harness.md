# v2 独立审查：原 harness 接口、预算与 gap

2026-09-24。审查 `longchain.py` 新 build、`longchain_events.py` EventCompiler，以及 `data/longchain-events-pilot-v2` 的已完成产物。只读，没有改变生成器、checker、原 harness 或 GPU 服务。此时产物状态为 `BUILT_UNVALIDATED`，本审查不是全量 GLM 渲染验收。

审查产物 manifest SHA256：`7e2bb7ac264dd1e18f4357f7b8222ea25e9bc152df65caea4df909aade31e2be`。构建时生成器 SHA 为 `ea99f3833432ea172e3527c963a8ecca73dde633e92b2011561e0182d0ff4542`；之后代码修复不自动改变该成品。

## 1. 确认的 bug：空间不足会绕过“自动重建”分支

**P1（扩大规模/更紧上下文时的生成阻塞，本 pilot 未触发）。** 审查版本 `longchain_events.py:260–262` 在没有 `d.increment < room` 的完整素材时直接抛 `ValueError`。`longchain.py:569` 调用该函数后，直到 575–583 才根据渲染长度走 context-pressure 重建。因而空间已经不足时常常先异常退出，不能兑现 manifest 中“context pressure may add explicit rebuild events”。

CPU 反例：只保留一个 `increment=10` 的 donor，调用 `EventCompiler.event(..., kind='intra', room=9)`，实际抛出 `ValueError: no complete compatible material fits context budget; explicit rebuild required`。不需要 tokenizer 或引擎就能复现，build 没有捕获该异常。

建议：用专门的“完整素材放不下”异常，由 build 转成明确的重建事件；或者在选择前显式检查。不能笼统吞掉所有 `ValueError`，也不能改用截断填满。重建后仍须真实 GLM 渲染并检查严格缩短、prompt+budget 不超限。若连重建也放不下应失败。被替换的计划事件及实际 phase 要保留在 sidecar，不能只在摘要写“配额已满足”。

另外，失败的试选可能已增加素材 usage；在真正引入内容之后再提交 usage 更符合来源次数口径。这对目前未触发 pressure 的 pilot 没有已知影响。

## 2. pilot 实测账本（由原 harness load_index / gate / gap_plan 复算）

| 项目 | 实测 |
|---|---:|
| 链 / 请求 / 独立 session ID | 24 / 455 / 24 |
| 原样正文 / 合成请求 | 70 / 385 |
| 链长 median / p95 / max | 8 / 89 / 125 |
| 合成 intra / turn_start / context_reset | 370 / 10 / 5 |
| 四门 fast / overall / turn / chain | 390 / 413 / 12 / 30 |
| 合成请求中的 fast | 352 / 385 |
| 总输出预算 / 所选 source 总输出预算 | 322,838 / 322,838 |
| 输出预算 median / p95 / max | 525 / 2,310 / 7,317 |
| prompt median / p95 / max | 72,237 / 142,038 / 225,596 |
| 总 prompt / 所选 source 总 prompt | 35,669,909 / 38,952,125 |
| 合成 gap median / p95 / max | 3.108s / 18.255s / 300s |
| 合成 gap 被 300s 封顶 | 1 |
| 3600s 累计 gap cap 影响的链 | 0 |
| 原始 replay_gap 合计 / cap 后合计 | 3,037,866ms / 3,037,866ms |
| 额外 context-pressure 重建 | 0 |

所有 385 条合成行均为 `gap_imputed=true`、`tool_union_ms=null`、`net_think_ms=null`；provenance 明确是 `estimated_capped_phoenix_end_to_start_proxy`。5 次新重建后的实际 GLM prompt 长度为重建前的约 28.3%–80.7%（本次从已生成的 token/LCP 账本计算，未重复渲染正文）。全体 8 次 reset 中，有 6 次紧接着出现 intra 且 prompt 继续增长；不意味着另两次异常，因为可以链尾结束或转用户轮。

注意：fast 数量大本身不能判错。fast 由冻结未缓存输入阈值决定，不由总上下文长度决定；链首优先进 chain。本集不能直接与旧 96 链成品四门数量比较，因为采样链集合不同。

## 3. 不是接口 bug，但必须透明保留的估计边界

### gap proxy 不是 task 的工具/思考分解公式

实现是 `round(min(Phoenix end-to-start gap, 300s)*1000)`（EventCompiler 209）；task 负载定义是 `min(tool_union,300s)+min(net_think,10s)`，然后由原 harness 执行每链累计 3600s 上限。

两者并不等价。例如纯用户思考 100s 的间隔，proxy 会保留 100s，而若可确定它全是 net_think，原规则应保留 10s；工具 500s 加思考 10s 的串行情形，proxy 为 300s，分解公式为 310s。缺失工具归因时不能声称误差只是 10s。

目前 manifest 与 provenance **已正确声明 estimated、分解未知，没有伪造 tool_union**，因此能作为明确命名的合成等待负载运行，runner 无需改动。但“接口合法”和“节奏贴近正式”是两件事：等待影响占槽、请求到达和缓存竞争，正式代表性结论仍需有工具区间证据的条件分布校准。建议报告 proxy 请求比例（本 pilot 合成部分 100%）、封顶比例、每链 cap 前后账，以及未来有归因数据时的替代敏感性；不要把这些 proxy 标成已恢复的原始工具耗时。

### source phase 配额不是上下文压力模型

EventCompiler 172–196 按源摘要减去公开前缀的事件数量安排 reset/turn，位置偏好连续模板中的压缩事件或模板边界；本集总 phase 精确等于 source：7 session_start、422 intra、18 turn_start、8 reset。这是**明确施加的计数约束**，不能当作观测验证成功。

3 个计划 reset 的参考边被替换成另一条压缩观测，sidecar 有 `replaced_reference_for_event=true`。这些位置前后的线上连续性已经打断，不能再把整段称为原封不动的线上联合序列。实际重建虽有物理缩短，频率和位置仍是推断；当前不存在“到某 GLM 上下文压力才触发”的统一模型。

保持原 runner 不变即可：标准行 phase 写实际编译事件，sidecar 记录计划 phase、实际 phase、替换原因、前后 GLM token、保留前缀和压缩比；汇总 source/plan/actual 三本计数账。pressure 额外 reset 应照实增加 chain 门样本，不能为了配额把真实重建伪装 intra。

### rank transfer 与总预算匹配不等于联合分布恢复

输出 rank 来自 Phoenix，映射到兼容 public donor 的预算，再按 source 链总预算缩放；增长 rank 映射到素材增量。这避免把不同模型的 token 直接冒充 GLM token，但不保证 phase×输入增长×输出预算的联合分布一致。`self.outputs/self.growths` 的 rank CDF 当前没有使用样本权重；模板选择虽使用 weight，也先筛选长度最近的 128 段。因此不能把最终负载称为无偏线上人口抽样。

本 pilot 最大输出预算仅 7,317，尚未覆盖 8k/16k/32k decode 尾部；拼进历史的长 assistant 文字不能代替当前请求的大输出预算。该点是机制覆盖缺口，不是让当前 pilot 失效的 schema 错误。

## 4. 与原 runner 的兼容性结论

- 仍导出原 `requests/chains/bodies/samples/cohort`，新增 event plans/provenance/manifest 为离线 sidecar；原 runner 只需换 root/set/cohort。未发现要修改原 runner 才能执行的新增字段依赖。
- 合成 session ID 已按接收链独立；公开正文未因 session header 改名而改变渲染。不能据此宣称物理 KV 隔离。
- `max_output_i` 驱动原 `/generate` 的实际 decode；下一 prompt 的素材正文仍冻结。预算总量匹配为实测，输出分布匹配为待核验。
- gap 为非负整数，原 harness 仍在上一响应完成后等待并占据该 worker 槽；时间和资源竞争由真实回放发生。synthetic dispatch 被标为排序坐标，不作为线上真实时间。
- 真正放行仍依赖全量独立 checker 的 body/cohort/hash/token/LCP/事件闭合校验及原 harness 自检；本文件只做接口审查和元数据复算。此前可选 `longchain_replay.py --self-check` 能在外部完成该防护，不要求 GPU 队列迁移到新框架。
- CPU 合法不代表正式分布已还原；也没有声称本 pilot 已有 GPU 分数或有效稳态 TPM。
