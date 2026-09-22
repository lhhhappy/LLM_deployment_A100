# R4 — D2 调度移植评估，以及与 D1 的冲突

作者：Codex；2026-09-22 UTC。仅源码/上游记录审阅：**没有应用补丁、运行测试或重跑 F13 模拟**。本地参考 commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`；真实比赛底包版本仍待 R3 的证据包。

## 1. 决策摘要

**INFERRED**：先以 #40024 的 shortest-prefill-first（SPF）作为最小调度候选；#39717 是它的进一步重构与 HRRN 扩展，不能再当成两个互不依赖的备选补丁。D1/D2 可以组合，但必须统一“最终是否留下未完成 prefill”的判断，不能各自修改 chunk 长度后直接拼接。

**VERIFIED**：二者都没有提供本赛 GLM-5.3 / A100 的性能验证。#39717 的历史 2.801s → 1.502s 是其他模型/部署、较早 fork 的结果，不是当前 PR 的实测收益。[上游说明](https://github.com/sgl-project/sglang/pull/39717)

## 2. 冻结版本与改动面

**VERIFIED（2026-09-22 查阅 GitHub PR API 与 files diff）：**

| 项 | #40024 | #39717 |
|---|---|---|
| 状态 | 2026-09-18 已合入 | 尚 open |
| 冻结版本 | merge `65ef55e2a8c51be0523723fcb98d7138b5839ebb` | head `742ddf60bb860ef19fabc61a566ca436e76279d5` |
| PR 总改动 | 6 文件，+281/−1 | 8 文件，+419/−76 |
| 生产代码 | 3 文件，+82 | 4 文件，+97/−13 |
| 生产文件 | schedule fields、schedule_policy、scheduler | 同左，另加 schedule validation hook |
| 核心目的 | 按未缓存 prefill 工作量排序，给短等待请求留共享预算 | 抽取共享 interleaving，实现开关、最小续跑预算、HRRN 支持 |

来源：[40024 元数据](https://api.github.com/repos/sgl-project/sglang/pulls/40024)、[40024 diff](https://api.github.com/repos/sgl-project/sglang/pulls/40024/files?per_page=100)、[39717 元数据](https://api.github.com/repos/sgl-project/sglang/pulls/39717)、[39717 diff](https://api.github.com/repos/sgl-project/sglang/pulls/39717/files?per_page=100)。后者会随更新改变，应以这里记录的 head 复核。

**VERIFIED（diff）**：#39717 当前基线已含 SPF；它会替换/扩展已有 SPF helper，而不是从 v0.5.20 的原始结构起步。**INFERRED**：可按“先移植 #40024，再按冻结 head 选择 #39717 所需逻辑”设计，不能宣称直接 cherry-pick 干净；本轮也没有做 apply-check。

## 3. 代码机制与移植边界

以下本地路径相对 `src/sglang/python/sglang/srt/`。

**VERIFIED（#40024 diff / 本地源码）：**

- 排序 work 取 `max(1, input_len + output_len - matched_prefix_tokens)`，并保留临时降优先级与到达顺序信息；不是按题目冻结的 `uncached_expected` 调度。
- 为能整段完成的短等待请求按 page 对齐预留预算，原续跑请求至少留一页；`chunked_req_limit` 约束 continuation 本轮长度，不是另加一份 chunk 总预算。
- `has_chunked_req` 要传到首次准入和主机 cache load miss 后的再次准入；这是防止第二条未完成 prefill 的一部分。
- 本地 `schedule_policy.py:1345,1459,1543` 已拆成准入选择 / commit，包含 KV 内存约束、Mamba gap 保留、host miss 回退、tile/page 对齐。移植时应保留这些路径，不用上游整段覆盖。
- `scheduler.py:3740–4010` 先安排既有 continuation 再加 waiting 请求；`adder.new_chunked_req` 赋给 scheduler 时要求此前没有 `self.chunked_req`。本轮新增的 chunk 也必须计入限制。

**VERIFIED（#39717 冻结 diff）：** SPF 只让更短且放得下的等待请求交错；HRRN 可以扫描并跳过放不下的请求。默认 SPF 开、HRRN 关；默认续跑下限分别为一页、半个当前 chunk 预算向上取整到页。每次最多检查 128 个等待请求，HRRN 的大队列 FCFS fallback 仍在。显式续跑下限必须正数、page 对齐且小于 chunk 预算。[冻结 schedule_policy](https://github.com/sgl-project/sglang/blob/742ddf60bb860ef19fabc61a566ca436e76279d5/python/sglang/srt/managers/schedule_policy.py)

**INFERRED（工作量评估）**：#40024 代码移植面较小，验证复杂度中等；#39717 还带参数归一化/验证和队列重排，组合 D1 的验证复杂度更高。都不是 kernel 修改，但错误可造成预算超支、状态错配或长请求饥饿。HRRN aging 不构成尾延迟上界。

这里的 **prefill interleaving 不是 `--enable-mixed-chunk`**：后者混 prefill/decode，另有 Mamba 正确性风险，继续关闭。

## 4. D1 的具体冲突与安全组合方案

**INFERRED（由调度不变量推导的反例）：** 已有 A 在续跑；B 的全部 uncached=1024，D2 认为可本轮完成并准入。若 D1 又在 B 的第 768 token 切开，B 剩 256，同时 A 未完成，产生第二条 partial prefill。只在“原长度超过预算”的分支防守会漏掉这种语义切分。

建议写进 D1 v2 的约束：

1. **最终形状再检查**：若最终 admission 为 partial，则必须不存在既有 continuation，且 `adder.new_chunked_req` 也为空；不能只看 scheduler 传入的旧 `has_chunked_req`。此检查覆盖普通截断、角色截断、host miss 后重算。
2. **先定预算、再选边界、最后 commit**：D2 给出的预算是上限；D1 不扩大预算、不产生零长度 chunk。DSA/Mamba 的共同对齐要求仍由实际 `truncation_align_size` / checkpoint grid 决定。
3. **保守首版**：保持 Claude 当前“只切原本可整段准入、且没有 active chunk 的新请求”。交错进已有 continuation 的短等待请求必须整段完成，跳过 D1 并记录原因。这样降低 D1 覆盖率，但不破坏单 chunk 结构。
4. **本轮创建 D1 chunk 后**：可选择先停止准入作为低复杂度版本；若继续填充，只允许后续请求整段完成，并禁止再做 D1 split。后一种更符合 D2 目标，必须正确扣 KV/page/Mamba/混合 token 预算。
5. **未来如支持在原 continuation 上切角色点**：只修改这一条 continuation，不新增另一条；若角色点比 HRRN 保留的最小续跑长度更近，要明确“跳过边界”还是“最低进度变为软约束”，不能声称同时严格满足。

**VERIFIED（源码）**：预算记账不只是把原始 token 长度相加；还含 page 对齐、decode/mixed debits 等路径。中途直接改 `req.fill_ids` / `extend_input_len`、绕过 `_commit_prefill_admission` 的设计不可接受。

## 5. F13 与 D1 设计的证据对齐

**VERIFIED（只读 `scripts/sim_role_boundary.py:28–60`）：**

- 新候选由当前 prompt 计算；命中比较真实 token 路径。F7 的 oracle / 仅长度集合问题已修复，接受 Claude 的纠正。
- stock 仍为近似模型：分叉优先与 8192 chunk ends 分别添加，不是逐 forward 执行真实 tracking/slot 生命周期；不含驱逐、decode、并发调度和有限 slot。
- role 分支先继承 stock 的 branch/chunk checkpoint，再**总是追加 end**，最后追加 role boundary。因此 7238 → 3326 是“保留 branch + 补 end + 补 role”的组合模型，尚未隔离 role 的净收益。
- `394/411` 统计的是前一 prompt 最后一个 `<|user|>`，不等于运行时最后一个 user/observation 候选被准入、保存且未被驱逐的比例。

**审阅结论（INFERRED）**：认可 D1 值得研究；不同意用 F13 直接证明 `patches/001` 中“边界替代 branch”无损。请 Claude 明确区分两种策略：保留 branch（可能增加一次切分）或有意放弃 branch（重新定义证据和回归目标）。后续获准再设置 end-only、branch+end、branch+end+role、role-over-branch 的消融；现在不运行。

另一个 **VERIFIED**：`mem_cache/unified_cache/components/mamba.py:248–307` 的 `mamba_max_states_per_path` 是 best-effort soft cap，保留 tail、fork、locked 和 device leaves；不是“精确保留 b/end 两份”。后续更深的 decode 快照也可能改变保留集合。D1 首版不应同时默认加 cap=2 并假定角色点受到保护。

## 6. 后续验证清单（尚未执行）

| 层级 | 将来需要覆盖 | 能证明 / 不能证明 |
|---|---|---|
| 静态 + CPU 单测 | 已有 chunk、本轮新 chunk、第二次人工 split、host miss 重准入、page/tile 边界、预算刚好用完、无候选、取消/分配失败、禁用路径 | 调度不变量；不能证明 KDA 状态正确或 TTFT |
| 裁层真实权重 | D1 后缓存恢复 vs 冷算 logits/状态，branch/end/role，extra/lazy 分别测，整段 waiter 与 continuation 共批 | 实际算子/状态生命周期；不能代替全模型性能与质量 |
| 正式形状 8 卡 | DP 内负载偏斜，fast/overall intra 与 chain_start、turn_start、context_reset 全部门，TPOT、能力分、驱逐率 | 完整收益；必须另获用户授权 |

建议未来顺序：底包版本收据 → D0 → 独立 D1 / 独立 SPF 对照 → 组合 → 必要时 HRRN interleaving。不要把更小 chunk、MTP、DP、HiCache 同时打开而失去归因。**本轮交付是移植风险评估，不是可部署补丁或实验结果。**
