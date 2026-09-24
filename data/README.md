# 数据

## 四种数据，不能混用

| 名称 | 位置 | 谁给的 | 规模 | 有没有正文 | 用途 | 不能用于 |
|---|---|---|---|---|---|---|
| 主办方开发集：正文 | `s1-dev/data/dev-combined-v1/requests.jsonl`、`bodies/`（只读） | 主办方 | 311 链、722 请求，都是各链的**开头片段**（task.md：链前缀抽样） | 有 | 与原 harness 一起回放；比较自己两次部署的相对变化（A/B、回归） | 预测正式 N@SLO |
| 主办方开发集：整链统计 | 同目录 `chains.jsonl`（只读） | 主办方 | 同样 311 条链的**完整长度**统计：`n_requests` 合计 5601，另有各类请求数、token 总量、输出预算 | **没有**。例如最长链记录 240 次调用、持续约 14 小时，正文只给了开头 4 条；311 条链中有 245 条的正文不全 | 了解真实会话结构：链长、各类请求比例、复用比例 | 直接回放 |
| 合成长链集（我们生成） | [`s1-dev-longchain/`](s1-dev-longchain/)，快速子集 [`s1-dev-longchain-lite/`](s1-dev-longchain-lite/) | 数据 Codex 生成 | 完整集：311 链、5601 请求，保留主办方 722 条正文，**合成补齐 4879 条**；lite：64 链、1123 请求 | 有，合成 | 在"长链、高复用、缓存会被挤满"的状态下比较我们自己的配置 | 预测正式 N；当作真实观测 |
| 正式评测集 | 不在本地 | 主办方隐藏 | 341 链、5150 请求（task.md） | 没有 | 只由官方评测 | — |

正式集和主办方这 311 链统计是什么关系，我们不知道。两者都来自同一个 S1-Real-v2 来源，结构相近是推断。

## 合成集里哪些是真的，哪些是合成的

- **沿用主办方真实统计**：每条链的长度，以及各类请求（链首、轮首、链中、重建）的数量。
- **来自真实正文**：原 722 条请求原样保留；新增正文的素材（system/tools、用户与助手文本、工具调用与结果）取自 s1-dev，可以跨 session 借用，也可以局部改写。
- **合成或估计**：
  - 事件出现在链里的什么位置；
  - 等待时间的分解：新增事件用 Phoenix 观测的相邻间隔 `min(gap, 300 s)` 估计，没有真实的工具和思考分解；
  - 重建的具体形式：用接收历史的摘录，而不是语义摘要。
- **尚未恢复或未覆盖**：
  - 每条请求新增 token 的分布；
  - 链内 system/tools 版本切换（真实数据中这很常见，会切出新链）；
  - 等待时间的真实分解。

详见[分布审计](../evidence/longchain-design-20260924/full-distribution.md)与[生成设计](../scripts/longchain/longchain.md)。

## 完整集 `s1-dev-longchain/`

- **规模**：311 链、5601 请求；新增 139 次用户追问、118 次历史重建。中位链长 8 次，p95 为 82 次，最长 240 次。"次"指模型调用，包含工具续跑，不是用户轮数。
- **验收**：独立的 GLM 渲染检查 **VALID**（5601/5601，0 错误）；原 harness 全量自检 **PASS**（0 失配）。见[验收收据](../evidence/longchain-design-20260924/full-acceptance/acceptance.json)。manifest 里的 `BUILT_UNVALIDATED` 是构建当时冻结的状态，之后的验收由这份独立收据记录。
- **GPU 成绩**：暂无。

| 每链模型调用 | 链数 | 请求数 |
|---:|---:|---:|
| 1–4 | 119 | 253 |
| 5–8 | 48 | 316 |
| 9–15 | 64 | 734 |
| 16–30 | 30 | 671 |
| 31–99 | 43 | 2453 |
| 100+ | 7 | 1174 |

## lite 子集 `s1-dev-longchain-lite/`

- **构成**：从完整集整链抽出 64 条链、1123 个请求，每条链的正文、输出预算、事件和等待都完整保留。含 44 次 `turn_start`、27 次 `context_reset`（其中包括链首本来就带的标签）。
- **链长分层**：1–4 次 25 条，5–8 次 10 条，9–15 次 13 条，16–30 次 6 条，31–99 次 9 条，100+ 次 1 条。平均等待 7.36 秒。
- **校验状态**：按用户要求，**没有运行 token 渲染和 harness 自检**。
- **使用范围**：只用于 lite 与 lite 之间的配置比较，不等同于完整集。
- **复现**：`python3 -B scripts/longchain/longchain_subset.py`（目标目录必须不存在）。抽样收据见 `subset-selection.json`。

## 回放

用原 runner，只替换三个参数，其他部署参数保持不变：

```text
--root data/s1-dev-longchain        （lite：data/s1-dev-longchain-lite）
--set s1-dev-longchain              （lite：s1-dev-longchain-lite）
--cohort data/s1-dev-longchain/cohort.json
```

- 评分时指定本集的 `requests.jsonl`。
- 每档重新 flush KV，并使用新的输出目录。
- 不要加 `--max-chains` 或 `--no-gap`。

可以直接复制的命令见[运行入口](../scripts/longchain/README.md#直接回放当前成品)。

## 目录约定

- `data/` 只保留完整集、lite 和本说明。
- 旧副本、失败产物、Phoenix 原文缓存和临时正文缓存都已清除。
- 生成命令与复现入口见 [longchain.md](../scripts/longchain/longchain.md)；结构输入与检查记录保存在 `evidence/longchain-design-20260924/`。
