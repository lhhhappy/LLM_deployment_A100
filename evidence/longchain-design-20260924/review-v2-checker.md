# v2 重建与后续增长：独立 checker review

2026-09-24。仅数据与 CPU；无 GPU、服务或线上 API 请求。重点审查 `_rebuild_errors` 和调用它的 synthetic 分支，不把“收到 reset 标签”视为发生了重建。

## 发现及修复

修前用独立构造的完整微型数据集、原 GLM Renderer 和冻结依赖运行，确认以下 5 个结构反例返回 VALID，另 1 个直接崩溃。每个 fixture 的 artifact/cohort/body hash 和实际 token/LCP 收据均重新计算，排除了陈旧 hash 干扰。证据为 `v2-checker-before.json`，不是假 tokenizer 结果。

| 问题 | 修前影响 | 修复 |
|---|---|---|
| prefix_end 可落在 assistant call 与 result 之间 | 留下没有结果的调用；只检查 tail 未覆盖 prefix | 验证两端都为完整工具组边界，检查 prefix 与 tail 的完整工具组 |
| prefix_end=0 可删除原始开场任务 | 与声明的 preserve-task method v1 不符 | 独立重算 opening prefix，要求精确保留 |
| 摘录摘要可夹入未记账的大段正文 | “extractive”收据不能证明摘要来源或长度形态 | 摘录 1–8 条、每条 1–240 字符、真实源前缀；摘要必须等于固定头尾和有序摘录 |
| 摘要 user 对象可含额外 tool_calls | 可藏入未配对调用，甚至被模板忽略而逃过渲染 | 摘要对象只允许 role/content 两个字段 |
| 摘录索引可重复、乱序 | 伪造重复块而不被发现 | 索引严格递增 |
| summary_excerpts=[null] | AttributeError 中断 checker，无结构化 INVALID | 校验每个 excerpt 类型，畸形 receipt 返回错误 |

对应实现：`scripts/analysis/longchain_check.py:174` 的 `_rebuild_errors`。用户授权后，本审阅者直接修复这一函数；生成器由主会话负责。

另修复持久摘要被当作尾提醒删除的风险：摘要采用普通 user 文本，但摘录本身仍可能包含 `system-reminder` 字样。checker 的 `_new_message_suffix` 独立排除以 `历史上下文摘录（中间记录已归档）：\n` 开头的持久摘要，使其不能通过“替换最后一条提醒”的特例消失；正常系统提醒仍允许原来的替换规则。此改动位于 `longchain_check.py:102`。

主会话新增的压缩重试（保留 tail=0，摘录 120×4／32×1／1×1）符合本收据合同。checker 不相信“尝试过压缩”的自报字段：每次仍核完整替换、来源、实际渲染和 prompt 严格缩短。摘录多少可以不同，不能少检查任何一项。

## 已经正确防住的情况

- 仅换 phase 的假 reset、无前驱 reset、未知 method、正文没有实际替换。
- removed/tail hash 不符、尾部偷偷改写、tail 开头是孤立 tool result。
- 在正文插入第二份摘要消息，导致正文与 receipt 指定的拼接不一致。
- reset 后直接恢复旧完整快照：下一次请求不再是新历史的合法追加，会被拒绝。
- 对重建后新历史继续追加完整工具组则通过。正常跨来源借素材不是错误，不能把所有相同文本再次出现都叫“历史恢复”。

## 验证与复现

```sh
python3 -m unittest discover -s tests -p test_longchain_events_check.py
python3 -m unittest discover -s tests -p test_longchain_check.py
uv run --offline --with-requirements scripts/analysis/requirements-longchain.txt \
  python evidence/longchain-design-20260924/reproduce-v2-checker.py
```

新增 `tests/test_longchain_events_check.py` **16 项通过**，含多组畸形输入、持久摘要、缩短重试；原 checker **27 项通过**。

修后真实 Renderer 证据 `v2-checker-after.json`：正常对照 VALID，原来 6 个坏例全部 INVALID，不再崩溃。环境 transformers=5.12.1、tokenizers=0.22.2、jinja2=3.1.6；正常例实际 prompt 2941→636 token。修前、修后摘要格式按新合同从提醒改成普通文本，所以正常例缩短后的 token 数不同，不能把二者当性能 A/B。

一次宽泛的系统 Python unittest discover 因系统 Python 缺 pytest 而不能导入另外两个 pytest 模块；随后仅针对本审查负责的两个 unittest 文件执行，全部通过。主会话可在 uv pytest 环境执行完整生成器套件。本报告未复跑 24 链成品，也不声称证明线上代表性、语义正确性或 GPU 成绩。
