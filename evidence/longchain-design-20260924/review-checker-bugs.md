# 长链 checker 独立 bug review

2026-09-24。审阅范围：`longchain_check.py`、现有单测、素材盘点和原 harness 消费路径。只读生成核心与冻结数据；未操作 GPU、服务或网络。以下行号是修复前审阅版本，主会话将集中修复。

结论：跨 session 借素材本身可行，但旧 checker 有可复现漏检，不能把其 `VALID` 扩展成下一版事件数据的完整验收。已确认 5 类应修代码问题；另有若干新设计尚未实现的检查，不混称已有成品损坏。

## 复现方式与证据限度

运行：

```sh
python3 evidence/longchain-design-20260924/reproduce-checker-bugs.py --fake-renderer
python3 -m unittest discover -s tests -p test_longchain_check.py
```

脚本为每个场景创建临时的 2 请求 generated 数据集，重算完整 artifact hash、cohort hash、逐请求 token/LCP 收据与链摘要，排除“只是陈旧 hash 导致失败”的混淆。临时文件自动清理。修前结果见 `checker-bugs-before.json`：正常对照和 8 个负例均返回 `VALID`；现有 21 项单测全部通过。

本机 `.venv-longchain` 的配置是已搬迁的 Linux Python 3.10 路径，实际 `python` 落到 Anaconda 3.12，缺可导入的 transformers。离线缓存的 transformers 4.56.0 无法加载现有 `TokenizersBackend`。因此使用明确标注的 `CharacterRenderer` 测试替身，验证 checker 控制流；这不是 GLM 长度/格式兼容验收。脚本不加 `--fake-renderer` 时走原 Renderer，具备正确冻结依赖的环境可以复跑。下列漏洞不依赖具体 tokenizer 的输出。

## 已确认代码漏洞

### P1：缺失 provenance，或 kind 拼错，可绕过生成数据专属约束

- 位置：`scripts/analysis/longchain_check.py:486–499`、`:574–576`、`:617–634`；工作账本 `:218–219`。
- 触发：generated root 没有整份 `provenance.jsonl`，artifact 清单也未列该文件；或 `kind="synthethic"` 拼错。
- 复现：`missing_provenance` 删除整个来源文件且把第二条替换为全新历史；`unknown_kind` 同时引入未标记的历史替换和错误 frozen uncached，均 `VALID`。
- 原因：文件存在才校验 ID 集合，kind 没有枚举约束；非 synthetic/adapted_original 不校验追加与冻结标签。
- 影响：来源账本断裂和错误 fast 分桶标签可能被放行；删除文件和小拼写错误不是罕见的攻击前提。
- 建议：generated 强制 provenance 文件存在、逐请求恰好一条、kind 属于有限集合；检查未知 kind 时直接 INVALID。原数据没有 provenance 的兼容模式应保留。

### P1：缺失 session_id 可通过 checker，但回放必然 KeyError

- 位置：checker 生成行检查 `:406–427`；原 harness `s1-dev/harness/s1_loadgen.py:347` 直接读取 `row["session_id"]`（`:309` 也读取）。
- 复现：`missing_session_id` 仅删第二条请求 session_id，仍 `VALID`。
- 影响：数据被判可用后，实际回放失败，无法得到完整请求集合；这是直接可用性漏洞。
- 建议：generated 请求校验 harness 必填字段和类型；同一链接收 session_id 应稳定。`session_changes_mid_chain` 也被放行，虽不必导致 KeyError，但违反当前明确的接收 session 契约。

### P2：合成追加可偷偷改变 system/tools，仍按 intra 接受

- 位置：checker `:578–594` 只比较 messages；`:155–162` 的 system/tools 比较只影响 append 汇总，不会报错。
- 复现：`silent_system_change` 保持 messages 为合法追加，第二条 system 整体替换，仍 `VALID`；链摘要如实记 append_only=0 即可通过。
- 影响：前部 token 变化造成额外重算，与声明的稳定历史续跑不同；下一版若混入这类错误，缓存策略对照会偏离事件计划。
- 建议：旧追加模式要求 system/tools 不变；未来明确版本变更事件应另外验证，不通过只改 phase 放行。

### P2：新增完整工具组可重复使用历史 call ID

- 位置：`_tool_errors` `:87–90` 把所有 ID 变为集合；`_new_tool_block_errors` `:132–144` 只检查单个组内重复。
- 复现：`duplicate_call_across_blocks` 在已有闭合 call_1/result 后，再追加一个 call_1/result 完整组，仍 `VALID`。
- 影响：借入新实例未正确改 ID 时不被发现；上下文内工具引用有歧义。不同请求快照正常携带历史不应误判，但同一快照中两个不同调用实例不是“快照重复”。
- 建议：对新增调用 ID 同时与前驱已有 ID、该 suffix 早先新增 ID 比较；只对新增实例执行严格规则，避免未经核实地拒绝原样源历史的重复 ID。

## 需补的设计检查，不作为已生成数据的实证 bug

- 来源校验当前主要是“数据对自身 metadata 自洽”。`nonexistent_original_source` 仍 VALID：original 的 source_req_id/source hash 未回查实际源文件；adapted_original 的 parent 缺失只 warning。新生成器应提供可离线核验的素材 source hash/ID 账本，不能把自洽 hash 当来源真实性证明。
- `invalid_gap_decomposition` 的 999999999 ms gap 仍 VALID，因为 `:415–422` 只校验有限非负。它可以被 harness 的每链 3600 秒 cap 再缩放，意外改变整条链节奏。当前源转换约定单请求不超过 310 秒，但 checker 没有原始 tool_union/net_think → replay_gap → effective_gap 的关系校验。把其列为数据时序契约缺口；不要把所有长等待强行认定为缓存 miss。
- 新设计的 context_reset 仍会被旧 checker 的“合成必须追加”规则拒绝。这是已披露未实现能力，不能直接放宽标签；需要独立确认实际删除/改写、完整工具组和重建后增长。
- 跨 session 长前缀、完整 prompt/新增大块克隆检查尚无实现；目前 `donor_reuse` 仅统计自报 fingerprint 使用次数，且 token LCP 只比较链内相邻请求。因此 `VALID` 不证明反克隆约束。跨源借素材和正常共享 system/tools 不应被简单当作违规。
- `material-inventory.json` 明确把 433 user、2381 工具组列为候选，保留 ID 去重且允许隐式配对。它没有声称真人数、独立语义材料数或可接受续接数；当前措辞合理。组合数量不证明有效分布和覆盖，不能据 69899 配对数推断真实性。

## 旧冻结候选的范围

本次轻量复查旧 `data/longchain-screen/requests.jsonl` 共 1718 请求：缺 session_id=0，链内 session_id 混用=0，replay_gap>310s=0。其原哈希/全量渲染验收证据仍保留；本次没有重复大集 token 渲染，也没有证明旧数据包含上面所有负例。

旧 CPU VALID 能证明其已检查的结构、账目、哈希、token/LCP 自洽；不能证明新机制已实现、来源逐块核验齐全、无意外共享前缀、线上分布已复原、真实 MTP/MoE 行为相同或 GPU SLO 通过。优先修复上述明确漏洞，再给新事件机制添加专属失败反例。

## 修复后独立复查

主会话修复 checker；本审阅者仅修改 `tests/test_longchain_check.py` 增加 6 项回归测试组。重新执行上面的两个命令：**27 项 checker 单测全部通过**；包含真实文件 hash/cohort 的 fixture，原有正例不被粗暴禁用。新增覆盖整份来源文件缺失、未知 kind、system 或 tools 隐变、缺失/混用 session、新工具 ID 与旧历史冲突及 suffix 内跨组冲突。另覆盖另一位审阅者发现的重复 polish：新 adapted_original 必须继承祖先 source hash/frozen labels，不能误拿已改写的直接 parent 当前 body/labels 比较。

`checker-bugs-after.json` 结果：正常对照仍 VALID；6 个明确结构负例全部 INVALID，错误原因与预期一致。超大正数 gap、不存在原始来源两个例子仍 VALID，保留为**尚未实现的观测/策略审计**，没有声称修好。所有复查依旧使用 CharacterRenderer 隔离控制流；没有用它替代成品的真实 GLM 渲染验收。新事件核心与跨 session 反克隆仍需后续实现。
