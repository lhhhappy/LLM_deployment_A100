# 独立审查：原 harness 与部署任务兼容性

2026-09-24。只读审查 `task.md`、原 harness、当前 `longchain.py`、独立 checker 与现行设计；未改源代码、未接触 GPU、未发网络请求。复现只使用临时目录和 CPU。这里审查的是当前追加式实现及下一版接入风险，不把尚未实现的事件核心说成通过。

结论：从 A session 取 query/完整片段，适配后续接 B，**与固定轨迹回放契约兼容**。正文来自哪条源 session 不影响接口能否执行；必须重新冻结接收实例的完整输入、实际 GLM token、前缀关系与输出预算。兼容不等于改写前后性能等价。以下有两个原 harness 缺口和两个当前生成器契约/元数据问题，须在外层封住；不能只凭原自检的 PASS 放行。

## 发现

### H1，P1，原 self-check 可在正文完全缺失时返回 PASS

- 位置：`s1-dev/harness/s1_loadgen.py:472–485`，尤其 475–477 的 `if b is None: continue`。
- 触发：cohort/requests 有请求，正文分片缺失、损坏，或缓存返回了另一批 ID。所有缺正文请求不进入计数，`bad=0`，返回码仍为 0。
- CPU 反例：一个请求、零正文，真实 `loadgen.main --self-check` 返回 `自检结果：失配=0；共=1 / SELF_CHECK: PASS`。为避免加载 tokenizer，反例注入了一个禁止 render 的 Renderer stub；因为没有正文，该 stub 不会参与判断，也未模拟引擎。
- 影响：若将 self-check PASS 当作完整性验收，缺失正文会漏过。正式回放中的 `drive` 会记录 `HARNESS_DATA:body_missing`（355–356），所以并非正式计分一定漏报，而是验收假阳性会浪费测量并误导完成声明。
- 修法：原 harness 保持只读；生成交付入口先强制独立 checker 的 `VALID`，核对 cohort ID、正文 ID、哈希和渲染覆盖数均一致，再做原 self-check。独立 checker 的 459–460 已有缺正文检查；旧冻结集做过此独立检查，本发现不推翻旧集验收。

### H2，P1，正文缓存不绑定数据内容，切换同名合成集时可能回放旧正文

- 位置：`s1-dev/harness/s1_common.py:321–337`；`s1_loadgen.py:468–469` 的缓存路径仅为 `<out-dir>/body_cache_<set>.jsonl`。
- 触发：复用同一 out-dir 与 set，换 root、重新生成或改写同 ID 正文。`ensure_bodies` 只判断 `len(bodies) < len(wanted)`，既不核对 ID 集合，也不核对 root/hash。相同行数就使用旧缓存；甚至同样多、但完全不同 ID 的缓存也接受。
- CPU 反例：当前分片 `same-id=NEW`，缓存 `same-id=OLD`，函数返回 OLD；想要 `new-id` 时仍返回只有 `same-id` 的缓存。若旧/新文本 token 数相同，纯长度自检也无法识别内容错误。若 ID 不同，则与 H1 组合成缺正文假 PASS。
- 影响：A/B 可能测到旧材料而非已验收材料，cohort hash 也不能兜底，因为原 cohort SHA 只绑定有序 chain/request ID 等结构（`s1_common.py:284–285`），不绑定 body 字节。
- 修法：不改只读原 harness；每个冻结 artifact 分配独立且内容寻址的输出目录/set，执行前绑定 manifest/body 哈希；直接调用 loadgen 时传 `--no-body-cache`。`run_dev.py` 不透传该开关，因此使用它时尤其应采用新的输出目录，并保证其内缓存仅服务这一 artifact。同一 artifact 的 JIT 预热/测量可共享该缓存。

### H3，P2，旧生成器尚未提供设计要求的独立接收 session ID

- 位置：`scripts/analysis/longchain.py:490–492` 直接保留原行；530 继续使用 `target['session_id']`；chain ID 也沿用 source。
- 实测旧成品：96 条 chain 只有 53 个 `(pack, session_id)`；11 个 session 对应多条 chain，最大 15 条。原 harness 在 `s1_loadgen.py:347` 取此 ID，并在 113 作为 `X-S1-Session-ID` 发送。
- 影响：这不是证明 KV 自动串链，也不是原 s1-dev 错误（源数据本来有一 session 多 chain）。但与新设计“每个合成接收实例独立 session/request/tool ID”的明确契约不一致；未来若部署按 session 做亲和/限流/路由，同一来源的多个合成实例会被当作同一 session。
- 修法：新事件核心分配稳定的 artifact+instance session/chain/request ID，另存 source_session/source_chain/source_req。正文内已存在调用引用应成组处理；不能靠给 prompt 加盐实现 ID 独立。新 checker 检查实例标识与来源标识分离。旧冻结集不原地改。

### H4，P2，合成 dispatch 字段混入了并非真实时序的值

- 位置：`scripts/analysis/longchain.py:533–534` 用“上条 dispatch + donor gap”生成下一 dispatch，却未包含上一请求执行时间；end 留空。
- 实测旧成品：59 条首次合成续接的 dispatch 早于前一条真实请求 end。例如 `biomaster:canon:lc_20260924_000:llm:0005`：prev dispatch 311214392 ms，prev end 311223783 ms，next dispatch 311215082 ms，gap 690 ms。
- 影响边界：不会让原 harness 并行跑同链；它只用 dispatch 排序（`s1_common.py:174`），`drive` 仍是前请求完成后等待 gap 再执行下一条（`s1_loadgen.py:334–391`）。问题在于这些字段看起来像真实相对时刻，若用它们分析连续性、工具等待或用户行为，会得到负间隔/错误执行时间。当前 manifest 未明确逐字段说明排序时刻不是真实发生时刻。
- 修法：新产物明确记录 `event_index`、原始观测时间与合成排序坐标的区别；为原 harness 保留严格递增排序字段，并写明它只排序、不驱动到达。不能凭空补 execution duration 伪装真实时间。实际时间只取测量 raw。

## 可复现证据

运行 `python3 -B evidence/longchain-design-20260924/repro-harness-bugs.py`，实测通过三个断言：旧正文缓存被返回；等数量错 ID 缓存被接受；零正文的 self-check 返回 0/PASS。脚本使用临时目录，结束自动清理，不修改只读 harness，也不调用任何服务。

## 已核对成立的接口和下一版验收要求

- 分桶：链首不论 phase 都进 chain；内部 turn_start 进 turn；内部 context_reset 进 chain；其余进 overall，冻结 `uncached_expected<=4096` 再进入 fast。见 `s1_common.py:107–146`。不能从实际 cache_read 决定桶，也不能只改 reset 标签放宽门。
- N：worker 持有整链直到结束，gap 期间继续占槽；不等于 HTTP 并发。见 `s1_loadgen.py:580–599`。实际交错会随部署速度变化，固定轨迹不等于固定跨链到达顺序。
- 时间：每条 `replay_gap` 在该请求发送前等待，包括链首；每链 cap 3600 秒由原 harness 执行。只记录“gap p95”不足以证明负载近似，还需检查缩放后的 gap、长链位置与联合关系。
- 缓存：同一测量共用 cache_namespace，session header 不天然隔离 token cache。不能用新 session ID 保证无跨链共享；应审核实际渲染 LCP 与克隆材料。
- 输出：原 `/generate` 使用 `max_new_tokens=max_output_i, ignore_eos=True`（`s1_loadgen.py:100–106`）。下一 prompt 内拼长 assistant 文本不会增加本次 decode 预算；两者都应单独建模。实际文本仍可影响 MTP、MoE 与生成速度，不能宣称内容置换无性能影响。
- 渲染：生成器复用了原 `Renderer`，当前路径为 apply_chat_template 后 encode，无手写模板；当前旧成品 prompt+output 最大 260393，不存在已发现的超部署上下文例。新候选依然必须按选定部署上限验收，不能仅因为 tokenizer 的 1M 上限就认为部署支持。
- 评分：`scripts/score_formal.py:469–471` 默认 requests 是原 722 条 dev，合成集调用必须显式 `--requests <新root>/requests.jsonl`；`level_verdict.py --data-root` 可传新 root。默认不是源码错误，但新交付命令遗漏就会 INVALID。
- 稳态 TPM：原 scorer 默认窗口 10–70 分钟，且六个子窗口均需有请求、最后请求至少达到窗口终点（`s1_score.py:110–142`）。24–32 链小集不保证有有效 TPM，可用于结构和机制诊断；不能为了窗口人为添加空等或把全程平均冒充正式口径。
- 新方案的“重建→继续增长”、受控用户追问尚未实现；旧 checker 577–590 仍强制合成都为 append/reminder。实现后必须补独立反例测试：真重建可识别、只换标签拒绝、孤立 tool 拒绝、重建后的旧历史不能无意长回。此为已知未实现范围，不是本审查发现的成品随机失败。

不能由这些 CPU 契约证明隐藏正式负载已还原；本审查也没有给出任何 GPU 分数/性能等价结论。
