# 130 — 完整分词移出事件循环 + S1 路由键（T42 / W16）

状态：CPU 验证通过；待 Claude 交叉审阅与 L2。基线：`build/base_exact/sglang` 的副本，依次打 000 → 101 → 110 → 111 → 130，全部 `patch -p3 --fuzz=0`。不适用于 v0.5.20。

## 行为与回退

- `SGLANG_AX_ASYNC_TOKENIZE` 缺省开启；启动前设 `0` 恢复原始同步分词策略。现有 `--enable-dynamic-batch-tokenizer` 优先：启用该参数时，130 不创建新线程池，所有输入维持原策略。
- 每个 TokenizerManager 一个按需启动的专用线程、一个异步信号量。所有 regular text 调用（包括小输入、batch、cross-encoder）经同一个 worker 串行执行，避免 HF wrapper 的 padding/truncation 状态并发写入。没有文本大小阈值，不会在另一个长输入编码时穿插一个主线程小输入编码。
- 用原始完整文本、原始 kwargs 调原来的 `tokenizer(...)` / slow `encode(...)`；token_type_ids、EmbeddingGemma EOS 和结果形状逻辑保留。没有重套模板，没有截断，没有改变输出预算。
- 请求取消只取消调用方的等待，不停止 Rust 编码。`shield` 保留底层 future，实际完成（含异常）后才释放槽位并消费废弃异常；排队取消不会提交 executor 工作。executor 至多有一个任务，避免取消风暴累积 native 编码队列。正常关闭调用 `shutdown(wait=False, cancel_futures=True)`，已运行任务允许结束。
- HTTP `/generate`：保留 body 中非 `None` 的 `routing_key`（包括显式空串）；否则取非空 `X-S1-Routing-Key`，再取非空 `X-S1-Session-ID`；没有提示则仍为 `None`。HTTP 框架的 Headers 查找不区分大小写；不 strip 或解释值。该接线独立于 `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES`。
- Session-ID 只作为 routing fallback；不写 native `session_id/session_params`、cache_salt 或 priority。修复 `GenerateReqInput.__getitem__` 原来遗漏 routing_key 的复制，使 batch 子请求保留同一键（也覆盖显式 body 键）。不修改 scheduler；沿已有 TokenizedGenerateReqInput → Req 字段透传。
- 开关只回退分词；完整撤销（含路由接线）在同一副本执行 `patch -R -p3 --fuzz=0 < patches/130-async-tokenize.patch`。

## 正确性与边界

**VERIFIED（源码与 CPU）**：原分词代码仅被提取到 `_tokenize_texts_sync` 并移动执行位置。相同 tokenizer、输入列表、kwargs、分支和完整 encode 次数，返回相同 IDs；后处理没有变化。测试直接提取并执行 baseline/candidate 的生产方法，使用真实 tokenizer 验证，不用另写的分词模型代替生产方法。CPU 测试不导入完整 GPU 服务依赖。

不实现可选前缀分词缓存。即使前缀字符串逐字节一致，`encode(prefix) + encode(suffix)` 一般也不等于 `encode(prefix + suffix)`，词尾 BPE merge/预分词边界会变化。固定回退若干字符不能构成通用证明；本补丁每次重新完整编码，自然避免边界、Unicode、特殊 token 与前缀失配问题。真实 GLM tokenizer 的反例见 `real_results.json` 的 `prefix_merge_counterexample`。没有新增持久缓存，因此不需要额外 flush 分词缓存。

000 的 ASGI 接收时间戳、tokenize_finish、scheduler 的 prefill/decode 时钟、所有 token 计数及 `/flush_cache` 实现均未改。线程池排队/分词开销仍包含在真实请求耗时内；不把分词耗时从 TTFT 中扣除。

**INFERRED（须 L2）**：线程能释放 HTTP loop，预期减少并发 SSE 被长分词延迟的时间；不能由本地 CPU 数据推出 N@SLO 或 TPOT 收益。单线程没有提高 CPU 分词吞吐，小请求仍排在正在编码的大请求后；排队时间可能成为下一个瓶颈。Python wrapper/IDs 物化仍有短暂 GIL 占用，不能承诺零 loop lag。其他后端、慢 tokenizer、多模态处理与 OpenAI chat-template 渲染未量化；本补丁不移动模板渲染、JSON 解析或 IPC。

## CPU 复现与证据

本次历史 CPU 复现使用 `transformers==5.12.1`、`tokenizers==0.22.2` 和 `jinja2`，不需要 torch/GPU。一次性 runner 已从现行 `scripts/` 清理，不能直接照旧命令重跑；[evidence/T42/](../evidence/T42/)保留逐请求比较、覆盖率、性能结果、依赖锁定和当时的运行日志。

脚本自动复制只读底包到临时目录，按顺序应用补丁，执行生产方法抽取测试。真实输入用未修改的 `s1-dev/harness/s1_common.py:Renderer` 渲染；对所有 bodies 验证开启/关闭/原版的 IDs 完全相同，并对照 requests 的冻结 `glm_tokens`。逐请求只保存 ID、长度和 token IDs 的 SHA256，不保存 prompt 正文。

- `patch_apply.log`：完整补丁链；`cpu_tests.log`：最终 CPU 测试与进度。
- `token_comparison.jsonl`、`coverage.json`：逐请求三路径相等和数据集覆盖。
- `real_results.json`：版本、架构、数据量、循环心跳与耗时。
- `requirements-lock.txt`：实际 CPU Python 依赖；`receipt.json`：输入、补丁和测试脚本哈希。
- `cpu_tests_dependency_mismatch.log`：首次 transformers 4.x 不支持 TokenizersBackend，随后按底包 pyproject 对齐 5.12.1；`cpu_tests_metadata_attempt.log`：测试脚本首次假定 metadata 有 req_id，修正为 pack:view:logical_call_id 后全量重跑。均非生产补丁失败。

**VERIFIED（M3-01…05）**：14/14 单测；722/722 原样对话（合计 34,416,777 tokens，最长 256,733）新/旧/关闭逐 token 一致，冻结 glm_tokens 全同；7 边界文本、21 并发请求和 3 batch/pair 对照通过。无完整 SGLang/GPU 服务导入。

本地 Linux aarch64 / Python 3.10.12 / transformers 5.12.1 / tokenizers 0.22.2 / TokenizersBackend（is_fast=True）。1ms 心跳，交替次序各3次，两个执行路径预热；**原样对话**数据（`real_prompt_benchmark.json`）：

| 原样输入 tokens | 路径 | 分词耗时中位数 ms | 每次最大 loop lag 的中位数 ms | 三次最坏 loop lag ms |
|---|---|---|---|---|
| 100214 | sync | 69.148 | 68.201 | 90.027 |
| 100214 | thread | 54.771 | 2.483 | 3.026 |
| 256733 | sync | 195.517 | 194.570 | 195.166 |
| 256733 | thread | 176.632 | 8.590 | 9.470 |

原样同步分词期间每次均 0 个心跳；线程分词期间 10 万输入每次 26–28 次、25 万输入每次 80–87 次。线程路径仍有约 9ms 的短暂峰值，不能写成“完全无阻塞”。本地耗时有 CPU/调度噪声，不宣称吞吐加速。

**GIL 实测**：直接 `backend_tokenizer.encode_batch([text])` 在线程内执行，249,967 token 耗时 204.4ms，等待期间 loop 运行 102 次心跳，最大 lag 4.89ms（`real_results.json:rust_backend_thread`）。该 native 编码调用未持续持有 GIL；包括 Python wrapper 的完整生产路径也已实测改善。不能外推所有 tokenizer/版本均释放 GIL。

`real_results.json:benchmark` 另保留从真实文本派生的 96,262 / 249,967 token CPU 压力输入，各3组交替：最大 loop lag 中位数分别 115.62→2.67ms、297.30→8.36ms。仅此压力基准在测试进程中构造文本；722 条正确性输入和上表原样性能输入没有修改，也从未发送给模型或冒充 SLO 成绩。


## 8 卡验证方案（未执行，由 Claude 审阅并安排）

1. 核对底包实际 transformers/tokenizers 版本及 `is_fast`；相同底包链与参数，仅切换 `SGLANG_AX_ASYNC_TOKENIZE=0/1`。先做完整导入、启动和退出测试。
2. 选公开集原始短/长（约 10 万和 25 万 token）多轮对话，同时维持生成流；核对每条 prompt_tokens / return_prompt_token_ids、ignore_eos 的完整输出长度、thinking/tools、无路由提示/有提示/body 覆盖/batch。
3. 检查接收→tokenize_finish→prefill→response 时序及实际 SSE 间隔；测空闲与繁忙 flush、分词中断开客户端、排队取消、长短混合与持续取消时的 CPU/RSS/线程数。
4. 相同配置分别全量原版 dev 回放（每档真 flush），交替顺序并重复；比较全部 TTFT 桶、TPOT、错误率和 tokenizer CPU 排队。CPU loop-lag 改善不能代替此步；不把当前补丁自动写入 RELEASE、构建脚本、提测或提交队列。

交叉审阅：待 Claude。
