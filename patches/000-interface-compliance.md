# 000 — 接口合规（D0）设计说明

作者：Claude · 2026-09-22 · 状态：已实现，属于 S0 基线，所有 8 卡运行都包含
依据：SGLang v0.5.20 源码（`src/sglang`，只读）、`llm-challenge-arena-v1/task.md`「服务接口规范」、F9/F10。

## 1. 题面要求与 v0.5.20 原生行为逐项对照

| 要求（task.md） | v0.5.20 原生行为 | 结论 |
|---|---|---|
| `POST /generate` SSE，`text` 原样送入引擎、不套模板 | 原生 `/generate` 接收 `text`，不套 chat template（V：原生接口语义） | 合规，保持不动 |
| `meta_info.completion_tokens` 为**累计值** | 流式输出的 meta_info 带累计 `completion_tokens`（V：`tokenizer_manager.py:2350-2461`） | 合规 |
| `prompt_tokens`、`cached_tokens` 出现在首个产出 token 的事件里 | 每个事件都会写这两个字段（V：同上 `:2356,2401`） | 合规 |
| `request_received_ts`（取自末事件） | 由 `APIServerReqTimeStats.convert_to_output_meta_info` 写入，**只在请求结束时**合并进去（V：`req_time_stats.py:494-503`、`tokenizer_manager.py:2603`） | 末事件里有，合规。值取自 `created_time`：请求未带 `received_time` 时，等于 `_init_req_state` 执行时的 `perf_counter`（V：`tokenizer_manager.py:3571,3609`，`req_time_stats.py:404-406`） |
| `prefill_finished_time`（取自首事件） | 由调度器侧 `convert_to_output_meta_info` 写入，**只在 `--enable-metrics` 打开时才有**（V：`tokenizer_manager.py:2359-2362`、`req_time_stats.py:1177-1186`） | **必须加 `--enable-metrics`**，否则 TTFT 回退到客户端口径，网络抖动会算进成绩 |
| 发增量 `text`，不要发累计 | 默认发累计 `text`（`incremental_streaming_output=False`，V：`arg_groups/fields/serving.py:265`） | **加 `--incremental-streaming-output`**，免得长输出拖慢 SSE、抬高 tpot_mean |
| `ignore_eos=true` 生效、输出正好 `max_new_tokens` 个 token | 原生支持 `ignore_eos` | 合规（开 MTP 后需实测：输出不能超过 max_new_tokens） |
| `POST /flush_cache` 返回 2xx 和 JSON `{"success": true}` | 返回**纯文本**（`"Cache flushed.…"`），状态码 200/400（V：`http_server.py:990-1005`） | **不合规，要改**。题面明确要求 JSON，平台会不会解析 body 未知，按最严理解处理 |
| flush 必须真的清掉前缀 KV | 调度器要完全空闲才执行清理；UnifiedRadixCache 的 reset 会清到主机 L2，但不清 L3 外部存储（F9）。DP 下 `tokenizer_control_mixin.py:301-310` **只取第 0 个 worker 的结果** | **DP>1 时要改成"所有 worker 都成功"再返回 success**；不启用 L3 存储 |
| `GET /v1/models` 返回 200 | 原生支持 | 合规 |
| 未知的 `X-S1-*` 请求头不能导致拒绝 | 原生忽略未知头 | 合规 |

## 2. 改动方案（最小化，写在引擎进程里，不加反向代理）

**2.1 `/flush_cache` 返回 JSON，并汇总所有 worker 的结果**
- `http_server.py` 的 `flush_cache`：成功时返回 `JSONResponse({"success": true, "message": ...}, 200)`；失败时返回 `JSONResponse({"success": false, "message": ...}, 400)`。
- `tokenizer_control_mixin.flush_cache`：把 `[0]` 改为遍历全部结果，`success = all(r.success)`，message 拼接各 worker 的信息。
- 空闲语义保持不变（有在途请求时拒绝并返回 400）。平台在每档测量前才调用，此时应已空闲；遇到瞬时忙碌可以在服务端做有界重试，例如等到 `timeout` 参数给定的上限（原生已支持 `timeout` 查询参数）。
- **不得**在未清空时返回 success（红线 3）。

**2.2 启动参数（submission.json 的 command）**
- 必加：`--enable-metrics`（为了拿到 `prefill_finished_time`）、`--incremental-streaming-output`。
- 需确认：`--enable-metrics` 的额外开销（Prometheus 计数），在 8 卡上与关闭时对比 tpot。

**2.3 时间戳语义自查（红线 2）**
- `request_received_ts` 应尽量接近"服务端收到请求"的时刻。目前取在 `_init_req_state`，而这一步发生在 FastAPI 解析完约 1 MB 的 JSON body 之后，**会略晚于真实收到的时刻**，偏向对我们有利的方向。
- 最诚实的做法：在 HTTP handler 入口把 `obj.received_time = time.perf_counter()` 写进去（OpenAI 接口已经这么做，见 `serving_base.py:78-98`），让 `/generate` 与之对齐。**这是合规加固，不是优化**；需要 Codex 审阅，确认口径。
- `prefill_finished_time` 在 `batch_result_processor.py:345/465` 设置，即 prefill 批次结果处理完、首 token 产出的时刻，语义与题面一致。

**2.4 rid 重复**
- harness 把 `rid` 设为请求 ID（`s1_loadgen.py:108`）。在 `rid_to_state` 中重复会直接抛 ValueError（`tokenizer_manager.py:3601`）。preflight、warmup 和正式测量是否会复用同一个 rid，需要读 `run_dev.py` 确认。如果会，必须保证前一个已结束；否则就是 ENGINE 错误。

## 3. 本地评分补丁（不改 harness）
- 新写 `scripts/score_formal.py`：读取 `run_dev.py` 产出的 `raw_*.jsonl`，在 dev 的 10 条门之外补上正式的第 11 条门 `tpot_p95 ≤ 0.10`（F10），同时输出 `tpot_mean`，以及 `cached_tokens` 与冻结 `uncached_expected` 的偏差分布，用于 D1 对账。
- 统计余量规则（95% 单侧下界）按 task.md 实现，报告同时列出点估计结论和带余量的结论。

## 4. 验收清单（等用户批准实验后执行）
1. `curl -X POST /flush_cache` 返回 200 和 `{"success": true}`；有在途请求时返回 400 和 `{"success": false}`。
2. 同一条 prompt 连打两次：第二次 `cached_tokens` 上涨；flush 之后归零。
3. SSE 首事件带 `prefill_finished_time`，末事件带 `request_received_ts`；harness 报告里 `ttft_source=server`。
4. 输出 token 数恰好等于 `max_new_tokens`（MTP 开启与关闭两种情况都测）。
5. DP>1 时，故意让其中一个 worker 失败（例如制造在途请求），flush 必须返回 false。

## 5. Codex 交叉审阅（2026-09-22 UTC，追加，不改作者原稿）

**结论：有条件同意 D0 方向；请修订下面的口径后再进入实现。当前仍只有设计，没有实现/实验授权。**

1. **同意 JSON + 全 worker 汇总（VERIFIED）**。现有 `managers/communicator.py:13–120` 已有 fan-out 收齐机制和 `merge_results`，优先复用。注意该 helper 对空列表的 `all([])` 为 true；不能把空/缺响应、异常、超时当作成功。保留预期响应数和有界失败处理；接收列表按到达顺序，不能用列表下标伪装真实 DP rank。成功后才清 tokenizer 的本地 preprocessing cache；当前 L3 不启用。
2. **时间戳入口需更正（VERIFIED / INFERRED）**。FastAPI 的 typed handler 入口已经在 body 解析之后；仅在此处设置 `received_time` 不能覆盖 JSON 解析开销。若按真正服务端入口加固，应在 ASGI 请求入口、消费 body 之前记录服务端 monotonic 时刻，经 request state 传入原生时间转换链，不能使用客户端可控的 `received_time`。不要把 epoch 值塞进期望 `perf_counter` 的字段。现有 `req_time_stats.py:494,1177` 已做输出的 epoch 转换；保留其时钟域一致性。该设计是口径澄清，不能让 admission/排队从 TTFT 中消失。
3. **metrics 是候选方案要求，不是题面硬性要求（VERIFIED）**。`task.md:202–205,556` 允许缺服务端时间戳时回退客户端计时；因此“必须加 enable-metrics”应限于本方案选择服务端计时的前提。可以为减少噪声选择它，但不能说 metrics 关闭就必然不合规。首 token 验收以真正有模型 token 的事件为准（累计 completion_tokens 开始大于 0），空心跳/role/usage 不算。
4. **wrapper 不宣称复刻正式判分（VERIFIED）**。题面给出四道 TTFT 的超标率 95% 单侧下界规则，但未给出置信区间估计器的完整实现。wrapper 应同时保留原 dev 结果、TPOT 硬门和明确标注方法的统计估计；方法未确认前不标“formal PASS”。不要把 TTFT 的余量自动套到 `tpot_p95 <= 0.10`。coverage、错误率、无样本桶均不能被统计过滤掩盖（`task.md:519,565,577`）。
5. **后续验收补项（INFERRED）**：DP flush 混合成功/失败、缺失响应、超时、非空 HiCache 主机层（如将来启用）分别测试；原生 request id 重用先只读核对生命周期，不改 harness。单次同 prompt 前后 cached_tokens 对照不足以证明全 worker 都清了，需要覆盖每个可路由 worker。

作者：Codex。以上是文档审阅，不包含新的执行结果。

### 5.1 针对再次交接的时间戳确认（Codex，2026-09-22 UTC）

**VERIFIED**：`entrypoints/http_server.py:911–918` 明确由 FastAPI 先把 JSON 转成 `GenerateReqInput` 再进入 typed handler。因此“从 `_init_req_state` 移到 handler entry”可以提前计时，但**仍不能计入此前的 body 解析时间**；请不要在修订稿中把这两个入口等同。[FastAPI 官方 middleware 文档](https://fastapi.tiangolo.com/tutorial/middleware/)说明 middleware 先于路由执行。

**INFERRED**：若选择服务端应用接收口径，在消费 body 前的应用入口记录一次真实 monotonic 时刻，handler 只传递它，不重采样覆盖；保留原生 epoch 转换与 admission/队列耗时。应用入口也不应宣称为网络首字节时刻。SSE 返回后不要缓存/聚合整条响应来测量，避免破坏增量流式。

JSON flush、全 worker 成功归并、服务端时间戳路径启用 metrics、增量 text 的方向继续同意；§5 的失败处理、可信时间源与评分估计边界仍是确认条件。此处仅补充证据，没有实现 middleware 或接口修改。

## 6. Codex T29：v1.1代码复核（2026-09-22）

冻结补丁SHA256=`e0d7924989ebb538e331110f4458ccd4e95a0bfa3cbd87dfb8324ac0328960f6`，基线94602c9；D1配套版本仍是001 v1.1，不包含002/003/004。上面的“未实现/仅设计”是历史状态；本节审阅已实现补丁，E2已验证单worker空闲JSON路径，其余live结果追加到experiments/TEST_PLAN，不提前宣布通过。

- **VERIFIED / 有条件同意**：ASGI middleware在调用下游之前采样perf_counter到请求scope私有键；typed `/generate`用此值覆盖客户端的received_time，沿用原epoch换算，不去掉排队时间。不消费body、不聚合SSE；应用层入口不等于网络首字节，外层middleware的耗时仍不在此时间内。
- **VERIFIED**：HTTP flush使用ORJSONResponse和真实布尔值，保留200/400；tokenizer层要求非空且all成功，任一失败不清本地多模态预处理缓存；timeout原样传到调度器。`scripts/test_d0_v11_review.py`对实际build/d0 AST的6项CPU检查通过：全成功、混合失败、空结果、无processor、body前私有scope采样、非HTTP透传。它们不是DP live验证。
- **VERIFIED / 未解决的可靠性边界**：`FanOutCommunicator`依赖配置的fan_out响应计数收齐，D0没有新增通信超时或rank身份校验。`timeout=N`约束调度器等待空闲，不给失联worker的communicator await提供硬截止。客户端必须有独立HTTP截止；不能把卡住/连接中断视为flush成功。错误字符串`worker{i}`只是响应到达序号，**不是DP rank**，排障不要据此定位具体卡。
- **验收范围**：本轮可在原qfull TP1验证IF-08忙碌400/等待200及flush后归零；IF-09多worker混合失败、HiCache/L3清理仍未测，HiCache/L3保持关闭。D0不改变模型输出或修复D1-04数值门；既有D1-02/04失败不因接口检查通过而解除。
- **T29结果回填**：W6 IF-08实际4096-token流验证busy400/false、等待54.409s后200/true、同prompt冷命中0；同流经过严格DONE/逐事件字段与计数/首末时间戳检查，IF-03附带通过。只限L2 TP1、无MTP/HiCache，不代替IF-09或IF-12；证据`evidence/T29/first/if08.json`，完整说明见experiments T29/F47。
