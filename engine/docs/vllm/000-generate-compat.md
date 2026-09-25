# vllm 000 — 评测接口：`/generate` 与 `/flush_cache`

task.md 要求引擎根路径提供 SGLang 形的 `POST /generate`（SSE）和 `POST /flush_cache`。官方 vLLM 没有这两个口，
但 `main` 已有 `vllm.endpoint_plugins` 扩展点（`vllm/plugins/endpoint_plugins/interface.py`），task.md「引擎适配指引」推荐的也是这种进程内插件：
与 OpenAI 接口同进程、同端口，直接用引擎自己的 token 计数，不加反向代理。

## 实现

代码在 `vllm/entrypoints/generate_compat/`，入口 `generate_compat` 登记在 `pyproject.toml` 的 `vllm.endpoint_plugins` 组。
端点插件默认不加载，启动时必须设 `VLLM_PLUGINS=generate_compat,lora_filesystem_resolver,lora_hf_hub_resolver`
（该变量同时收紧所有插件组，所以把底包自带的两个通用插件一并列上）。不改 vLLM 核心代码。
插件加载后在 API 进程打印一行 `[ax] vllm mechanisms: base=a811738a6 000=on ...`（`vllm/ax_mechanisms.py`），任务据此核对实际生效的机制。

| 项目 | 行为 |
|---|---|
| 请求体 | `text`（或 `input_ids`）、`sampling_params`、`stream`、`rid`；未知字段和未知采样字段一律 400，不静默忽略 |
| 采样字段 | 只接受两边含义相同的：`max_new_tokens→max_tokens`（缺省 128，同 SGLang）、`min_new_tokens→min_tokens`、`temperature`、`top_p`、`top_k`、`min_p`、三种 penalty、`stop`、`stop_token_ids`、`ignore_eos`、两个 special-token 开关、`sampling_seed→seed`；`n` 只能为 1 |
| prompt | 原样交给引擎的补全渲染器（与 `/v1/completions` 同一路径），不套 chat template |
| 请求 id | `rid` 原样作为引擎请求 id（vLLM 内部另加随机后缀保证唯一），服务日志可按 harness 的请求 id 追踪 |
| SSE | 每个产生新 token 的引擎输出发一个事件 `data: {"text": 增量, "meta_info": {...}}`，最后 `data: [DONE]`；增量文本，不发累计文本 |
| 中途出错 | 发 `data: {"error": {...}}` 再 `[DONE]`；客户端断开时引擎请求被 abort（`AsyncLLM.generate` 的取消路径） |
| `/flush_cache` | `reset_prefix_cache(reset_running_requests=False, reset_connector=True)`。引擎侧：块池除 null 块外有任何占用（运行/等待中的请求、在途传输持有的块）即拒绝；否则删除全部前缀哈希（`vllm/v1/core/block_pool.py` `reset_prefix_cache`）。前缀索引只在引擎核心进程的调度器里，各 TP worker 不持有索引，所以没有"部分 worker 未清"的状态。成功 200、拒绝 400，响应体是下表的收据。不动代码缓存与 CUDA graph |

### flush 收据（响应体与服务日志同一个 dict）

服务日志写一行 `[ax] flush_cache {JSON}`，与响应体内容相同：

| 字段 | 含义 |
|---|---|
| `success` | 引擎确认已清空（见上）；`false` 时另有 `message` |
| `flush_started_ts`、`flush_finished_ts` | API 进程 epoch 秒，包住对引擎的调用 |
| `unfinished_requests_at_start` | 开始时 API 进程仍在流式输出的请求数（`AsyncLLM.get_num_unfinished_requests`） |
| `reset_connector` | 恒为 `true` |
| `kv_connector` | 配置的 KV connector 名称；无则 `null` |
| `engine`、`engine_version` | `vllm` 与实际安装版本（打包版本含 `.ax.<提交>`） |

边界：`kv_connector` 非空时，`success` 只证明引擎侧索引清空与 connector 的 `reset_cache()` 未返回失败；
OffloadingConnector 的二级层（FS/网络）不被清除、已排队的传输任务在后续步骤才消化（Codex 源码复核），
所以目前只有"无 connector"这条路径的 flush 语义是闭合的；启用主机层前须另行验证。

## `meta_info` 字段（每个事件都带）

| 字段 | 来源 | 说明 |
|---|---|---|
| `prompt_tokens` | `len(RequestOutput.prompt_token_ids)` | 引擎实际输入 token 数 |
| `cached_tokens` | `RequestOutput.num_cached_tokens` | 调度器首次调度时记录的命中数 = 本地前缀缓存命中 + connector（主机层）命中（`PrefillStats.set`） |
| `completion_tokens` | 累计输出 token 数 | MTP 一步出多个 token 时跳变，属正常 |
| `finish_reason` | `CompletionOutput.finish_reason` | 结束时 `{"type":"length","length":N}` / `{"type":"stop","matched":...}`；未结束为 null |
| `request_received_ts` | 插件安装的纯 ASGI 中间件在请求进入应用时打的 `time.time()`，早于读取请求体 | 与 SGLang 000 的 `_ArenaRecvTimeMiddleware` 同为读请求体之前；vLLM 的 CORS/鉴权等核心中间件在插件之后加入、包在外层（SGLang 的打点在 CORS 之外），差值是这些中间件自身的微小时间，未量化；TTFT 包含请求体接收、JSON 解析与分词 |
| `api_server_dispatch_finish_ts` | 分词完成、交给引擎客户端前的 `time.time()` | 诊断用 |
| `forward_entry_time` | 引擎 `scheduled_ts`（首次被调度进 batch 的那一轮调度开始时刻）换算到墙钟 | 与 SGLang 同名字段含义相同（首次入批，早于 GPU 执行）；诊断用 |
| `queue_time` | `scheduled_ts − queued_ts` | 进入调度器等待队列到首次入批的间隔；诊断用 |
| `prefill_finished_time` | 引擎 `first_token_ts`（首 token 所在输出批生成时刻）换算到墙钟 | 评分用的首 token 时刻 |

引擎核心的时间戳是 `time.monotonic()`（Linux `CLOCK_MONOTONIC`，同一主机所有进程共用）。换算方法：
ASGI 入口同时记 `time.time()` 和 `time.monotonic()`，墙钟 = 入口墙钟 +（引擎单调时间 − 入口单调时间）。
这要求 API 进程与引擎核心在同一主机（我们的单机 TP8 部署总是如此）。换算结果若不在 [收到请求, 当前时刻] 内，说明假设不成立，
该值不采用：首 token 时刻退回为 API 进程收到首个输出的时刻（只会更晚，不会把 TTFT 报小），`forward_entry_time` 不报。
没有开统计（`--disable-log-stats`）时同样退回。

## 验证

- CPU：`tests/entrypoints/generate_compat/test_generate_compat.py`（20 项），假引擎 + 真实路由/解析/SSE：逐事件累计计数与 MTP 跳变、
  首 token 时刻位于收到请求之后且全程不变、异时钟来源被拒绝、原文 prompt 不加模板、流中错误、非流式、flush 收据（成功/拒绝、时间窗、connector 与未完成数）、
  收到请求的时刻早于请求体到达（ASGI 层打点）、插件合同。
- 待 GPU：真实分词与冻结 token 数逐条相等、`ignore_eos` 精确、同一 prompt 二次命中 `cached_tokens` 上涨、flush 后归零、原 harness 跑通。
