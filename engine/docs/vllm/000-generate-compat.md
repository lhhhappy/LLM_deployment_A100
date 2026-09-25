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
| `/flush_cache` | `reset_prefix_cache(reset_running_requests=False, reset_connector=True)`：清设备前缀缓存，并清任何 connector 管理的主机/外部层。成功 200 `{"success": true}`；仍有请求或传输占用块时引擎拒绝，返回 400 `{"success": false}`，不做部分清理。不动代码缓存与 CUDA graph |

## `meta_info` 字段（每个事件都带）

| 字段 | 来源 | 说明 |
|---|---|---|
| `prompt_tokens` | `len(RequestOutput.prompt_token_ids)` | 引擎实际输入 token 数 |
| `cached_tokens` | `RequestOutput.num_cached_tokens` | 调度器首次调度时记录的命中数 = 本地前缀缓存命中 + connector（主机层）命中（`PrefillStats.set`） |
| `completion_tokens` | 累计输出 token 数 | MTP 一步出多个 token 时跳变，属正常 |
| `finish_reason` | `CompletionOutput.finish_reason` | 结束时 `{"type":"length","length":N}` / `{"type":"stop","matched":...}`；未结束为 null |
| `request_received_ts` | 路由函数入口的 `time.time()`，在读取请求体之前 | 服务端收到请求；比 vLLM 自己的 `arrival_time`（分词前）更早，TTFT 包含解析与分词 |
| `api_server_dispatch_finish_ts` | 分词完成、交给引擎客户端前的 `time.time()` | 诊断用 |
| `forward_entry_time` | 引擎 `scheduled_ts`（首次被调度进 batch 的那一轮调度开始时刻）换算到墙钟 | 与 SGLang 同名字段含义相同（首次入批，早于 GPU 执行）；诊断用 |
| `queue_time` | `scheduled_ts − queued_ts` | 进入调度器等待队列到首次入批的间隔；诊断用 |
| `prefill_finished_time` | 引擎 `first_token_ts`（首 token 所在输出批生成时刻）换算到墙钟 | 评分用的首 token 时刻 |

引擎核心的时间戳是 `time.monotonic()`（Linux `CLOCK_MONOTONIC`，同一主机所有进程共用）。换算方法：
路由入口同时记 `time.time()` 和 `time.monotonic()`，墙钟 = 入口墙钟 +（引擎单调时间 − 入口单调时间）。
这要求 API 进程与引擎核心在同一主机（我们的单机 TP8 部署总是如此）。换算结果若不在 [收到请求, 当前时刻] 内，说明假设不成立，
该值不采用：首 token 时刻退回为 API 进程收到首个输出的时刻（只会更晚，不会把 TTFT 报小），`forward_entry_time` 不报。
没有开统计（`--disable-log-stats`）时同样退回。

## 验证

- CPU：`tests/entrypoints/generate_compat/test_generate_compat.py`，假引擎 + 真实路由/解析/SSE：逐事件累计计数与 MTP 跳变、
  首 token 时刻位于收到请求之后且全程不变、异时钟来源被拒绝、原文 prompt 不加模板、流中错误、非流式、flush 成功与拒绝、插件合同。
- 待 GPU：真实分词与冻结 token 数逐条相等、`ignore_eos` 精确、同一 prompt 二次命中 `cached_tokens` 上涨、flush 后归零、原 harness 跑通。
