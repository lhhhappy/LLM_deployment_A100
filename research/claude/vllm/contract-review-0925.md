# vLLM 路线评测契约复核（2026-09-25，Claude）

对象：`engine/vllm` @ `fb18e488`（底包 a811738a6 + 000 + 010；101 另列）。Codex 的 SGLang 侧复核见
[contract-implementation-review-0925](../../../notes/reports/contract-implementation-review-0925.md)，vLLM 源码与主机层研究见 [R29](../../codex/R29_vllm_contract_and_host.md)。
每条注明依据层级：**源码**（逐行读过）、**CPU**（单元测试/离线复算）、**两卡**（开发机 TP2、真实配置截 8 层 + MTP、dummy 权重）、**TP8**（未做）。

## 逐项结论

| 契约 | vLLM 实现 | 依据 | 未验证边界 |
|---|---|---|---|
| prompt 原文、不二次套模板 | `/generate` 把 `text` 作为补全 prompt 交给引擎渲染器（与 `/v1/completions` 同路径），不走 chat template。GLM tokenizer 的 post-processor 只有 ByteLevel，`add_special_tokens` 不加任何 token | 源码；CPU：长链集 5601 条经 harness 渲染 + vLLM 渲染器分词，与冻结 `glm_tokens` 逐条相等（0 条不一致）；两卡：探针请求 `prompt_tokens == glm_tokens` | TP8 真实权重下同样由渲染器决定，不依赖权重 |
| 逐请求 `ignore_eos` 与输出数 | `max_new_tokens→max_tokens`，`ignore_eos` 原样传入。`ignore_eos` 时不设主 EOS，generation_config 的额外 EOS（含 `<|user|>`/`<|observation|>`）也不加入停止词（`sampling_params.py` `update_from_generation_config`）；MTP 一步多 token 逐个追加并在达到 `max_tokens` 时截断（`scheduler.py` `_update_request_with_output` + `check_stop`） | 源码；两卡：全部探针请求末事件 `completion_tokens == max_new_tokens` | 两卡 dummy 权重很少采到 EOS，不能单凭它证明；源码路径已排除 EOS 停止 |
| 首 token 时刻（`prefill_finished_time`） | 引擎核心在**首个带新 token 的输出批**生成时打的 `EngineCoreOutputs.timestamp`（`time.monotonic()`，GPU 结果取回后），经 ASGI 入口的单调/墙钟对换算为墙钟。调度器只为有新 token 或结束的请求发输出（`should_emit_output`，"no partial prefill outputs"），所以不会记成首次调度或中间分块的时刻。换算值不在 [收到, 当前] 内则退回 API 进程收到首个输出的时刻（只会更晚） | 源码；CPU：异时钟被拒、首 token 时刻晚于收到且全程不变 | 同主机 CLOCK_MONOTONIC 是前提（单机 TP8 成立）；未在 TP8 上与客户端首事件时间做分布对照 |
| 收到请求时刻（`request_received_ts`） | 插件安装的纯 ASGI 中间件在请求进入应用、读请求体之前打 `time.time()`（`fb18e488` 起），与 SGLang 000 的 `_ArenaRecvTimeMiddleware` 同为"读请求体之前"。层次不完全相同（Codex 勘误）：SGLang 先加 CORS、后加打点中间件，打点在 CORS 之外；vLLM 先挂插件、后加 CORS/鉴权等核心中间件，打点在它们之内 | 源码；CPU：请求体延迟 0.3 s 到达时，收到时刻早于分发完成 ≥0.3 s | vLLM 多算的只是外层中间件自身的处理时间，未量化；Codex 与我都不主张为此改引擎 |
| 客户端 TPOT 与累计 token | 每个产生新 token 的引擎输出发一个 SSE 事件，`completion_tokens` 为累计值（MTP 跳变），`text` 为增量；harness 用客户端首/末事件时间和末事件计数算 TPOT | 源码；CPU：逐事件累计单调、结束值正确；两卡：探针全部通过 | TP8 负载下事件发送延迟未测 |
| `cached_tokens` 语义 | `PrefillStats.num_cached_tokens` = 首次调度时本地前缀命中 + connector（主机层）命中，上限为 prompt−1；被抢占后重新调度不更新该值 | 源码；两卡：同一 prompt 二次命中上涨、flush 后为 0 | 抢占重算不体现在该值里（两引擎都如此，需在 TP8 日志里看抢占次数） |
| `/flush_cache` | 引擎块池除 null 块外只要有占用（运行/等待请求、在途传输）即拒绝；否则删除全部前缀哈希。前缀索引只在引擎核心的调度器里，TP worker 不持有索引。响应体 = 收据，日志同一 dict（字段见 [000](../../../engine/docs/vllm/000-generate-compat.md)），`level_verdict` 已按收据校验（Codex `04b6d100`） | 源码；CPU：收据字段、成功/拒绝 | **只有无 connector 路径闭合**。OffloadingConnector：二级层（FS/网络）不清、reset 后已排队任务在后续步骤才消化、重复 reset 在队列未消费前会断言（Codex 源码复核）——启用主机层前须另行设计与验证 |
| 优化默认关闭 | 000 只在 `VLLM_PLUGINS` 点名时加载；010 按硬件（`support_deep_gemm()`/`supports_fp8()`/capability 8）生效，H100+ 路径不变；101 在 `VLLM_AX_MAMBA_ROLE_CHECKPOINT_TOKEN_IDS` 为空时关闭（行为等于底包） | 源码；CPU：上游调度器/前缀缓存测试在底包树与本版树（101 默认关）上逐项对照，新增失败 0（[收据](../../../evidence/vllm-a0-20260925/upstream-regress-055122/summary.txt)） | — |

## 101（角色边界 KDA 检查点）状态

- 独立候选，默认关闭，不属于冻结基线。机制与测试见 [101 说明](../../../engine/docs/vllm/101-role-boundary-checkpoint.md)。
- CPU：17 项专门测试通过（真实调度器分块函数 + 真实 KVCacheManager）。开发与复核中修正了三处：
  1. 只登记 Mamba 状态不够，全注意力类各组还要在 `⌊r/u⌋·u` 处登记附加键，追问请求才能匹配到该检查点（我的测试发现）。
  2. 附加键只在提示词算完那一步登记时，解码把该块填满、升级为满块会删掉块上全部键，追问命中退回（Codex 独立 CPU 复核发现，
     证据 `evidence/vllm-review-fb18e488/`）。改为与上游末尾键相同的生命周期：算完后每步登记一次，已存在时不做事。
     Codex 的 6 个复现用例在修复后的源码上重跑，预填后、解码跨块后、拥有者释放后三次命中都等于检查点位置（u32/u64，提示 2301–2560）。
  3. 前缀缓存关闭而开关设了时，旧代码静默当作关闭、机制行却报 on（Codex 指出）；现在启动即拒绝。
- 未验证：GPU 上从该检查点恢复的 KDA 状态数值正确性（块中间分块结束处的状态经写时复制保存）；TTFT 代价（每请求多 1–2 步前向）；八卡显存。

## 主机层（OffloadingConnector）边界（供 HiCache 经验迁移）

- 容量口径：vLLM `--kv-offloading-size` 是整个 TP 组的 GiB 总预算；SGLang `--hicache-size` 按每个 rank。两边填同一个数不是同容量（Codex 指出）。
- 命中粒度：OffloadingConnector 在有 EAGLE 组（MTP）时禁用部分尾块（`supports_partial_tail`），主机命中只按块边界。
- 完整检查点合同：一个可复用位置要求 MLA KV、索引器 K（kpool 及其尾缓冲）、KDA 递归与卷积状态、MTP 草稿层 KV 同时有效。vLLM 按组分别管理，前缀命中取各组最小值；
  kpool 尾缓冲组不可前缀缓存，命中点需落在 `index_kpool` 的整数倍上（`u` 必须是其倍数，启动校验）。
- GLM 的 KDA 没有接入 prompt 内检查点导出（FlashKDA 导出器只接在 Kimi-K3；且 FlashKDA 仅 SM90+），A100 上 KDA 状态只在分块末尾写出（Codex 指出）。
- flush：见上表，启用主机层前须补"清所有层 + 等在途任务排空"的实现与证据。

## 两路共同评测合同（与 Codex 对齐；2026-09-25 用户澄清）

目的是在**同一评测**下比较哪条路更好，不是证明两边实现等价。

- 固定：赛规（task.md）、原 harness（`run_dev.py`/`s1_loadgen.py`）与 `score_formal.py`、模型、同一冻结数据（manifest 19a7e5a6…）与逐请求输出预算、原顺序与 gap、
  冷热流程（rep16 预热 + 真 flush）、同档 N、同样的硬件与资源边界；两边各自过质量门。
- 允许不同：生成文本、缓存命中、调度顺序、内部参数，以及闭环下的实际到达时刻。评测工具没有生成文本一致门。
- 口径：四桶用冻结标签；TPOT 取客户端逐请求值；`cached_tokens` 如实；TTFT 用服务端"收到请求 → 引擎首 token 输出批"；判分含 Codex 新增的逐请求 token 契约检查。
- 开发取向：SGLang 上已证明有效的设计能用就复用，不在 vLLM 上重复探索；vLLM 原生功能更好就用原生（例如 KDA 检查点保留用原生开关 `--prefix-cache-retention-interval`/`--prefix-match-unit`，
  101 只补原生开关覆盖不到的"最后一个角色边界"）。不把实现等价或补齐 SGLang 全部功能作为评测的前置条件。

## Codex 评测工具补丁复核（`19673ef4`、`04b6d100`）

- `validate_replay_tokens`：输出预算缺省值与原 loadgen `int(max_output_i or 512)` 一致；成功行要求输出数 = 预算、prompt 数 = 冻结值、`0 ≤ cached ≤ prompt`；错误行不查、仍走原错误率门；原 harness 写入的都是 int。
  唯一可能误拒的情形是 prompt+输出超过 `max_model_len` 被截断——我们的配置为 524288，数据最大 267,773，不会发生。**未发现误拒或误放。**
- `check_vllm_flush`：与我的实现字段一致；响应体与日志行用同一 dict、`sort_keys` 序列化，JSON 往返浮点精确，逐字符比较成立；时间窗比较要求 runner 与服务同主机同时钟（Pod 上成立）。**未发现问题。**
