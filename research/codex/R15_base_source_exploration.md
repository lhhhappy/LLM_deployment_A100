# R15 — 实际底包：快照、请求语义与 DP 路由的只读探索

2026-09-22，Codex main，T36。用户要求先理解，不实现。**未编写/应用 102–104，没有安装、模型导入、实验、GPU、服务或构建操作。** 下文 VERIFIED 是源码/静态数据证据，INFERRED 是设计推论，不是速度或正确性实测。

## 0. 先给结论

1. **VERIFIED**：底包已有“单 forward 取内部状态”的完整上层链路；102 有较小的单点选址入口，但“替换一个点”不等于 101 的“角色点 + 后续末尾点”。不能默认 strict_append 不退步。
2. **VERIFIED**：实际底包读取普通中间张量 `h`；之前 R7/R13 的参考树则另有 FP32 `h_track_buf`。**底层 `sglang.kernels` 未包含在本次 407 文件中，实际 h 的精度尚不能确认。** “一行改选点即可无损替代”证据不足。
3. **VERIFIED**：S1 routing key 是 sys/tools 前缀族，不是 session。公开 722 请求中有 136 个 session、156 个族，18 个 session 跨族；最大的族涉及 19 个 session。103/104 的第一道问题是语义，不是 hash 函数。
4. **VERIFIED**：`routing-key` 排队策略、`prefix_affinity` DP 分发、cache namespace 是三个不同机制。只把头写入 `routing_key`，不会自动拥有 DP 亲和或缓存隔离。
5. **INFERRED**：最值得借鉴的是 vLLM 的“位置合法性、状态导出、槽位归属共同契约”，以及 SGLang PR 的“负载观测 + 亲和性折中”；不是照搬几处函数。

## 1. 证据来源与版本边界

**VERIFIED**：读取 `build/diag3/log4.txt` 的 JSON `data.log`，在内存中重组 DIAGTGZ_BEGIN/END 之间的 base64/xz 归档，SHA256 为 `025ee541e1747f620ad4ae0964f17a470697b7c2f312de175df2fbeb6c236984`，1343324 bytes。逐文件与 `build/base_src_full/` 比较：407 文件，0 缺失、0 内容不符。没有解压覆盖文件。

日志固定 FROM digest 为 `f24781f02a81d96f51e27741fc67d7056302c23a37d6e87b9d7c1ebe7e8580ab`；这是来源收据，不等于该完整配置已经在 A100/DP 下验收。诊断构建最后的 `exit 1` 是提取脚本主动退出，不能当模型启动失败。

本地 refs HEAD 分别为 vLLM `7565389f848994d5271986f74aab2af7ede0cbab`、SGLang `ec6e4e8b5fe4aad4b1549c68e01db1d99e185598`。本轮分析固定快照，不重新宣称在线 PR 状态。[收据与静态统计](../../evidence/T36/source_receipt.json)。`refs/pr*.diff` 是精选核心改动，不含完整测试/所有依赖；不是可直接 patch 的标准完整补丁。

`base_src_full` 是这些 srt 子树的完整提取，**不是整个安装包**：`sglang.kernels`、部分 configs、model_executor 等依赖没有一并取回；`base_src_partial/srt/arg_groups` 是另一次提取。不能用 `src/sglang` 中的同名依赖默默补齐，随后宣称就是实际底包。

用户引用的“底包 F49”与既有 T34/F49 重号，已只把新条目更名 F51、保留原标记和正文。本报告收窄其“即可替代 101”的推论，不否认已有内部取状态能力。

## 2. 102：实际链路与最重要的三个契约

### 2.1 链路已经存在，不需要为单点选择再切 forward

**VERIFIED**：

```text
schedule_batch：选择逻辑深度 b + track slot
  → hybrid_linear_attn_backend：生成 conv 窗口、h/final-state 索引
  → kda_backend：保存 raw-qkv conv 历史；完整 prefill；复制 SSM 到 track slot
  → batch_result_processor：prefill 后调用 cache_unfinished_req
  → MambaRadixCache / Unified MambaComponent：把所选 slot 捐给树
```

源码锚点（以下路径均相对仓库根）：

- `build/base_src_full/srt/managers/schedule_batch.py:2631` 每个 extend 收集 track entry；`:2766–2860` 选末尾或 branch，只有一个 track_index/track_seqlen。
- `.../layers/attention/hybrid_linear_attn_backend.py:333–402` 分别生成 conv 与 SSM 索引，`:857–878` 复制所选状态。
- `.../layers/attention/linear/kda_backend.py:744–810` 从**卷积前** mixed_qkv 取历史，调用完整 extend 后跟踪状态。
- `.../managers/scheduler_components/batch_result_processor.py:340–346` 正常未完成请求在 prefill 后入树。
- `.../mem_cache/mamba_radix_cache.py:686–735` 或 `.../unified_cache/components/mamba_component.py:529–606`：缓存长度来自 `req.kv.mamba_last_track_seqlen`，捐赠对应 slot。

**INFERRED**：102 放在这个入口，可以不再人为制造第二个 partial prefill，因此绕开 101 为拆分新增的准入/续跑分支。但天然 budget chunking、retraction、overlap 等并没有消失，仍需单独覆盖。

### 2.2 “序列深度 b”与“传给 backend 的 b+1”不能混淆

**VERIFIED**：`_force_track_h(b)` 返回 `b+1`，是让 backend 走 h 分支的编码，不是多缓存一个 token。实际 `mamba_last_track_seqlen` 仍为 b。

令 p 为已命中前缀长度，e 为本次 extend 结束深度，b 为要缓存的边界：

- b 对树是真正可命名的深度；`mamba_checkpoint_grid(tree.page_size)` 为 cache chunk 与真实 tree page 的 lcm（`runtime_context.py:1920–1925`），不能直接写死 64。
- h 的索引按 **b−p 的相对位置**；`_force_track_h` 断言其 cache-chunk 对齐。多请求打包后，还要加前面请求的 h 行数偏移（backend `:371–395`）。
- 内部点应满足 p < b < e；b=e 应走真正的 final-state 路径。否则把末尾强制编码为 b+1，可能尝试读取并不存在的下一个 h 行。
- conv 与 recurrent 必须指向同一个 b。conv 用原始投影输入 `[b−conv_len,b)`，不是卷积后的 q/k/v，也不是 end 时的 live conv。

**INFERRED**：应把“选点”与“合法性/索引编码”分开；skip reason 要分无角色、点在 prefix 中、点不在本次 extend、grid 不合法、branch 冲突等。还要先定义扫描的是整个请求的最后角色，还是每个 chunk 内最后角色；这会改变生成多少快照，不能混用同一预测。

### 2.3 单点替换会改变缓存语义，ping-pong 两格不是自动双断点

**VERIFIED**：本次一个 track entry 只能写入一个目标。`memory_pool.py:1461–1463` 只保留 `mamba_last_track_idx`；`:1508–1529` 将该 slot 捐给树并替换，不会把两格都插入。unfinished 入树后清掉 last_track_seqlen（`mamba_radix_cache.py:802`、MambaComponent `:650`）。live recurrent state 会继续 decode，不是树里的 end 快照。

**INFERRED**：

- **102-A 单点替换**：没有有效 branch 时把默认 end 换成 role，改动面较小；但牺牲了本次 end 的缓存机会。短 decode 未达到下一次 track grid 就结束时，尤其不能指望 decode 自动补回。
- **102-B 双点保留**：role + 原有 end/branch 都保留，才接近 101 的缓存语义；需第二组 conv/SSM 导出、持久 slot、两次入树及 cleanup/eviction 预算，不是只改 schedule_batch。
- **branch-first** 能保护已有 branch，却不能证明 strict_append 不退步，因为 end 仍可能被替换。F24/101/E2b 的单调改善证据不能直接转给 102-A。
- **lazy** 路径没有显而易见的单点结构冲突：prefill 跳过 swap、keep_idx 取显式 last_track_idx、捐赠后同步槽位映射已存在。但 lazy 初始只分配一格（另一格 −1），更不能把它当“双点免费槽”。尚未证明实测兼容。

## 3. 102 的精度与 kernel 成本：这次底包带来的新信息

**VERIFIED**：实际 `kda_backend.py:777–810` 传 `return_intermediate_states` 与 `track_ssm_h_src`，最终由 `hybrid_linear_attn_backend.py:872–874` 执行 `h[...] → ssm_states.dtype`。它没有参考树中的 `h_track_buf/track_chunk_idx` 导出路径。

对照只读 `src/sglang`：`kda_backend.py:864–891` 明确另分配 FP32 track buffer，并检查实际 kernel 支持；`kernels/ops/attention/fla/chunk_delta_h.py:396` 的普通 h 则用 `k.new_empty`。**这只证明两份源码路径不同，不证明未取回的底包 kernel 也用 BF16 h。** 若底包 h 已经降精度，事后 `.to(float32)` 不能恢复丢失精度；实际 dtype 必须补证据。

还存在选点改变 kernel 路径的问题：

| 实际底包 wrapper | 内部快照约束（VERIFIED） |
|---|---|
| Triton，`kda_triton.py:218–250` | 调用包外 `sglang.kernels...chunk_kda` 并请求中间 h；精确 dtype/存储实现缺失 |
| FlashKDA，`kda_flashkda.py:123–143` | tracked prefill 转 Triton；配置名不等于实际执行的是 FlashKDA |
| NVIDIA KDA，`kda_nvidia.py:255–265` | 内部快照使 fused fast path 不再 eligible |
| CuteDSL，`kda_cutedsl.py:117–126` | 无中间状态接口，直接拒绝 extra_buffer tracking |
| PTX，`kda_ptx.py:186–199` | 对内部 h 的 stride/chunk grid 有额外约束 |

**INFERRED**：少一次 forward 不必然更快——可能换成更慢的 kernel，或增大状态写回。以后应观测 actual backend/fallback、h/track bytes、kernel 次数，而不只记总 TTFT。A100 候选应先锁住已支持的路径，不把 Hopper/Blackwell 示例当作现成后端。

## 4. 从 vLLM #56960 学的是接口契约，不是抄一种选点

**VERIFIED / 本地 refs**：

- `vllm/v1/kv_cache_interface.py:1094–1127` 统一 checkpoint 位置与合法性：需严格在 query 内、满足 hash/后端 alignment、checkpoint block 不覆盖 initial-state block。
- `vllm/model_executor/layers/mamba/checkpoint.py:21–58` 统一算 offset 与 block-table column，`:79–123` 仅 align 模式/声明支持的 spec 才生成 metadata；投机/非投机 request 行序明确映射。
- `.../kda_checkpoint.py:16–17` FlashKDA alignment 为 16，**不是本底包固定可用的 grid**；`:26–130` 统一存 recurrent checkpoint 与 raw-qkv conv 历史。
- `vllm/models/glm5next/common/kda.py` 保留完整 forward 的最终状态，另有 checkpoint workspace/export；不能据此认定两个状态都已经按我们的 end/role 策略入树。
- `tests/models/glm5next/test_kda_recurrent.py:198–362` 检查多个 offset、conv 布局、混入投机请求、raw_qkv 不被改写、无效槽不写、checkpoint 恢复 suffix/final-state；数值容差为非零，不是逐 bit 一致。

**INFERRED**：最有价值的是“分配了一个 checkpoint，就必须证明 backend 在同一位置写了两类状态”。可以借其 state-level 测试设计，补足只看 cached_tokens/logits 的盲区；不能直接搬它的容差解除 D1-04，也不能把 vLLM slot 管理按字段名硬移到 SGLang。

## 5. 103：先区分四种请求元数据

**VERIFIED**：`s1-dev/harness/s1_loadgen.py:110–120,347–350` 与 `task.md:169–191` 的契约一致：

| 字段 | 原义 | 不应混同 |
|---|---|---|
| X-S1-Request-ID / rid | 本次请求标识 | 不是跨轮会话键 |
| X-S1-Session-ID | 会话标识 | 路由亲和 ≠ 自动开启引擎 session 缓存语义 |
| X-S1-Routing-Key | `sys_tools_hash`，公共前缀族 | 不是 session ID；同族有多个会话，同会话也可能跨族 |
| X-S1-Cache-Namespace | 各 N 档/运行的缓存隔离标识 | 只放到路由 hash 中，不会隔离 radix cache |

静态读取完整公开 requests.jsonl：722 请求 / 136 session / 156 family；最大两族分别 79 请求跨19 session、64请求跨18 session；18 个 session 有多个 sys_tools_hash。**不是并发占用分布，也不据此推断隐藏负载。** 摘要及输入 hash 见收据，不输出会话原值。

底包 `/generate`（`:908–911`）只在环境开关启用时调用通用 header overrides；`request_headers.py:10–19` 没有任何 X-S1 映射，也没有 routing_key/cache_salt 映射。因此不能“只开 overrides”就接通比赛信息。但 **JSON body 的 routing_key 已有字段与透传能力**，不能泛称 `/generate` 完全不支持 routing key。

已追通单请求路径：GenerateReqInput `io_struct.py:294` → tokenizer `:1411–1413` → TokenizedGenerateReqInput → scheduler `:2673–2675`。其中 routing_key、extra_key、cache_salt 分开保留。`GenerateReqInput.__getitem__` 的 batch 子请求构造没有复制 routing_key（`:925–967`）；赛题单请求不受此路径影响，但泛化为批量接口必须另审。

**INFERRED / 设计建议，未实现**：

- 将“族 key”“会话 affinity key”“隔离 salt”视为独立概念。若复用同一个 routing_key 做 DP 与 batch 排序，要明确接受哪种取舍；更清晰的独立字段则意味着显式修改序列化/透传链。
- namespace 候选映射到真正 radix key 的 `cache_salt`（已有与 extra_key 分离的机制），需定义与 body salt/LoRA 的组合和优先级。不是在 hash(session) 前拼 namespace 就完成缓存隔离。
- 不要为了获得路由键，顺手打开 session radix cache 或写入 session_params。底包 scheduler `:2617–2626,2683–2686` 会按这些配置改变会话生命周期，和单纯 affinity 不同。
- 只解析白名单元数据；未知头仍可忽略。不要为三项提示开放覆盖 rid、priority、bootstrap_host 等一整套通用 overrides；不要碰 text、采样参数、时间戳口径。

## 6. 104：两个同名概念容易看错，PR 的依赖也不止 controller

### 6.1 原生 routing-key 只排序，不负责“去哪个 DP rank”

**VERIFIED**：`schedule_policy.py:439–465` 统计 running batch 中每个 routing_key 的频次，给 waiting_queue 排序。它是已经进入某个 scheduler 后的策略，不是 DP rank chooser；也不是按 LCP 匹配树，更不是 SLO 感知调度。103 即使接通 family key，也只会让这一排序拥有有效输入（且需实际启用该 policy）。

controller `:85–99,162–174` 只有 RR/bootstrap/total_requests/total_tokens。`:744–757` 的 routed_dp_rank 是**指定目标的直接发送**，优先于其他策略；它不执行亲和负载折中，也不增加正常 DPBudget dispatch 的预测计数。以它作临时静态绑定可以研究语义，但不能冒称等价于 #31170 的负载保护。

### 6.2 PR 真正可学的实现

**VERIFIED**：本地 PR controller `:829–956`：显式 key 优先、缺 key 可 hash token 前缀；稳定 rank ID 的 rendezvous 排序；遍历存活 rank，以 running+waiting 计数做负载上限，再预测性 +1 请求/+tokens。它是有回退的软亲和，不保证每轮绝对回同一卡。

精选 diff 涉及 **5 个生产文件**：arg_groups/field_order.py、fields/parallel.py、serving_hook.py、managers/data_parallel_controller.py、managers/load_snapshot.py。重要的不只是 enum/新函数，还包括参数注册、验证、每次 dispatch 前刷新负载、ZMQ reader ownership。漏掉 load_snapshot 接线，可能让过载保护基于错误/陈旧观测。

实际底包 `controller.py:306–321` 已有 20ms refresh 节流，`DPBudget.update_budget :109–118` 用 timestamp 跳过旧快照，避免突发请求的 +1 计数反复被覆盖。移植应保留这些已有防护，不用旧版 controller 整文件覆盖。PR 测试中包含稳定 rank ID、掉卡、无 key、过载回退、负载快照/突发计数等可借鉴项目。

### 6.3 仍需自己决定的策略，而不是 PR 帮我们决定

**INFERRED**：

- 会话键优先有利于深历史；族键优先有利于共享 system/tools。公开数据说明两种目标会冲突，不能默认拿 S1 routing key 就等于 session affinity。
- 其负载度量不是“剩余 GPU 工作量”：一个冷长 prefill 和一个缓存命中的短请求都算1。只有当避开的排队成本大于换卡带来的重算成本时，迁移才划算。
- 因实时负载变化，固定 key 也可能在不同轮次被导向不同备选 rank；PR 没有记录“上次溢出后新 home”的粘滞表。是否需要更强 stickiness/滞回是待测假设，不是当前方案定论。
- 小 N 时保护阈值的整数效应明显；应该对照纯 RR、稳定 session hash、带 guard affinity 三者，分开量化缓存/排队，而不把多个变动绑在一次 A/B 里。

## 7. 102 与 DP 会相互影响，不能按两个独立倍数相乘

**VERIFIED**：底包 `glm5_next.py:345–365` 正常路径按 `attn_tp_size` 切 KDA heads；controller `:680–685` 有 `attn_tp_size = tp_size / dp_size / attn_cp_size`。因此固定 TP8/CP1 时，DP 增大使每个 shard 上的单请求 KDA state 更宽。

**INFERRED**：单 checkpoint 的每卡成本会变化；请求也分散到更多 shard，总额取决于活跃/缓存槽分配与预留，不能简单说总 KDA 内存一定乘 DP。102 双点增加 slot，104 改变这些 slot 的驻留位置；DP 分散 MLA 的收益必须与 KDA/权重/临时内存共同记账。源码存在这些路径只是可行性线索，不是完整 GLM+DSA+A100+DP 已跑通的证据。

## 8. 可以先确认什么，哪些必须留给验证

本轮已用静态证据确认：源归档一致、调用链、单点槽位契约、header 元数据真实含义、公共数据中的族/会话关系、PR 依赖面和 backend fallback 条件。

| 后续问题（不是本轮执行任务） | 最小有用证据 |
|---|---|
| 102 能否保持数值契约 | 取得**同 digest** 的 kernels/config/model_executor 相关文件；state/conv 恢复对照、slot sentinel，不能只看命中计数 |
| 102 单点能否替代 101 | 同负载对照 role-only vs role+end；尤其 strict_append、短输出、已存在 branch、长冷头 |
| 102 对齐/生命周期是否安全 | p/b/e 与 page/grid 的边界；multi-request offset；lazy、abort/retract、flush、slot 重用 |
| 103 是否正确透传 | 单请求到 TokenizedGenerateReqInput 的元数据契约；缺头/未知头/body冲突；namespace 真隔离 |
| 104 亲和性是否有收益 | DP rank、实际 cached_tokens、排队、迁移次数、冷重算成本；负载快照突发/过期测试 |
| 完整模型是否更好 | A100 实际 kernel、完整输出/能力、TTFT与TPOT各门；不能用短输出替身或外部PR跑分代替 |

**结论**：103 是元数据接线问题，104 是亲和性与负载的决策问题，102 是状态位置、精度与持久化契约问题。102 的入口虽小，语义和验证面最大。可以先把三者各自的契约写清，再决定实现；本报告不改变现有计划、不批准部署，也不解除既有数值门。
