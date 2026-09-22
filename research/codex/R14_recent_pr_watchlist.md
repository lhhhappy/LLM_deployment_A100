# R14 — SGLang / vLLM 近期 PR 与 issue 定向复核

2026-09-22，Codex main，T35。**仅调研：没有应用补丁、启动服务或实验。**

## 1. 范围与证据口径

- 以前已做：R6/R7 的缓存先例及单 forward 快照；W1 的 R8 是 66 行 / 130 PR 清单；R13 精读 AgentX / MLPerf。清单覆盖不代表主会话逐一精审所有 PR。
- 本次：按 GLM-5.3、KDA、cache、DP 定向搜索近期更新，精读下表 10 个条目的正文与 REST 状态；另完整读取 #56960 六文件及 #40517 两文件 diff。不是全仓穷尽扫描。
- **VERIFIED** 只表示公开正文/API/源码已核对；PR 作者跑分仍是作者报告，未在本赛复现。可迁移性与优先级标 **INFERRED**。
- 状态快照与 head SHA：[evidence/T35/github_status.json](../../evidence/T35/github_status.json)。`closed` 不等于 merged；open PR 的 `merge_commit_sha` 可能只是测试合并，不用它判断已合入。
- 本地只读基线 `src/sglang` = `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。上游 main 有特性，不代表本地或主办方镜像已具备。

## 2. 候选总表

| 上游条目 | 截至 09-22 状态 | 实际作用 / 适用边界 |
|---|---|---|
| [vLLM #56960](https://github.com/vllm-project/vllm/pull/56960) | open，09-15 提出 | **GLM 本身的 KDA prefill 内部快照**；Plan B 的直接实现参考，不是 A100 即插即用 |
| [SGLang #31170](https://github.com/sgl-project/sglang/pull/31170) | open，07-14 提出、09-22 更新 | 单实例 DP attention 内部的会话亲和路由；本地无 `prefix_affinity` 实现 |
| [SGLang #40517](https://github.com/sgl-project/sglang/pull/40517) | **09-21 merged** | 为 GLM 接通 ReplaySSM 的池分配/容量核算；主要涉及投机验证，不是跨轮角色快照 |
| [SGLang #40680](https://github.com/sgl-project/sglang/pull/40680) | open，09-22 提出 | HiCache write_back 淘汰内部节点时保全 Mamba 状态 |
| [SGLang #39156](https://github.com/sgl-project/sglang/pull/39156) | **仍 open**，09-12 提出 | hybrid HiCache 恢复 DSA indexer 数据；旧正确性前置项未因其他 PR 自动解除 |
| [SGLang #40685](https://github.com/sgl-project/sglang/pull/40685) | **09-22 merged** | GLM PTX prefill 漏 beta sigmoid 的正确性修复；作者测试是 GB300 |
| [vLLM issue #56868](https://github.com/vllm-project/vllm/issues/56868) | open，09-14 提出 | 长时间累计 decode 后退化报告；根因未定，非我们的 D1 已被证伪或证实 |
| [vLLM #57261](https://github.com/vllm-project/vllm/pull/57261) | open，09-17 提出 | “容量 +14.7%”是**报告口径**修正，分配器/池大小未变；另有 sm80 私有后端线索 |
| [SGLang #40310](https://github.com/sgl-project/sglang/pull/40310) | **09-21 merged** | GLM **权重** CPU offload 与 PD index mapping；不是 HiCache DSA 恢复补丁 |
| [SGLang #34299](https://github.com/sgl-project/sglang/pull/34299) | open，08-10 提出、09-22 更新 | Kimi K3 / Cake native checkpoint、packed decode；41 文件，GB300 数据，不宜当小补丁搬运 |

## 3. 最贴近当前设计的代码机会

### A. #56960：在一次完整 prefill 内导出中间快照

**VERIFIED / diff**：GLM KDA cache spec 声明 checkpoint 支持，metadata 携带位置和槽位；FlashKDA 同时输出最终状态与内部 checkpoint，`FlashKDAPrefillCheckpointExporter` 保存匹配的 convolution 历史。测试覆盖两个 conv 布局、多个 offset、含/不含投机请求的行序，以及从 checkpoint 恢复 suffix；作者采用非零数值容差，不能当 bitwise 相等证明。

动机是原先为保留 1152-token 快照而把 1159-token prefill 分成 1152+7；新路径无需额外 forward。作者报告相同 cached_tokens、少一个调度步骤；没有足够受控端到端数字可写成固定百分比提升。测试配置引用 `FLASHINFER_MLA_SPARSE_SM90`，不能证明 A100 可用。

**INFERRED / 本赛映射**：这是 Plan B 很直接的参考，但上游按对齐位置存快照，不等于我们的角色选点与 role+end 双持久快照。本地 SGLang 已有单点 `h_track_buf/track_chunk_idx`（`kda_backend.py:865–930`），可借鉴 metadata、conv 与恢复测试，不必先换引擎。双点的持久槽、入树与生命周期问题仍见 R7；既有 D1-04 失败不因上游合入或 PR 测试解除。

### B. #31170：DP 容量增长之后，还需要请求回到原分片

**VERIFIED / PR 正文**：新增 native `DataParallelController` 的 `prefix_affinity`。优先使用已有 `routing_key`，缺失时可 hash 前 4096 tokens，再以负载策略兜底；采用 rendezvous hashing 与过载保护。这不是仅在外层网关选服务器，外层网关不能直接选择单实例内部 DP rank。

作者的 8×H100、Qwen3.5-122B、并发 64、共享前缀测试：req/s 5.47→7.88，mean TTFT 3511→789 ms。**这些是该合成负载的作者数据，既不是 GLM/A100 结果，也不是本赛 p95。** 同文的 TP→DPA 容量对比改变了并行方式，不能归因于亲和路由本身。

**INFERRED / 本赛映射**：DP 实验应把“容量”和“亲和性”分开验收。该 PR 借 OpenAI endpoint 的 `x-smg-routing-key` 透传；我们的 `/generate` harness 不因此自动有稳定会话键，不能假设逐请求 rid 就是会话 ID，更不能修改只读 harness。若依赖 token fallback，共享 system/tools 开头可能把不同会话集中到同一 rank。需要单独设计服务端可合法取得的会话标识/内容路由与过载退让。本地 `srt` 搜索未找到 `prefix_affinity`。

### C. #40517：ReplaySSM 的 GLM 接线，服务于投机验证

**VERIFIED / 两文件 diff**：增加 `hybrid_kda_config = kimi_linear_config or glm5_next_config`，替换 request pool 和预算核算中的 Kimi 专属判断；不是重写整个算子。基线 `kv_cache_configurator.py:2406–2420` 仍是旧 Kimi 判断，不能仅因存在 flag 就认定 GLM 完整接通。

`--enable-linear-replayssm-spec` 用原始输入窗口加接受后重放，代替每个 draft step 的全状态快照。它优化的是投机 verify 的状态保存，不是用户会话重放。基线 `fields/exec_.py:441–450` 和 `attention_hook.py:364–435` 已约束线性 draft chain、后端、PD 模式、dtype 等。

另一个 `--enable-linear-replayssm` 是普通 decode 缓冲路径，**不要混用**：本地说明明确警告 KDA 的 ring 较大、通常可能比 packed decode 慢，且该模式要求 no_buffer，和我们的 extra_buffer 不兼容（`attention_hook.py:331–337`）。不能把这个限制不加区分套到 spec 模式。

**INFERRED**：若 profiling 显示 MTP/KDA verify 的临时状态写回占显著成本，可列后续独立候选。现有资料不证明本赛 A100 上增益；两文件接线合入也不等于完整 GLM 质量/性能验收。

### D. #40680：快照不仅要选对位置，还要活到下一轮

**VERIFIED / PR 正文**：HiCache write_back 当前会备份设备叶节点，但内部节点的 Mamba 状态可能在淘汰时直接失效；MLA KV 尚在也无法从缺失状态恢复。PR 在 tombstone 前尝试主机备份，缺空间/失败时保留旧丢弃行为。新增 helper 在本地基线搜索不到。

**INFERRED**：角色边界将来成为树内节点时，这个生命周期问题值得纳入压力验证，不能只测冷启动后的第一次命中。它不是 #39156 的替代：DSA indexer 与 recurrent state 是不同恢复对象，修一个不代表另一个安全。暂不据此放行 HiCache。

## 4. 新 issue 的价值：明确应该怎么测，而不只是找加速数字

- **VERIFIED / #56868 是作者报告，根因未确认**：其私有 W4A16 GLM 在长时间累计 reasoning decode 后退化，而失败请求冷跑正常；prefill-only 回放未复现同样行为，关闭 prefix cache/MTP 也未解决。设备与权重不同，不能外推到我们的 FP8/A100，更不能直接断言是 KDA 槽复用 bug。**INFERRED**：短输出替身回放只证明那段 cache/scheduler 行为；完整模型必须有持续长 decode、同输入冷/暖对照与真实能力检查。不得用截历史、缩输出或删 reasoning 的方式规避赛题。
- **VERIFIED / #40685**：GLM 的 raw beta logits 在 PTX prefill 路径漏 sigmoid，补丁传递对应开关；作者在 GB300 检查错误输出与修复后行为。说明新 kernel 的数学语义要核对，但没有证据说我们的 Triton 路径也有此 bug；不能泛化为“所有 KDA prefill 出错”。

## 5. 两类容易误读的“新优化”

1. **#57261 的 +14.7% 不是实际新腾出显存**。正文明确 allocation unchanged：同样 11.88 GiB/rank 池，修正 speculative scratch 应按 max_num_seqs 而非理论驻留并发计费，报告容量 635699→729088 tokens；作者 TTFT/decode 无噪声外变化。另一个 +35.6% 是 PP4 算式而非实测。其有用的新线索是作者声称用 4×CMP170HX（GA100/sm80）+ **out-of-tree Ampere sparse-attention backend** 跑过 GLM W4A16；该 backend 不在 PR 内，不能据此认定上游 A100 路径就绪或主办方补丁来源已知。
2. **#34299 不能只看吞吐**。作者固定 head 的 GB300/Kimi K3 测试中，2k64/c32 吞吐 +8.34%、TTFT p50 −41.18%，但 TPOT p50 **恶化 62.22%**；另一组负载结果不同。本赛卡 TPOT 门，不能用吞吐一项替代验收。此 PR 有 41 文件及外部 backend 依赖，不是低风险几行优化。
3. **#40310 的 CPU offload 主要是权重 offload**：修 tied parameter/缓存 view，并处理 PD index mapping。不是我们的 HiCache KV/DSA 恢复完整解决方案，不能用它撤销 #39156 风险。

## 6. 仅供后续决策的阅读/验证顺序

**INFERRED，未改变既定计划**：

1. Plan B 实现参考：#56960 + 本地 R7 单点路径，先拆清选点、conv、持久槽与数值门。
2. 若评估 DP：同时评估 #31170 的会话键可得性、热点保护、真实缓存命中，不只抄并行参数。
3. 若评估 HiCache：#39156 正确性 + #40680 状态保留分开审计，完整 GLM/DSA 才能确认。
4. 若实测 MTP 状态带宽是瓶颈：再单独评估 #40517/ReplaySSM spec；不直接叠加所有新特性。

本轮不拉仓库/镜像，不改 `src/sglang`、harness、补丁或任务队列，不运行 GPU。公开 PR 不能代替 D1-04、004 alignment、完整模型能力与 N@SLO 的既有验收门。
