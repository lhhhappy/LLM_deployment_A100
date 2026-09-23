# R8 · 下一步方向（2026-09-23，Claude）

依据：赛题原文 `llm-challenge-arena-v1/task.md` 全文重读；开发集 `s1-dev/data/dev-combined-v1/requests.jsonl` 统计；
最新公开成绩 `data/all_att_2026-09-23.json`（551 条）；research/ 全部报告（两个子代理提炼 + 本人复核）；8 卡实测 F59/F62/F63。
标注：VERIFIED = 数据/源码/实测；INFERRED = 推断。

## 1. 比赛到底在比什么（VERIFIED，task.md）
- 排名：`n_at_slo`（梯子 2/6/10/14/18/22/26…，**从 N=10 起爬**，过了 +4，首次在已过档之上失败即停）→ `tpot_mean`（发压侧首/末 SSE 时刻算）。
- 单档 11 条硬门：coverage=100%、harness_data/render=0、engine_error<1%、infra_error<1%、四道 TTFT p95（fast_intra≤3s〔链中间且冻结未命中≤4096〕、overall_intra≤5s、turn_start≤15s、chain_start≤30s）、gated_phases_have_samples、tpot_p95≤0.10s。
  TTFT 门**带统计余量**：超标率 95% 单侧下界 >5% 才判失败。单档约 4 小时；隐藏整链集。
- **进入压测前的能力门**：aime26 与 gpqa-diamond 两科 points **严格 >90**（`/chat/completions`，不能压输出/thinking）。
- 赛后复核镜像、参数、代码，核对时间戳、token 计数、flush 真实性。
- 五条官方底包：SGLang（我们用的，A100 原样起不来）、vLLM 官方、TokenSpeed、Transformers、KTransformers；另有主办方跑通全链路的 **vLLM sm80 backport**（示例提交，带 MTP 3 步）。

## 2. 负载形状（VERIFIED，开发集 722 请求 / 311 链 / 136 会话）
| 量 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| prompt token | 36,576 | 91,549 | 224,581 | 256,733 |
| 冻结未命中 token（全部） | 2,867 | 37,661 | 138,847 | 256,733 |
| 未命中 token（chain_start 类，约 400 条） | 17,879 | 58,633 | 194,991 | 256,733 |
| 未命中 token（链中间/轮首，约 322 条） | 1,561 | 4,950 | 36,279 | 37,947 |
| 输出 max_new_tokens | 198 | 554 | 2,453 | 5,644 |
| 请求间隔 replay_gap | 2.0s | 9.2s | 33.9s | 300s |
- edge_type：append-only 441、**system-tools-changed 148**（结构性零命中）、chain-head 104、unexplained-break 23、compact-rebuild 6。
- 总量：prompt 3442 万 token，其中未命中 1198 万；输出仅 21.6 万。**这是预填充主导的负载**。
- 注意：开发集是链前缀抽样，chain_start 占比远高于正式集（正式集链更长、中间请求更多）。

## 3. 前排（VERIFIED，2026-09-23 拉取；无人过 N=26）
| 选手 | N | tpot_mean | fast | overall | turn | chain |
|---|---:|---:|---:|---:|---:|---:|
| LewyM | 22 | 0.0273 | 1.79 | 3.00 | 10.9 | **61.0** |
| Jinbo hu | 22 | 0.0370 | 1.61 | 2.35 | 8.2 | 29.4 |
| Mingjun Xu | 22 | 0.0509 | 2.11 | 2.67 | 7.3 | 30.2 |
| ccooddxx | 18 | 0.0242 | 2.19 | 3.46 | 11.4 | 44.7 |
- **所有 N=18/22 选手的约束门都是 chain_start**；fast/overall/turn 留 30–50% 余量；tpot_p95 最高也只到门槛的 0.78。
- LewyM/Mingjun Xu 于 09-22 晚各有新提交在评测中。Bamboo_0 N=14 但 tpot 0.0199（decode 最快）。
- 镜像名线索（INFERRED，无作者关联）：marlin 82 个（主流 sm80 量化路径）；`actquant`/int8 W8A8 3 个（MoE 激活量化）；`fp8kv` 1 个；
  调度名 edf/srpt/sjf/slack/aged-fifo 18 个（至少两队自研 SLO 调度）；dp2-affinity/balanced 8 个；rowchunk-kda-cpu/hicache/kpool 12 个；graphsafe/nextn 10 个。

## 4. 方向（按对"冲 N=26"的价值排序）

### P0 必做核验（不做就可能白费）
| # | 方向 | 为什么 | 放哪里 |
|---|---|---|---|
| V1 | **能力门自测**：我们的栈（Marlin W8A16、tilelang、112/113、140、160）跑 AIME/GPQA 样题，确认 >90 | 能力门不过则压测无成绩；从未测过 | 8 卡任务 `scripts/pod/jobs/cap_*.sh`；结果入 experiments |
| V2 | 8 卡基线 + profile（b113）：冷预填充 tok/s（2 万/6 万/19 万）、decode ms/step、显存账（启动日志） | 所有排序都以此为准 | 已在 autostart 队列 |

### P1 冷预填充吞吐 = chain_start（约束门）
| # | 方向 | 机制 | 放哪里 |
|---|---|---|---|
| A1 | **INT8 W8A8 MoE（及稠密层）** | A100 int8 张量核 624 TOPS vs bf16 312；预填充 MoE 是算力瓶颈，Marlin W8A16 在大 batch 下只能跑 bf16 MMA。块量化 FP8→INT8 重量化 + 激活逐 token 量化。镜像名 `actquant` 显示有队伍在做 | 补丁 170；须过 V1 能力门 |
| A2 | 8 卡预填充 profile 驱动的逐项优化：KDA chunk 核（A100 调优配置）、Marlin MoE 大 M 配置、tilelang 稀疏注意力、allreduce | 先测再定 | 17x 系列 |
| A3 | chunked_prefill 8192→16k/32k（与 120 配合） | 减少轮数与每轮固定开销；单独调大会恶化 decode 停顿，必须与 120 的 cap/穿插一起调 | 参数，放在 120 A/B 之后 |
| A4 | EDF/按门截止期排序（在 120 之上） | 按请求所属门的剩余时间排序，chain_start 不被无限推后；利用统计余量把"必然超标"的少数请求留在预算内 | 补丁 121 |

### P2 容量（N=26 时 KV 很可能不够 → 驱逐 → 更多冷预填充）
- VERIFIED：每卡 KV 943k token（11.2GB，MLA+indexer 在 TP8 下每卡全量复制），KDA 状态 584 槽（9.7GB），decode CUDA graph 占 6.1GB（默认 max bs 远超实际并发）。
- INFERRED：N=26 × 平均上下文 ~5 万 ≈ 130 万 token > 94 万，容量将先于算力约束。
| # | 方向 | 机制 | 放哪里 |
|---|---|---|---|
| C1 | `--cuda-graph-max-bs` 收到实际并发（~64） | 预计腾出约 5GB → KV +40% | 参数，S |
| C2 | 状态池策略：`extra_buffer_lazy`、`mamba_full_memory_ratio` 重配、`--enable-int8-mamba-checkpoint` | KDA 状态与 KV 的预算再平衡；int8 需数值验证 | 参数 + 验证 |
| C3 | 驱逐策略：reminder 之后的 KDA 状态优先淘汰；回收"最深状态以下搁浅的 KV" | 同样显存下更高命中 | 补丁 141（与 140 同源） |
| C4 | FP8 KV cache（sm80 仅存储，核内反量化） | KV 容量 ×2；需改 tilelang/112 路径读 fp8 | 补丁 180，L |
| C5 | DP2 注意力 + 会话亲和路由（SGLang #31170） | 逻辑 KV 容量 ~1.9×；8 个镜像名显示有队伍在做 dp2-affinity | L，放在 C1–C3 之后 |

### P3 TPOT（同档名次；目前第一名 0.0273）
- 160 MTP（已备）；A100 Marlin/KDA decode 调优；SSE 增量输出（已开 `--incremental-streaming-output`）；allreduce 实现 A/B。

### 已明确放弃（不再重复）
HiCache（DSA indexer 未恢复→错误输出）、`--enable-mixed-chunk`（破坏 mamba 快照）、单机 PD 分离（权重放不下两份）、PDMux、DeepEP/TBO/deep_gemm EP（需 sm90）、
`--enable-linear-replayssm`（与 page_size=64 不兼容）、改写客户端 prompt（违反冻结负载）。

## 4b. 已调研的参考 PR 与方向对应（refs/）
| 参考 | 内容 | 对应方向 | 状态 |
|---|---|---|---|
| `refs/vllm-pr56960`（vLLM #56960） | GLM-5.3-Flash KDA 预填充中途快照：kernel 内导出 + 卷积历史 + 验证方法 | M2 → 补丁 140 | 已采用（W19/T45） |
| `refs/sglang-pr31170`（SGLang #31170） | DP `prefix_affinity` 路由：按 routing key（或前 4096 token 哈希）固定 DP rank；rank 负载 >1.5× 均值则绕开（fallback total_tokens） | C5 DP2 + 会话亲和；其"亲和+过载退让"思路也可用于单调度器内的会话感知排序（A4/121） | 未采用，C5 时移植 |
| `refs/sglang-fe236ea6c3` | 底包对应的公开提交（2026-09-01） | 所有补丁的基线 | — |
| （同一克隆内）SGLang main 至 09-22 | 底包之后三周的上游改动 | 可能含 GLM-5.3/KDA/DSA/sm80 修复与优化 → 见 R9 上游扫描 | 扫描中 |

## 5. 战略问题：引擎选择（待 V2 数据后决策）
主办方的 vLLM sm80 backport 开箱即跑通全链路（含 MTP），而 SGLang 底包需要 110–113 才能跑且性能待测。
若 V2 显示 SGLang 栈冷预填充吞吐明显落后，应评估"vLLM backport + 我们的机制移植（120/140 思路）"。在 V2 之前不下结论。
