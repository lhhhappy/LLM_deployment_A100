# Claude 编排报告

会话 `91cd0b57-498a-4428-b192-9ba6fe4f2404`，主做负载与服务层。合作方式见 [collaboration.md](../collaboration.md)，方向与卡点见 [roadmap.md](../roadmap.md)。

## 待执行：用户已同意（2026-09-24 14:05 UTC）

**1. 停 061s、撤 063s。** 原因：122 在 N30 把冷请求压成 64 token 一块。以下均为实测，来自各运行的服务日志：
- 061s（A+122）中 75% 的 prefill 批次小于等于 64 token（3,193/4,268），几乎都发生在有排队时。048（A+122，N22）为 9%，A 自身（047/058/059）约 0%。
- 13:50 起的一段：排队 25，运行 4，KV 用了 53%，同一冷请求每次只算 64 token，约 3 次/秒，待算 173 万 token。
- 122 每 30 秒的统计显示平均预算约 8,080 token，所以 64 来自 `_ax_pace_limits` 的 `cap = max(grid, budget - reserve)`：reserve 累加排队里所有新 token 不超过 8192 的短命中，排队一多就超过预算；而这些短命中本轮并未进批（原因正在用 CPU 复现）。
- 122 已修复（提交 `759a6eb`）：只预留能与一格续块并存的完整命中（执行层 Codex 发现并写的首个修复与测试）；被拒的命中在该续块结束前不再占预留。15 项 pace 测试通过（被拒命中测试在无第二个修复时失败），调度 32 项、SRPT 5 项通过；尚未上 8 卡。修好前不进任何组合。

**2. 本地三个任务**：全量 N30、70 分钟后排空，同一引擎提交 `c92acd5`，按顺序跑：
- a. 正式 A 原样，参数同 `submission/official-0923-A.json`；`G_EXPECT="120=on 122=off 180=off"`。
- b. a 加 `--mem-fraction-static 0.87`。依据（执行层复核后更正）：061s 前 21 分钟 8 卡采样的最大值为 70,446/81,920 MiB，但这段仍在预热，不能据此说约 11.2 GiB 从未用到，4 小时一档里可能出现更高峰值；0.87 时 `kv_cache_configurator._handle_max_mamba_cache` 按比例 0.9 同时扩大 KDA 槽，MTP 中间池还受 max_running=32 约束，三者须联合计算，不给 KV 增量的预估，以 065 启动日志的实测池大小为准；0.87 只是实验值，不承诺长时间不 OOM。
- c. b 加 `--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through`；`G_EXPECT` 中 `180=on`。

**3. 今天两个正式提交**，用户授权，用于校准本地尺子，不为刷分：
- 提交 1 = b，提交 2 = c，同一镜像：`scripts/build_image.sh c92acd5`，叠在 0923a 上，Dockerfile 38.7 KB。
- 每个配置在本地确认启动正常、机制核对通过、前 20 分钟无错误后即可提交。
- 与已有的正式 A（官方 N14）合成三个官方点，与本地 a/b/c 的好坏顺序对照。
- 风险：官方压测机主机内存未知（180 每卡需 32 GB）；0.87 在 4 小时一档里可能遇到更高的显存峰值；180 从主机搬回后的数值尚未在 GPU 上验证。

**4. 第二个问题（Claude 分析中）**：A 自身每批几乎只放 1 个请求。061s 有 93% 的批次是单请求批，排队长时这会成为瓶颈。

## 现状摘要（2026-09-24 夜）

- **测试口径（用户定）**：全量长链集（311 链、5601 请求）N30，第 70 分钟停止准入后排空；只比较方案，不要求本地过线上门，也不换算正式 N。判断补丁看它针对的机制量（缓存命中、重算、主机搬回、首执行前等待、TPOT），门照常全部报告。
- **引擎改为 git 源码**：`engine/sglang/`，标签 `engine-base`、`official-A-0923a`（与 0923a 镜像源码逐文件一致），每个机制一个或一组 `engine NNN:` 提交；启动打印 `[ax] mechanisms:`，任务 `G_EXPECT` 不符即不测。见 [engine/README.md](../../engine/README.md)。镜像构建以我们已注册的镜像为底，只嵌增量（`scripts/build_image.sh`）。
- **180（HiCache）**：已修正旧版在 HiCache 下悄悄关闭 120/122 的问题（059 因此不是单变量）。059 实测：每请求重算均值 4,661→4,068 token；fast/overall 变差来自 120 被关（有人排队时 8192 大块批次 10/1061→648/866，新增 fast 超时多出的时间全在准入到执行之间）。
- **8 卡当前**：061r（A+122）运行中，063r（A+122+新 180）排队；同一提交 c92acd5，两者只差 HiCache 参数。执行层挂 25 分钟监控。
- **下一步（Claude）**：
  - 两项出结果后，按请求类型、等待还是重算做失败归因，重点看 122 是否压住开场冷启动排队、180 的重算减少与主机搬回对 decode 的影响；
  - HEAD 用正式 A 参数与 official-A 的 8 卡等价确认（补做）；
  - 主机搬回计时与回退原因埋点（并入 180）。
- **暂停**：124 未迁移（与 122/123 冲突，功能在 122 里）；驻留账本在真实长链运行校准前不用于决策。

## 048 对照（A+122 冻结版 vs 047 A 原样，均 dev N22；单变量）2026-09-24

主会话取回的本地证据：`evidence/L048-official_a_122_n22/N22/compare_vs_047.txt/.csv`（两边 VALID，722 条，同一负载标识）。以下都是描述，不是因果；cache/wait/run 只是描述性分类。

| 门 | 047 | 048 |
|---|---|---|
| fast_intra | 33/23 FAIL（p95 5.23） | **10/23 过**（p95 2.08） |
| overall_intra | 55/27 FAIL（p95 10.80） | 28/27 FAIL（p95 7.78） |
| turn_start | 2/3 过 | 2/3 过 |
| chain_start | 73/22 FAIL（p95 93.09） | 62/22 FAIL（p95 76.63） |
| TPOT mean/p95 | .0620/.0870 | .0617/.0840；单请求 >0.10：8→2，最大 .115→.113 |

- **逐请求：**
  - fast 超限改善 25 条、新增 2 条；
  - overall 改善 32 条、新增 5 条；
  - chain 改善 16 条、新增 5 条；
  - turn 不变。
- **改善集中处：** 新 token 4097–8192 的缓存命中请求超限 41/81 → 2/77（`intra_attrib.csv`，两边同一口径）。这正是 122 中短命中预留规则作用的那一类。122 同时含进度控制，两者不能在这次运行里拆开；拆开需要 A+124。
- **执行块：** 测量窗内 prefill batch 2301→1585；≤4k 的 1805→89，≤8k 的 388→1217；partial 单跑 1815→1107；prefill token 9.92M→10.04M。
- **122 行为：** `[ax-pace]` 末行 decisions 8525，其中 forced_decode 6383，mean_budget 7257，**guard=0**（本次没有强制放行）。窗口 1170→1125 s，GPU 平均利用率 94%→95%。
- **仍挂的门：**
  - chain 62 条超限，等待中位 52.5 s，执行中位 4.9 s（047 为 48.0 / 3.5），仍以首次执行前等待为主；122 不改变 LPM 顺序。
  - overall 28/27 只差 1 条，在噪声范围内，不能据此判断能否过门。
- **其他解释与限制：**
  - 每边只有一次运行，N22 下没有 A 重跑。已知同配置重跑（036/042，S1 N22）的波动是 fast 7↔15、overall 15↔18。fast 与 overall 的变化量（23、27 条）超过这一次观测到的波动；chain 的变化（11 条）无从判断是否显著。
  - 缓存：总命中 24.51M→24.39M，129 条请求命中差 ≥1024，不支持"048 缓存更好"。
  - TPOT 均值基本不变，这与仿真中"会抬高 tpot_mean"的预期不同，原因未查。
- **结论（单变量、单次）：** A+122 在 dev N22 过了 fast，overall 接近门限，chain 仍远未过。chain 是下一个瓶颈。

## 047 本地审阅（正式 A 原样，dev N22；2026-09-24）

- **完整性：** 722/722 条唯一、0 错，VALID；与主会话重判一致。
- **11 门：** 挂三道。fast 33/23（p95 5.23）、overall 55/27（p95 10.80）、chain 73/22（p95 93.09）失败；turn 2/3、TPOT .0620/.0870 通过（8 条请求超 0.10）。
- **TTFT 各门以等待为主：** 超限的 chain_start 请求等待中位 48.0 s，执行中位 3.5 s（prompt 中位 65k）。等待指到首次执行前的时间，不能直接归因于调度。
- **intra 超限（`intra_attrib.py`）：**
  - 新 token 4097–8192 的缓存命中请求 41/81 超限；其中 28 条若按 T56 LCP（未核源诊断值）完整复用，新 token 会 ≤4096，所以缓存复用与块预算两者混淆。
  - 等待窗口里 691 个 prefill batch 中 572 个是 partial 单独运行、留位未用。
- **与 044r（同配置 N14）按请求对照（`compare_runs.py`）：**
  - chain 超限 32→73，新增 46 条中 25 条主要是等待变长，21 条的缓存命中与 N14 相差 ≥1024 token（14 条更少、7 条更多）；
  - 执行时间 p95 基本不变（12.7→13.0 s）；
  - 缓存总量 24.61M→24.51M，104 条请求的命中差 ≥1024。
- **证据：**
  - `evidence/L047-official_a_n22/N22/intra_attrib.csv`
  - `evidence/L047-official_a_n22/N22/compare_vs_044r.csv`
  - 工具 `scripts/analysis/compare_runs.py`：同请求对照；两边请求集合不一致或数据不完整时判 INVALID。
- **048 对照已准备：** 048 结束后运行 `compare_runs.py evidence/L047-official_a_n22/N22 <048目录>`，逐请求分开：
  - 缓存（命中差、LCP 缺口）；
  - 批形成与准入（等待）；
  - 执行块（exec→first、batch 大小分布、partial 单跑数、`[ax-pace]` 预算与 guard）。

## 里程碑：下一候选 124（短命中请求预留），2026-09-24，READY（草案，未入队）

1. **变量选择：** 短请求归因。工具 `scripts/analysis/intra_attrib.py` 可复用，输出逐请求 CSV：`evidence/L044r-cal_offA_n14/N14/intra_attrib.csv`、L045r、L042。
2. **044r（正式 A）：**
   - 新 token ≤4096 的缓存命中请求 0/274 超限；4097–8192 的 26/79 超限，等待中位 6.56 s（未超者 0.30 s）。
   - 超限的都在等别人的 partial：100%，未超者 49%，全程基线 91%。
   - 22 条超 3 s 的 fast 请求中 19 条属于这一类。4096 正是 A 的 chunk−COLD_CAP。
3. **045r（正式 B，剩余预算 0）：** 缓存命中 ≤8192 的请求 59/341 超限，都在等 partial。与 B 正式挂 fast 方向一致（描述性，不是因果证明）。
4. **补丁（未迁移，原文在 git 历史）：** `patches/124-short-hit-reserve.patch`，sha256 `3e256b48d0180e41bdf544d266b4a814dd6cb136b498680b8a738a56c7724c64`。
   - 规则：续算上限 = min(COLD_CAP, chunk − 等待中短命中请求的分页新 token)。没有这类请求时与 A 相同；batch 预算不变。
   - 开关 `SGLANG_AX_SHORT_RESERVE=1`。
   - 与 122 互斥：122 含同一规则并叠加了进度控制，124 用于拆分 048 的效果。
5. **CPU 测试：** 5 个用例，与 120/122 合计 44 个通过。
6. **草案：** `scripts/pod/jobs/drafts/offA_124_n22.sh`，A+124，dev N22，对照 047。
7. **风险：** 链首每让出一次，就多等一个短命中请求的 prefill 时间；A 的正式紧门是 chain。
8. **状态：** 124 冻结，不排队（主会话决定）。047/048 证据就绪后，用 intra_attrib.py 复核同一类请求。

## 124 反例检查（2026-09-24）：已冻结，停在候选，不追加队列

1. **19/22 口径：** fast 桶按冻结 `uncached_expected ≤4096`；这 19 条实际新 token 4097–8192，而按 T56 LCP（未核源诊断值）完整复用时都 ≤4096。**缓存复用不足与块预算混淆**，原先"19/22 属于留位问题"说法过强，已在 124 .md 更正。
2. **缓存：** 19 条的命中全部停在前一请求 prompt 末尾之前，缺口 1.4k–6.4k token（不是前一轮输出造成的）。原因未查；候选解释是 A 在 MTP 下关闭了 140，KDA 状态只在部分位置可复用。
3. **排队：** 26 条类别超限者的等待窗口里，595 个 prefill batch 中 502 个是 partial 单独运行，留位未用；只有 16 个是满块。不支持"被其他命中请求挤掉"。
4. **块预算：** 同类别未超限者 53 条，其中只有 49% 的等待窗口有 partial（超限者 100%），等待中位 0.30 s。
5. **结论：** 这批请求等待时的约束确实是固定留位，但其中 19 条同样可以通过修复复用来解决（依据是未核源的 T56 LCP 诊断，不作缓存因果归因）。124 与"复用修复"是针对同一批请求的竞争变量，等 047/048 证据后再决定是否值得占一档。
6. **可复现：** `scripts/analysis/intra_attrib.py`（新增列 frozen_expected、new_at_full_reuse、win_partial_only）；CSV 在 L044r/L045r/L042 目录。124 补丁未改，sha 仍为 3e256b48…。

## 细节

### 锚点时序（真实 overlap 循环，`scheduler.py` `event_loop_overlap`）
第 k+1 步的 `get_next_batch_to_run` 在第 k 步的 batch 启动之后、其结果处理之前调用。所以第 k 步 prefill 完成的请求，在第 k+1 步决策时已进入运行批，但输出尚未写回。122 在这一时刻建立锚点，时间取 max(同步时钟, 在飞受控 prefill 的预测结束)：

| 情形 | 锚点相对真实首 token |
|---|---|
| 产生它的 prefill 受控 | ≈ 预测结束；偏晚量 = 成本模型高估量，无上界 |
| 产生它的 prefill 不受控（当时无 decoder） | 早于首 token，保守 |
| 非 overlap | 晚了结果处理与调度的耗时，无上界保证 |

### MTP 产出计数
A 实际是 EAGLE/NEXTN 加 overlap（044r 的 server_args：`disable_overlap_schedule=False`）。122 按 `len(req.output_ids)` 计数；spec 在结果处理时 extend 接受的 token（`batch_result_processor.py:956`）。决策看到的计数最多滞后一步，只会让余量偏小。

### 各 rank collective 分支
`_ax_pace_now` 的 all_reduce(MAX) 只在以下条件同时成立时进入：
- 运行批非空且不是仅 prefill；
- 有未完成的请求；
- 有续算或等待中的请求。

这些量在各 rank 相同：
- 等待队列由 rank 0 广播；
- 运行批与 `finished()` 由相同的结果处理更新；
- 基线的 `SGLANG_REQ_WAITING_TIMEOUT` 默认 −1，不触发本地时间分支。

决策只依赖同步后的时钟与确定性的 token 计数；`busy_until` 由同步时钟和 `extend_num_tokens` 算出。日志只在 rank 0 打，不影响决策。

### 测试
- 12 个 122 用例覆盖：
  - 关闭时与正式 A 逐步相同；
  - 预算算术；
  - 落后时的 decode 轮数；
  - guard 放行整块；
  - 等待请求 fill ids 为空；
  - 首 token 未处理；
  - overlap 在飞 prefill 计入；
  - rank 时钟；
  - 两个假时钟仿真（机制自检，不是性能预测）。
- 另有 27 个 120 原有用例。

复现（测试直接从 git 取源码树：official-A-0923a 与 122 的提交）：
```
python3 -m unittest discover -s tests -p "test_tpot_paced_prefill.py"   # 12 OK
python3 -m unittest discover -s tests -p "test_sched_protect_chain.py"  # 32 OK
```

### 048 判据
- 全部 11 门与 047 逐门对照；
- `[ax-pace] on:` 的生效值；30 s 统计行中的 `mean_budget` 与 `guard`；
- prefill 块长分布、KV 使用率、在飞与排队数。

推翻条件：首 token 各门都没有改善，或 tpot_p95 > 0.10，或块并未变大。

### 相关文档
- 补丁说明：[122 .md](../../engine/docs/122-tpot-paced-prefill.md)
- 分析：[roadmap](../roadmap.md)、[introduction](../introduction.md)
