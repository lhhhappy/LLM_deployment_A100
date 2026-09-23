# R13 — Fable 独立方向评审（2026-09-23 夜）

依据：task.md、公开开发集 harness 源码（`s1-dev/harness/s1_loadgen.py`）、pod 上 7 个梯子档位的原始记录与 server.log（脚本 `/tmp/ax/codex/fable_runstat.py`，输出见附录）、`notes/ledger-8card.md`、R7/R8/R10/R12/R18、`evidence/R8/prefix_reuse_rows.json`、`data/all_att_2026-09-23.json`、用户今晚贴的榜单。
标注：【实测】直接来自数据；【算术】由实测数推出；【推断】我的判断。

---

## 0. 结论（先看这里）

1. **这个比赛是闭环负载，不是开环。** N 是并发"客户端槽"数，每个槽串行驱动一条链：发请求→等 TTFT→等 decode 完→睡 gap→下一条。【实测，loadgen 源码】结果：**decode 越快，客户端回来越快，prefill 需求越大**。我们所有 N18 档的未命中 prefill 吞吐都钉在 **8.1–8.5k token/s**，与配置无关【实测】——机器在 N18 就已经是 prefill 饱和的，省下的 decode 时间全部变成了 TTFT 排队【实测：026→034 decode 客户端秒 −4000s，TTFT 客户端秒 +1600s，墙钟只短 9%】。
2. **因此 MTP 在我们当前区间会降低 N**（028/034 的 TTFT 崩掉不是巧合），tpot 是一种可以花掉的预算，不是要抢先优化的指标。CalvinCao N26 / tpot 0.055 / tpm 偏低，正是"把等待均匀地放进 decode、用满 tpot 预算换 N"的画像。【推断】
3. **026 之所以 tpot_p95 0.219 不过，是 decode 的"时序"问题，不是吞吐问题。** 16k 大块每块停 1.1–1.6s，短输出请求（<100 token，占 11%）平均 tpot 0.122。修法是**大块 + 长 decode 间隔**（例如 `--prefill-decode-interval 16~24`，重排 decode 步而不减少 prefill 块大小）。这个象限**从来没测过**——团队测的是"小块 + 短间隔"（027/028c），小块的固定开销把吞吐吃掉一半，于是把"间隔"和"小块"一起否定了。【实测 + 算术】
4. **需求侧最大的杠杆没有做：跨会话的家族前缀复用。** 开发集冻结口径下 11.98M 未命中 token 里，`system-tools-changed` 边就占 7.97M（67%）。按"任何更早请求的最长公共前缀 + 4096 网格状态"的策略，这一类可降到 2.25M；整体未命中可从今天实测的 9.9–10.6M 降到约 6M（−40%）。今天的实测里 session_start/turn_start 已经拿到 55–63% 的跨链复用，**但 system-tools-changed 这一类复用为 0**（intra 实际 7.75M ≥ 冻结 7.40M）。这是我们和前排在"每个 prompt token 的 GPU 工作量"上差一倍的最可能来源。【实测 + 模拟（LCP，未计淘汰）+ 推断】
5. **R12 的 MFU 算术本身成立，但把它定为一级问题是错的框架。** 20–30% 的 MFU 在 A100 + fp8 权重 + Marlin W8A16 下 24 小时内提不上去（F70 已证 int8 不值）；而闭环 + 家族前缀 + decode 再定时三件事都是几十行改动或纯参数。**不换引擎、不换 sm80 栈；换的是负载模型和实验的度量方式。**

---

## 1. 独立判断：负载是闭环的，GPU 已在 N18 饱和

### 1.1 loadgen 的机制【实测，`s1_loadgen.py:334-394, 586-600`】
- `pick = min(N, len(chains))` 个线程，各自从链队列取链，`drive()` 串行发链内请求，每条请求前 `sleep(replay_gap)`。没有时间轴回放；正式集用同一套 harness（task.md 说"同一套 harness"）。
- 开发集：722 请求 / 311 链；gap 总和 3028s；输出 216k token；冻结未命中 11.98M。

### 1.2 客户端时间去哪了（N×墙钟 = gap + TTFT + decode）【实测，附录 A】

| 档 | tpot 均值 | gap | TTFT 秒 | decode 秒 | 墙钟 | 未命中 tok/s | fast / overall / chain p95 | tpot p95 |
|---|---|---|---|---|---|---|---|---|
| 026 N18（16k 块，无 MTP，140 开） | 0.083 | 3028 | 2222 | **16396** | 1270 | 8370 | 2.44 / 4.41 / 25.4 | 0.219 |
| 034 N18（8k 块，MTP，140 关） | 0.061 | 3028 | 3836 | 12375 | 1160 | 8534 | 13.2 / 17.2 / 35.5 | 0.210 |
| 028 N18（cap4k i2，MTP） | 0.053 | 3028 | 5682 | 11370 | 1191 | 8236 | 5.45 / 11.0 / 78.5 | 0.082 |
| 028b N18（+BCG 4k，MTP） | 0.048 | 3028 | 6788 | 10628 | 1196 | 8133 | 18.1 / 18.1 / 71 | 0.077 |
| 027 N18（cap2k i3） | 0.082 | 3028 | 9759 | 17126 | 1728 | 6579 | 14.7 / 18.9 / 107 | 0.111 |
| 025b N10 | 0.047 | 3028 | 2098 | 9056 | 1456 | 6857 | 2.11 / 3.13 / 26.4 | 0.13 |

读法：
- 026 里客户端 72% 的时间在等 decode。**decode 慢 = 到达率低 = prefill 队列短 = TTFT 过**。这不是设计，是撞上的。
- 四个 N18 配置的未命中吞吐都是 8.1–8.5k tok/s：**这是机器在这个负载混合下的容量**（含小块固定开销、decode 分走的时间）。谁把 decode 提速，谁就把省下的时间还给了 TTFT 排队。
- 027 更低（6.6k）是小块固定开销（F87）；这条结论仍成立，但它只说明"块不能小"，不说明"间隔不能长"。

### 1.3 容量的分解【算术】
- 026：墙钟 1270s，decode ≈ 216k token / 平均 bs 11 × 17ms ≈ 335s，其余 ≈ 935s 给了 10.65M 未命中 token 的 prefill ⇒ 全包 **≈88µs/token**（含固定开销、短块低效、前缀变长）。满负荷 prefill 上限 ≈ 11.4k tok/s。
- 冷块本身 60–100µs/token（026g/h 实测）；114 后 61–65µs。

---

## 2. 对手在做什么【推断，依据榜单与 R7】

- **所有 N≥14 的前排 tpm_all 都钉在 1.72–1.79M**（N14、N18、N22 一样）。闭环里 tpm 平台 = 饱和。他们也是饱和的，只是饱和点上 tpot 还能 0.024–0.028。
- 我们在 N18 的 tpm_all 等效值 1.64–1.79M，**和他们同量级**（开发集与正式集混合不同，只能粗比）。也就是说"链推进速度"接近，差别在**每个 prompt token 花多少 GPU**：他们 decode 能拿到 60% 的时间，我们只有 25%。同样的 tpm，一半的 prefill 工作 ⇒ **命中率高一倍或算力高一倍**。24 小时内算力翻倍不现实；命中率翻倍正是 §4 的家族前缀（−40% 需求）。
- **LewyM N22 / 0.027 / chain 61s**：短请求极快（fast 1.79）、冷启动排 61s 仍过统计余量 —— 典型 SRPT + 把等待推给 chain_start。
- **Jinbo hu N22 / chain 29.4s**：N22 下冷启动 p95 仍在 30s 内 ⇒ 冷启动基本不冷（家族前缀复用）或产能明显高。
- **CalvinCao N26 / 0.055 / tpm 1.46M / tpm_decode 18k**：tpm 比 N22 的人还低，说明他不是更快，而是**把等待均匀地放进 decode**（tpot 是均值≈0.055 而 p95 仍 ≤0.10，说明没有长停顿），到达率被 decode 节流，TTFT 门反而好过。这是"用 tpot 预算换 N"的极端解，规则允许（N 第一顺位）。
- 组织方 vLLM sm80 backport 例子（piecewise CUDA graph + MTP）给了很多人一个小块固定开销低的起点，但从闭环模型看它并不是决定性的——王俊杰用它也只到 N18。

---

## 3. 我们真正的问题（按杠杆排序）

| 级 | 问题 | 证据 | 杠杆大小 | 代价 |
|---|---|---|---|---|
| **1** | **system-tools-changed 边零复用**：7.97M 冻结未命中，实测没降；家族前缀（system+tools，同 `sys_tools_hash`，开发集 156 族、44 个 prefix_family）本可被后来的链复用 | 附录 B：session_start 实际 1.01M vs 冻结 2.24M（已复用 55%），turn_start 0.84 vs 2.29；**intra 7.75M vs 7.40M（0%）**；R8 模拟：grid4096 下该类 7.97M→2.25M | **未命中总量 −30~40%** ⇒ decode 份额翻倍、tpot 自然下降、chain/turn 尾巴变短；这是唯一能同时改善四道门 + tpot 的杠杆 | 需要查清为什么现在不命中（状态不在边界 / 被淘汰 / 树结构），然后在 140 上加"首个角色边界（system+tools 末尾）快照"或按 4096 网格存状态；工作量：一个 worker 半天 |
| **2** | **tpot_p95 是时序问题**：16k 块停顿 1.1–1.6s，短输出请求被连续几块打中 | 026：out<100 的 tpot 均值 0.122、p95 0.55；>0.1 的请求占 18.8% | 把 026 直接变成 N18 通过，并有机会 N22 | **纯参数**：`--prefill-decode-interval K`（K=16~24），块不变。decode 步总数在闭环里守恒（同样的 token 要生成），只是从"突发"改成"均匀" |
| **3** | **产能**：114（长上下文 −20~33%/token，实测）从未以 026 为基线单测；170 在 16k 块下只值 ~4%；DCP 是容量（KV 峰值 0.97 已经在 N18 出现，N22/26 必淘汰） | 026g/h；F80/F81；026 KV 0.97 | 114：+10~15% 容量；DCP：N26 的 KV 前提 | 114 零代码；DCP 需 45 分钟探针 + 1 档 |
| **4** | **KDA 状态槽 200 与复用的耦合** | 027（2k 块）未命中 11.37M > 026 10.63M > 034（8k 块、默认槽）9.90M；块越小状态越多、200 槽被挤 | 中 | 一个参数探针即可判定 |
| 5 | MoE/indexer 大块 kernel（R12 §0.2 的 MFU） | F70/F71 | 理论大、24h 内小 | 不排进 24h |

---

## 4. 对 R12 的回答（只答有分歧的）

- **§0.2 MFU 20–30% 算术站得住**（16.4B 激活、33–45 GFLOP/token、峰值 2.5 PFLOP/s ⇒ 18µs 下限；实测 60–100µs）。**但它不是一级问题**：前排的 tpm 平台和我们一样，说明他们没有 2× 的算力，而是少做一半的 prefill。把"引擎 prefill 效率"定为一级会把 24 小时花在 kernel 上。
- **"prefill 吞吐 + cache 保留"为一级**：半对。更准确的一级是**"每个 prompt token 的 GPU 工作量"**，其中最大的一项是跨会话复用（§3-1），其次才是每 token 成本；而"cache 保留"（整段丢失的 0.9M）只是 9%。
- **对手靠什么**：最可能是家族前缀复用 + SRPT 型调度 + 均匀 decode（CalvinCao）。不是 CP（A100 上 CP 要求 deepep：`012g` 日志 `MARLIN requires a fused func for a2a backend deepep` ⇒ 死路）、不是 PD 分离（权重放不下）。
- **被误杀、该先捡回的**：① decode 间隔（和小块绑在一起被否定）；② MTP 的解读反了——它是 N 的敌人、tpot 的朋友，只在最后同档比 tpot 时上；③ `--max-mamba-cache-size 200` 的副作用没查；④ 家族前缀（R8 F1，"进行中"却没有 8 卡任务）。
- **P0/P1/P2 顺序**：P0 里"profile 16k 大块"可以砍掉（结论会是 MoE 31%/稀疏注意力 17%，F71 已有，24h 内改不动）；P1 应把"026+interval"放在最前，"026+MTP"放最后；P2 第一项换成家族前缀。

---

## 5. 整体路线对不对

- **sm80 SGLang 栈**：保留。它能跑、12/12、TTFT 在 N18 能过；闭环模型说明 host 开销不是决定项。换 vLLM backport 是 24 小时内不可控的赌博。
- **测试方式**：错在度量。梯子只报四门 + tpot，不报"客户端时间三分法、未命中 tok/s、decode 份额"。没有这张预算表，所有系统级预判都会继续错（R12 §0 的 4–12 条全是这种错）。附录 A 的脚本 4 行就能给出这张表，应进入每档自动诊断。
- **优化目标**：目前隐含地在优化"tpot 越低越好"（MTP 早上）。应改成：**先用 tpot 预算换 N，再在 N 定了以后压 tpot**。
- **开发机替身**：只用于数值/正确性，不再产出任何系统级结论（R11 已认）。

---

## 6. 接下来 24 小时（我的判断）

顺序：**(A) 026+decode 间隔 → 立即拿下 N18，试 N22**；**(B) +114** 做稳定线；**(C) 家族前缀复用**（worker 并行写代码，8 卡上一出来就测）——这是冲 N22/26 的主赌注；**(D) DCP 探针**做 N26 的容量前提；**(E) MTP 只在最后**。明天两次提交：A = 稳定线（B 的胜者），B = 稳定线 + C 或 D 中先验证通过的一个。

预期（【推断】）：A/B 达到 N18 通过、N22 五五开；C 如果把未命中降 30%，N22 稳、N26 可试；tpot 均值会落在 0.05–0.08，同档不占优——这是有意的取舍。

---

## 7. 9h 队列

约定：基线 **S0 = 026 栈与参数**（补丁 000 101 105 106 110 111 112 113 140 120；`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`；env `SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`）。每次只改一个变量。梯子用 `dev_ladder_template.sh`（LADDER_UP="18 22 26"，自动上爬）。时间含 3 分钟起服务。

| # | 任务名 | 类型 / 时长 | 相对基线的唯一改动 | 假设 | 预期结果 | 决定 |
|---|---|---|---|---|---|---|
| Q1 | `040-ifx_s0_i16` | 干扰探针 / 8 min | S0 + `--prefill-decode-interval 16` | 16k 块之间插 16 步 decode，冷期间 decode 间隔 max 从 ~1.6s 变成 ≤1.6s 但每 1.9s 出 16 个 token；短命中 TTFT p95 +0.3s 内 | `decode_gap_during_cold p95` 仍 ~1.5s 但 `stream_tpot` ≤0.09；`short_hit_ttft p95` ≤1.0；`cold_ttft` 比 026a 多 ≤20% | 与 Q2 比选 K；若 short_hit p95 >1.5s，K 太大 |
| Q2 | `041-ifx_s0_i24` | 干扰探针 / 8 min | S0 + interval 24 | 同上，floor ≈13 tok/s | stream_tpot ≤0.08；cold_ttft +25% 内 | 取"stream_tpot ≤0.085 且 short_hit p95 最小"的 K 进 Q3 |
| Q3 | `042-ladder_s0_iK` | 梯子 N18→22→26 / 35–105 min | S0 + interval K（Q1/Q2 胜者） | 闭环里 decode 步守恒，只是均匀化：TTFT ≈026，tpot_p95 0.219→≤0.10，tpot 均值 ≈0.08，未命中 tok/s ≈8.4k | **N18 过**；N22：chain_start p95 落在 28–35s（允许超 22 条）五五开 | 过 N18 ⇒ 这就是明天提交 A 的候选；N22 失败看是哪道门：chain/turn ⇒ 需求侧（Q6）；intra ⇒ K 减半重跑一次（唯一允许的参数回调） |
| Q4 | `043-ladder_s0_iK_114` | 梯子 / 35–105 min | Q3 胜者 + 114（env `SGLANG_AX_INDEXER_ROW_SHARD=1`，114 已在 028b 上 12/12） | 长前缀每 token −15~33% ⇒ 未命中 tok/s 8.4k→9.5k+，chain/turn 尾巴缩 | 四门 margin 变大；若 Q3 在 N22 失败，此处过 N22 | 过 ⇒ **稳定线 S1 = Q4 配置**，明天提交 A；任何门退步 ⇒ 114 有 TP8 负效应，退回 Q3 |
| Q5 | `044-dcp116_probe`（已有 `dcp116_probe.sh`，改成 S1 参数） | 探针 / 45 min | S1 + 114 115 116 + `--dcp-size 8` | 116 后 latent 真分片，容量 ×4.4，前缀命中不越界，数值与非 DCP 引擎一致 | 无 illegal memory；NUMCMP wrong=0；hiload 命中 cached>0；冒烟 ≥10/12；KV 行数 ~450k/卡（逻辑 ~3.6M） | 过 ⇒ Q8 排 DCP 梯子（N26 的容量前提）；不过 ⇒ DCP 出局，N26 靠需求侧 |
| Q6 | `045-ladder_s1_fam`（**需要代码**，见下） | 梯子 / 35–105 min | S1 + 家族前缀快照（141） | system-tools-changed 类从 7.97M 降到 ≤3M；总未命中 ≤7M；decode 份额升到 40%+，tpot 均值降到 ~0.06，chain/turn p95 下降 | 未命中 tok/s 不变（饱和）但墙钟明显缩短；**N22 过，N26 可试** | 过 ⇒ 明天提交 B；不过但未命中确实降了 ⇒ 看哪道门，通常是 KV 淘汰（转 Q8）；未命中没降 ⇒ 审计脚本（W-A）的结论错了，回到审计 |
| Q7 | `046-ladder_s1_mamba400` | 梯子（只跑 N18）/ 35 min | S1 + `--max-mamba-cache-size 400`（KV 约 −20%） | 状态槽压力是 027/026 多算 0.7M 的原因；KV 少一点在 N18 不致命 | 未命中 −5% 以上、tpot 略降；若 KV 峰值触顶则 chain 尾巴变差 | 决定 N22/26 的 KV 与状态槽如何分配；若 Q6 已把需求压下来则可跳过 |
| Q8 | `047-ladder_s1_dcp8` | 梯子 / 35–105 min | S1（或 Q6 胜者）+ DCP8（114 115 116） | KV 不再是 N22/26 的瓶颈；decode 稀疏注意力每卡只读 1/8 KV | KV 峰值 <0.5；chain_start 尾巴（整段丢缓存的 17s 请求）消失；冷 prefill 每 token 与 S1 持平或 −10% | 过 ⇒ N26 主力配置；冷 prefill 变慢 >20% ⇒ DCP 只在 N≥22 用 |
| Q9 | `048-ladder_best_mtp` | 梯子（只跑最高通过档）/ 35 min | 当前最佳 + MTP（160；140 必须关，除非 W-C 完成） | 同档比 tpot：MTP 把 tpot 均值降 30%，但到达率上升，TTFT 门变紧 | 若该档仍过 ⇒ tpot 从 ~0.06 降到 ~0.045；否则 MTP 不进提交 | 只影响同档名次；**排最后**，不许提前 |

时间：Q1+Q2 16 + Q3 ≤105 + Q4 ≤105 + Q5 45 + Q6 ≤105 + Q7 35 + Q8/Q9 各 35–105 ≈ 8.5–9.5h。Q6 的位置由 worker 交付时间决定（可与 Q5 互换）；Q7 可砍。

**需要并行准备的代码（现在派 worker）：**
- **W-A 审计（CPU，pod `/tmp/ax/codex`，luna/Sonnet，1–2h）**：对 026 与 034 的 raw，按派发顺序为每个 `system-tools-changed` / `chain-head` 请求计算与所有更早请求的精确 LCP（复用 `scripts/analysis/prefix_reuse_potential.py` 的渲染与分词），对照 `cached_tokens`，把差距分类：(a) LCP 处无状态（不在 16384 倍数、不在最后角色边界）；(b) 曾有状态但被淘汰（server.log 中该前缀的最近一次命中时间与 KV/mamba 高水位）；(c) 树里根本没有该路径。输出一张"损失原因 × token 数"表。**Q6 的代码方向由它决定。**
- **W-B 141 家族前缀快照（astra/Opus，3–5h）**：在 140 上加第三个导出点——请求内**第一个**角色边界（system+tools 末尾，向下取整到 G 网格），只在该边界位置树中尚无状态时申请额外槽；淘汰顺序里把"被 ≥2 条链命中过的节点"排在最后。若 W-A 判定是 (b) 淘汰为主，则改为只做淘汰保护 + 4096 网格状态（更小）。交付前提：CPU 树测试 + 8 卡冒烟 12/12 + 真实文本 numcheck（029 的随机 token 判据无效，改用 026 raw 里的 6 条真实 prompt 的 logprob 差）。
- **W-C 140 + MTP 共存（可选，terra）**：只有 Q9 需要。

**不做的**：prefill CP（A100 无 deepep，已死）；170 BCG（16k 块下只值 4%，且 TP8 未验）；MoE kernel；HiCache；DP attention；纯参数扫描。

---

## 8. 流程建议：让预判不再与结果背离

1. **每档自动输出"闭环预算表"**（附录 A 脚本，接进 `dev_ladder_template.sh` 的 diag）：gap / TTFT / decode 客户端秒、未命中 tok/s、prefill 忙碌估计、按输出长度分桶的 tpot、按门的 queue/exec 分解。上 8 卡前，用这张表的语言写预判（"未命中 tok/s 不变，decode 秒 −20%，TTFT 秒 +?"），跑完逐项对。
2. **系统级改动先过干扰探针再上梯子**（8 分钟 vs 35 分钟）；探针直接给出 decode 间隔和短命中 TTFT，能否定 90% 的坏想法。
3. **一次一个变量**，且每个变量的判据在任务名里写死（`ladder_s0_iK` 只回答"interval 是否修好 tpot_p95 且不伤 TTFT"）。
4. **不在混合变量的失败上杀机制**：027 杀了"间隔"，028/034 让 MTP 背了闭环的锅。凡是要"放弃"一个方向，必须能指出单变量证据。
5. **缓存复用按边类型做成门指标**：每档报 session_start / turn_start / system-tools-changed 三类的"实际未命中 / 冻结未命中"比值，这比 KV 使用率更能说明缓存在不在工作。
6. **先说清"N 第一、tpot 第二"的取舍**：任何降 tpot 的改动都要同时报 TTFT 秒的变化，不允许只报 tpot。
7. 开发机替身只出正确性结论。

---

## 附录 A：pod 上的原始统计【实测】

脚本：`/tmp/ax/codex/fable_runstat.py <run_dir> N18`（只读；用 harness 自己的 `s1_common.in_ttft_gate` 分桶）。摘录（完整输出在本次会话日志）：

```
026 N18: wall=1270s uncached_actual=10.63M expected=11.98M output=216k  uncached_tok/s=8370
  client-seconds: gaps=3028 ttft=2222 decode=16396 (N*wall=22860)
  fast_intra n=328 p95 2.44 | queue p95 1.27 exec p95 1.35 | uncached p50 1742
  chain_start n=314 p95 25.43 | queue p95 21.5 exec p95 9.8 | uncached p95 110775
  tpot mean 0.0829 p95 0.2189; >0.1: 18.8%;  out<100: mean 0.122 p95 0.554
  prefill batches 1234, mean chunk 8632 (<=1k:98, 1k-4k:205, 4k-8k:478, >8k:453); queue-req mean 1.3 max 13
034 N18: wall=1160s uncached=9.90M uncached_tok/s=8534; gaps 3028 ttft 3836 decode 12375
  fast p95 13.16 (queue p95 12.56, exec 0.96); chain p95 35.5; tpot mean 0.061 p95 0.210
028 N18: wall=1191s uncached=9.81M tok/s=8236; ttft 5682 decode 11370; fast 5.45 chain 78.5; tpot 0.053/0.082
028b N18: wall=1196s uncached=9.73M tok/s=8133; ttft 6788 decode 10628; fast 18.1 chain 71; tpot 0.048/0.077
027 N18: wall=1728s uncached=11.37M tok/s=6579; ttft 9759 decode 17126; 5064 batches mean chunk 2249
025b N10: wall=1456s uncached=9.99M tok/s=6857; ttft 2098 decode 9056; tpot 0.047/0.13 (out<100 p95 0.57)
```

## 附录 B：需求侧【实测 + 模拟】

开发集冻结未命中按 (phase, edge)：session_start/chain-head 2.24M；intra/append-only 1.51M；**intra/system-tools-changed 5.87M；turn_start/system-tools-changed 2.11M**；其余 <0.3M。合计 11.98M。

`prefill_waste.py` 实测（026 / 034 / 028）：session_start 1.01 / 0.73 / 0.69M（冻结 2.24）；turn_start 0.84 / 0.72 / 0.71（冻结 2.29）；**intra 7.75 / 7.45 / 7.41（冻结 7.40）**；context_reset 1.03（冻结 0.04，多算 1M，是另一个可查的点）。

`evidence/R8/prefix_reuse_rows.json`（与任何更早请求的 LCP，未计淘汰）按策略的总未命中：frozen 11.98M；grid16384 12.43M；grid8192 10.71M；grid4096 9.77M；exact 8.44M；按类：system-tools-changed 7.97 → grid16384 2.61 → grid4096 2.25 → exact 2.01；chain-head 2.24 → 1.03 → 0.52 → 0.35。intra/append-only 在网格策略下反而变差（8.49M vs frozen 1.56M），说明末尾状态必须保留——现实策略 = 末尾状态 + 角色边界 + 网格，冷类取 grid4096、append 取 frozen ⇒ 约 5.7M（−52%，上限）。

## 附录 C：其他核实
- prefill CP 崩溃根因【实测，`012g/server.log`】：`NotImplementedError: Runner backend MoeRunnerBackend.MARLIN requires a fused func for a2a backend deepep`——zigzag DSA CP 强制 `moe_a2a_backend=deepep`，A100 无此路径。
- `--prefill-decode-interval` 语义【实测，`base_exact/.../scheduler.py:1256-1276, 3456, 3500`】：每个 extend 批之后强制 K 轮 decode，计数器全局；120 的显式 interval 优先于它自带的 1 轮交替。
- 026 与 034 缓存差异中 140 开/关不是主因（034 关 140 反而未命中更少），块大小（状态网格 8k vs 16k）与状态槽（默认 vs 200）更可能【推断，见 Q7】。
