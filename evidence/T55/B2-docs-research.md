# T55 / B2-docs-research：入口文档、research/、plans/、docs/ 清理审查

审查人：Claude subagent（Opus），2026-09-24（容器日期 09-23）。只写本文件；未改任何其它文件、未提交。
判断依据：`llm-challenge-arena-v1/task.md`（行号）、F93/R19，以及审查期间新落盘的 F94–F96/R20（T56/T58），原始数据（`s1-dev/data/dev-combined-v1/requests.jsonl`、`s1-dev/run_dev.py`、`s1-dev/harness/*`），底包源码。每个 DELETE 都 grep 过被哪些保留文件引用（§3）。
范围：`README.md`、`HANDOFF.md`、`board.md`、`AGENTS.md`、`CLAUDE.md`、`rule.md`、`research/README.md`、`research/claude/*`、`research/shared/*`、`research/archive/*`、`research/codex/*`（Codex 的文件只给结论）、`plans/*`、`docs/*`。

本轮自己核实过的数据（可复现）：
- dev 集 `max_output_i`：p50 198、p90 554、max 5644，722 条中 199 条 >240；harness 按每条请求发送 `max_new_tokens`（`s1_loadgen.py:103,258`）。
- `s1-dev/run_dev.py:46–56,232`：`flush_kv` 只要 HTTP 不抛异常就算成功，返回值被忽略，不检查 JSON `success`（与 R19 S3 一致）。
- `data/all_att_2026-09-23.json`：551 条，最新 createdAt 2026-09-22T23:56Z；最高 N 仍是 22（LewyM 0.02728）。决策 39 所说"第一名 CalvinCao N26/0.0551"不在这份数据里（用户提供的榜单，见 §5 UNSURE-1）。
- 服务端 `scripts/check_records.py:27–36` 要求 `logs/codex/` 里的每个 `W*.log` 在 board 实例表都有一行。所以删 W1–W14 行之前，要先改这条检查。

---

## 0. 最会误导后续智能体的 10 条

1. **`HANDOFF.md` 全文**。README:7 要新会话"先读"它。内容写的是 09-23 早上：服务"在 Trisol 准入队列里"、`autostart.sh` 守护、"暂不正式提交"（实际 45979/45980 已交，`notes/submissions.md:30–31`）、旧补丁栈、队列 013b–025、以 `analyze_run.py` 为分析工具（它对空 raw 判 ALL_PASS，R19 S1）、优先级照 R8 §8。结论：DELETE。
2. **`research/claude/base/00-summary-mainline.md`**。`CLAUDE.md` 要求每次会话、每次压缩后先读它，但它仍是 09-22 的"新主线 M0–M4"：
   - M0 仍列候选 fa3，而 fa3 只支持 Hopper（F57）；
   - M3 仍列 130，前端并非瓶颈（F91）；
   - 顶部更正表写"先耗尽的是 KV""KDA 9.7GB"。R18 §1.5/§3 的结论是：10.05GiB，哪个池先满尚未证实；
   - 没有 F93/F95/F96，也没有 S0 的定义。

   结论：整篇 REWRITE，改成唯一的现行状态文档（见 §7）。
3. **`research/claude/R13_fable_direction_review.md`**。以下头条结论已被降级或推翻：
   - "8.1–8.5k tok/s 就是机器容量"（R19 §4.2）；
   - "闭环 ⇒ MTP 降 N"（R19 §4.1）；
   - "system-tools-changed 零复用"：依据是 `prefill_waste.py` 的按 phase 汇总，R19 S2 已指出其问题，F92 用同一份数据也得出相反结论；
   - 从榜单推断 CalvinCao 的做法（R19 §4.3）。

   另外，§7 把 S0 写成"补丁 … 140 120"，而现在的 `patches/120` 已是 v3，不是 026 用的 v2（`_context-0924` 第 7 条）。结论：DELETE。
4. **`research/claude/R8_next_directions.md`**。README:38 称它为"方向与问题清单"，research/README:9 标"现行"。问题：
   - L35 "所有 N=18/22 选手的约束门都是 chain_start"，属于推断；
   - L88 F1 的"开发集实测 −56%"，数据来自 `prefix_reuse_potential.py`。该脚本按原轨迹顺序排序、不计淘汰（R19 S7），只是情景推演，不是实测；
   - P0–P3、§8 的排序已被决策 39 取代；
   - R18 §9 逐行列出了它 9 处错误。

   结论：DELETE（有用部分先并入 00，见 §4）。
5. **`research/shared/pipeline.md`**。AGENTS.md:17 把它当"评测、打镜像、提交的流程"。其中 L33 写着「用完立刻删：`trisol inference delete`」，直接违反"服务不停不删"的规则（决策 34、用户指示）。此外阻塞项、Dockerfile 草图（000+001、v0.5.20 线）都已作废。结论：DELETE。
6. **`README.md` 目录表和状态段**：
   - L12、L22、L23 把实验引向 `tests/L2.md`、`scripts/l2.py`、`tests/queue/`，并写"守护进程在 GPU 机自动跑"。正是这个守护进程删掉了 `lh-arena-sess-a`，它已退役（决策 34 补记），B3 也判这些文件 DELETE；
   - L19 把 `l3_0922e/f` 称为"正式提交 A/B 实际运行的代码"，但这两个镜像从未部署成功（`submissions.md:20`）；
   - L32–38 的状态整体过时。

   结论：REWRITE。
7. **`research/claude/R14_fable_eval_review.md`**。以下说法都不成立或已被取代：
   - "tpot_mean 的 58–73% 是预填充停顿"，算法是 `1−p10/mean`，R19 §4.3 指出这不是实测份额；
   - "在飞 prompt 之和 = KV 驻留 ⇒ N22/26 撞容量墙"，R19 §4.3 已否定；
   - "intra 失败主因是缓存丢失"，依据是冻结代理值。F95 用真实 LCP 重算：可修缺口上限 850,432，只占全部 prefill 的 8.0%；10 条 fast 超时里，6 条是缓存与队头阻塞并存、3 条以缓存为主。

   结论：DELETE（burst、前端、dev↔正式、截断回放几点先并入，见 §4）。
8. **`base/02`、`base/04` 正文里的运行时断言**（`CLAUDE.md` 同样要求每次读 01–04）：
   - 04:46 "TPOT does not currently bind the score"：tpot_p95 是无余量硬门（task.md:519,577），S0 在 N18 以 0.219、在 N22 以 0.296 挂在这道门上（F96）；
   - 04:80 的第一条杠杆推荐 fa3（Hopper-only）；
   - 04:82 "no FP8 Marlin MoE path"：111 就是把底包已有的 Marlin FP8 MoE 接上（F58）；
   - 04:71 "36k 冷预填充 1–2 s"：按 F76 约 1 万 tok/s 推算约 3.6 s，属粗推；
   - 02:57 "Outputs here are ≤240 … unverified whether the harness sends max_tokens"：实际 p50 198、max 5644，harness 按条发送 max_new_tokens，见上方核实。

   结论：REWRITE。
9. **`board.md`「进行中」和实例表**：
   - L37 仍写 035 在跑，实际已结束（F94/F96）；
   - L41 T50/W24 已由 T50b 完成；
   - L9、L43 的 T37 指向 `R16_submission_evidence_audit.md`，该文件从未存在，dispatch 里 T37 仍是 accepted；
   - L44 写"45734/45735 等待出分"，它们在部署阶段就失败了；
   - L45 写 sess-a 和 tmux 守护进程，前者已被删、后者已退役；
   - L46 M0 早已完成；
   - L34 T56 已完成（L51）。

   结论：REWRITE。
10. **`rule.md` §0 L11**：
    - "目标：N=26，或 N=22 且 tpot_mean<0.0273"。按决策 39，第一名已是 N26/0.0551，这个目标过时（UNSURE-1）；
    - 把排名规则写成"能过全部硬门的最大档"，缺了关键规则：从 N=10 起爬，过了 +4、没过 −4，在已过档之上首次失败即停（task.md:500–515）。

    结论：REWRITE。

（`R11`、`R12` 也有严重误导，未进前十只是因为没有入口文档指向它们。见 §1.3。）

---

## 1. 逐文件结论

### 1.1 根目录入口文档

| 路径 | 结论 | 理由 |
|---|---|---|
| `README.md` | REWRITE | 目录表 L12/19/20/22/23/25 与现状不符；L7 指向要删的 HANDOFF；L32–38 状态过时（服务、提交、F 编号、R8）。见 §2。 |
| `HANDOFF.md` | DELETE | 全文是 09-23 早上的状态：准入队列、autostart、暂停提交、补丁栈缺 106/114/160/170、队列 013b–025、analyze_run。真实状态以 F94–F96、`submissions.md:30–31`、决策 39 为准。有用的规则已在 rule.md §4 与 AGENTS.md 里。 |
| `board.md` | REWRITE | 「进行中」L37、L41–47 和实例表 L9、L34 与事实不符；「已完成」L63–109 全是已退役路线的历史（v0.5.20、E1/E2、SPF、模拟器、守护进程、session_a），其中 T11、T12 还各出现两次（L84/86、L85/87）。 |
| `AGENTS.md` | REWRITE | L17 指向 `research/shared/pipeline.md`（含删服务指令）和 `tests/L2.md`（已退役的守护进程流程）；缺少 R19/`_context` 的评测方法规则。红线部分与 task.md:699,707–708 一致，保留。 |
| `CLAUDE.md` | KEEP | 只做导航。它要求先读 00 和 01–04，所以 00 必须按 §2 改写，否则每次会话都会先读到过时主线。 |
| `rule.md` | REWRITE | §0 L11 的目标与爬坡描述见 Top10-10；L34 的 shared 行、L40"双方确认"（与 §4.6、决策 25 矛盾）、L104 的 archive 指针、L110–130 的旧变更记录都需要改；L76 的 b0–b3 为 UNSURE-2。§4 红线本身与 task.md 一致。 |

### 1.2 `research/README.md` 与底包源码地图

| 路径 | 结论 | 理由 |
|---|---|---|
| `research/README.md` | REWRITE | L9–L15 把 R8、R7 标为现行，并列出要删除的 R1–R6、shared、archive；缺 R10、R19、R20。 |
| `research/claude/base/00-summary-mainline.md` | REWRITE（整篇） | 见 Top10-2。行级错误见 §2。 |
| `research/claude/base/01-request-path.md` | REWRITE（小） | L68 末句"harness 的 {success:true} 检查会失败"：公开 harness 并不检查 JSON（`run_dev.py:46–56,232`，R19 S3），要求来自 task.md:226；L76 仍写 MambaRadixCache，而实际是 UnifiedRadixCache（03 §0）；L120"前端最大开销 0.1–2 s"已被 F91 实测否定。 |
| `research/claude/base/02-scheduler.md` | REWRITE（小） | 页首 ③"先满的是 KV（N6 峰值 50%）"与 R18 §1.1/§3 矛盾；② 缺"patches/120 现为 v3、S0 用 drafts v2"；L57 的输出 ≤240、harness 未发 max_tokens 两点均已被数据否定；L133 建议"让 harness 发 max_tokens"，但负载冻结，做不到（task.md:355,699）。 |
| `research/claude/base/03-hybrid-cache.md` | REWRITE（小） | R18 §9 已给出核实过的更正，尚未落实：L103–107 容量估算（实测 584 槽、10.05GiB、KV 943,360、上限 116）；L139/L240 "never served""Low risk"；L206/L208 "No new risk"；L221 "≈1.5× KV"（应为 ≈1.267×）。 |
| `research/claude/base/04-model-kernels.md` | REWRITE | 页首更正没有写回正文。L15/L29/L46/L71/L80–L82/L85–L86 的错误见 Top10-8 与 §2。 |

### 1.3 `research/claude` 的 R 报告

| 路径 | 结论 | 理由 |
|---|---|---|
| `R1_model_and_engines.md` | DELETE | 基于 v0.5.20（`src/sglang`），不是底包（决策 29）。§0.3"主办方镜像必带私有补丁"被 F53/F54/F56 推翻；§3.1 "~2.0M tokens"，实测 943k/1.57M（F80）；§4 DP8 已放弃；§5 待办早已完成。模型结构已在 base/04 §1，HiCache/mixed-chunk 禁用已在决策 #5。 |
| `R2_serving_techniques.md` | DELETE | 前提"v0.5.20 = 底包"是错的（L3，F53）；排序实验（移植 #40024 等）按决策 29 作废；有价值的放弃项已在 R8"已明确放弃"清单，该清单将并入 00。 |
| `R3_prior_art_en.md` | DELETE | 是 D1 方案 A/B 的先例调研，D1 已由 101/140 落地；D1/D2/D3 这套标签已随 directions.md 作废。 |
| `R4_similar_competitions.md` | DELETE | AgentX 等外部参考，其中 +141% 已被页首更正；"D2/D3 补充"已过时；Codex R8/R13 与 Claude R9 覆盖更全。 |
| `R6_harness_template_review.md` | DELETE | 采纳表与现状矛盾：写"不另建 histories"，现已有 `docs/histories/`；写"board-archive.md"，从未建立。流程已落在 AGENTS.md。 |
| `R7_top_players_analysis.md` | DELETE | 09-22 榜单；§5 结论"tpot_p95 never binds"、"MTP 只作同分决胜"与我们的实测相反（026/035 挂在 tpot_p95）；从榜单推断对手做法，R19 §4.3 认为推不出；每日 3 次与决策 21（2 次）矛盾。 |
| `R8_next_directions.md` | DELETE（先合并） | 见 Top10-4。§1 规则摘要、§2 负载形状（本轮已核对 p50/p90/max）、"已明确放弃"清单要先并入 00。 |
| `R9_upstream_since_base.md` | REWRITE（小） | 上游候选清单仍有效（基于底包）。L367"大头仍在…INT8 W8A8"：INT8 已被 F70 否定；L88 #38522 需注明"已移植为 170"。 |
| `R10_prefill_fixed_overhead.md` | REWRITE（小） | T51 实测报告，方法与数据可复现，保留。L9、L18、L20 的"真机约 150ms"来自 F85 墙钟反推（INFERRED），需要与 F87 026g 的直测（P=0 时约 107ms + 60µs/token）并列；§6 需注明 170 的现状（F87/F88/F90）。 |
| `R11_retro_unverified_assumptions.md` | DELETE（先合并 §1） | §2.1"冒烟 <11/12 退出"，现为 6/12 且不是能力门（决策 39）；§2.2 把 numcheck/numcheck_cmp 当正确性闸门，但 numcheck_cmp 对截短、分叉都判 ok（R19 S4、F93）；§3 的"补齐错位"推断已被 F90 纠正。 |
| `R12_full_experiment_review.md` | DELETE（先合并） | "一级问题 = prefill MFU"已被 R13/R14 否定；§0 #12"026/028 缓存丢失几乎相同"用的是 cachecmp，R19 S2 已指出其问题；§2"实际比冻结少 13%"没有按链首拆分（F93）；§4 计划已被决策 39 取代。§0 的 12 条预判表和 §1.3 的疑似误杀清单值得留下。 |
| `R13_fable_direction_review.md` | DELETE（先合并附录 C） | 见 Top10-3。附录 C 的 prefill CP 崩溃根因（012g 日志）和 interval 语义要先迁入 00 或 findings。 |
| `R14_fable_eval_review.md` | DELETE（先合并） | 见 Top10-7。§2.5 burst、§2.6 前端、§3.1 dev↔正式、§5 截断回放（仅作筛选）要先迁入。 |

### 1.4 `research/shared`、`research/archive`

| 路径 | 结论 | 理由 |
|---|---|---|
| `research/shared/pipeline.md` | DELETE | 见 Top10-5。打镜像与提交的约束 task.md:272–434 写得更权威；我们的实现在 `scripts/build_image.sh`、`scripts/submit_official.sh`，台账在 `notes/submissions.md`。 |
| `research/archive/STATUS-2026-09-22.md` | DELETE | 自称已归档；按政策不留归档目录。内容是 v0.5.20 替身、SPF、"只有 LewyM N=22"。 |
| `research/archive/directions.md` | DELETE | 旧方向清单（D1–D3、"~2M token"）。 |
| `research/archive/rule-handoff-2026-09-22.md` | DELETE | 自称"已过时，基于底包 = v0.5.20 假设"。 |
| `research/archive/R5_patch_review.md` | DELETE | 对 v0.5.20 的 000/001 草案的审阅；001 在 `patches/v0520/`，B3 判 DELETE。 |

### 1.5 `research/codex`（只给结论；由 Claude 通知 Codex 后处理）

| 路径 | 结论 | 理由 |
|---|---|---|
| `README.md` | REWRITE | L5、L7、L9、L13、L14、L28 链到的 R1/R2/R3/R4/R9/R10/R12 都已在 `archive/`，链接已失效；L19 链到 `patches/001`（已移到 v0520，B3 判删）；L30–32 是 E1/archive 说明。应改成只列 R17–R20 的 5 行索引。 |
| `R5_dp_memory_accounting.md` | DELETE | 基于 v0.5.20 源码；DP 已放弃；几何数字在 base/03 §3 与 R18 §7 已有。 |
| `R6_prior_art_cn_github.md` | DELETE | D1 时期的先例调研，已由 101/140 落地；rid 复用事实已在 F23。 |
| `R7_kda_internal_checkpoints.md` | DELETE | Plan B 的设计调研，已实现为 140（设计见 `patches/140-*.md`）。 |
| `R8_agentx_checklist.md` 与 `R8_agentx_pr_inventory.json` | DELETE | 存在性核对针对 v0.5.20 `src/sglang`，不是底包，容易误读为"底包已有"；上游清单已被 Claude R9（基于底包）取代。 |
| `R13_agentx_mlperf_reading.md` | DELETE | 基于 v0.5.20，按 D1/D2 分项；没有现行动作依赖它。 |
| `R14_recent_pr_watchlist.md` | DELETE | 09-22 的 PR 快照，被 R9（09-23，基于底包，逐项核验）取代。 |
| `R15_base_source_exploration.md` | DELETE（先合并 §0.3–0.4） | 基于 407 文件的部分提取；"h 精度不能确认"已被 base/03 §7 解决；102 已被 140 取代。routing key = sys/tools 前缀族（136 会话、156 族）要先并入 base/01 §4。research/README:13 把它标"现行"（UNSURE-7）。 |
| `R17_nextn_sm80.md` | KEEP | 基于底包，是补丁 160 的现行路径说明，被 `patches/160-nextn-sm80.md:3`、`tests/TEST_PLAN.md:262` 引用。 |
| `R18_cache_loss_and_capacity.md` | REWRITE（小） | 主体标注严谨，与 R19 一致，保留。§9（L313–343）是给 Claude 的更正清单：base/00/03/04 按本文 §2 改完、R8 删除后，§9 就成了指向已删内容的待办，应删除。 |
| `R19_progress_and_cache_review.md` | KEEP | 本轮判定标准（F93）。 |
| `R20_true_lcp_attribution.md` | KEEP | T56/F95，真实 LCP 归因，取代 R14 §2.1–2.2 和 F91 的代理数字。Claude 对它的独立复核尚未记录（UNSURE-8）。 |
| `archive/`（21 个文件：R1–R4、R9–R12、README、SHA256SUMS、audit.*、contract_probe、prepare_patches、remote_smoke、make_reduced_checkpoint、test_research、fixtures/*、patches/0001–0002） | DELETE（整目录） | 政策是"不建归档目录"；自身 README 已声明"均非当前候选、不要直接运行"。grep 结果：`scripts/`、`scripts/pod/`、patches、jobs 均未引用。 |

### 1.6 `plans/`

| 路径 | 结论 | 理由 |
|---|---|---|
| `active/2026-09-23-8card-selftest-ladder.md` | DELETE | 标为 active，但里程碑 024–026 已过期；判据是 analyze_run 的正式估算（R19 S1）；决策记录写"首档失败再下探 14/10"，与用户"不下探"的指示相反（提交 2276240）。 |
| `active/2026-09-23-offline-queue-3h.md` | DELETE | 已执行完毕，结果在 F89/F90；"[ ] 收结果并写 F89+"早已完成。 |
| `active/T50-dcp-prefix.md` | DELETE | T50/T50b 已完成（F90）。唯一未了项"116 的 8 卡复验"要移到 board，并注明 `dcp116_probe.sh` 不强制门槛（R19 S5）。 |
| `completed/102-role-track.md` | DELETE | 自称 superseded by 140。 |
| `completed/2026-09-22-d1-role-boundary.md` | DELETE | 自称 dropped（v0.5.20/SPF）。 |
| `completed/2026-09-22-evaluation-strategy.md` | DELETE | 自称 dropped；它引用的 session_a/守护进程流程已退役（B3 判删）。 |
| `completed/2026-09-22-first-8gpu-session.md` | DELETE | 自称 dropped；stock/v0.5.20 计划。 |
| `completed/112-sm80-indexer.md`、`113-prefill-indexer.md`、`160-nextn-sm80.md`、`2026-09-23-T45-kda-dual-snapshot.md`、`T47-indexer-runtime-shapes.md` | KEEP | 对应补丁都在 0923a 栈里，是准确的完成记录，有助追溯（UNSURE-6）。 |
| `prompts/T41`–`T50`（11 个：T41、T42、T43、T44、T45、T46、T47、T48、T49-brief、T49-codex-astra、T50） | DELETE | 一次性派发文本，任务全部完成。里面仍要求先读 R8、HANDOFF、`dev_template.sh`、"正式规则估算"，重用会误导。 |
| `prompts/T54a`、`T54b`、`T55`、`T56`、`_context-0924.md` | KEEP（临时） | 属现行任务。T56 已完成（F95）。T54/T55 结束后，把 `_context-0924.md` 并入 00 再删这 5 个文件，避免两份事实清单各自漂移。 |
| `templates/plan.md` | KEEP | 模板。 |

### 1.7 `docs/`

| 路径 | 结论 | 理由 |
|---|---|---|
| `histories/2026-09/20260923-0150-sm80-bringup-tilelang-marlin.md` | KEEP | 与 F57/F58/F59 一致。 |
| `…-0230-fix-101-double-partial.md` | KEEP | 与 F62 一致。 |
| `…-0400-sm80-indexer-kernels.md` | KEEP | 与 F63–F68、F72 一致。 |
| `…-0500-pod-verify-and-analysis-tools.md` | REWRITE（小） | L19 把 analyze_run 描述为"硬判定 + 正式规则估算"，未注明缺陷（R19 S1）。 |
| `…-0650-sm80-sparse-attn-dcp.md` | REWRITE（小） | L21"KV 逻辑 ×7.8"、L25"DCP 同时解决 KV 容量"已被 F90 纠正。 |
| `…-0720-scheduler-120-v2.md` | REWRITE（小） | L21、L30 指向将删除的 v1 草稿（B3）；缺"patches/120 现为 v3、S0 用 drafts v2"。 |
| `…-0830-infra-safety.md` | KEEP | 与决策 34、rule §4.7 一致。 |
| `…-0900-docs-after-8card.md` | KEEP | 属历史记录：它列出的被改文件本身会被删或改，但这条记录不误导。 |
| `…/T57-readonly-log-collector.md` | KEEP | 新增的 `collect_run_logs.py` 记录（Codex）。 |
| `histories/template.md` | KEEP | 模板。 |

---

## 2. REWRITE 明细（行号，原句不超过 20 字，改成什么）

### README.md
- L7「HANDOFF.md — 最新交接」：删去此项，改为"`research/claude/base/00-summary-mainline.md`：唯一现行状态与已核实事实"。
- L11「顶部有"当前事实基线"」：改为"F93 起为现行；旧条目以 F93/R19/F95/F96 的 supersedes 为准"。findings 顶部那块是 09-22 的旧基线，由 B1 处理。
- L12「tests/L2.md（L2 队列用法）」：改为"`scripts/pod/README.md`：8 卡实验只走 pod 队列（podq/qpush）"。
- L19「正式提交 A/B 实际运行的代码」：改为"0922e/f 从未部署成功（`submissions.md:20`）；现行镜像是 0923a，补丁清单见 `build/image/0923a.patches.txt`（45979/45980）"。
- L20「现行 000、101（RELEASE）」：改为"默认清单见 `patches/RELEASE`（Tier 1 共 8 个）；S0 精确栈见 00"。
- L22「l2.py（L2 队列）」：删去 l2.py，改列 `scripts/pod/`（podq、qpush、pread、pexec_codex、stopjob）和 `scripts/score_formal.py`。
- L23「守护进程在 GPU 机自动跑」：删除整句（守护进程已退役，决策 34）。
- L25「22b.json，523 条」：改为 `all_att_2026-09-23.json`（551 条）。
- L32–38「当前状态（2026-09-23」整节：删除，换成一行"当前状态只在 00 维护"。整节内容全部过时，逐行如下：
  - L34「能力冒烟 12/12」若要保留，须写"12 题冒烟只是崩溃筛，不是能力门"（task.md:484、决策 39）；
  - L36「正在测：120（调度）N6」已过期；
  - L37「正式提交暂停」：45979/45980 已提交；
  - L38 指向 R8，而 R8 要删除。

### AGENTS.md
- L17「research/shared/pipeline.md」与「tests/L2.md：L2 队列」：改为"实验：`scripts/pod/README.md`；打镜像：`scripts/build_image.sh` + `patches/RELEASE`；提交：`scripts/submit_official.sh` + `notes/submissions.md`"。
- 在"必须遵守"后新增 5 条评测方法规则，照 `_context-0924.md`「方法规则」：
  1. 门与分桶只用 `s1_common.in_ttft_gate`、`s1_score.evaluate`（经 `score_formal.py`）。
  2. 按 `idx_in_chain`、前驱是否在本 raw 中拆分。
  3. 冻结的 `uncached_expected` 只是代理，真实缺口用相邻 prompt 的真实 LCP（F95）。
  4. 空数据、不完整、缺指标、非零退出、flush 未确认一律判 INVALID。
  5. VERIFIED 必须附脚本、输入 SHA256 与输出。另加一句：dev 集只做 A/B（task.md:354）。

### rule.md
- L11「3 人 N=22（LewyM」「目标：N=26，或 N=22 且」：删掉榜单和目标数字，改为"目标与榜单快照见 00（决策 39）"。
- L11「能过全部硬门的最大档」：改为"从 N=10 起，过了 +4、没过 −4，已过档之上首次失败即停；n_at_slo 为通过的最大档（task.md:500–515）"。
- L34「research/shared/」这一行：删除（目录清空后不再存在）。
- L40「提交前需双方确认」：改为"Claude 按决策 25 批准，每次向用户汇报"（与 §4.6 一致）。
- L41「lb.json, all_att.json」：改为"`all_att_<日期>.json`，取最新快照"。
- L59「codex exec -m gpt-6-astra」：改为"worker 用 `scripts/codex_worker.sh`（决策 12），日志在 `logs/codex/`"。
- L76「profile 为 b0–b3」：UNSURE-2。确认 0923a 实际使用的 serve 脚本后再改。
- L94–100：编号重复（两个"4."、两个"8."），重新编号。
- L104「已归档到 research/archive」：删除这句。
- L110–130「变更记录」：删除 09-22、09-23 全部条目。它们引用 directions.md、D0/D1、"仅调研"等已废约定，git 里有历史。从 09-24 重新开始记。
- 新增 §4.8 评测方法红线（内容同 AGENTS 新节；或让 AGENTS 只指向这里）。

### board.md
- L6「截至 W12，全部 worker」：删除。
- L9「T37 A/B 正式提交证据链审计」「进行中」：改为 main 的实际状态（T53、T57、T58 均已结束），并在 dispatch 里关闭 T37（UNSURE-3）。
- L34「T56 … 进行中」：改为"已结束（R20/F95）"。
- L37「pod 035：S0…直接测 N22」：移到已完成，写"F94/F96：11 门过 10；TPOT p95 0.296 未过，TTFT 按统计余量估算通过"。
- L41「[Codex W24] T50 DCP」：删除（T50b 已完成）；另起一行"116 的 8 卡复验（探针须强制门槛，R19 S5）：待排"。
- L42「8 卡跑通：补丁 110」：删除（F59/F73）。
- L43「R16_submission_evidence」：删除（文件从未存在）。
- L44「45734（A）/45735（B）等待出分」：改为"45979/45980 待出分（约 09-24 09:20 UTC）"。
- L45「lh-arena-sess-a 排队等卡」：删除（该服务已被删，守护进程已退役）。
- L46「主线 M0：GPU 机装」、L47「~~[未决] D6」：删除。
- 已完成 L63–L109：删除，只留 L50–L62（T58、T56、T57、T53、T49、T48、T47、T46、T45、T44、T43、T41、T42）。
- 实例表 L10–L23（W1–W14）：要删须先放宽 `check_records.py:33–36`，否则保留。

### research/README.md
- L9「R8_next_directions」、L10「R7_top_players」、L11「claude/R1, R2」、L12「R3, R4, R6」、L14「shared/pipeline.md」、L15「archive/、codex/archive/」：删除这些行。
- L13「codex/R5, R6, R7, R8」：改为"codex/R17（MTP/160）、R18（缓存/容量）、R19（审阅标准）、R20（真实 LCP）"。
- 新增两行：claude/R9（上游候选）、claude/R10（T51 固定开销实测）。

### research/claude/base/00-summary-mainline.md（整篇改写）
- L1「（2026-09-22）」：改为"当前主线（2026-09-24）"。
- L3–12 更正表：删除，改成的内容直接写进正文。其中：
  - L8「方向对但比例不对」：改为"状态约占池 47% 原本正确；KDA 10.05GiB；哪个池先满未证（R18 §1.5/§3/§9）"；
  - L9「长空闲（244–300s）几乎全丢」：改为"真实 LCP 归因见 F95：可修缺口上限 850,432（8.0%）"；
  - L11「M1=补丁 120 正在 8 卡」：删除。
- L20–26 §一.1「运行时未证」「候选 fa3」：改为"已证实底包在 A100 起不来（F56）；由 110–113 与 tilelang 解决（F57/F59）"。L25 的「A/B（45734/45735）」删除（这两个提交从未部署）。
- L32「与领先者画像一致」：删除（从榜单推断对手做法，R19 §4.3）。
- L40「先耗尽的是 KV 不是状态槽」：改为"哪个池先满尚未证实（R18 §3）"。
- L45「25 万 token 可能 0.4–2s」：改为"实测前端不是瓶颈：客户端与服务端 TTFT 之差 p95 约 0.12s（F91）"。
- L48–52 §一.5：补充"MoE 实际走 111 Marlin W8A16；prefill BCG 由 170 移植（v2 修了 scatter，TP8 未复验）"。
- L54–65「二、新主线」表：整段删除，换成决策 39 的路线：
  1. 先修测量门（T54）；
  2. S0 直接测 N22，每次只改一个变量；
  3. 当前约束是 S0 的 N22 TPOT p95 0.296（F96）。
- L67–84「三、本地探索（L1）复盘」：删除，其中方法教训并入 rule.md 方法节。
- 新增各节，内容见 §7 第 3 项。

### research/claude/base/01-request-path.md
- L68「harness's {"success": true}」：改为"task.md:226 要求 2xx + JSON success；公开 run_dev 只看 HTTP、忽略返回值（`run_dev.py:46–56,232`）；000 已改为 JSON 且覆盖全 worker"。
- L76「for MambaRadixCache」：改为 UnifiedRadixCache，与 03 §0、§5 一致。
- L120「likely the largest front-end」：改为"已实测不是瓶颈（F91：客户端与服务端 TTFT 差 p95 约 0.12s）"。
- L123「stamped late」：补一句"000 已改为在 ASGI 入口打点（见 `patches/000-*.md`）"。
- §4 新增 R15 §0.3–0.4 的语义：S1 routing key 是 sys/tools 前缀族，不是 session；routing-key 策略、prefix_affinity、namespace 是三回事。

### research/claude/base/02-scheduler.md
- L4 ② 末尾：补"当前 `patches/120` 为 v3；S0 使用 `patches/drafts/120-sched-protect-chain-v2.patch`（sha ef1744b3…）"。
- L5「先满的是 KV（N6 峰值 50%）」：改为"50% 是扣除可淘汰后的占用，完整测量峰值 0.92；哪个池先满尚未证实（R18 §1.1、§3）"。
- L57「Outputs here are ≤240」与「unverified whether the harness」：改为"harness 按条发送 `max_new_tokens = max_output_i`（`s1_loadgen.py:103,258`）；dev p50 198、p90 554、max 5644"。
- L133「have the harness send」：删去该半句（负载冻结，task.md:355、699）。

### research/claude/base/03-hybrid-cache.md（落实 R18 §9）
- L103–107「Estimate, INFERRED」「~690 slots」「~138」：改为实测值：584 槽、KDA 10.05GiB、KV 943,360 token/卡、上限 116（F59、R18 §1.5–1.6）；并注明扩容参数后 KV 为 1.57M（F80），026 为 1.32M。
- L139「never served」：改为"当前匹配不用它；但后代状态或锁可能依赖祖先 KV（R18 §9）"。
- L206「which today insert no decode」：改为"取决于绝对序列长度是否跨过网格，不只看输出长度"。
- L208「No new risk」：改为"复制频率与 overlap 压力需实测"。
- L221「about 1.5× the KV tokens」：改为"固定总预算下约 1.267×（943k→约 1.196M，R18 §9）"。
- L240「Low risk」：改为"只能回收无后代、无锁的部分"。
- Levers 段首补状态：L2、L3、L5 已由 140 实现（双 fp32 快照与尾部优先淘汰）；140 拒绝 lazy 和 int8（R18 §8.2）。

### research/claude/base/04-model-kernels.md
- L3–4 页首：删除，内容写回正文。
- L15「resolves to MoeRunnerBackend.TRITON」：保留"底包原样"，补"sm80 实际经 111 接到底包已有的 Marlin W8A16 FP8 MoE（F58、F73、F74）"。
- L29「Possible hard failure on A100」：改为"已证实：原版在 A100 起不来（F56）；用 tilelang 后端（F57）"。
- L46「TPOT does not currently bind」：改为"tpot_p95≤0.10 是无余量硬门（task.md:519,577），S0 在 N18 为 0.219、N22 为 0.296 均未过（F96）；MTP 在 N18 实测 0.082（F89）"。
- L71「on the order of 1-2 s」：改为"F76 约 1 万 tok/s，36k 粗推约 3.6s（非实测）"。
- L80「--dsa-prefill-backend fa3」：删除这条杠杆（fa3 仅 Hopper，F57）。
- L81「Tune the Triton FP8 MoE」：删除（实际走 Marlin，MoE 只到峰值 27–30%，F70）。
- L82「Not available: there is no FP8」：删除（111 已接通）。
- L83 杠杆 4：补"chunk 16384 会把自动 mem_fraction 降到 0.646，须显式设定（F81）"。
- L85 杠杆 6：改为"160 已实现；8 卡实测见 F88/F89"。
- L86 杠杆 7：补"140 拒绝 int8 与 lazy（R18 §8.2）"。

### research/claude/R9_upstream_since_base.md
- L367「大头仍在我们自己的方向」：删除这句（INT8 W8A8 已被 F70 否定；R8 删除）。
- L88 的 #38522 行：补"已移植为补丁 170（T52）；v2 修了 scatter，TP8 未复验（F88/F90）"。

### research/claude/R10_prefill_fixed_overhead.md
- L9「每块约 150ms 的固定开销」：改为"替身拟合截距 119–147ms（P 为 0–180k）；8 卡 026g 直测 P=0 时约 107ms + 60µs/token（F87）"。
- L18「真机 | ~220ms」、L20「真机约 150ms」：标为 INFERRED（由 F85 墙钟反推，P 约 95k），并列出 F87 的实测值。
- L186 §6.1 下补状态："170 = #38522 移植；BCG 只覆盖 MoE/mHC/norm，KDA/DSA 仍是 eager（F87）；chunk 4096 下固定开销 77–92ms → 43–58ms（F88）；TP8 未复验 v2"。

### docs/histories
- `0500` L19「analyze_run（harness 硬判定」：补"已知缺陷：空 raw 判 ALL_PASS、缺 TPOT 记 0（R19 S1）；以 T54 评分器和 `score_formal.py` 为准"。
- `0650` L21「KV 逻辑 ×7.8」、L25「DCP 同时解决 KV 容量」：补"F90 更正：底包 norope latent 写入未分片，×7.8 不成立；116 修复后约 ×4.4，为算术值，TP8 未验"。
- `0720` L21「v1 存 patches/drafts/」、L30 v1 路径：注明 v1 已删除；补"`patches/120` 现为 v3；S0 用 drafts v2（sha ef1744b3…）"。

### research/codex/README.md（Codex）
- 删除 L5、L7、L9、L13–L17、L19、L23–L28、L30–L32（链接失效，或指向要删的文件）。只保留 R17、R18、R19、R20 的索引。

### research/codex/R18_cache_loss_and_capacity.md（Codex）
- L313–343「## 9. 请Claude更新的过时」：待 base/00、03、04 按上文改完、R8 删除后，删掉整节。

---

## 3. DELETE 引用检查

grep 范围是全仓库，排除 `.git logs src build refs s1-dev`。"保留的引用方"指不会被删的文件。

| 删除对象 | 保留的引用方（文件:行） | 会不会断掉在用的工具、job、补丁、提交追溯 | 处理 |
|---|---|---|---|
| `HANDOFF.md` | `README.md:7`；`docs/histories/…0150:13`（只是用户原话）；`…0900:30`（历史文件清单）；`plans/prompts/T55…`（任务文本） | 不会 | 改 README:7；histories 保留原样 |
| `research/claude/R1` | `notes/findings.md:35`（F6 标题引用）；`board.md` 已完成行（L106，本来就删） | 不会 | 由 B1 决定 findings 是否注明"文件已删，见 git" |
| `R2`、`R4` | `board.md:108`、`board.md:92`（在删除范围内） | 不会 | — |
| `R3`、`R6`、`R11`、`R12`、`R13` | 只有 `notes/README.md`（`scripts/index_notes.py` 自动生成） | 不会 | 删完后重跑 `index_notes.py` |
| `R7` | `research/README.md:10` | 不会 | 改 research/README |
| `R8` | `README.md:38`、`research/README.md:9`、`research/codex/R18:321–341`（§9 更正清单）、`docs/histories/…0900`（历史） | 不会 | 改 README 与 research/README；R18 §9 随后删 |
| `R14` | `notes/findings.md:555`（F91"详见 R14"） | 不会 | 由 B1 决定 |
| `research/shared/pipeline.md` | `AGENTS.md:17`、`research/README.md:14`、`board.md:97`（在删除范围内） | 不会；在用的是 `scripts/build_image.sh`、`scripts/submit_official.sh` | 改 AGENTS 与 research/README |
| `research/archive/*` | `rule.md:104`、`research/README.md:15`；`notes/dispatch.md` 历史行 | 不会 | 改 rule 与 research/README |
| `plans/active/2026-09-23-8card-selftest-ladder.md` | `patches/RELEASE:15`（Tier 2 的门槛说明） | 不会影响构建（只是注释）；但注释会指向不存在的文件 | 并入 B3 的 RELEASE REWRITE：删掉这个路径 |
| `plans/active/…offline-queue-3h.md` | 只有 notes/README | 不会 | — |
| `plans/active/T50-dcp-prefix.md` | `notes/dispatch.md:343,346`（T50/T50b 行） | 不会 | dispatch 历史引用可悬空（B1）；未了项转 board |
| `plans/completed/102-role-track.md` | `patches/README.md:6`（B3 REWRITE）；`scripts/archive/make_102_role_track.py:4`（B3 判 DELETE） | 不会 | 跟 B3 的改动一起处理 |
| `plans/completed/2026-09-22-*`（3 个） | `board.md:71`（在删除范围内）；`notes/dispatch.md:45,57,70`（历史） | 不会 | — |
| `plans/prompts/T41–T50`（11 个） | `notes/dispatch.md` T41–T50 各行（历史） | 不会 | dispatch 引用可悬空 |
| `research/codex/R5–R8、R13–R15` | `notes/findings.md:72,92,115–134,237,245,262`（出处注记）；`notes/dispatch.md` T4–T11、T34–T36 行；`board.md:64–66,84,86,93,96,102`（在删除范围内）；`research/codex/README.md`（重写） | 不会；patches、jobs、scripts 均未引用。R17 被 `patches/160-*.md` 引用，保留 | 由 Codex 删；B1 决定 findings 出处注记 |
| `research/codex/archive/`（21 个） | `research/codex/README.md:30`、`research/README.md:15`、决策 #17（历史）、`board.md:81`（在删除范围内） | 不会；`scripts/` 与 `scripts/pod/` 没有 import 或调用 | 由 Codex 删 |

删除后需要跑：`python3 scripts/index_notes.py`，然后 `python3 scripts/check_records.py`。check_records 的第 3 项只检查 `plans/active` 下现有文件，所以 active 变空也能通过。

---

## 4. 删除前必须先并入现行文档的内容

| 来源 | 内容 | 并入位置 | 注意 |
|---|---|---|---|
| `_context-0924.md` | 已核实事实 1–7、方法规则、赛题口径（逐条带 task.md 行号） | 00（改写后） | 并入后删 prompts 副本，避免两份清单漂移 |
| F94–F96、R20 | 035：S0 在 N22 11 门过 10，TPOT p95 0.296 未过；真实 LCP 可修缺口上限 850,432（8.0%） | 00 | — |
| R8 §1、§2、"已明确放弃" | 规则摘要；负载形状（p50/p90/max 已核对）；放弃清单：HiCache #39156、mixed-chunk #39526、单机 PD、PDMux、DeepEP/TBO、ReplaySSM、改写 prompt | 00 | 放弃清单每条都要附证据 |
| R13 附录 C | prefill CP 崩溃根因：`MARLIN requires a fused func for a2a backend deepep`（012g server.log）；`--prefill-decode-interval` 的语义（`scheduler.py:1256–1276,3456,3500`） | 00 放弃清单，或 findings | 日志只在 pod 上；用 pread 摘录到 `evidence/` 后才算 VERIFIED（UNSURE-5） |
| R12 §0 表、§1.3 | 12 条预判的对错统计（3 对 9 错）；疑似误杀、仍待单变量测试的方向（026+114、116 TP8、170v2 TP8） | experiments 教训（B1）；00 的待测清单 | — |
| R14 §2.5、§2.6、§3.1、§5 | 开局 N 条链首同时起跑的 burst（`s1_loadgen.py:586–596`）；前端不是瓶颈；dev↔正式结构差异；6 分钟截断回放只能当筛选（n=4，未证实） | 00 | 不要带上 §2.3、§2.4 的降级说法 |
| R11 §1、§2.4–2.6 | 失败清单；梯子不下探；依赖探针的任务等结论出来再入队；写明开发机没覆盖什么 | rule.md 方法节 | §2.1–2.3、§3 不要带过去 |
| Codex R15 §0.3–0.4 | routing key 的语义 | base/01 §4 | 需要 Codex 同意 |
| `plans/active/T50` | 116 的 8 卡复验待办 | board | 注明探针要强制门槛（R19 S5） |

---

## 5. UNSURE 项与确定办法

1. **第一名是否为 CalvinCao N26/0.0551**。出处是决策 39（用户提供），本地数据截至 09-22T23:56Z，里面没有这条。办法：重新拉榜，存成 `data/all_att_<日期>.json`，00 引用时写明日期和来源。
2. **rule.md:76 与 `build_image.sh` 的 b0–b3**。b2、b3 使用 `shortest-prefill-first`，而底包没有这个策略（决策 29）。办法：B3 读 `build/image/Dockerfile` 里 0923a 实际用的 serve 脚本，再定 rule 的写法。
3. **board 上的 T37/R16**。文件从未交付，dispatch 里仍是 accepted。办法：问 Codex main，然后在 dispatch 里关闭。
4. **R10 的头条数字**："150ms"与 F87 的"约 107ms"来自不同的 P 和方法。办法：两个数都保留，并分别注明 P 与来源（§2 已写）。
5. **prefill CP 的死因**。只在 R13 里引用了 pod 日志，本地没有证据。办法：`scripts/pod/pread grep` 012g 的 server.log，存进 `evidence/`。
6. **112/113/160/T45/T47 这 5 个 completed 计划是否保留**。它们准确，但按"只留现行文档"最严格的读法也可以删。办法：由统筹或用户决定；没有工具引用它们。
7. **Codex R15**。research/README 标"现行"。办法：通知 Codex，确认先合并 §0.3–0.4 再删。
8. **R20 的独立复核**。按"逐条核对原始数据"的规则，Claude 还没有记录复核。办法：重跑 `evidence/T56/attribute.py`（T56-01..04），在 F95 下注明结果。

---

## 6. 计数

| 范围 | KEEP | REWRITE | DELETE |
|---|---|---|---|
| Claude 维护的文件（69 个） | 19 | 15 | 35 |
| research/codex（13 个 + archive 21 个，只给结论） | 3 | 2 | 8 + 21 |
| 合计（103 个） | 22 | 17 | 64 |

---

## 7. 最小入口文档集（新智能体只读这些，按顺序）

总原则：状态只写在一处（00）。其余入口文件只做导航和规则，不写时效数字。findings、dispatch、histories、plans/completed 都是账本，不是入口。

1. **`CLAUDE.md` / `AGENTS.md`**（自动加载，每份不超过 30 行）。必须写：
   - 唯一准绳是 `llm-challenge-arena-v1/task.md`；
   - 阅读顺序：README → 00 → rule §4 → board；
   - 红线：task.md:699、707–708；服务不停不删；pod 只用 pread 和 pexec_codex；四个只读目录；
   - 5 条评测方法规则（§2 AGENTS 新节）；
   - 记录规则：`next_id`、`check_records`、证据放 `evidence/<T>/`。

   不写任何状态或榜单数字。
2. **`README.md`**：只放目录索引和指针（00、rule、board、`notes/submissions.md`、`scripts/pod/README.md`、构建与提交脚本）。只允许一行"状态见 00"。
3. **`research/claude/base/00-summary-mainline.md`**：唯一的现行状态与已核实事实文档，取代 HANDOFF、R8、R12–R14 的头条，以及 `_context-0924`。必须包含：
   - (a) 目标和带日期、带来源的榜单快照；
   - (b) 赛题口径摘要，逐条附 task.md 行号：11 门、统计余量、tpot_p95 无余量、从 N10 起 +4/−4、单档约 4 小时、能力门严格 >90、dev 只做 A/B、flush 必须真清；
   - (c) 已核实事实：F93、F95、F96 和 `_context` 第 1–7 条，包括 S0 精确栈及 120-v2 的 sha；
   - (d) 明确的"作废或降级说法，勿用"清单：F92 的 5.3M；F91 的 43 条；8.1–8.5k = 容量；闭环 ⇒ MTP 降 N；在飞 prompt 之和 = KV；1−p10/mean = 停顿占比；从榜单推断对手方法；12/12 = 能力门；DCP ×7.8；BCG"补齐错位"；system-tools-changed 零复用；analyze_run 的判定；numcheck wrong=0 即正确；`patches/120` = S0 的 120；
   - (e) 放弃的方向，每条附单变量证据和"什么情况下会重新打开"；
   - (f) 当前约束（S0 在 N22 卡在 TPOT p95）、下一个决定性实验、未决问题；
   - (g) 01–04 源码地图与 R9、R10、R17–R20 的索引。
4. **`rule.md`**：只放协作约定、目录归属、§4 红线和评测方法节。不放榜单，不放旧变更记录。
5. **`board.md`**：只放进行中、已排队、阻塞项和在役实例。已完成只保留最近一天。W 行的去留受 `check_records.py` 约束。

有赛规问题时直接读 task.md 原文，不读任何转述。
