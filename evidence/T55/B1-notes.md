# T55 · B1-notes 审查（`notes/` 全部）

- 审查者：Claude subagent（B1）。只写本文件，未改任何其它文件、未提交。
- 快照：2026-09-23 18:12 UTC（= 09-24 02:12 北京）。审查期间 notes 仍在被追加（F94–F96、T56→done、T58/T59 新行），本报告覆盖到此快照；之后新增的条目未审。
- 判断依据：`llm-challenge-arena-v1/task.md`、F93/R19、F95/F96、原始数据、源码。凡本文标 **VERIFIED** 的，均在 §12 给出输入 SHA 与复现方法；其余标 INFERRED/UNSURE。

---

## 0. 最会误导后续智能体的 10 条（按危害排序）

1. **F92 仍在 findings 正文**：「append-only 多算 5.3M、占全部 prefill 一半、修好就够 N22」。F93 已撤回：5.29M 中 4.52M 来自本轮链首（没有本轮前驱）；真实 LCP 正缺口是 0.863M（F95，占实际未命中 8.1%）。→ DELETE F92。
2. **F91 的第 2 条与 Fable 指标**：
   - 「43 个 fast 超标、97.7% 未命中、中位 80,607」用错了分桶。按 `in_ttft_gate` 算，fast n=328、>3s 只有 10 条（VERIFIED；F93）；这 10 条的归因见 F95：6 条缓存缺口与队头阻塞并存、3 条以缓存重算为主、1 条以队头阻塞为主。
   - 「在飞 prompt 和 1.2–1.3M = KV 驻留」「1−p10/mean = 停顿占比 58–73%」「140 只减两成」三条已被 R19 §4.3 降级；第三条是多变量比较（026/027 对 028/034），不能归因。
3. **F78（及 D38 第 2 项、quality.md:22）的 DCP 结论不成立**：
   - 「逻辑 KV ×7.8」并不存在（F90/T50b：norope latent 写入没有分片）。
   - 「19 万冷预填充 −19%，64 头读 KV 更高效」其实是 114 的效果：012f 只有 114、没有 DCP，190k 为 15.44s；013b 是 DCP+114+115，190k 为 15.58s；b113 为 19.29s（VERIFIED：ledger + job 文件）。
4. **F84「缓存损失 468 万 → 46 万」是口径变化造成的假象**：
   - 468 万来自 N6 b120on 的分析，那时分析器还包含链首（最大一条 prompt 252,115、cached=64）。
   - 46 万来自 025b N10，分析器在 244b9a7 之后已排除链首（VERIFIED：两份 analysis.txt + git log）。
5. **submissions.md 的状态和预测是错的**：
   - 45766/45767 仍写「queued」，实际部署失败、不计额度（VERIFIED：all_att_2026-09-23.json changelog）。
   - 写「完整状态在 data/submissions.json」，该文件实际是 `[]`。
   - 「预期 N10–14」是用 dev 预测正式档位，违反 task.md:354。
   - B 写「未测，搏一把」；034（=B 配置）后来 dev N18 失败，其中 tpot_p95 为 0.210（R14 表；本地无 raw，UNSURE）。
   - decisions「2026-09-23 晚」一节有同样的预测。
6. **用榜单或镜像名推断对手方法（F40、D20、F39、F35）**：
   - 「N22 选手牺牲链首 ⇒ EDF/I6 P0」：R19 §4.3 已指出榜单推不出方法；而且 3 名 N22 通过者里有 2 人的 chain p95 为 29.4s/30.2s，并没有牺牲链首（VERIFIED）。
   - F35「卡 TTFT 而不是 TPOT」：我们自己多次卡在 TPOT，026 p95 0.219、034 0.210、035 0.296（F96）。
7. **有操作危险的文字**：
   - F33 写「delete promptly」（及时删除服务），违反 D34 和用户红线（未经批准不得删/停服务）。
   - experiments.md 的 T24/T27 节保留了已退役守护进程的 nohup 启动说明；这个进程带 IDLE_RELEASE 删除路径，09-22 曾删掉 lh-arena-sess-a。
   - D23 仍写「守护进程迁 GPU 机」。
8. **quality.md 全表过时**：
   - 「115 使 DCP 可用」：DCP 实际崩溃，根因由 116 修复。
   - 「140/150/160 8 卡未测」：140/160 已多次上 8 卡；150 从未启用。
   - 「000 运行行为未测」：已在 8 卡验证。
   - 「submit_official 已成功提交 45734/45735」：两次都部署失败。
   - 「pod 工具链 B」：判定器 fail-open（R19 S1–S8）。
9. **被取代的机制结论仍按事实写着**：
   - F83 的根因「indexer 按全量 KV 寻址」是错的，实际是 norope latent 写入（T50b/116）。
   - F85/F86 和决策「梯子不降级」仍用「约 150ms/块」；F87 实测为 107ms。
   - F86/F92「调度只能在 TTFT/TPOT 之间换、产能钉死 8.1–8.5k」已被 R19 §4.2 降级。
   - F89「MTP 解决了解码门」是多变量比较。
   - F53「只有一处差异」不对：底包与 fe236ea6c3 有两个文件不同，含 MTP draft 的 `deepseek_nextn.py`（VERIFIED：diff）。
10. **记录基础设施本身在误导**：
    - dispatch 状态：T52b 写 in-progress，实际已 done（SUMMARY_v2 22/22）；T39 写 in-progress，pod 流程早已运行；T37 写 accepted，但 R16 从未交付。
    - notes/README 索引只解析决策表格行，D38、D39 和其它节式决策都不可见，看起来最新决策是 #37。
    - ledger-8card.md 已过时：没有 034/035；CP_/IFX/ab170 这几类 job 的参数和环境是空的（013b 看不到 `--dcp-size 8`）；LADDER 行里的「harness=」其实是 analyze_run 的「硬 p95+TPOT」，不是 harness 判定。

---

## 1. 文件级结论

| 路径 | 结论 | 理由 |
|---|---|---|
| notes/README.md | KEEP（清理后重新生成） | 由 `scripts/index_notes.py` 自动生成，不手改；但索引只解析 `^\| N \|` 决策表格行（index_notes.py:15），漏掉 D34 节、D38、D39 和两个无号节式决策。请 B3 修生成器后再重跑 |
| notes/findings.md | REWRITE | F1–F96 逐条见 §2：KEEP 31、REWRITE 28（另加文件头）、DELETE 25 |
| notes/decisions.md | REWRITE | 逐条见 §3。文件头「被推翻的决策不删」与 D39 冲突；「决策 34」小节与表格 #34 重号（check_records 只查表格行，查不出来）；新旧顺序混乱 |
| notes/experiments.md | REWRITE（几乎全删） | 16 节全是已退役的 v0.5.20 替身实验（E1/E2/E2b/T28/T29）、退役工具（T24/T27 守护进程、T20/T25 模拟器）或重复内容（T43=F64）。8 卡试验本的「推理半」按记忆要求本应在这里，实际散落在 F76–F96；应改建为逐 job 的推理台账（§4） |
| notes/ledger-8card.md | REWRITE（修 `ledger_dump.py` 后重新生成） | 生成于 15:37 UTC，缺 034/035；部分 job 的参数和环境为空，关键变量被隐藏；「harness=」列标签有误（§5） |
| notes/dispatch.md | REWRITE（只改状态和事实不符的行） | T37、T39、T52b 状态与事实不符；T52 没有结论；表头注明日期为 UTC，但 T54/T55/T56 写的是北京日期 09-24（§6） |
| notes/submissions.md | REWRITE | 45766/45767 状态错误；引用的 data/submissions.json 是空的；有 dev→正式的预测；B 配置的 dev 结果没回填（§7） |
| notes/submission_attempts.log | KEEP | `scripts/submit_official.sh:34,37` 仍在追加；6 个 attempt ID 与 all_att 记录一致 |
| notes/tech-debt.md | REWRITE | 已关闭条目仍保留（删除线）；多行过时；第 24 行空行把 T53 两行切出了表格（§8） |
| notes/quality.md | REWRITE | 两张表都停在 09-22/23 早上的状态，多处与 8 卡事实相反（§9） |
| notes/archive/findings-superseded.md | DELETE | 政策不保留归档目录；内容（F5/F7/F13…F41）是模拟器和 v0.5.20 线的旧事实，早已被取代。引用检查见 §11 |
| notes/e2_d1/（README.md、comparison.json、numeric_on/off/cross.json、splits_on.jsonl） | DELETE | v0.5.20 替身 E2 的原始证据；本就违反「原始证据放 evidence/」（AGENTS.md）。没有脚本读这个路径（grep scripts/ tests/ 为 0）。引用见 §11 |

---

## 2. findings.md 逐条

### 2.1 文件头（第 3–8 行）→ REWRITE

- 第 5 行「提交线：000…+ 101」「45734/45735 待出分」→ 改为：两次都部署失败；45766/45767 也部署失败；当前在跑 45979/45980（0923a，15 个补丁），出分约在 09-24 09:20 UTC。
- 第 6 行「仍成立的核心：…F4、F40/F35、F45」→ 删掉 F4、F35、F40 推断和 F45。改列：
  - 分桶只认 `in_ttft_gate`（F10、F93）；
  - dev≠正式（F31、task.md:354）；
  - S0=026 精确栈（_context §7）；
  - 没有任何候选 N18 全门通过（F89–F91、F96）；
  - 判定器 fail-open，要等 T54a（F93）。
- 第 7 行「以下 F36/F37/F42/F43/F47…」→ 删除（这几条都要删，见 2.3）。
- 第 8 行「已归档…notes/archive/…」→ 删除（该归档文件要删）。

### 2.2 KEEP（31 条）

| F | 理由 |
|---|---|
| F8 | 12,716 B/token 与底包启动账吻合（F79：KV 943360×12716B=11.17GiB） |
| F10 | 核对 s1_common.py:40 GATES 有 10 项；s1_score 硬门（:451 起）不含 TPOT，TTFT 按硬 p95 判；与 task.md:519 的 11 门对照正确 |
| F11 | `extra_buffer_lazy` 在底包的同一行（arg_groups/mamba_hook.py:110）；#39156/#39526 仍支撑 D5 |
| F21 | 底包镜像 160721 = 我们的 FROM；事实 |
| F23 | rid 复用、run_dev 不解析 flush JSON（:46–56、:232）与 R19 S3 一致 |
| F27 | 我们实际在用 `--prefill-decode-interval`；+141% 的来源更正仍有效 |
| F29 | HiCache 的 DSA indexer 缺口仍是 D5 的依据 |
| F32 | A/B 命令实际含 glm45/glm47；答案落在 content 已由 8 卡冒烟证实（F73、tech-debt 第 13 行）。行号属 v0.5.20，属小瑕疵 |
| F44 | 平台写每日 3 次、用户确认 2 次，这一事实有效。小瑕疵：data/all_att.json 刷新后符合条件的 attempt 是 42 条而不是 49 条，不影响结论 |
| F52 | 收窄 F51；harness 只在 header 带 routing key（s1_loadgen.py:112–120，body 无 routing_key），已核对 |
| F55 | changelog 原文核对一致；task.md:283、374 建议钉 digest，:442 的示例用纯 tag |
| F57、F58、F59、F60、F62 | 8 卡或源码事实，至今成立 |
| F64、F65、F66、F67、F68、F69 | Codex 算子交付事实，有收据 |
| F73、F74 | 8 卡事实（F74 的收据路径见 §12 UNSURE-4） |
| F79 | T49/R18 已纠正容量口径 |
| F81、F82 | 与 ledger 024 一致（KV 633,536；N10 ENGINE_DEAD） |
| F87 | ledger 026g「FIT 0 fixed 107.5ms、60.2µs/tok」、026h 核对一致 |
| F93、F95、F96 | VERIFIED。F96 的 TPOT p95 0.2962、均值 0.1040 已按 evidence/T58/score_formal.json 核对 |

### 2.3 DELETE（25 条；引用检查见 §11）

| F | 行 | 理由 |
|---|---|---|
| F4 | 32–34 | v0.5.20 add_chunked_req/SPF 的表述；底包上的现象已由 F76/F77/F85 实测取代 |
| F6 | 35–40 | 「主办方镜像必带私有 sm80 补丁」错误（F56：原版底包在 A100 启动即崩）；「只能 extra_buffer」已被 F11 更正；有用部分已在 D5/F11 |
| F12 | 63–68 | v0.5.20 源码行号的 D1 设计约束；底包等价事实见 F52/F62/F79 |
| F18、F20 | 74–82 | v0.5.20 `components/mamba.py` 路径（底包里是 mamba_component.py）；只服务于已退役的 D1 v0.5.20 设计 |
| F22、F26 | 88–93、100–105 | role-boundary/内部快照先例调研；方向已由 101/140 落地，细节在 research/codex R6/R7 |
| F28 | 117–122 | DP 亲和（v0.5.20）；底包的 DP 路由事实在 F51/F52；DP8 已暂缓（D30） |
| F30 | 129–135 | 前端方向已由 130 落地；R14 §2.6 实测前端差 p95 0.124s（VERIFIED：026 raw），结论已过时 |
| F35 | 148–152 | 引用的是旧快照中位数（例如 N6 写 44 条，文件实为 47）；框架是已退役的模拟器；「TTFT 卡而非 TPOT」被我们自己的 TPOT 失败推翻（026 0.219、034 0.210、035 0.296） |
| F36、F37 | 153–163 | v0.5.20 随机替身 E1 |
| F39 | 164–173 | 由镜像名推断路线，R19 §4.3 指出榜单连方法都推不出，镜像名更不行；D2/I6 标签已退役 |
| F42、F43、F45、F47 | 180–192、197–204、213–223 | v0.5.20 替身 E2/T29。F45 仍有价值的教训（零容差冷/暖 logits 门没有判别力）应并入 tech-debt 的「数值门缺失」条（§8） |
| F48 | 224–232 | 讲的是已退役的 trisol_test_daemon |
| F49、F50 | 233–246 | 外部 PR 调研小结，决策已做；细节在 research/codex R13/R14 |
| F61 | 336–342 | 120 v1 的 CPU 收据；R19 S6 指出不能为 v2/v3 背书；101 双 partial 部分已在 F62 |
| F63 | 351–356 | 「110 torch indexer 占 60%」已被 112/113 解决（F76：indexer 4.3%） |
| F75 | 437–443 | 容量口径已被 F79/F80 更正取代；每块成本见 F87；decode 步长见 F85/F88 |
| F92 | 557–565 | 已撤回（F93、D39） |
| F94 | 575–582 | 被 F96 完整取代：F96 用 score_formal 重算、精确 p95，F94 的 INFERRED 已被证实；其余提醒已在 F93/R19 S8 |

### 2.4 REWRITE（28 条）

| F | 行、原句（≤20 字） | 改成 |
|---|---|---|
| F1 | 12「~1.5M tokens capacity/GPU (R2」 | 改为实测每卡 KV：默认 943,360（F59）；graph-bs64+mamba200 为 1,569,152（F80）；chunk16384 自动时 633,536（F81）；026/028 PRECHECK 为 1.32M/1.04M |
| F2 | 16「Hidden-split chains (178): 4030」 | 改为：dev 集 722 请求/311 链（hidden 178 链/422 请求、dev 125/291、validation 8/9）；chains.jsonl 的 n_requests（合计 5601）是原始轨迹长度，本轮不回放（VERIFIED） |
| F2 | 18「Gates: fast_intra uses FROZEN」 | 补上：分桶只用 `s1_common.in_ttft_gate`；idx_in_chain==0 或 context_reset 一律进 chain_start（dev 桶 328/388/20/314，VERIFIED） |
| F3 | 21「57% of intra requests」 | 按实测改：phase=intra 为 255/503=51%，gate intra 为 235/388=61%（VERIFIED） |
| F3 | 26「a hit needs a saved KDA state AT」 | 改为「LCP 处或之前最近的有效快照；复用量 = 该快照深度」（archive F7 已更正；R18）；v0.5.20 的描述换成底包事实（F51/F52：每次 extend 一个 track，branch 覆盖 end） |
| F3 | 30「TO MEASURE:」 | 删除（已由 R18/F95 完成） |
| F9 | 行号 990/301 | 改为底包 `http_server.py:980–994`、`tokenizer_control_mixin.py:304–310`（已核对，仍返回纯文本，扇出取 [0]）；注明 000 已修且 8 卡验证过 |
| F16 | src/sglang 路径 | 改指底包 `kv_cache_configurator.py:2169/2336/2393`（底包同样按 attn_dp_size 下取整，已核对） |
| F31 | 139「the binding gates are fast」 | 删除「binding gates」一句（TPOT 同样构成约束：026/034/035）；其余数字 VERIFIED 保留 |
| F33 | 108「delete promptly」 | 删除，改为「服务不得删/停，须用户批准（D34）」；删掉 109 整行（interval +141% 的错误归因已由 F27 更正） |
| F40 | 176–177「strongly suggests」「Implication: the formal」 | 只保留公开观测（LewyM 45417 各门数值）；补充另外两名 N22 通过者 chain p95 29.4/30.2（Jinbo hu 45461、Mingjun Xu 45443，VERIFIED）；删掉对手方法和 I6/SPF 的推断（R19 §4.3、F46） |
| F46 | 209「MODEL OUTPUT / WHAT-IF」、211「003草稿」 | 删这两段（模拟器和 003 已退役）；保留第一段统计余量：n=20/65/314/328/388/808/9023 对应 3/6/22/23/27/51/485（VERIFIED 重算） |
| F51 | 253「即可替代 101 的拆分预填充」 | 删除该推论（F52 已收窄；102 已由 140 取代） |
| F53 | 271「一处与我们无关的」、273「唯一差异」 | 改为：`eagle_worker_v2.py` 与 `deepseek_nextn.py` 两个文件不同（外加 `_version.py`），都是多模态嵌入对齐，纯文本路径等价（VERIFIED：diff base_exact 与 refs/sglang-fe236ea6c3）；276 起的「102 精度」一段只保留「底包分叉点快照取 bf16 h」 |
| F54 | 288「正式评测跑的源码：A =」 | 0922e/f 从未运行（部署失败），改为 0923a = 底包 + 15 个补丁（build/image/0923a.patches.txt）；并注明 build/image/Dockerfile 的 FROM 是纯 tag（未钉 digest），「L3 代码 = base_exact」对 0923a 待证（§12 UNSURE-3） |
| F56 | 304「--dsa-prefill-backend fa3」 | 删掉 fa3 参数和「真实 8 卡验证中」；补充 45766/45767 结果：部署失败、不计额度（VERIFIED） |
| F70 | 413「须在 pod 复测」 | 改为「F74 已在 pod 确认」 |
| F71 | 421 句末「须 pod 复测」 | 改为「F76 8 卡 profile 吻合」 |
| F72 | 424「8 卡待测」 | 改为「8 卡：180k 时每 token −33%（F88）、190k 冷启动 19.29→15.44s（012f）」 |
| F76 | 448–449「19 个 intra」「KV 峰值仅 50%」 | 按 F79 改为 18 intra+1 turn_start；非可淘汰 KV 峰值 92%；最大几条「丢失」是链首 |
| F77 | 452 标题「四门全过（自估规则）」 | 改为「formal-est 四门过；harness overall 5.29>5 FAIL」；删掉 455「缓存丢失模式不变…」（链首假象，F79） |
| F78 | 459 标题、461–462 | 删掉 DCP「×7.8」与「−19%/64 头更高效」；写明「190k 的 −19% 由 114 单独即可复现（012f 15.44s 对 013b 15.58s，单次无重复），且 013b 跑在 116 之前的错误写入路径上」；保留 mHC scatter 数据（020） |
| F80 | 473 标题「零风险」、477「更正（09-23，Codex R18」 | 去掉「零风险」（F79：不能称零风险；tech-debt 第 18 行：mamba200 可能加剧淘汰）；477–478 的更正移到 F75/F76/F77 各自条目（F75 删除后并入 F76），不要挂在 F80 下 |
| F83 | 491「INFERRED：DCP 下每卡」 | 根因改为 T50b/116：GLM rope=0 走 norope latent 写入未分片，分配器高水位越过每卡行数（F90） |
| F84 | 496 行末「缓存损失 468 万 → 46 万」 | 删除（口径变化：244b9a7 起分析器排除链首，VERIFIED）；499「第一名 0.0273」删除（榜单已变，见 §12 UNSURE-1） |
| F85 | 510「截距 ≈150ms/块」 | 改为「见 F87：8 卡实测固定约 107ms + 60µs/token（026g）」；注明 a 行与 b–e 行差多个变量（chunk 8192、无 140/120） |
| F86 | 521「约 150ms 的固定开销（F85）」「调度只能在 TTFT」 | 改为 107ms（F87）；「只能…换」「吞吐掉一半」标 INFERRED（R19 §4.2） |
| F88 | 532「拟合固定开销 43–58ms」 | 改为 43–64ms（026j FIT0=63.9）；补上 BCG 每 token 斜率更高（83.9–123.9 对 76.3–115.5 µs/tok，chunk 4096 切块，仅供参考）；补上 026l（BCG 不带 scatter）12/12，以及 v2 修复（T52b，TP2 22/22；TP8 待测） |
| F89 | 537「MTP 解决了解码门」 | 改为 INFERRED：028 对 027 同时改了 MTP、114、140 关、cap4096/i2、池 1.04M、fp32 SSM，不能单独归因 |
| F90 | 540 附近「四门里三门 FAIL」 | 改为「四门全 FAIL」：turn 5/3 超出允许、p95 180s（ledger 028c，VERIFIED）；「块从 8192 缩到 4096，吞吐的损失…」标 INFERRED（同段自称不能归因）；116/170v2 的 TP8 验证仍待做（R19 §3） |
| F91 | 547–552 | 只保留：034=B 配置，N18 fast 91/23、overall 75/27、chain 19/22、tpot_mean 0.0612、**tpot_p95 0.210（也 FAIL）**（R14 表，本地无 raw，§12 UNSURE-2）；B 在自测前提交。删除 548 整条（43/97.7%/80,607，错桶；正确归因见 F95）；删除 550–552 三条（R19 §4.3 降级 + 多变量）；删除「线上预期低于 N18」（task.md:354）。保留经 026 raw 复核的三点：前端差 p95 0.124s；<60 token 输出 43 条的 TPOT p95 0.554；开头 18 条链首同时到达、其中 7 条 >30s（VERIFIED） |

---

## 3. decisions.md 逐条

- **文件头（第 3 行）** 原句「被推翻的决策不删」→ 改为按 D39：过时决策直接删，git 保留历史；全文统一为「最新在上、唯一编号」，节式决策也编号（例如把现在的「决策 34」节改号为 40）。
- **表格行（#1–#37）**：
  - KEEP 14 行：#5、#9、#10、#12、#14、#25、#26、#29、#31、#33、#34、#35、#36、#37。
  - REWRITE 7 行：
    - #1（理由「v0.5.20 原生有 glm5_next」改为底包 fe236ea6c3）；
    - #15（删「以及定期归档」，D39 起不设归档）；
    - #21（删「自动提交守护进程每天上限 2」：logs/submit_daemon.events 为 0 行，所有提交都走 submit_official.sh 手工提交）；
    - #23（保留「Claude 批准自测」；删「守护进程迁到 GPU 开发机」：测试守护进程已退役，D34）；
    - #27（入口 `scripts/l2.py` 服务于已退役的守护进程队列，改为 pod 队列 `scripts/pod/qpush|podq`；原则「与提交同代码、同命令」保留）；
    - #28（保留「只排机制补丁」；删 102/DP 亲和等已过时的具体项）；
    - #32（这条描述的是 120 v1。改为当前三个版本及各自归属：S0=v2 草稿 `patches/drafts/120-sched-protect-chain-v2.patch` sha ef1744b3…；镜像 0923a=v3；make_120.py 生成的是旧逻辑，R19 S6）。
  - DELETE 16 行：
    - 已被取代：#4、#6、#11、#18、#22；
    - v0.5.20 替身配置：#7、#8、#16、#19、#24；
    - 已无意义的撤回或旧方向：#2、#3；
    - 旧 directions 与模拟器：#13、#17；
    - #20：依据是 F40 推断，已被 R19 §4.3 和数据推翻（2/3 的 N22 通过者 chain ≈30s）；
    - #30：M0–M4 全部完成或已被 D38/D39 取代。
- **节式决策**：
  - T12 工具选择（45–48）：KEEP。score_formal 是指定评分器。
  - D39（50–58）：REWRITE 一处。51 行「第一名 CalvinCao：N26、0.0551」本地没有快照（data/ 里最高 N22；R8 在 09-23 还写「无人 N26」），来源是 R13 Fable，应标「未核实」或补一次只读榜单快照（§12 UNSURE-1）。
  - 「决策 34」节（60–65）：REWRITE，改号，避免与表格 #34 重号；把 D38 的「补记（09-23…）」一段（守护进程与 run_forever 退役、删除调用核验 0 处）并入本节。
  - D38（67–80）：DELETE。第 2 项「--dcp-size（KV 容量 ×N）」错误（F90）；第 4 项「追平前排」基于榜单推断；第 0/1 项已完成（114）。
  - 「决策（2026-09-23，N6 基线后）」（82–85）：DELETE。「KV 在 N6 已 50%」错误（F79：92%）；「114 价值下调」被 012f/F88 推翻；排序已执行完。
  - 「决策（2026-09-23）— 采纳 Fable T49」（87–91）：DELETE。「120 v2」的定义并入改写后的 #32；「零风险杠杆」措辞错误（F79）；CP 不做主线的依据并入 tech-debt 或 D39。
  - 「2026-09-23 — 梯子不降级」（93–95）：REWRITE。保留用户规则「N18 失败即停、不下探」，注明已由 D39「直接 N22、N18 作对照」更新；删掉「每块约 150ms」（F87 实测 107ms）、「调度只能…换」（R19 §4.2）和「下一步机制 T53（KDA/DSA…）」（T53 编号已被 R19 审阅占用）。
  - 「2026-09-23 晚」（97–99）：REWRITE。删掉「⇒ 预期 N10–14」（task.md:354）和「MTP 过了解码门」（多变量）；补一句 B=034 dev N18 FAIL（含 tpot_p95 0.210，UNSURE-2）；「明天迭代」的待办移到 plans/board。
- **计数**：KEEP 15、REWRITE 11、DELETE 19。

---

## 4. experiments.md（REWRITE：几乎全删）

- 删除以下各节：
  - E1（11–116）；
  - E2（117–166）；
  - T23 Session A runner（170–186）；
  - T19（188–208；用法已在 tests/T19_USAGE.md）；
  - 提交包工具（210–213）：写有「前提未核实：底包…就是 v0.5.20」，但 F53 表明底包不是 v0.5.20；check_submission 的 flag 检查对底包失效（submissions.md:32 用 SKIP_FLAG_CHECK=1）；
  - harness 自带测试（233–234）；
  - T16（236–271）；
  - T20（273–280）、T25（346–352）：模拟器/SPF/003 已退役（D29）；
  - T22（282–339）：文档应在 submission/queue/README.md，且守护进程从未用于提交；
  - **T27（341–344）、T24（354–359）**：已退役的删服务守护进程，保留启动说明有危险；
  - T28（361–389）、T29（391–422）、E2b（424–466）：v0.5.20 替身；
  - T43/112（468–474）：与 F64 重复。
- 第 1–9 行的表头和 E 表：改为「8 卡试验本·推理半」（与 ledger-8card.md 的事实半配对，符合 ledger.py:2–3 的注释）。每个 8 卡 job 一条：假设、相对已验证基线的精确 diff、两种判定下的结果、结论、保留或放弃的方向。从已按本报告修正的 F76–F96 抽取，至少覆盖：
  - 012、013、024、025、025b；
  - 026（S0）；
  - 026a–l；
  - 027、028、028b、028c；
  - 029a–d；
  - 033、034、035（S0 N22）。
- T12 节（215–231）：只留 3 行评分器说明（`score_formal.py` = harness 10 门 + 无余量 tpot_p95 + Clopper–Pearson estimated），标明「pod 梯子尚未接入，待 T54a」。
- 「放弃方向」清单（记忆要求），可直接写入的证据：
  - INT8 MoE 已放弃，证据充分（F70/F74）；
  - 随机 token 数值指纹已放弃，工具无效（029d 噪声对照 6/18）；
  - LL128 NCCL 无收益：021 单次测量，60k 6.18s 对 6.37s、190k 19.06s 对 19.29s，≤3%，tech-debt 还写着「待分析」；
  - **以下不能算已放弃**：
    - DCP：崩溃根因已修；116 只有开发机 PASS，TP8 未测；
    - 170：v1 错误，v2 在 TP2 上 22/22，TP8 未测；
    - MTP：「闭环 ⇒ 必降 N」已被 R19 §4.1 否定。

---

## 5. ledger-8card.md（REWRITE = 修工具后重新生成）

- 第 1 行「自动生成 2026-09-23 15:37 UTC」：已过时，缺 034（B 配置）和 035（S0 N22，F96），也缺之后的 job。
- 参数和环境列为空，关键变量被隐藏：
  - 47 行 012e、73 行 013b：实际带 `CP_ARGS="--dcp-size 8"`（coldprobe_b115_dcp8.sh），ledger 显示「参数：``」；
  - 217/242 行 026f/026i：MTP 参数没显示；
  - 249/260/271 行 026j/k/l：job 文件里有字面参数，ledger 也为空。
  - 原因在 `scripts/pod/verify/ledger_dump.py` 只认 G_ARGS/G_ENV（B3 修）。
- 177/184/286 行等 LADDER 行里的「harness=False」：是 analyze_run 的「硬 p95 + tpot_p95≤0.10 + 错误率」，**不是** dev harness 的 ALL_PASS（harness 没有 TPOT 门），应改称 `hardp95+tpot`。这些行都来自 analyze_run（R19 S1 fail-open）；当前列出的档都是完整运行，没有被翻案，但 T54a 之后应改为 score_formal 判定。
- 026 [failed]：实际是 N18 跑完后人为停止（F84），「descending: 14 10」并未执行；状态应区分「首档失败」与「人为停止」。

---

## 6. dispatch.md（只看状态与事实不符的行）

| 行 | T | 现状态 | 应改为 | 依据 |
|---|---|---|---|---|
| 12 | T37 | accepted | dropped（或 blocked：未交付） | 预期产出 `research/codex/R16_submission_evidence_audit.md` 不存在；45734/45735 已有结论（部署失败，F55） |
| 174 | T39 | in-progress | done | pod 队列从 001 跑到 035；备注「队列暂停中…待恢复」已过时 |
| 347 | T52b | in-progress | done | evidence/T52b/SUMMARY_v2.txt：12:04 UTC v2 判定 TP2 scatter 22/22、51/51 走图；170 v2 已进 0923a；F90 已记录 |
| 345 | T52 | done（没有结论） | 补结论 | F87：170 = #38522 移植 + 140 字段直通；替身 40→27ms；TP8+scatter 结果错误，转 T52b |
| 343/346 | T50/T50b | dropped/done | 注明「T50 由 T50b 取代」 | 两行都把 patches/116 列为产出 |
| 350–352 | T54/T55/T56 | 日期 09-24 | 09-23 | 表头写「派发时间 (UTC)」，快照时 UTC 仍是 09-23；T57–T59 用的是 UTC |

其余行状态与事实一致。T56 已由他人改为 done。T15 的一句话结论「TPOT 常先绑定」是已退役模拟器的输出，可在行尾注明「模拟器已退役」，不改状态。

---

## 7. submissions.md（REWRITE）；submission_attempts.log（KEEP）

| 行 | 原句 | 改成 |
|---|---|---|
| 3 | 「full status history…in `data/submissions.json`」 | 删除；该文件是 `[]`（VERIFIED），提交守护进程从未用过 |
| 5–6 | 空的结果表 | 填入实际结果：45734/45735 为 image 格式 422、45766/45767 为 trisol service failed，四次都部署失败、不计额度；45979/45980 在跑 |
| 13 | 「Daily quota 2/2 used for 2026-09-22」 | 改为「部署失败不计额度（changelog 原文）」 |
| 18–19 | 「queued」×2 | 改为「failed（部署：trisol service entered 'failed'；镜像没有 sm80 补丁，F56）」 |
| 24 | 「⇒ 预期 N10–14」 | 删除（task.md:354） |
| 25 | 「未测，搏一把」 | 补充：B=034 配置（ladder_B2_mtp_c8k.sh 的参数和环境与 B 的 submission.json 逐项相同，VERIFIED）；034 dev N18 FAIL，含 tpot_p95 0.210（UNSURE-2） |
| 30 | A「MTP+114+v3…」 | 注明：A 的镜像含 170（未启用）和 150（未加 `--warmups ax_shapes`，未启用），而 028 运行时没有 170。作为 dev↔正式校准对时，需先确认两者未启用即等价（UNSURE-5） |

---

## 8. tech-debt.md（REWRITE）

| 行 | 原句 | 改成 |
|---|---|---|
| 3 | 「历史见 git/归档」 | 改为「历史见 git」 |
| 7、8、13 | 删除线的已关闭条目 | 删除行 |
| 9 | 「中途状态取自 bf16 的 h；数值门 D1-04」 | 改为：101/140 在底包 TP8 上没有有判别力的数值验证（numcheck_cmp 截断和分叉都判 ok，R19 S4；随机 token 指纹无效，029d）；并入 F45 的教训（零容差冷/暖门也会被普通缓存触发）；交 T54b |
| 10 | ROLE_BOUNDARY_STATS | 保留，补上 140 每请求快照成功/跳过/驱逐事件缺失（R19 §3） |
| 11 | 「L1 替身跑的是 v0.5.20」 | 删除（L1 已换成底包 rank 模型：T51/T52/F71） |
| 12 | 「check_submission.py 按 v0.5.20」 | 改为：对底包用注解字段定义的参数全报不存在，0923 用 SKIP_FLAG_CHECK=1 跳过（submissions.md:32） |
| 17 | analyze_run 自实现估算 | 并入第 25 行（改用 score_formal 且失败即失败，T54a） |
| 18 | 「梯子对比 DCP 路线」 | 保留，改为「待 116 TP8 验证后再比」 |
| 20 | 「150/160…尚未在 8 卡上验证」 | 改为：160 已在 8 卡运行（026f/i、028、034）；150 从未启用（所有 job 和 A/B 都没有 `--warmups ax_shapes`） |
| 21 | 「021 已跑，待分析」 | 写入结果：LL128 单次 60k 6.18s 对 6.37s、190k 19.06s 对 19.29s（≤3%，无重复），没有收益证据 |
| 23 | 「残留一条未处理的 T49」 | 删除（UNSURE-6：Codex main 之后已执行 T53/T57–T59） |
| 24 | 空行 | 删除，使 25–26 两行回到表内 |

---

## 9. quality.md（REWRITE，合并成一张现状表）

| 行 | 原句 | 改成 |
|---|---|---|
| 8 | 「底包上的运行行为未测」 | 000 升为 A/B：8 卡 flush 返回 JSON（F59），服务端时间戳 722/722 完整（F96） |
| 9 | 「45735 成绩 + L2 02 项」 | 101+105 为 B：有 105 后 8 卡不崩；数值未经有判别力的验证 |
| 10、11、13 | 102 行、L2 守护进程行、L1 v0.5.20 行 | 删除（已被 140 取代或已退役） |
| 12 | 「已成功提交 45734/45735」 | 改为：45979/45980 已提交；此前 4 次部署失败 |
| 14 | 「未接真实 8 卡结果」 | 改为：T58 已用它重算 035（F96）；pod 梯子待 T54a 接入 |
| 21 | 「v2 待梯子验证｜梯子 024」 | 改为 120 C：v1/v2/v3 的实测归属，生成器不一致（R19 S6） |
| 22 | 「115 使 DCP 可用」 | 拆开：114 为 B（8 卡 −33%/token）；115/116 DCP 为 D（025 崩溃，116 只有开发机结果，×7.8 不存在） |
| 23 | 「8 卡未测｜梯子 026」 | 140 为 C：026/035 开启，缺快照事件，MTP 下关闭 |
| 24 | 「150 预热 / 160 MTP｜D｜未上 8 卡」 | 150 为 D（从未启用）；160 为 B（8 卡运行、冒烟 12/12，能力门未测） |
| 25 | pod 工具链「B」 | 判定器降为 D：R19 S1–S8 fail-open，待 T54a |
| 新增 | — | 106 为 B（025b KV 三次打满不崩；进展性未证，R19 §3）；170 v2 为 C（TP2 22/22，TP8 未测） |

---

## 10. notes/README.md

KEEP，在全部删改后运行 `python3 scripts/index_notes.py` 重新生成。已知问题（生成器属 B3）：只解析表格行决策，看不到 D34 节、D38、D39、无号节，并把 experiments.md 描述成「实验与工具台账」。

---

## 11. DELETE 前的引用检查（grep -rn，排除 src/、build/base_exact、llm-challenge、evidence/T55、notes/README）

- **F4/F6/F12/F18/F20/F22/F26/F28/F30/F36/F37/F39/F42/F43/F47/F48/F49/F50/F61**：只有文字引用，出现在：
  - board.md 历史行、rule.md 变更记录（:115–120）、dispatch 历史行；
  - research/archive/*、research/codex/{R5,R7,R8,R15,README,archive/*}；
  - patches/v0520/*、plans/completed/*；
  - tests/TEST_PLAN.md（F45/F47 在 D1-04 等 v0.5.20 行）；
  - patches/000-interface-compliance.md:81（T29 回填提到 F47）。
  - 没有任何脚本、job、补丁或提交追溯依赖这些编号。
- **F35**：还被 scripts/archive/sim_envelope_fit.py 及其测试（已退役模拟器）和 research/claude/base/04-model-kernels.md:46 引用，都是文字，不断工具。
- **F63**：patches/112-sm80-indexer-kernels.md:7 以「F63 profile」作动机；docs/histories/2026-09/20260923-0400-sm80-indexer-kernels.md:19；research/claude/base/00-summary-mainline.md:10；R18:336。建议改引 `evidence/T43/profile_decode_b112_n6_TP0.txt`（B2/B3 执行）。
- **F75**：R18:319/339（列为过时条目）；00-summary-mainline.md:8；plans/prompts/T49-*。都是文字。
- **F92**：D39、F93、tech-debt:26、TEST_PLAN:293、R19、_context 都以「已撤回」的语气引用，删正文不影响。
- **F94**：F96 文中写「补全 F94」，删后 F96 改为「supersedes F94」。
- **决策** #2/#3/#4/#6/#7/#8/#11/#13/#16/#17/#18/#19/#20/#22/#24/#30、D38、两个无号节：
  - #30 被 tests/L2.md 引用（该文件本身属退役 L2 流程，B3）；
  - #22 被 board.md/dispatch 历史行引用；
  - #24 被 experiments（也要删）引用。
  - 不断工具。
- **notes/archive/findings-superseded.md**：findings.md:8（文件头要改）、dispatch T38 行（历史）。不断工具。
- **notes/e2_d1/**：
  - 引用方：board.md:73/75、plans/completed/2026-09-22-d1-role-boundary.md:40、patches/v0520/001-…md:218、experiments（要删）、dispatch T13 行/节（历史）、findings F42/F43/F45（要删）。
  - scripts/compare_e2.py、analyze_e2_numeric.py 用参数 RUN_ROOT，不读这个路径（grep scripts/ tests/ 为 0）。
  - 不断工具，也不涉及提交追溯。
- **experiments.md 各节**：
  - 被 board.md、research/codex/README.md:3、evidence/E1_stock/README.md:14、evidence/T29/README.md:29、plans/completed/*、patches/v0520/* 按节名引用，都是文字；
  - `scripts/index_notes.py` 只抓标题，`scripts/pod/ledger.py` 只有注释引用。
  - 文件本身保留（AGENTS.md:18、rule.md:37、TEST_PLAN:7 引用了这个路径）。

---

## 12. UNSURE（及怎样才能确定）

1. **D39「第一名 CalvinCao N26 / 0.0551」与 F84「第一名 0.0273」**：本地 data/all_att_2026-09-23.json 最高是 N22（LewyM 0.0273、Jinbo hu 0.037、Mingjun Xu 0.0509），CalvinCao 只出现在 R13（Fable）。解决办法：做一次只读榜单快照，带时间戳存入 data/。
2. **034 N18 各数值与「纯排队」归因**（F91、R14 表：fast 91/23、overall 75/27、chain 19/22、mean 0.0612、p95 0.210）：来自 Fable 读取的 pod raw，本地没有。解决办法：用 pread 取回 034 raw 放进 evidence/，像 T58 一样用 score_formal 重算，再用 in_ttft_gate 拆分排队和执行。
3. **0923a 的 FROM 是否钉了 digest**：build/image/Dockerfile 用的是纯 tag，而 0922 的 A/B.Dockerfile 钉了 f24781f0…；这里无法确认 0923a 实际用的是哪个。解决办法：`bohr image dockerfile 164197`（我们自己的镜像，只读），对比 FROM 行。
4. **F74「verify_kernels 11/11 PASS」**：ledger 003 没有结果行，findings 也没给收据路径。解决办法：取回 /tmp/ax/runs/003-*/job.log 放进 evidence/。
5. **A 与 028 是否等价**：A 的镜像多了未启用的 170 和 150。解决办法：审阅 170/150 补丁在默认关闭时是否改动执行路径（170 含「140 字段直通」修改）。
6. **tech-debt 第 16 行（CUDA13 compat 只影响 L1）和第 23 行（残留 T49 消息）**：解决办法：看开发机 `/sjtu/linhang/arena/env.sh` 当前的底包环境是否仍依赖 compat 库；问 Codex main 的队列状态。
7. **submit_daemon 是否仍打算使用**：它决定 D21/D23 的措辞和 T22 文档的去留。现状是事件为 0、队列空、全部手工提交。需要用户或 Claude 决定。
8. **F3 原文的「57%」出处**：我复算得到 51% 或 61%（取决于分母）。原口径未知，按 VERIFIED 的复算值改写即可。
9. **45979/45980 成绩**：约 09-24 09:20 UTC 出分后回填 submissions.md 和 findings 文件头。

---

## 13. 核实收据（输入 SHA256 前 16 位 → 结论；全部为本地 CPU、只读）

| 输入 | SHA256 | 用途 |
|---|---|---|
| evidence/T53/026_N18_raw.jsonl | 486f041027abde2a… | 026 N18 raw，用于 fast/TPOT/前端差/链首突发的核对 |
| s1-dev/harness/s1_common.py | a7238b5f1adf728b… | 分桶函数 in_ttft_gate |
| s1-dev/data/dev-combined-v1/requests.jsonl | 00c79cb68219a954… | dev 集组成 |
| evidence/N6_b113/analysis.txt | 7c6a8ea0e2e0a78e… | N6 b113 分析，F76 数字 |
| evidence/N6_b120on/analysis.txt | 76f172354b6f819d… | 旧口径 lost 4,676,934（含链首） |
| evidence/L025b/N10_analysis.txt | 77a0383ce3b2f8e0… | 新口径 lost 455,687 |
| notes/ledger-8card.md | 509d2772272cba35… | 012b/012f/013b/020/021/022/026g–l/027/028/028b/028c/029/033 结果行 |
| scripts/pod/jobs/coldprobe_b114.sh | 2304ba1495f909b2… | 012f 配置：CP_ARGS="" |
| scripts/pod/jobs/coldprobe_b115_dcp8.sh | f462760c77671fe4… | 013b 配置：CP_ARGS="--dcp-size 8" |
| data/all_att_2026-09-23.json | f66b610425ba149e… | 45734–45767 的 changelog；N18/N22 通过者 |
| data/all_att.json | ef03157d96b5e500… | F35 中位数；配额 changelog 数 42 |
| data/submissions.json | 37517e5f3dc66819… | 内容为 `[]` |
| evidence/T52b/SUMMARY_v2.txt | d312ec65bdd2f945… | 170 v2 TP2 22/22 |
| evidence/T58/score_formal.json | 0a23ce96f39b91fe… | 035：tpot_p95 0.29619、mean 0.10396 |
| build/base_exact/.../deepseek_nextn.py | 6c1900a4636bd042… | 与 fe236ea6c3 不同 |
| build/base_exact/.../eagle_worker_v2.py | 6a89f1b5936db4e6… | 与 fe236ea6c3 不同 |

在 026 raw 上的复现结果（python3，`sys.path` 加入 s1-dev/harness，用 `s1_common.in_ttft_gate`）：

- 分桶：fast n=328，>3s 为 10 条。
- 这 10 条的实际未命中：[3120, 4869, 9330, 14095, 41071, 46728, 65308, 73660, 80607, 80803]。
- ttft_client_s−server_ttft_s：p50 0.004、p95 0.124、max 0.449。
- 输出 <60 token 的 43 条：TPOT p95 0.554。
- 按 client_dispatch_at_s 排序的前 18 条：全部是 idx0，2.1s 内发出，其中 7 条 >30s，未命中合计 607,132。
- dev 分桶（requests.jsonl）：fast 328、overall 388、turn 20、chain 314、idx0 311；append-only/reminder-replaced 在 phase=intra 中占 255/503，在 gate intra 中占 235/388。

其它复现：

- allowed_over：用 log 域二项尾概率 + 二分求 Clopper–Pearson 下界，得 n→k 为 20→3、65→6、314→22、328→23、388→27、808→51、9023→485。
- 底包差异：`diff -rq build/base_exact/sglang refs/sglang-fe236ea6c3/python/sglang`，结果是 2 个文件不同 + `_version.py`。
- 口径变化：`git log -- scripts/pod/verify/analyze_run.py` 显示 244b9a7「analyzer excludes chain heads」。
- `check_records.py` 结果为 0 error；但它的决策查重只看表格行（check_records.py:64），查不出「决策 34」重号。

---

## 14. 审查期间观察到的工作区变动（不属本分区，只做提醒）

- 18:15 UTC 前后，工作区已有暂存的补丁重构：
  - 101 改名为 101-role-boundary-split；
  - 105、112、113、116 的 .patch 删除或并入其他补丁；
  - `patches/drafts/` 已清空；
  - 新增 `121-sched-cap-while-decoding.patch`。
- 已核对：现 `patches/120-sched-protect-chain.patch` 与 HEAD 中的 `patches/drafts/120-sched-protect-chain-v2.patch` 逐字节相同（sha256 ef1744b3…），S0 基线没有丢。
- 因此 §3 #32 和 §11 里的补丁路径（例如 `patches/112-sm80-indexer-kernels.md:7` 现为 `patches/110-sm80-dsa-indexer.md:8`）应以重构后的名字为准。
- 0923a 的清单 `build/image/0923a.patches.txt` 仍用旧名。追溯必须锚定 git 提交 c405467 与 `build/image/0923a.patch_sha.txt`，出分前不要让这两处失效。
