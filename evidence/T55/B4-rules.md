# T55 · B4-rules：赛规转述与判定实现核对（以 task.md 为唯一准绳）

- 作者：Claude subagent（B4 分区）；2026-09-23T18:10Z（UTC）。只读审查：除本文件外未改任何文件，未提交，未碰 pod。
- 快照：HEAD `5734843` + 工作区。T54a 正在改 `scripts/pod/jobs/dev_ladder_template.sh`、`scripts/pod/verify/make_kit.sh`，并新增未跟踪的 `scripts/pod/verify/level_verdict.py`。模板按 HEAD 版和工作区版分别说明。
- 读过的内容：
  - `llm-challenge-arena-v1/` 全部：task.md 1–710 逐行；challenge.json、config.json、datasets/…/public-manifest.json 与 README.txt。
  - `s1-dev/`：`README.md`、`run_dev.py`、`harness/s1_common.py`、`s1_score.py`、`s1_loadgen.py`（计时段 44–175）。
  - 判定与校验：`scripts/score_formal.py`、`scripts/pod/verify/*`（全部 20 个文件，含在制的 level_verdict.py）、`scripts/check_submission.py`、`scripts/ladder_search.py`、pod 梯子 job。
  - 关键词检索：860 个已跟踪文本文件（不含只读目录、build/scratch、tests/queue_archive）。关键词覆盖排名、门、余量、TPOT、TTFT 口径、爬坡、单档时长、开发集用途、能力门、flush 和诚实要求；命中处逐条读了原文。
- 输入哈希（sha256 前 16 位）：
  - task.md `6d572313067c60ed`；challenge.json `2af588e4c76ddb30`
  - s1_score.py `8da35984ce6bc88e`；s1_common.py `a7238b5f1adf728b`；run_dev.py `4e5fdbd479b22ba0`；s1-dev/README.md `c0c518ea6ed4d982`
  - score_formal.py `f337997092db01e2`；analyze_run.py `5e8fc4e512bdf4f1`；ladder_search.py `806ff99e8277049f`；check_submission.py `c00738717adc50fa`
  - dev_ladder_template.sh（工作区）`e09f70dc25a874e4`
  - 026 N18 raw `486f0410…bfdb59`（完整值见 §8）

**计数**：与 task.md 不一致 **38 处**（高 10 / 中 18 / 低 10）。另有相邻问题 4 处（不属于 task.md 规则，但会误导）、UNSURE 3 处。转述正确的地方列在 §6。

## 0. 最会误导后续智能体的 10 条

1. **pod 梯子的档判定不是 11 道门**（B1–B4）。判定依据是 `analyze_run.py` 的 `formal_est_all_pass`：
   - 缺 TPOT 会被记成 0。026 N18 的 tpot_p95 是 0.2189，本应 FAIL；删掉 tpot_s 后判 PASS。
   - 空桶、7 条 HARNESS_DATA 错误也都判 PASS（VERIFIED，见 §8）。
2. **所有 pod 梯子都不确认 flush 是否成功**（A9/B10/B11）。`run_dev.py:232` 丢弃了 flush 的返回值；HEAD 模板、13 份 job，以及 T54a 在制的 `level_verdict.py`，都没有补上这一步。task.md:230 规定：清不掉，该档失败。
3. **`score_formal.py` 不检查整集完整性**（B7）。只有 400/722 条的 raw 也判 `estimated PASS`，coverage=1.0（VERIFIED）。它只能放在完整性校验之后使用，例如 level_verdict.py。
4. **`ttft_decomp.py` 和 `cachecmp.py` 按数据集的 `phase` 分桶**（B5/B6）。在 026 N18 上，187/722 条用错了门限：它们报"intra 超限 68 条"，而 harness 分桶下 overall_intra 实际是 18 条、fast_intra 是 10 条（VERIFIED）。这与 F91 的口径错误是同一类。
5. **dev 的 ALL_PASS 不能当档结论**（A7/A8）。`summary.json` / `report_*.json` 里的 `ALL_PASS`、`evaluation_status` 只算 10 道门，用点估计 p95，没有 TPOT 门，也没有统计余量。
6. **`s1-dev/README.md` 的旧口径全部作废**（A3–A6）：
   - :6 "10 条硬门"
   - :50–55 "TPM 排名"
   - :57 门清单少两道
   - :64 "客户端计时"
7. **`challenge.json:2` 摘要的排名过时**（A1）。它写的是"N@SLO → TPM(decode) → chain_start p95"。它的 `content` 字段与 task.md 逐字节相同，过时的只有摘要。
8. **`ladder_search.py` 默认 `--gate-policy dev`**（B8），即 10 门、点估计、无 TPOT 门。
9. **把 dev N 换算成正式 N**，违背 task.md:354（C3/C4）。出处：`tests/TIERS.md:22`、`R12:118`。
10. **`research/shared/pipeline.md:45,65` 让人钉 digest、写 `@sha256`**（C1）。平台拒收这种写法（F55；45734/45735 因此部署失败）。

## 1. 只读输入里的过时或不一致内容

这些文件不能改。对策都是不再引用，一律以 task.md 为准。

| ID | 位置 | ↔ task.md | 问题 | 一行更正 | 级别 |
|---|---|---|---|---|---|
| A1 | `llm-challenge-arena-v1/challenge.json:2`（abstract） | 72, 74, 593, 599–606, 610–614 | 写"排名按 N@SLO → TPM(decode) → chain_start p95" | 排名只看 `n_at_slo` → `tpot_mean`，TPM 与 chain_start 都不参与排名。content（:5）与 task.md 逐字节相同（diff 为空），过时的只有 abstract | 中 |
| A2 | `challenge.json:45,52`；`datasets/paper2arm-…/1/resources/reference/README.txt:1` | 357–379, 466–478 | 写"human_review_only / No automated scoring"，还要求"bundle synthetic fixtures" | 这是平台模板占位。计分由平台自动完成，提交物只有 submission.json | 低 |
| A3 | `s1-dev/README.md:6` | 519 | "v7，10 条硬门" | 每档是 11 道硬门 | 中 |
| A4 | `s1-dev/README.md:50–55` | 72–74, 548–549, 593, 599–606 | "排名只看两项：n_at_slo、tpm_all" | 排名是 `n_at_slo` → `tpot_mean`（越小越好）；TPM 只回报，不排名 | 高 |
| A5 | `s1-dev/README.md:57` | 519, 577 | 门清单缺 `gated_phases_have_samples` 与 `tpot_p95 ≤ 0.10` | 以 task.md:519 的 11 道门为准 | 中 |
| A6 | `s1-dev/README.md:64` | 205, 556 | "TTFT 从发请求前开始计时" | 正式口径是服务端的 `prefill_finished_time − request_received_ts`；只有缺任一字段才回退到客户端计时 | 中 |
| A7 | `s1-dev/harness/s1_common.py:40–43`；`s1_score.py:10,451–464,495` | 519, 577 | GATES 只有 10 道，SERVICE_GATES_PASS 不含 TPOT | dev 判定缺 `tpot_p95` 门，只能由 score_formal 补上 | 中 |
| A8 | `s1_score.py:461`（TTFT 门 = 点估计 `p95 <= lim`）；`run_dev.py:266–271,289–291`（lane=dev、exit-code-mode=execution，summary 取 ALL_PASS） | 519, 565–567 | dev 的 ALL_PASS / evaluation_status 没有统计余量和 TPOT 门；门失败时 run_dev 也返回 0 | 档结论只用 score_formal 的估计门，且必须先过完整性校验；dev ALL_PASS 只作点估计参考 | 高 |
| A9 | `s1-dev/run_dev.py:46–56,231–232` | 226–230 | flush 不读 JSON，返回值被丢弃；flush 失败也照样进入测量 | 外层必须自己确认 `{"success": true}`，否则判 INVALID。ladder_search.py 已做到，pod 梯子没做（见 B10） | 高 |
| A10 | `s1_score.py:465–469` | 205, 519, 556 | `--lane formal` 额外要求服务端打点覆盖 100% 和 strict basis，task.md 的 11 道门里没有这两项 | 不要把 harness 的 formal 通道当成 task.md 口径；task.md 允许逐请求回退到客户端计时 | 低 |

## 2. 判定与校验实现

`score_formal.py` 和 `ladder_search.py` 是 Codex 的文件，这里只报告，由 Claude 转交 Codex。

| ID | 位置 | ↔ task.md | 问题 | 一行更正 | 级别 |
|---|---|---|---|---|---|
| B1 | `scripts/pod/verify/analyze_run.py:31`（`hard = (not t) or p95 <= lim`；空桶时 `allowed_over=0`、`over=0`，`formal = over <= ao` 为真） | 519（gated_phases_have_samples） | 空桶判 PASS；空 raw 整体判 ALL_PASS（evidence/T53/empty_verdict.json） | 任一桶 n=0 时，该档判 FAIL 或 INVALID | 高 |
| B2 | `analyze_run.py:9,36`（`errors <= 0.01*len(rows)`） | 519 | 5 道数据/错误门被合并成"总错误 ≤1%"：harness_data/render 没要求等于 0，engine/infra 没分开算，没查 coverage=100%，`<` 也写成了 `≤`。7 条 HARNESS_DATA 错误加删 TPOT，仍判 PASS（VERIFIED） | 直接用 harness 的 5 道门（经 score_formal 的 `dev.gates`） | 高 |
| B3 | `analyze_run.py:34–36`（用 `if r.get("tpot_s")` 过滤；列表为空时记 0） | 546–547, 577 | 缺 TPOT 时 tpot_p95=0，判通过，tpot_mean 也是 0。026 N18 因此由 FAIL 翻成 PASS（VERIFIED） | 多 token 请求缺 TPOT 时判 INVALID 或 FAIL（`score_formal.py:291–301` 的做法是对的） | 高 |
| B4 | `analyze_run.py:26,38`，自称"formal-est = task.md rule"。拿它当档判定的有：<br>- `dev_ladder_template.sh:47`（HEAD 版）<br>- 11 份 `ladder_*.sh`（best、best106、best140、dcp、v3、mtp114、bcg114_c2i3、bcg114_c4i2、bcg114_mtp、B1_mtp_c16k、B2_mtp_c8k）的 run_level 末行<br>- `s0_n22.sh:58`、`s0_114_n22.sh:58`<br>- pod 上的 `build/verify_kit/analyze_run.py` 副本 | 519 | 只有四道 TTFT 门的余量算法与 task.md 一致；整体 PASS 缺 5 道门，TPOT 门还可以被绕过 | 档判定改用 level_verdict.py（完整性 + score_formal）；旧的 `LADDER … formal_est=` 行一律不采信（`s0_n22.sh:3–4` 已注明） | 高 |
| B5 | `scripts/pod/verify/ttft_decomp.py:5,8–13`（`LIM={"intra":5,"turn_start":15,"session_start":30,"context_reset":30}`，按 `phase` 分组） | 560–563 | 没有 fast_intra 的 3s 桶。phase 为 intra 或 turn_start 的链首本该按 30s 判，实际分别按 5s（115 条）和 15s（72 条）判，026 N18 上共 187/722 条用错门限。它报"intra over-limit 68"，harness 桶下是 overall 18、fast 10（VERIFIED） | 分桶一律用 `s1_common.in_ttft_gate`，按本轮回放的 `idx_in_chain` | 高 |
| B6 | `scripts/pod/verify/cachecmp.py:15,18` | 560–563, 519 | 按 `phase` 分组报 TTFT p95；缺 TTFT 时记 0（`or 0`），把 p95 压低 | 分桶同 B5；缺值判 INVALID，不得记 0 | 中 |
| B7 | `scripts/score_formal.py:259–308`（coverage 由 dev evaluate 按已有记录计算） | 152（每档整集回放）, 519（coverage=100%） | 不核对 raw 是否恰好覆盖 cohort 全部请求、每条一次。只有 400/722 行也判 `estimated PASS`，coverage=1.0（VERIFIED） | 调用前先做完整性校验：req_id 集合等于 `s1_common.load_index`，且无重复。level_verdict.py:54–76 已实现，待并入所有入口 | 高 |
| B8 | `scripts/ladder_search.py:21,468`（默认 `--gate-policy dev`） | 519, 565–567, 577 | 默认档判定是 10 门、点估计、无 TPOT 门 | 默认改为 `estimated`，dev 只作并列参考（交 Codex） | 中 |
| B9 | `scripts/session_a/remote.py:410`（`gate_policy="dev+tpot"`） | 565–567 | TTFT 门不带统计余量，比 task.md 严，会误判 FAIL | 若复用 session_a，改为 `estimated`；否则随工具退役 | 低 |
| B10 | `scripts/pod/jobs/dev_ladder_template.sh:34–39`（HEAD 版）及 13 份副本（例如 `ladder_best140.sh:22–26`、`s0_n22.sh:43–49`） | 226–230, 152, 519 | 不看 run_dev 的退出码（s0_n22 只打印，不判定）；取目录里最新的 raw；不确认 flush | 包装层自己 POST `/flush_cache`，校验 2xx 和 `{"success": true}`（与 `ladder_search.py:162–` 的 guard 相同）；或解析 run_dev.log，要求有 `flushed KV via` 且没有 `flush failed`。否则判 INVALID。raw 一律从 summary.json 取 | 高 |
| B11 | `scripts/pod/verify/level_verdict.py`（未跟踪，T54a 在制）:42–76，:97 | 228–230 | 已补上退出码、summary 定位和完整性检查，但仍不检查 flush。另外，tpot_p95 为 None 时 :97 的格式化会抛异常，score_formal 抛 ValueError 时也一样；进程以退出码 1 结束，被当成"门 FAIL"而不是 INVALID（INFERRED：读码，未运行） | 加 flush 确认；任何异常一律退出码 2，判 INVALID | 中 |
| B12 | `scripts/check_submission.py:4,29,115–119,184–185,188–189,196–199` | 144, 376–377, 387, 457 | 缺 `env`、`model_name` 或 `--served-model-name` 时报 ERROR，并标成"task.md rule"。按 task.md，env 与 model_name 可选（默认 `default`），served-model-name 保持一致只是建议 | 这几项降为 WARN。SGLang 在 env 里设 `SGLANG_OPT_USE_TOPK_V2=0` 是 task.md:424 的硬要求，保持 ERROR | 中 |
| B13 | `check_submission.py:6,137–141`（禁止 `@sha256`） | 283, 374（"建议钉 digest"） | 与 task.md 的建议相反 | 保留这项检查（有平台实测依据，F55：`notes/findings.md:291–295`），但标注为"平台实测例外，不是 task.md 规则" | 低 |
| B14 | `check_submission.py:13,69–77,101`（默认对照 v0.5.20 的 `src/sglang`，且只解析 `arg_groups/fields`）；`scripts/submit_official.sh:18`（`SKIP_FLAG_CHECK=1` 旁路） | 387 | 对真实提交的 `submission/official-0923-A.json` 误报 ERROR：<br>- 默认源（v0.5.20）不认识 `--cuda-graph-max-bs`；<br>- 换成 base_exact 后，17 个 flag 报 unknown，因为 base_exact 的 flag 定义在 `server_args.py` 的 `name: A[` 字段里。<br>结果整个校验器被旁路，task.md:387 的格式检查在 L3 提交上没有执行（VERIFIED，已运行）。这是工具正确性问题，不是 task.md 文字本身的问题 | 默认用 `--sglang-src build/base_exact/sglang/srt` 并解析 server_args.py 的 A[ 字段；旁路只跳过 flag 检查，不跳过格式检查 | 中 |

## 3. 文档转述

| ID | 位置 | ↔ task.md | 问题 | 一行更正 | 级别 |
|---|---|---|---|---|---|
| C1 | `research/shared/pipeline.md:45,65` | 283, 374（+F55） | "拿到镜像后钉 digest"；示例写 `@sha256:<digest>` | image 用唯一、永不重建的 `name:tag`；平台拒收 tag@sha256（F55） | 中 |
| C2 | `research/shared/pipeline.md:12` | 500, 519 | "沿 2/6/10/14/18/22 梯子爬坡"；"（v7 口径）" | 改为"从 N=10 起，过 +4、不过 −4，已过档之上首次失败即停；11 道门"，删掉"v7 口径" | 低 |
| C3 | `tests/TIERS.md:22` | 354 | "攒成换算关系（开发集 N 对应正式 N 多少…）" | 配对结果只作 A/B 背景记录，不推导正式 N（R14:105 已说明不存在换算系数） | 中 |
| C4 | `research/claude/R12_full_experiment_review.md:118` | 354 | "要用它校准 dev 集与线上的关系" | 同 C3：正式成绩只用来检验 dev 上 A/B 的方向，不做换算 | 中 |
| C5 | `dev_ladder_template.sh:1–2`（HEAD 版，T54a 已在工作区修改）及 13 份副本的头注：`ladder_*.sh:5–7`、`s0_n22.sh:10`、`s0_114_n22.sh:10` | 500–515 | 头注写"Self-test ladder like the formal climb … (default 10 14 18 22 26) … stop at the first failure"。实际默认是 `18 22 26`、不下探；而正式爬坡从 10 起，首档失败会下探 | 改为"内部自测：从 N18/N22 起、不下探（用户 09-23 规定），不是正式爬坡" | 中 |
| C6 | `docs/histories/2026-09/20260923-0500-pod-verify-and-analysis-tools.md:21` | 500–515 | "dev_ladder_template（像正式爬坡一样 N=10→26）" | 历史记录可以保留，但加注"不是正式爬坡"，或删掉这一句 | 低 |
| C7 | `plans/active/2026-09-23-8card-selftest-ladder.md:17,43` | 519 | "爬坡按正式规则估算"，实际判据是 analyze_run 的 formal_est_all_pass（见 B1–B4） | 改为"判据 = level_verdict.py（完整性 + score_formal，11 门）" | 中 |
| C8 | `research/claude/R7_top_players_analysis.md:77` | 565 | "turn_start (n≈65) tolerates about 11.5%" | 整数允许数 `allowed_over(65)=6`，即 9.2%；7/65 时下界为 0.0516，判 FAIL | 低 |
| C9 | `research/claude/R14_fable_eval_review.md:96` | 560, 567（+F31） | 正式 fast 允许超标 485 条，这是把 fast 桶当成全部 9023 条 intra 算出来的 | fast 只是 intra 中冻结未命中 ≤4096 的子集，规模未定（见 U1）；不要用 485 | 中 |
| C10 | `HANDOFF.md:36` | 519（+F77 的更正，`notes/findings.md:457`） | "120 N6（四门全过，自估）" | 改为"四道 TTFT 门估算通过；harness 判 overall 5.29s FAIL；其余 7 门未判"，或随 HANDOFF 一起重写 | 低 |
| C11 | `README.md:34`（"**能跑 + 能力**…能力冒烟 12/12"）、`notes/quality.md:19,22`、`docs/histories/2026-09/20260923-0650-sm80-sparse-attn-dcp.md:21` | 482–484, 535–537 | 把 12 题冒烟写成了"能力" | 写明"12 题冒烟，不是能力门"。能力门是 aime26（44 题）和 gpqa（156 题）的 points 都严格 >90，至今未测 | 中 |
| C12 | `patches/000-interface-compliance.md:27` | 228 | 说平台只在"每档测量前"调用 flush | 平台在两处调用 flush：每档正式测量前，以及切换并发档前 | 低 |
| C13 | `notes/findings.md:150`（F35） | 577–581 | "TTFT gates bind, not TPOT"。依据是对手已通过档位的中位数，有选择偏差 | 加注"只对前排已通过档位成立"。我们的 025b N10（tpot_p95 0.1315）、026 N18（0.219）、035 N22 都挂在 TPOT 门上（`findings.md:494–496,577–579`） | 中 |
| C14 | `research/archive/directions.md:55`、`research/archive/rule-handoff-2026-09-22.md:22` | 593–606 | "榜单显示能力分平均，与 N@SLO 排名不一致，需向主办方确认" | 排名以 task.md:593–606 为准；这个悬而未决的问题随 archive 一并处理 | 低 |

## 4. 相邻问题

这几处不属于 task.md 规则，但会误导后续智能体。

- **D1（高风险）** `research/shared/pipeline.md:33` 写"用完立刻删 `trisol inference delete`"。task.md:133–136 的这句话针对一次性的 hang 服务。本项目的常驻 8 卡服务，按用户规定不得删、不得停，停也须用户同意（HANDOFF.md:14、用户记忆）。应改写或删除。
- **D2** `rule.md:11` 写"3 人 N=22 … 目标 N26 或 N22 且 tpot<0.0273"，`README.md:37` 写"无人过 N=26"。这两处都已过时：`notes/decisions.md:51` 记录第一名 CalvinCao 已到 N26，tpot_mean 0.0551。
- **D3** `plans/active/2026-09-23-8card-selftest-ladder.md:45` 写"首档失败再下探 14/10"，与用户 09-23"不下探"的规定冲突（`dev_ladder_template.sh:25–26`）。
- **D4** `research/claude/R7_top_players_analysis.md:84` 写"quota is 3 scored runs/day"，而用户确认的是每天 2 次（`notes/findings.md:296`）。

## 5. UNSURE

- **U1 正式 fast_intra 桶的规模与余量。** 两处来源矛盾：
  - task.md:567 说 fast 允许条数比点估计多约 19%（约相当于 3.0s→3.6s）。用 Clopper–Pearson 反推，n_fast 约 1.4k，允许约 84 条。
  - F31（`notes/findings.md:137–138`）按 cohort 文件算，fast 占 intra 的 85.2%，n 约 7.7k，允许 417 条，只多 8%（即超标率 5.4%）。R7:78 用的就是这个规模。

  两者差别决定 fast 门到底有多少余量。确定办法：45979/45980 出分后，看报告里的 `allowed_over`、`rate_ci_lower`（task.md:567 说会给出）和 fast 桶的 n。
- **U2 p95 的取法。** harness 的 `q()` 取 `sorted[floor(0.95n)]`（`s1_common.py:98–100`）。当 0.95n 恰为整数（n 是 20 的倍数）时，它比最近秩法高一位；例如 n=20 时取的是最大值。task.md:52 的措辞两种理解都说得通，但 task.md:322 说 dev 与正式同口径，所以大概率一致。影响范围只有 `pass_point` 和不带余量的 tpot_p95 门。确定办法：用正式报告的 `pass_point` 与同档 p95 对照。
- **U3 主办方的置信区间算法没有公开**（task.md:565–567 只给了规则描述）。现有两套实现一致：score_formal 与 analyze_run 都用精确 Clopper–Pearson，在 n=1…388 上结果逐一相同。结果也与仓库里引用的数吻合：dev 23/27/3/22，n=808→51，n=9023→485，分别见 R14:96 和 TEST_PLAN SLO-01。确定办法同 U1。

## 6. 已核对一致，不需要改

- **排名与目标**：`rule.md:11–12`（口径部分）、`AGENTS.md:3,21`、`README.md:3`、`research/claude/R8_next_directions.md:8–12`、`notes/decisions.md:51–54`、`research/claude/base/00-summary-mainline.md:62`、`plans/prompts/_context-0924.md:31–37`。
- **红线**：`rule.md:63–67`（不关 thinking、不压输出预算、不截历史、不删 tools，时间戳如实，flush 真清，command 是 argv，TOPK_V2）。
- **流程文档**：`research/shared/pipeline.md:11,13,19,75`（能力门 >90、11 门加余量、正式余量实例、单档约 4 小时）。
- **测试计划**：`tests/TEST_PLAN.md:65,69,71,108,147`；`tests/TIERS.md:12`（dev 不能确认正式 N）。
- **补丁 000 说明**：`patches/000-interface-compliance.md:12–17,43–44,59–60`。
- **findings 与计划**：F2、F9、F10、F23（`notes/findings.md:18,47–53,94–98`）；`plans/active/2026-09-23-offline-queue-3h.md:36`；`plans/active/2026-09-23-8card-selftest-ladder.md:13`。
- **引用与定义**：
  - 仓库里 15 处 `task.md:<行>` 引用全部指向正确内容。
  - 所有文档对 fast_intra 的定义都用冻结的 `uncached_expected ≤ 4096`（task.md:560）。
  - 没有任何工具在测量时使用 `--max-chains/--no-gap/--include-all`（task.md:355）。
  - 没有任何文档照抄 challenge.json 摘要里的排名。
- **实现层**：
  - score_formal：共 11 门；只对四道 TTFT 门加 Clopper–Pearson 余量，超标用严格 `>` 判；tpot_p95 不带余量；tpot_mean 按逐请求均值。除 B7 外，与 task.md:519、546–547、565–567、577 一致。
  - ladder_search：official-climb 状态机与 task.md:500–513 的全部样例路径一致；每档前严格校验 flush JSON，符合 task.md:226–230。
  - loadgen：TTFT = 首事件 `prefill_finished_time` − 末事件 `request_received_ts`；TPOT = (末 − 首)/(n−1)，按 token 计。与 task.md:199–205、546 一致。
  - interference.py 与 coldprobe.py：TPOT 按 token 计，TTFT 用服务端口径（task.md:521、556）。
  - `notes/ledger-8card.md` 里没有任何 `formal_est=True`，所以 B1–B3 迄今没有造成假 PASS。

## 7. 按文件建议（供 T55 汇总）

| 文件 | 建议 | 条目 |
|---|---|---|
| `llm-challenge-arena-v1/challenge.json`、`s1-dev/README.md`、`s1-dev/harness/*`、`s1-dev/run_dev.py` | 只读，保留；在 AGENTS / rule 中注明"只信 task.md，不要引用这几处" | A1–A10 |
| `scripts/pod/verify/analyze_run.py`（及 `build/verify_kit/` 副本） | 不再用作判定：去掉 ALL_PASS，或改为只输出诊断 | B1–B4 |
| `scripts/pod/verify/ttft_decomp.py`、`cachecmp.py` | REWRITE：改用 `in_ttft_gate` 分桶，缺值不记 0 | B5, B6 |
| `scripts/pod/jobs/dev_ladder_template.sh` 及 13 份副本 | 模板由 T54a 改写，但还缺 flush 确认；已跑完的旧副本若留作追溯，需加注 | B4, B10, C5 |
| `scripts/pod/verify/level_verdict.py`（在制） | 补上 flush 确认；异常一律退出码 2 | B11 |
| `scripts/score_formal.py`、`scripts/ladder_search.py`（Codex 的文件） | KEEP；B7、B8 交 Codex | B7, B8 |
| `scripts/check_submission.py`、`scripts/submit_official.sh:18` | REWRITE：调整 ERROR/WARN 分级、改 flag 来源、缩小旁路范围 | B12–B14 |
| `research/shared/pipeline.md` | REWRITE（:12、:33、:45、:65） | C1, C2, D1 |
| `tests/TIERS.md`、`R12`、`R14`、`R7` | REWRITE 对应段落 | C3, C4, C8, C9, D4 |
| `HANDOFF.md`、`README.md`、`notes/quality.md`、`rule.md:11` | REWRITE 对应句子 | C10, C11, D2 |
| `plans/active/2026-09-23-8card-selftest-ladder.md` | REWRITE（:17、:43、:45） | C7, D3 |
| `notes/findings.md:150`（F35） | 追加更正条目（findings 只追加，不改原文） | C13 |
| `patches/000-interface-compliance.md:27`、`docs/histories/…0500:21` | 小改或加注 | C12, C6 |
| `research/archive/*` | 按 B2 分区的结论处理 | C14 |

## 8. 复现（VERIFIED 条目的脚本、输入与输出）

- **输入**：`evidence/T53/026_N18_raw.jsonl`，722 行，sha256 `486f041027abde2a1702e0c19190ce11e7001eb0cc4055d3773aff8c95bfdb59`。
- **运行条件**：在仓库根目录执行；只用 CPU，设 `PYTHONDONTWRITEBYTECODE=1`，不写只读目录。
- **变体**（run 元数据为合成的 `{"wall_s":1260}`）：
  - B：删除全部 tpot_s。
  - BC：删 TPOT，并去掉 turn_start 桶的 20 行。
  - BD：删 TPOT，并把前 7 行改成 HARNESS_DATA 错误。
  - G：只取前 400 行，tpot_s 统一设为 0.05，以排除 TPOT 的影响。

```bash
set -euo pipefail; export PYTHONDONTWRITEBYTECODE=1
T=$(mktemp -d); RAW=evidence/T53/026_N18_raw.jsonl; sha256sum $RAW
python3 - "$T" "$RAW" <<'PY'
import json, sys
sys.path.insert(0, 's1-dev/harness'); import s1_common as C
T, RAW = sys.argv[1:3]
rows = [json.loads(l) for l in open(RAW) if l.strip()]
def dump(name, rs):
    with open(f'{T}/{name}.jsonl', 'w') as f:
        for r in rs: f.write(json.dumps(r) + '\n')
drop_tpot = lambda rs: [{k: v for k, v in r.items() if k != 'tpot_s'} for r in rs]
tpot005 = lambda rs: [dict(r, tpot_s=0.05) if r.get('tpot_s') is not None else r for r in rs]
herr = [dict(r) for r in rows]
for r in herr[:7]: r.update(error='HARNESS_DATA:synthetic', error_class='HARNESS_DATA', ttft_s=None)
dump('A_orig', rows); dump('B_no_tpot', drop_tpot(rows))
dump('BC_no_tpot_no_turn', drop_tpot([r for r in rows if not C.in_ttft_gate(r, 'turn_start')]))
dump('BD_no_tpot_harness_err7', drop_tpot(herr)); dump('G_trunc400_tpot005', tpot005(rows[:400]))
json.dump({'wall_s': 1260, 'config': {'N': 18, 'set': 'dev-combined-v1'}}, open(f'{T}/run_synth.json', 'w'))
LIM = {"intra": 5, "turn_start": 15, "session_start": 30, "context_reset": 30}   # ttft_decomp.py:5
bad = sum(LIM.get(r['phase'], 30) != (5 if C.phase_gate(r) == 3.0 else C.phase_gate(r)) for r in rows)
over_phase_intra = sum(r['phase'] == 'intra' and r['ttft_s'] > 5 for r in rows)
over_h = {g: sum(C.in_ttft_gate(r, g) and r['ttft_s'] > l for r in rows)
          for g, l in (('fast_intra', 3), ('overall_intra', 5), ('turn_start', 15), ('chain_start', 30))}
print(f'ttft_decomp: rows with wrong limit={bad}/722; "intra over-limit"={over_phase_intra}; harness over={over_h}')
PY
for t in A_orig B_no_tpot BC_no_tpot_no_turn BD_no_tpot_harness_err7; do
  python3 scripts/pod/verify/analyze_run.py s1-dev/harness $T/$t.jsonl $T/$t.v.json >/dev/null 2>&1
  python3 -c "import json;d=json.load(open('$T/$t.v.json'));print('analyze_run $t formal_est_all_pass=%s tpot_p95=%s errors=%s turn_n=%s'%(d['formal_est_all_pass'],round(d['tpot_p95'],4),d['errors'],d['gates']['turn_start']['n']))"
done
for t in A_orig BC_no_tpot_no_turn BD_no_tpot_harness_err7 G_trunc400_tpot005; do
  python3 scripts/score_formal.py --raw $T/$t.jsonl --run $T/run_synth.json --out $T/$t.s.json >/dev/null
  python3 -c "import json;d=json.load(open('$T/$t.s.json'));g=d['estimated']['gates'];print('score_formal $t passed=%s rows=%d coverage=%s failed=%s'%(d['estimated']['passed'],d['dev']['n_attempted'],d['dev']['coverage'],[k for k,v in g.items() if not v]))"
done
python3 -c "
import sys; sys.path.insert(0,'scripts'); import score_formal as s
print('allowed_over', {n: s.allowed_over(n) for n in (20,65,314,328,388,808,1400,7700,9023)}, 'L(7,65)=%.4f'%s.binomial_lower(7,65))"
```

输出（2026-09-23 约 18:10Z 实跑，rc=0）：

```
486f041027abde2a1702e0c19190ce11e7001eb0cc4055d3773aff8c95bfdb59  evidence/T53/026_N18_raw.jsonl
ttft_decomp: rows with wrong limit=187/722; "intra over-limit"=68; harness over={'fast_intra': 10, 'overall_intra': 18, 'turn_start': 0, 'chain_start': 12}
analyze_run A_orig formal_est_all_pass=False tpot_p95=0.2189 errors=0 turn_n=20
analyze_run B_no_tpot formal_est_all_pass=True tpot_p95=0 errors=0 turn_n=20
analyze_run BC_no_tpot_no_turn formal_est_all_pass=True tpot_p95=0 errors=0 turn_n=0
analyze_run BD_no_tpot_harness_err7 formal_est_all_pass=True tpot_p95=0 errors=7 turn_n=20
score_formal A_orig passed=False rows=722 coverage=1.0 failed=['tpot_p95<=0.10']
score_formal BC_no_tpot_no_turn passed=False rows=702 coverage=1.0 failed=['gated_phases_have_samples', 'turn_start(<=15s)', 'tpot_p95<=0.10']
score_formal BD_no_tpot_harness_err7 passed=False rows=722 coverage=1.0 failed=['harness_data==0', 'tpot_p95<=0.10']
score_formal G_trunc400_tpot005 passed=True rows=400 coverage=1.0 failed=[]
allowed_over {20: 3, 65: 6, 314: 22, 328: 23, 388: 27, 808: 51, 1400: 84, 7700: 417, 9023: 485} L(7,65)=0.0516
```

这组输出同时核对了 `_context-0924.md` 第 5 条：026 N18 的 fast 10/23、overall 18/27、chain 12/22，tpot_p95 0.2189，FAIL。

**check_submission（B12、B14）的复现：**

- 执行 `python3 scripts/check_submission.py submission/official-0923-A.json --final`，报 `ERROR flags not found … ['--cuda-graph-max-bs']`。
- 加上 `--sglang-src build/base_exact/sglang/srt` 再跑，报 17 个 flag "not found"。
- 另做一份删掉 env 和 model_name 的最小 submission，得到 5 个 ERROR，其中 3 个针对的是 task.md 规定为可选的字段。
