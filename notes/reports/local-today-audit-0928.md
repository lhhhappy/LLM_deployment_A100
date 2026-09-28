# 09-28 本地 N30 原始记录独立复核

2026-09-28；状态：离线 raw 复算完成。作者：Codex。仅复核本地 N30，未上传、未运行 GPU、未修改队列或共享报告。

**chain 结论：eznb→eznc→ezno 在三者共同的 1687 条请求上是 12→7→6；修好 6、没有新坏、持续 6。** 钉池阶段修好 5 条；118+scatter 组合再修好 1 条；随后 MoE 与 KDA 阶段没有减少 chain 超标条数。本轮同 ID 的 TPOT p95 从钉池后的 120.941 ms 回到最终 96.730 ms。这里是定时诊断的实测，不是完整 cohort 成绩，也没有同配置重复支持每一段小变化的因果或稳定性。

## 方法和请求完整性

直接导入仓库原工具 [compare_window.py](../../scripts/analysis/compare_window.py) 与 [window_gates.py](../../scripts/analysis/window_gates.py)，加载原 `score_formal.load_harness()`。每一对先取 `req_id` 交集，按原 `in_ttft_gate`、原分位数方法统计；对每条相同 ID 检查 chain/phase/index/edge、冻结 token 数、输出预算和 effective gap 完全一致。成功请求的 prompt/output 数分别等于 `glm_tokens`/`max_output_i`。所有已读行唯一、无 error、时间值有限，TPOT 非负。没有修改 harness 门或统计余量。

九份 `N30/raw_*.jsonl` 的 run 元数据与 `loadgen.log` 内 TIMED_WINDOW 都满足：`n_attempted = dispatched = completed = raw 唯一行数`，2400 秒准入后 `DRAINED`、`outstanding=[]`、runner_rc=0、flush HTTP 200。cohort 哈希均为 `b78593bdea138f58`，workload 哈希 `97175a1e2ea92d15`，N=30；原 cohort 311 链/5601 请求，全部明确 `full_cohort_complete=false`。这是对排空收据与 raw 的一致性核对；本地未存完整派发 ledger，未重新计算其 ledger_sha256。归档 `level_verdict.json` 仍是 INVALID（缺失/额外 ID 的原诊断），不能覆盖成正式 PASS。

j/k/p 只有旧 `window/raw.jsonl`；核对其 SHA256 与 snapshot 的 downloaded_raw_sha256、行数与 completed_rows 一致。快照写着 running，不能冒充后来停止时的终态；慢的未完成请求可能缺失。ezni 在本地 evidence 没有找到 raw，因此不编造量化比较。

| run | 实际读取行数 | 证据范围 | chain 超标 | TPOT p95 ms（各自全 raw） | 原始文件 |
| --- | --- | --- | --- | --- | --- |
| eznb | 1689 | 40 min admission，已排空 | 12 | 98.387 | [raw](../../evidence/L130eznb-tail_rot150_n30_chainmax16k_fix_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790559367.jsonl) |
| eznc | 1726 | 40 min admission，已排空 | 7 | 121.140 | [raw](../../evidence/L130eznc-tail_rot150_n30_chainmax16k_fix_mamba400_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790562930.jsonl) |
| eznd | 1680 | 40 min admission，已排空 | 18 | 89.987 | [raw](../../evidence/L130eznd-tail_rot150_n30_chainmax16k_fix_133_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790566218.jsonl) |
| ezne | 1679 | 40 min admission，已排空 | 12 | 108.925 | [raw](../../evidence/L130ezne-tail_rot150_n30_124m_on_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790569827.jsonl) |
| eznf | 1689 | 40 min admission，已排空 | 14 | 96.555 | [raw](../../evidence/L130eznf-tail_rot150_n30_124m_off_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790572950.jsonl) |
| eznj | 956 | 旧 OPEN 快照，22.45 min | 8 | 165.469 | [raw](../../evidence/L130eznj-tail_rot150_n30_124m_on_mamba400_40m/window/raw.jsonl) |
| eznk | 1239 | 旧 OPEN 快照，28.31 min | 7 | 126.146 | [raw](../../evidence/L130eznk-tail_rot150_n30_chainmax16k_fix_mamba400_scatter_40m/window/raw.jsonl) |
| eznl | 1778 | 40 min admission，已排空 | 7 | 113.985 | [raw](../../evidence/L130eznl-tail_rot150_n30_chainmax16k_fix_mamba400_118_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790586237.jsonl) |
| eznm | 1827 | 40 min admission，已排空 | 6 | 103.322 | [raw](../../evidence/L130eznm-tail_rot150_n30_chainmax16k_fix_mamba400_118_scatter_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790591008.jsonl) |
| eznn | 1844 | 40 min admission，已排空 | 6 | 95.543 | [raw](../../evidence/L130eznn-tail_rot150_n30_chainmax16k_fix_mamba400_118_scatter_moetune_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790594907.jsonl) |
| ezno | 1869 | 40 min admission，已排空 | 6 | 94.627 | [raw](../../evidence/L130ezno-tail_rot150_n30_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N30/raw_s1-dev-longchain-v5g-tail-rot150_N30_1790598252.jsonl) |
| eznp | 886 | 旧 OPEN 快照，19.46 min | 8 | 97.611 | [raw](../../evidence/L130eznp-tail_rot150_n30_chainmax16k_fix_mamba400_118_scatter_moetune_kda_133tuned_40m/window/raw.jsonl) |

完整派生量、源 raw SHA256、配置与收据摘要见 [audit.json](../../evidence/review-0928-official-local/local-today/audit.json)。逐 ID CSV 保留所有 gate 修好/新坏/持续、phase、边类型、两边派发时间、TTFT/TPOT 与缓存量。`compare_window.compare()` 原函数的固定“all windows open”说明在派生摘要中改为本次实际证据范围，仅修说明文字，不改统计。

## 三者“共同 1700 多条”的精确分母

eznb、eznc、ezno 各自总数为 **1689 / 1726 / 1869**。两两交集 b∩c=1687、c∩o=1726、b∩o=1689；**三者交集为 1687**，不是 1700 多条。1726 只适用于 eznc 与 ezno；把它用于 eznb 是分母错误。共同 ID 完整列表在 [triple-b-c-o-req_ids.txt](../../evidence/review-0928-official-local/local-today/triple-b-c-o-req_ids.txt)。

| 同一批 1687 请求 | chain >30s | fast >3s | overall >5s | turn >15s | TPOT mean ms | TPOT p95 ms | TPOT >100ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eznb | 12 | 248 | 209 | 11 | 54.932 | 98.387 | 83 |
| eznc | 7 | 40 | 21 | 6 | 59.996 | 120.941 | 118 |
| ezno | 6 | 26 | 13 | 2 | 52.219 | 96.730 | 78 |

该共同集的 gate 样本数固定：chain 238、fast 1238、overall 1383、turn 66，TPOT 1687。chain p95 分别 30.305 / 24.969 / 18.859 s；统计余量不能替代 chain 第一的逐 ID 取舍，也不能据此声明正式通过。

## 逐步变化：每行严格使用本行共同 ID

| 对照 | 改动/范围 | 共同 ID | chain | chain 修/新/持续 | fast / overall / turn | TPOT p95 ms | TPOT >100ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b-c | 钉池 400 | 1687 | 12→7 | 5/0/7 | 248→40/209→21/11→6 | 98.387→120.941 | 83→118 |
| c-l | 118 单项 | 1726 | 7→7 | 0/0/7 | 40→44/21→16/7→7 | 121.140→116.470 | 122→108 |
| c-k | scatter 单项，旧 OPEN | 1239 | 7→7 | 0/0/7 | 34→34/21→18/5→3 | 158.419→126.146 | 105→83 |
| l-m | 118 上加 scatter | 1778 | 7→6 | 1/0/6 | 47→34/16→9/7→3 | 113.985→105.248 | 108→99 |
| c-m | 118+scatter 组合 | 1726 | 7→6 | 1/0/6 | 40→30/21→9/7→3 | 121.140→106.463 | 122→99 |
| m-n | 再加 MoE 调参 | 1827 | 6→6 | 0/0/6 | 34→40/9→18/3→4 | 103.322→96.142 | 100→83 |
| n-o | 再加 KDA prefill | 1844 | 6→6 | 0/0/6 | 40→28/18→13/4→3 | 95.543→94.627 | 83→83 |
| c-o | 钉池→最终组合 | 1726 | 7→6 | 1/0/6 | 40→26/21→13/7→3 | 121.140→97.599 | 122→82 |
| b-o | 47043 配置→最终组合 | 1689 | 12→6 | 6/0/6 | 248→26/209→13/11→2 | 98.387→96.730 | 83→78 |

**分母更正：** c→m 的严格同 ID 候选 p95 是 **106.463 ms**，不是 m 全 raw 的 103.322 ms；l→m 是 105.248 ms。m→n 的严格同 ID 候选是 **96.142 ms**，不是 n 全 raw 的 95.543 ms。c→k 的基线在共同 1239 条是 **158.419 ms**，不能拿 c 全 raw 121.140 ms 来配。上述更正不改变 chain 条数，但会影响 TPOT 改善幅度；表中从不混合分母。

b→c 的 5 条修复为 3 个 context_reset、2 个切段 intra，其中开场修 1、60 秒后修 4。l→m 唯一再修好的是 52,218-token 切段 intra `biomaster:canon:member-v5-subagent-4b341feb33056b32c9433b51:llm:7`：40.624→28.307 s（c 中 42.126 s）。c→l、c→k 各自都没有修好 chain，不能把组合增益拆开相加；本次组合观察到过门，不证明交互机制。

m/n/o 持续的 6 条是同一组：5 个切段 intra + 1 个 context_reset，全部在开场首 6 秒内派发；没有 session_start 或 turn_start 的 chain 新坏。n→o 的这 6 条 TTFT 各下降 1.472–1.868 s，但仍在 30 s 外。KDA 步的 TPOT p95 仅下降 0.916 ms，单次差值不能解释为稳定收益。

## 开场、后续与链首类型

为避免 closed-loop 调度变化使同一 ID 换桶，下面按**基线**派发时间固定 cohort：开场 [0,60) 秒、中段 [60,600) 秒、暖稳态 ≥600 秒。既有记录说 b 的“稳态 4 条”实际是开场后的 4 条，其中一条在 596.5 秒，严格十分钟稳态是 3 条。p 中一个新坏 session_start 的基线派发是 66.2 秒、候选却是 7.0 秒；不能把两个运行各自时钟的开场桶当同一批 ID。

| 对照 | 开场 chain | 60–600s chain | ≥600s chain | 按 phase 类型：修/新/持续 |
| --- | --- | --- | --- | --- |
| b-c | 8→7 | 1→0 | 3→0 | context_reset:3/0/1; segmented_intra:2/0/6 |
| c-l | 7→7 | 0→0 | 0→0 | context_reset:0/0/1; segmented_intra:0/0/6 |
| c-k | 7→7 | 0→0 | 0→0 | context_reset:0/0/1; segmented_intra:0/0/6 |
| l-m | 7→6 | 0→0 | 0→0 | context_reset:0/0/1; segmented_intra:1/0/5 |
| c-m | 7→6 | 0→0 | 0→0 | context_reset:0/0/1; segmented_intra:1/0/5 |
| m-n | 6→6 | 0→0 | 0→0 | context_reset:0/0/1; segmented_intra:0/0/5 |
| n-o | 6→6 | 0→0 | 0→0 | context_reset:0/0/1; segmented_intra:0/0/5 |
| b-o | 8→6 | 1→0 | 3→0 | context_reset:3/0/1; segmented_intra:3/0/5 |
| c-o | 7→6 | 0→0 | 0→0 | context_reset:0/0/1; segmented_intra:1/0/5 |
| b-d | 8→11 | 1→1 | 3→6 | context_reset:3/2/1; segmented_intra:2/5/6; session_start:0/1/0; turn_start:0/3/0 |
| b-f | 8→9 | 1→0 | 3→5 | context_reset:2/1/2; segmented_intra:1/3/7; turn_start:0/1/0 |
| f-e | 9→9 | 0→0 | 5→3 | context_reset:1/1/2; segmented_intra:2/0/8; session_start:0/1/0; turn_start:1/0/0 |
| c-j | 7→8 | 0→0 | 0→0 | context_reset:0/1/1; segmented_intra:0/0/6 |
| o-p | 6→7 | 0→1 | 0→0 | context_reset:0/0/1; segmented_intra:0/1/5; session_start:0/1/0 |

`segmented_intra` 指 phase=intra 且 idx_in_chain=0；保留原 harness 的 chain 桶，不用主办方 session_start 人群替换。raw 的 context_reset 仍是 reset，即使出现在本地开场。此合成负载把切段头纳入 chain，不能把剩余 6 条直接外推为正式线上 chain 形态。

## 负例与机制范围

| 对照 | 改动/范围 | 共同 ID | chain | chain 修/新/持续 | fast / overall / turn | TPOT p95 ms | TPOT >100ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b-d | 133 默认 | 1673 | 12→18 | 5/11/7 | 248→256/208→209/11→9 | 98.396→89.932 | 83→27 |
| b-f | 124m OFF 对基线 | 1675 | 12→14 | 3/5/9 | 248→238/209→201/11→8 | 98.396→94.572 | 83→77 |
| f-e | 124m OFF→ON | 1674 | 14→12 | 4/2/10 | 238→241/200→194/8→11 | 94.537→108.925 | 76→96 |
| c-j | 钉池+124m 计数，旧 OPEN | 956 | 7→8 | 0/1/7 | 31→29/20→13/4→4 | 189.491→165.469 | 101→92 |
| o-p | 最终组合+133 调参，旧 OPEN | 886 | 6→8 | 0/2/6 | 17→27/12→20/1→2 | 138.973→97.611 | 67→30 |

133 默认（b→d）净新增 chain 6，但实际是修好 5、新坏 11；不能只写 12→18 而漏掉失败集合换手。其 TPOT p95 变好不能抵消 chain 变坏。eznp 的现有 19.46 分钟快照对 ezno 是新增 2 条、持续 6：52k 切段 intra 25.088→56.112 s，以及 15,479-token session_start 0.443→30.580 s；持续长头也显著变慢。该快照已经提供拒绝证据，但不充当最终 40 分钟结果。ezni 仅有阶段文档称第 12 分钟停，本地没有原 raw，本报告不复述未独立复算的数字。

124m e/f 的 job.log 各自确认 on/off，完整 e 的 server.log 中 `[ax-124m]` 事件数确为 **0**，f 也是 0；因此 f→e 的 chain 14→12 不能归因为让路机制。该单次配对还新坏了 2 条且 TPOT p95 94.537→108.925 ms。b→f 这个 OFF 对照自身 chain 修3/新5，也直接显示不同运行会发生失败集合换手。j 的旧快照 chain 新坏 53,221-token reset：29.010→30.365 s。j 的 1836 次 plan 拒绝明细在阶段笔记中有记录，但本地未归档其完整 server.log，本次只独立核其 raw，不把二手计数冒充重新核验的事实。

## 各门失败集合变化与复现

| 对照 | fast 修/新/持续 | overall 修/新/持续 | turn 修/新/持续 | 逐请求 CSV |
| --- | --- | --- | --- | --- |
| b-c | 227/19/21 | 195/7/14 | 6/1/5 | [paired](../../evidence/review-0928-official-local/local-today/b-c-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/b-c-chain.csv) |
| c-l | 28/32/12 | 16/11/5 | 3/3/4 | [paired](../../evidence/review-0928-official-local/local-today/c-l-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/c-l-chain.csv) |
| c-k | 25/25/9 | 13/10/8 | 2/0/3 | [paired](../../evidence/review-0928-official-local/local-today/c-k-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/c-k-chain.csv) |
| l-m | 40/27/7 | 12/5/4 | 5/1/2 | [paired](../../evidence/review-0928-official-local/local-today/l-m-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/l-m-chain.csv) |
| c-m | 34/24/6 | 14/2/7 | 4/0/3 | [paired](../../evidence/review-0928-official-local/local-today/c-m-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/c-m-chain.csv) |
| m-n | 24/30/10 | 2/11/7 | 0/1/3 | [paired](../../evidence/review-0928-official-local/local-today/m-n-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/m-n-chain.csv) |
| n-o | 30/18/10 | 9/4/9 | 1/0/3 | [paired](../../evidence/review-0928-official-local/local-today/n-o-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/n-o-chain.csv) |
| b-o | 237/15/11 | 199/3/10 | 9/0/2 | [paired](../../evidence/review-0928-official-local/local-today/b-o-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/b-o-chain.csv) |
| c-o | 32/18/8 | 13/5/8 | 4/0/3 | [paired](../../evidence/review-0928-official-local/local-today/c-o-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/c-o-chain.csv) |
| b-d | 163/171/85 | 146/147/62 | 3/1/8 | [paired](../../evidence/review-0928-official-local/local-today/b-d-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/b-d-chain.csv) |
| b-f | 151/141/97 | 136/128/73 | 6/3/5 | [paired](../../evidence/review-0928-official-local/local-today/b-f-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/b-f-chain.csv) |
| f-e | 148/151/90 | 130/124/70 | 3/6/5 | [paired](../../evidence/review-0928-official-local/local-today/f-e-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/f-e-chain.csv) |
| c-j | 18/16/13 | 11/4/9 | 2/2/2 | [paired](../../evidence/review-0928-official-local/local-today/c-j-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/c-j-chain.csv) |
| o-p | 14/24/3 | 9/17/3 | 0/1/1 | [paired](../../evidence/review-0928-official-local/local-today/o-p-paired.csv) / [chain](../../evidence/review-0928-official-local/local-today/o-p-chain.csv) |

复现：在仓库根目录执行 `python3 evidence/review-0928-official-local/local-today/audit.py`。脚本只读原数据，派生 JSON/CSV 写入脚本所在证据目录；报告正文不改共享台账。

N34 eznh/eznq 不在这个 N30 同 N 对照中。并发、KV 压力与 admitted 请求集合都不同，不能用 N34 与 N30 的时间差证明某一内核作用。47266 的正式提交/队列状态与 N30 本地诊断分开记录，正式结果只认平台终态。本报告不替代 47043 的正式成绩核验，不预报 47266 的结果。
