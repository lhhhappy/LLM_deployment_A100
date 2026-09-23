# 共享背景（2026-09-24 凌晨，Claude 统筹）——所有 T54–T56 任务先读这份

仓库 `/workspace/Agentic_science_challenge`。先读 `AGENTS.md`（红线）、`rule.md` §4。本轮的核心要求（用户原话）：**评测和分析一定要正确**。

## 已核实的事实（按这些为准；与之冲突的旧文档都算过时）
1. **F93 / `research/codex/R19_progress_and_cache_review.md`**（Codex T53；Claude 已在隔离副本重跑 `evidence/T53/reproduce_review.py`，输出逐字节一致）：
   - 026 N18 raw（722 条，0 error，SHA256 486f0410…，本地 `evidence/T53/026_N18_raw.jsonl`）：全档实际 prefill 10.63M。
   - **本轮链首（`idx_in_chain==0`）311 条，实际 8.10M = 76%**；其中 53 条原始标签是 append-only，但本轮没有发送前驱（dev 集是链前缀抽样），实际 4.61M、冻结 0.08M——**不是同链缓存丢失**。
   - 本轮真正的后续请求 411 条：实际 2.53M、冻结 1.67M；相对冻结值的**正差 0.913M（8.6%）**，这只是代理指标，不是真实 LCP、不是可省算力。
   - F92「同链多算 5.3M、占一半、修好就够 N22」**已撤回**；F91「43 个 fast 超时」口径错误：按 `s1_common.in_ttft_gate`，fast_intra n=328，>3s 只有 10 条。
2. R19 §2 的工具缺陷 S1–S8 均已复现或读码确认：空 raw → ALL_PASS、缺 TPOT 记 0、梯子忽略 run_dev 退出码并取"最新 raw"、flush 不验 JSON、numcmp 截断/分叉判 ok、116 探针只打印不断言、`make_120.py` 与 120 v3 不一致、job 文件复制旧模板、`engine_current.log` 是静态副本、logstat 不按档切窗口。
3. R19 §4 已降级的说法：8.1–8.5k tok/s 是"未命中量/全档墙钟"，不是纯 prefill 产能；"闭环 ⇒ MTP 必降 N"不成立；在飞 prompt 求和 ≠ 物理 KV 驻留；`1−p10/mean` 不是已测停顿份额；榜单推不出第一名的方法；12/12 冒烟 ≠ 能力门。
4. dev 与正式集结构不同：dev 722 请求 / 311 链（43% 是链首）；正式集（F31）9896 请求、808 chain_start（8.2%）、9023 intra、65 turn_start。dev 档约 20 分钟，正式档数小时。
5. 实绩：026 N18 四道 TTFT 门过（fast 2.44s 10/23、overall 4.41s 18/27、chain 25.4s 12/22），tpot_p95 0.219 FAIL，tpot_mean 0.083；028 N18 tpot 0.053/p95 0.082 过，但 fast/overall/chain FAIL；027/028b/028c/034 N18 均 FAIL。没有任何候选 N18 全门通过。
6. 正式提交 45979（A=028 配置）/ 45980（B=034 配置），镜像 lh-img:0923a，预计 09-24 约 09:20 UTC 出分——这是第一个 dev↔正式校准点。
7. **S0 = 026 的精确栈**：补丁 `000 101 105 106 110 111 112 113 140` + `patches/drafts/120-sched-protect-chain-v2.patch`（sha256 ef1744b3…，与提交 b55081f 时的 120 逐字节相同；当前 `patches/120` 是 v3，不能当 026 用）；参数/环境见 `scripts/pod/jobs/ladder_best140.sh` 头部。

## 方法规则（R19 的做法，必须遵守）
- 门与分桶只用 harness 自己的代码：`s1-dev/harness/s1_common.py:in_ttft_gate`、`s1_score.evaluate`（经 `scripts/score_formal.py`）。不要再写第二套简化门。
- 按**本轮回放的链内位置**（`idx_in_chain`、前驱是否在本 raw 中）拆分；数据集的 `phase`/`edge_type` 只描述原始轨迹。
- 冻结 `uncached_expected` 只是代理；真实可复用量要用本轮实际相邻两个 prompt 的真实 token LCP。净差和正差分开报。
- 失败即失败：空数据、不完整 cohort、缺指标、runner 非零退出、flush 未确认 → INVALID，绝不能判通过。
- 标 VERIFIED 必须附可复现的脚本 + 输入 SHA256 + 输出；否则标 INFERRED。每个声称的工具行为都要有最小反例测试。

## 红线
- 8 卡 pod：只能用 `scripts/pod/pread`（只读）与 `scripts/pod/pexec_codex`（仅 CPU，只写 /tmp/ax/codex）；**禁止**停/删服务、杀进程、向引擎发请求、写 /tmp/ax 的 queue/src/runs/patches/bin、用 GPU、入队。本轮任务都**不往 pod 推送任何东西**（验证通过后由 Claude 推送）。
- 只读目录：`s1-dev/`、`src/sglang/`、`build/base_exact/`、`llm-challenge-arena-v1/`。
- `research/codex/*`、`scripts/score_formal.py`、`scripts/ladder_search.py`、`scripts/make_120.py` 是 Codex 写的：只能 import/包装/在自己的输出里提意见，不要改写（发现问题写进你的交付说明）。
- 往 findings/decisions 追加编号前先 `python3 scripts/next_id.py F|D`，写完跑 `python3 scripts/check_records.py`。原始证据放 `evidence/<任务号>/`。

## 赛题口径（以 `llm-challenge-arena-v1/task.md` 为唯一准绳，Claude 09-24 逐条核对）
- 排名：`n_at_slo` → `tpot_mean`（越小越好）。**TPM 只回报不排名**（task.md:74,548–549,593–614）。`s1-dev/README.md:50–55`（TPM 排名、10 道门、客户端计时）是旧口径，作废。
- 每档 11 道硬门（task.md:519）：coverage=100%、harness_data=0、harness_render=0、engine_error<1%、infra_error<1%、四道 TTFT p95（fast_intra 3s / overall_intra 5s / turn_start 15s / chain_start 30s；判定带统计余量：超标率 95% 单侧下界 >5% 才失败，task.md:565–567）、gated_phases_have_samples、**tpot_p95 ≤ 0.10（无余量）**。TTFT 用服务端口径（meta_info 两个时间戳，task.md:205,556）。
- 正式爬坡从 N=10 起，过 +4、不过 −4，已过档之上首次失败即停；单档约 4 小时（task.md:476,500–515）。
- 开发集是链前缀抽样，**只能比自己两次部署的相对变化（A/B、回归），不能当 N@SLO 预测值**（task.md:354）；不许用 `--max-chains/--no-gap/--include-all`（task.md:355）。
- 能力门：aime26、gpqa-diamond 的 points 都**严格 >90**（task.md:484），即约 ≥40/44、≥141/156。
- 质量与诚实：不关 thinking、不压输出预算、不截历史、不删 tools（task.md:699）；meta_info 时间戳/计数如实、`/flush_cache` 真清，清不掉该档失败（task.md:226–230,707–708）。
