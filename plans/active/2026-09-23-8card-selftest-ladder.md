# 8 卡自测梯子：找出当前最佳配置能爬到的 N

## 目标
在 L2 8 卡服务 `lh-arena-sess-b` 上，像正式评测一样沿 N=10→14→18→22→26 爬坡，确定当前各候选配置（TP 最佳 / DCP / +140）能稳定通过的最高档，并据此决定首个正式提交的配置。

## 范围
- 包含：pod 队列中的梯子任务 024（TP 最佳：120v2 + mHC 输入分散 + 扩容）、025（DCP8 + 114 + 115 + 120v2）、026（024 + 140）；每档的逐请求分析与日志指标；按结果调整队列。
- 不包含：正式提交（用户指示先把自测跑通）；新的内核开发（开发机并行）。

## 背景
- 相关文档：`research/claude/R8_next_directions.md` §8、`notes/findings.md` F73–F80、`research/codex/R18_cache_loss_and_capacity.md`、`HANDOFF.md` §3。
- 相关代码路径：`scripts/pod/jobs/dev_ladder_template.sh`、`ladder_{best,dcp,best140}.sh`、`scripts/pod/verify/analyze_run.py`。
- 已知约束：8 卡一次只能跑一个配置（模型 328GB 放不进 4 卡）；每档约 35 分钟；开发集只能比较相对变化，不能直接换算正式 N@SLO。

## 风险
- 风险：服务被误停/删除。缓解：旧守护进程已退役；审阅者只能经 pread / pexec_codex（拒绝停删与杀进程）。
- 风险：开发集 harness 用硬性 p95，正式评测带统计余量，两者判定可能不同。缓解：analyze_run.py 同时报告两种判定；爬坡按正式规则估算，结论注明口径。
- 风险：KDA 状态槽缩到 200 增加状态淘汰。缓解：梯子对比 DCP 路线（不缩槽）；观察缓存效率与 chain_start 尾部。

## 里程碑
1. 梯子 024 完成（TP 最佳配置的最高通过档）。
2. 梯子 025 完成（DCP 路线的最高通过档与 TPOT）。
3. 梯子 026 完成（140 对缓存丢失与 intra 门的贡献）。
4. 选定首个正式提交配置，报用户决定是否提交。

## 验证方式
- 命令：`scripts/pod/pread grep "^LADDER" /tmp/ax/runs/02{4,5,6}-*/job.log 50`；`scripts/pod/pread analyze /tmp/ax/runs/<job>`（单档）。
- 手工检查：每档 harness report_*.md 的门禁表；能力冒烟保持 12/12。
- 观测检查：logstat.py 的 decode ms/step、KV/KDA 池、排队长度；analyze_run.py 的排队/执行拆解与缓存效率。

## 进度记录
- [x] 2026-09-23：8 卡准入；b113 探测、能力冒烟 12/12、真机 kernel 复核 11/11。
- [x] 2026-09-23：N6 基线（intra 两门 FAIL，排队）→ 120 N6（intra 排队 6.4→0.36s；harness overall_intra 5.29s FAIL，正式估算 PASS）。
- [x] 2026-09-23：探针：mHC 输入分散 −6~9%；DCP8+114+115 KV ×7.8、长预填充 −19%；扩容参数 KV +66%。
- [x] 梯子 024（TP 最佳，chunk16384 使 KV 降到 63 万）：N10 引擎崩溃（Prefill OOM）→ 补丁 106。
- [x] 梯子 025（DCP8）：N10 前缀命中 illegal memory access → DCP 退出梯子，T50（补丁 116）修复中。
- [ ] 025b（024 + 106，回归，从 N10 起）进行中。
- [ ] 026（+140 +106，mem 0.75，从 N18 起）。

## 决策记录
- 2026-09-23：自测改为梯子式爬坡（用户要求积极自测、循序渐进）；爬坡判据用正式规则估算，同时记录 harness 硬判定。
- 2026-09-23：CP 不做主线（Fable：每轮只准入 1 请求、关闭 120）；DCP 因实测容量与长预填充双收益进入梯子对比。
- 2026-09-23：梯子改为从 N18 起爬（目标 N22/26，高并发才暴露问题且墙钟更短），首档失败再下探 14/10（用户建议）。
- 2026-09-23：补丁 106 升为第一层（必带）；DCP 在 116 修好前不上线。
