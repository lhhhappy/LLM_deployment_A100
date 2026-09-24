# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到结果；协作见 [collaboration.md](collaboration.md)。

## 当前安排（2026-09-24，用户最新决定）

今天只比较以下两项，使用**新源码流程、全量长链集、N30、70分钟准入后排空**。
不再跑 lite、额外 A 基准或单独 180。正式提交候选是这两项，晋档只能由平台确认。

| Job | 配置 | 状态 |
|---|---|---|
| 061r-official_a_122_full_n30_70m | A 的参数与 MTP + 122，HiCache 关闭 | 已确认running，正在启动新源码引擎；执行层 Codex 跟踪 |
| 063r-official_a_122_180_full_n30_70m | 同 061r，仅加新版 180 的三个 HiCache 参数 | 已确认pending，排061r后；不要求前一项SLO通过 |

两项固定同一个引擎提交 `c92acd57a61eb6f9eed3222cc048877eef7963d9`，源码含正式 A 全部13项改动。
171/172/123/DCP关闭；122以 `SGLANG_AX_PACE_TPOT=0.085` 开启。
新版180为主机层保留120/122；启动日志须与任务的 `G_EXPECT` 一致。
063r只加 `--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through`；暂不加NUMA绑定。

## 测量与比较

- 数据：`data/s1-dev-longchain/`，311链、5601请求；已同步Pod。保留原链顺序、正文、输出预算和gap。
- 计时从测量第一条请求发出开始，4200秒关闭准入，已发请求全部排空。启动、preflight、warmup不计入70分钟。
- 每项独立输出目录，预热后真实 `POST /flush_cache`，必须2xx且 `success=true`；不加重复生成一致性门。
- 发送台账与完成记录逐条闭合；缺失请求为INVALID。70分钟可能只走到全量集的一部分，明确标为定时诊断，不冒充完整5601请求VALID。
- 比较完成量、排空时间、TTFT四桶、TPOT均值/p95、错误率、缓存/恢复与等待。报告共同请求的配对差异，注明两轮到达的请求集合可能不同。
- 原评分统计保留；10–70分钟的客户端准入吞吐另列诊断口径，不冒充原评分器稳态TPM。
- 每25分钟只读监控，运行中完成记录的窗口标open；不因部分样本超过门限自动停止。

## 已撤下的旧任务

- 060旧180中断：旧开关会同时关闭调度保护，部分记录仅诊断。
- 060z原样A中断：用户收窄测试范围，无测量raw。
- 061旧流程lite中断：用户要求全面迁移，无测量raw。
- 062单独新版180取消；063旧流程组合lite取消，均无测量成绩。
- 28个过时job/重复草案已从可执行入口移除。含未提交内容的原文保存在 [清理快照](../evidence/source-workflow-20260924/retired-jobs.json)，原始实验数据保留。

## 已成立的参考与边界

- 047/048原dev N22：122使fast超标33→10，overall55→28、chain73→62；仅一次对照，不外推正式N。
- 058原样A lite N14：1123条VALID，chain12/8失败，TPOT .01585/.04425；冷开场与后段排空影响明显。
- 059旧180 lite N14：1123条VALID，chain11/8，fast8→26；隐含关闭调度保护，不能当作新版180单变量结果。
- 171/172仅测过原dev N22，未证明稳定整体净收益；今天不额外叠加。
- 源码迁移应用检查：从底包按提交差分生成的4692文件与git源码完全一致；CPU测试不替代TP8验证。

完整历史结果见 [experiments.md](experiments.md)，当前机制与未决边界见 [knowledge.md](knowledge.md)。
部署、清理与检查证据见 [source-workflow-20260924](../evidence/source-workflow-20260924/README.md)。
