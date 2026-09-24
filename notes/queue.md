# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到结果；协作见 [collaboration.md](collaboration.md)。

## 当前安排（2026-09-24，用户最新决定）

按已准备的三项对照方案（来源：[执行说明](reports/claude-orchestration.md)，80abc38），
带122缺陷的061s已停、063s已撤。旧双会话和错误消息投递已排除，当前Claude已按用户要求恢复队列，pread独立确认064运行、065/066排队。**尚未正式提交**。准备的实验为**全量长链集、N30、70分钟准入后排空**的三个逐项对照。

| Job | 配置 | 状态 |
|---|---|---|
| 064-official_a_full_n30_70m | 正式 A 参数与 MTP；120 on、122 off、180 off | running，14:11 UTC开跑；执行层 Codex 跟踪 |
| 065-official_a_mem087_full_n30_70m | 同064，仅加 `--mem-fraction-static 0.87` | pending，排064后 |
| 066-official_a_180_mem087_full_n30_70m | 同065，仅加新版180的三个HiCache参数 | pending，排065后 |

三项同一个引擎提交 `c92acd57a61eb6f9eed3222cc048877eef7963d9`，含正式 A 全部13项改动。
171/172/123/DCP关闭；显式 `SGLANG_AX_PACE_TPOT=0` 关闭122。
066加 `--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through`；不加NUMA绑定。
122的修复由Claude接续复核/提交；不混入此次冻结源码。

两个拟议正式校准候选为065与066，提交JSON和38749字节Dockerfile已准备并通过离线检查。按报告所载用户安排推进；每项本地启动、机制检查及真正测量的前20分钟无错误后再提交。
本地结果不能替代正式能力门或宣布晋档。正式上传进度单独记 [submissions.md](submissions.md)。

## 测量与比较

- 数据：`data/s1-dev-longchain/`，311链、5601请求；已同步Pod。保留原链顺序、正文、输出预算和gap。
- 计时从测量第一条请求发出开始，4200秒关闭准入，已发请求全部排空。启动、preflight、warmup不计入70分钟。
- 每项独立输出目录，预热后真实 `POST /flush_cache`，必须2xx且 `success=true`；不加重复生成一致性门。
- 发送台账与完成记录逐条闭合；缺失请求为INVALID。70分钟可能只走到全量集的一部分，明确标为定时诊断，不冒充完整5601请求VALID。
- 比较完成量、排空时间、TTFT四桶、TPOT均值/p95、错误率、缓存/恢复与等待。报告共同请求的配对差异，注明两轮到达的请求集合可能不同。
- 原评分统计保留；10–70分钟的客户端准入吞吐另列诊断口径，不冒充原评分器稳态TPM。
- 每25分钟只读监控，运行中完成记录的窗口标open；不因部分样本超过门限自动停止。

## 已撤下的旧任务

- 061s：在预热中主动停止，队列标failed；122为无法与续块同批完成的短命中预留预算，出现连续64-token续块。CPU真实调度器反例已复现；无正式测量成绩。
- 063s：pending撤为cancelled；无测量。061s独立显存与阶段证据见 [memory-audit](../evidence/L061s-official_a_122_full_n30_70m/memory-audit/snapshot.json)。显存数是预热采样峰值，不能证明全程安全余量；扩大预算还会增加KDA状态池，不能全部折算KV token。

- 061r：引擎正常就绪，但任务把NEXTN别名错误地与内部EAGLE名称比较，测量前误判退出；现修正G_EXPECT为spec=EAGLE。063r有同一检查，已停，无测量raw。证据见 [修复记录](../evidence/nextn-alias-20260924/README.md)。

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
