# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到结果；协作见 [collaboration.md](collaboration.md)。

## 当前安排（2026-09-24，用户最新决定）

当前Codex单人持续迭代SGLang，vLLM暂缓。围绕工作量、单位成本、调度改善做闭环；
统一规则见[evaluation.md](evaluation.md)，过程见[Codex迭代日志](iterations/codex.md)。

| Job | 配置 | 状态 |
|---|---|---|
| 067-official_b_full_n30_shortwarm | 正式A + mem0.87 + 新版180 + 修复122；MTP保留 | running；短预热101.7s完成，15:15:31 UTC真flush成功，进入N30测量 |
| 064-official_a_full_n30_70m | 原A参数 | 已按用户改变迭代方式停止；停止前仍在预热，无测量成绩 |
| 065 / 066 | 仅mem0.87 / 再加新版180 | 已从pending撤销，不再阻挡组合验证 |

067引擎固定759a6ebb8e31723519ad5daf438e26e24b32501a，运行工具296caaa；
120/122/180 on，171/172/123/DCP off。配置对应已上传正式46174/0924d，**不重复提交官方attempt**。
正式46173/46174的上传与成绩只看[submissions.md](submissions.md)及official_status，不从本地推断。

## 测量与比较

- 全量`data/s1-dev-longchain/`：311链/5601请求、N30，原顺序、正文、输出预算和gap；不设70分钟截止。
- preflight一条链 → rep16-v1固定16完整请求短预热 → 真flush → 全量测量 → 原评分器11门。
- 预热计划、错误、实际耗时和flush收据留证；当前rep16-v1不保证全部形状覆盖，不与原预热结果冒充单变量。
- 每分钟健康观察，首次15分钟、随后每30分钟窗口快照；30分钟重点检查严重bug。运行中窗口open，不宣称整档通过。
- 全量5601请求每个恰好一次、runner/flush/指标完整才判VALID PASS/FAIL；否则INVALID。
- 证据明确的严重bug先保存并报告，再用stopjob停单个job，修复后新run ID重跑；8卡服务不可停删释放。
- TPM保留原固定稳态窗口。记录等待/执行、prefill块长、缓存重算、MTP和内存压力，不将高GPU利用率等同效率。

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
