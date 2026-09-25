# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到结果；协作见 [collaboration.md](collaboration.md)。

## 当前安排（2026-09-24，用户最新决定）

当前Codex继续SGLang，Claude Code接手vLLM，共同探索N38。vLLM首阶段为基本开发/GPU调通和相同评测合同；复用已有有效设计，各自优化实现。短探针先筛选，完整回放后判分，见[交接](handoffs/vllm-claude-code.md)。071冻结不变，其后为vLLM预留072启动/冒烟位置，尚未发布到Pod队列。围绕工作量、单位成本、调度改善做闭环；
统一规则见[evaluation.md](evaluation.md)，过程见[Codex迭代日志](iterations/codex.md)。

| Job | 配置 | 状态 |
|---|---|---|
| 071-official_b_host64_full_n30_shortwarm | 恢复后沿用070引擎/参数/全量数据；新Pod、RAM工作目录及编译缓存，冷启动 | 09-25 07:03 UTC：权重/KV分配完成，首次CUDA graph编译/捕获中，仍pending、尚未测量；数据及32项运行文件SHA已通过。cgroup327GiB/1509、RAM工作区0.92GiB、根/tmp1.32MiB；临时盘仍20Gi。等待引擎READY后放行。见[容量收据](reports/pod-capacity-0925.md) |
| 072-vllm_tp8_real_smoke | 官方main a811738a6 + 000/010，基线冻结fb18e488（不含101）；TP8完整真实权重、MTP、接口/长上下文/缓存冒烟；部署产物与工具哈希待核验 | **计划预留在071之后，未入队、未安装环境**；无connector的000接口与flush判分CPU联调已通过；Claude准备独立venv容量/兼容性收据及vLLM任务入口，Codex协调共享队列；06:14 UTC新Pod可exec，优先核容量与写入落点。见[安排](reports/vllm-tp8-slot-072.md) |
| 070-official_b_host64_full_n30_shortwarm | 对照069，只开122 τ=.085；host64、GPU预算与其余配置不变 | 基础设施中断，无测量：22:13:28 UTC旧Pod临时存储超20Gi被驱逐；09-25 06:14 UTC revision2新Pod已可exec；070没有恢复执行 |
| 069-official_b_pace_off_host64_full_n30_shortwarm | 对照068，只扩HiCache host预算32→64GB/rank；122off，GPU预算不变 | 完成5601条/107.59分钟，VALID FAIL；10/11通过，仅chain31/29失败，TPOT .028824/.055902 |
| 068-official_b_pace_off_full_n30_shortwarm | 对照067，只关闭122；其余引擎配置不变 | 完成5601条/约125分钟，VALID FAIL；fast/overall/chain失败，TPOT通过，turn仅CP余量通过 |
| 067-official_b_full_n30_shortwarm | 正式A + mem0.87 + 新版180 + 修复122；MTP保留 | 完成5601条/约123分钟，VALID FAIL；四类TTFT失败，其余7门通过，详见experiments |
| 064-official_a_full_n30_70m | 原A参数 | 已按用户改变迭代方式停止；停止前仍在预热，无测量成绩 |
| 065 / 066 | 仅mem0.87 / 再加新版180 | 已从pending撤销，不再阻挡组合验证 |

067引擎固定759a6ebb8e31723519ad5daf438e26e24b32501a，运行工具296caaa；
120/122/180 on，171/172/123/DCP off。配置对应已上传正式46174/0924d，**不重复提交官方attempt**。
正式46173/46174的上传与成绩只看[submissions.md](submissions.md)及official_status，不从本地推断。

068的唯一引擎开关变化为`SGLANG_AX_PACE_TPOT=.085→0`，`G_EXPECT 122=off`，源码仍759a6eb。
检验122整体在本负载下的取舍：关闭也会恢复固定decode interval与4096冷块上限，不能把差异只归给某一个参数。
数据manifest SHA256 `19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c`。
判据为全部5601条同ID、四桶修复/新增坏例与TPOT回归；未预设关闭会更好。
新运行库e77933d补采已有HiCache分层指标（同10秒频率）并收紧预热校验；067的16条原预算已另行严格核对通过。

069部署queue-20260924T195929-4936已见RUNTIME_DEPLOYED和DONE rc=0；运行库29文件与068完全一致。
069仍固定源码759a6eb与同一数据manifest、rep16、N30；不把122off当作最佳版本。
只改`--hicache-size 32→64`，KV/indexer和KDA host同比扩，8卡增加约256GB主机内存；不改device池预算。
CPU源码尺寸函数及cgroup余量检查通过，独立review无阻断理由；收据见[budget-probe](../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/budget-probe.json)。
启动实测device KV=1,397,760、KDA=418不变，host FULL=2,903,808 token、KDA host=23.73GB；122off/180on符合G_EXPECT。
判据：完整11门、589个固定坏例及新增坏例、cached/重算量、等待/执行与TPOT。高host占用是干预线索，未证明某请求遭淘汰。

070部署queue-20260924T220817-20847已核RUNTIME_DEPLOYED与DONE rc=0，队列已恢复；29运行文件与069一致。
070启动时Pod被平台驱逐（ephemeral_exhausted），不是SLO FAIL；监控显示的running/startup是旧缓存。
恢复更新已接受：同service2102486579267252224，revision2=2103249397960679424；仅新增AX_WORKSPACE_ROOT，资源限额/8卡/模型镜像不变（GPU显示名由平台补NVIDIA前缀）。bootstrap在任何上传前验证并链接/tmp/ax。已实测RAM可执行，JIT/cache/tmp迁至RAM；新Pod冷编译缓存是对照混杂，恢复后用新run071并明确记录。
070固定引擎759a6eb，只将`SGLANG_AX_PACE_TPOT=0→0.085`与`G_EXPECT 122=on`对应调整。
host64、mem0.87、MTP、原数据/rep16/N30不变；这是122整体机制对照，不是只改变τ常数。
069工作量降低后，检验更大prefill预算能否减少chain等待且守住TPOT；31条chain中20条queue≥80%，但尚未识别具体阻塞原因。
完整比较311个固定TTFT坏例和新增坏例，单轮临界跨门不认定稳定收益。设计收据见[070配置](../evidence/L070-official_b_host64_full_n30_shortwarm/config-plan.json)。

## 测量与比较

072是有限功能探针，不是完整N30或N38判分；启动/捕图与请求探针分别设预算，确认错误留证后立即结束自己的测试。071终态并取证、模型按任务边界退出后再运行072；不并驻两个完整模型、不停删共享service。现有qpush/启动器只支持SGLang，不能直接用于072，须先准备独立入口。072通过后再安排完整对照，不预先占用后续全量档。

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
