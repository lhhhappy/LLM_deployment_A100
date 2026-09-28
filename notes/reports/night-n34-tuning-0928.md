# 今晚 N34 → N38 参数试验（2026-09-28，Codex）

用户授权：每个配置每档派发 1 小时、排空；成功后同配置自动上 N38。只有一组 TP8，性能臂串行，参数与坏例分析并行。固定引擎 bf6b66fa、KDA 全候选、rot150 冻结 cohort、rep16-v1 预热与每档真实 flush，不改变请求、输出预算或判分规则。

| 顺序 | 任务 | 相对 eznq 的唯一变化 |
| --- | --- | --- |
| 1 | 130eznr / pdi3 | 普通 prefill 后 decode 轮数 2→3 |
| 2 | 130ezns / pdi4 | 普通 prefill 后 decode 轮数 2→4 |
| 3 | 130eznt / relief1 | 积压 relief 阶段 decode 轮数 0→1，普通阶段仍 2 |
| 4 | 130eznu / repeat | 原配置重复，观察 1h 窗口与运行波动 |

前两臂仍保留 125 relief=0、131 risk=1 覆盖，不能保证 chain 没有间接代价；第三臂会直接在积压时插 decode，更可能影响 chain，独立测、不与前两项叠加。现有 N34 有 102 条 TPOT>100ms，开场前5分钟51条；后续51条中41条输出≤128 token，因此先检验 prefill 后的 decode 间隔。暂不调整池大小、不重试已否决的133。

自动晋档依据是 `timed_verdict.json` 的四个原统计余量 TTFT 门、TPOT p95≤0.10 和零错误/已排空；另外与冻结 eznq N34 的268个chain ID配对，要求参考ID全部完成、没有新增chain超标。新出现的其他请求仍纳入本臂整体TTFT门，但无相同ID参考，不能冒称它们已配对。参考原 raw SHA为6125d6629e69f4766eb514222bbac5032e3f80d17f4793b2b5b9b8e01f94db86。

原 timed_score 的退出0只表示已排空，不能直接当PASS爬坡；因此模板增加仅本轮显式启用的晋档检查，保留promotion.json。门不满足跳过本臂N38、继续下个独立参数；证据INVALID则该任务失败，不上N38。1h通过仍称本地诊断通过，不称正式档位。

预计4–8小时测量时间，另加启动、预热与排空。首臂是pdi3；每个通过N34的配置立即测自己的N38，再处理下一个参数。结果按共同ID逐项列chain修好/新坏、fast/overall/turn、TPOT及显存峰值。完整阶段日志及晋档收据保留在各任务runs/N34、N38目录。

发布前检查：4个job shell语法；同固定引擎及数据，分别只改目标参数；晋档控制用真实N34失败收据、正例fixture、TTFT失败、新chain坏例、缺参考ID、raw SHA不符共6项CPU检查通过。另一参与者复核控制流；不修改原评分器。
