# D1 角色边界 KDA 快照：从设计到可提交

> 2026-09-22 归档（dropped）：基于 v0.5.20 线 / SPF / 旧自测设计，已被决策 27–29、tests/L2.md、plans/active/102-role-track.md 取代。

- 状态：active　负责人：Claude（补丁），Codex main（审阅、E2）　创建：2026-09-22
- 关联：F3/F13/F24、`patches/001-*`、decisions #6 #11、T13、E1/E2

## 目标
让链中间请求在尾部 reminder 被替换后仍能命中 KDA 快照：fast_intra 实际重算 p95 从约 7.2k 降到约 3.3k（模拟值）。在 8 卡上证实它能把临界 N 往上推，并且能力分不回退。

## 范围
- 包含：方案 A（在边界处切 chunk，role_conservative）；统计导出；E2 替身验证；8 卡 A/B。
- 不包含：方案 B（同一 forward 导出多个快照）；保留策略（cap）调优；HiCache。

## 风险与缓解
| 风险 | 缓解 | 回滚 |
|---|---|---|
| 状态恢复出错，导致输出错误 | E2 做 logits 对照；8 卡能力抽检 | 去掉环境变量即恢复 stock |
| 多一轮调度拉高 TTFT | E2 测"第二段"的时延；8 卡 A/B | 同上 |
| 与 D2 和 decode 间隔相互影响 | D2 在 D1 验证之后再加（F27） | — |

## 里程碑
1. [x] 设计 v1、v2（§9），Codex 审阅（§8），对抗性审查（R5）→ v1.1（§11）
2. [x] Codex 审阅 v1.1（T13，§10：无HiCache E2有条件可测，2个major）
3. [x] E1 stock 替身基线（72请求；F13预测70个精确匹配，2个偏差，详见experiments）
4. [x] E2 v1.1执行完成：三组缓存+三对raw logits；D1-01/03通过，D1-02/04失败（不是验收通过，详见F42/F45）
5. [ ] 统计导出（tech-debt）
6. [ ] 8 卡 A/B（首次 8 卡会话的里程碑 4）
7. [ ] 纳入候选镜像

## 验证方式
- E2：逐请求 `cached_tokens` 与 F24 预测对账；strict_append 的命中不下降；logits 差值在容差内。
- 8 卡：同一 N 下 fast_intra/overall_intra p95 下降；能力抽检不回退。

## 进度记录
- 2026-09-22 Codex main：T26/E2b完成，三case143项控制=旧v1.1，004无退步且全部精确符合role预测。reminder p95 8135→2439，strict67项不变，unset smoke等stock，flush池均恢复；D1-01/02/03及N1 D1-08通过。004额外alignment major/旧D1-04失败仍未解除，不批准完整正确性；54CPU pass+1已知失败。自有服务已停，两卡空闲，今日仅此收尾后idle，证据evidence/T26。
- 2026-09-22 Codex main：T26按wrap-up仅恢复004审阅/E2b。审阅见004.md：新鲜adder续跑读不到额外alignment（已复现major），当前无DSA/无额外alignment配置可限定测试；002覆盖host-miss第二partial漏洞。批次000+001+002/FCFS控制→+004候选→unset smoke，冻结原E2三case/参数，D1-04既有FAIL保持；结束后停自有服务。
- 2026-09-22 Codex main：T29补测完成，F47/experiments T29：IF-08/IF-03/D1-10原配置通过；为完整cold_heavy启context262144/KV524288，105条N4/0error、306轮partial≤1，D1-08池回启动值通过。D1-05统计仅覆盖105新准入而不含229续跑，整体门wip待明确；D1-02/04失败保持。50CPU通过，所有自有服务已停，GPU0/1各4MiB/0%，其他会话保留。
- 2026-09-22 Codex main：T29响应重复派发，只补E2遗留D1-05/08/10、IF-08；新run `E2_T29_20260922`，000+001 v1.1/同qfull哈希已核对，原矩阵不重跑。只读进程内trace统计每轮partial和flush前后池；N4按cold_heavy与reminder_heavy完整前缀并集回放。新增10项CPU工具/AST检查通过；D0代码审阅补到000 §6。既有D1-02/04失败保持，live结果待追加；不混入后续补丁。
- 2026-09-22 Codex main：T13/E2已交付，v1.1无cache退步但reminder p95仍8135，stock20 4750→3962；角色恢复已通过trace证明，严格数值门未过，普通off缓存也存在数值/greedy差异。当前不能称安全或可部署。未混入后续004；需单独验证冷请求最后chunk覆盖和重新预定义精度判别基线。GPU已释放，完整记录见experiments E2与notes/e2_d1。
- 2026-09-22 Codex main：T13/E2开始。三组原stock / D0+D1 unset / D0+D1 set，使用同qfull权重与同smoke/reminder_heavy/strict_append/stock20。统一context131072容纳完整用例，page64/chunk8192/extra_buffer/槽位配置不变；只在arena/code独立clone应用。raw logits诊断与无trace回放分开，GPU仅占空闲GPU0。
- 2026-09-22：v1.1 补丁 148 行，可以干净应用（`patch -p3 --fuzz=0`）。
- 2026-09-22 Codex main：E1完成；随机qfull Kimi-Linear（3KDA+1MLA，无DSA），stock20=72请求/0错误，真实cache与flush smoke通过，GPU已释放。证据见`evidence/E1_stock/`和experiments；不是GLM性能/数值正确性证据。E2用同一qfull权重，采用新cases时须重跑stock；F13/F24需区分驻留FULL-KV与抽象LCP，不能作为严格oracle。

## 决策记录
- 2026-09-22：策略选 role_conservative（F24：只跳过 2/712，收益与 role_all 相同）。
- 2026-09-22：切分后继续准入能整段完成的请求，并加"每轮最多一个 partial"的守卫（R5 major 2）。
