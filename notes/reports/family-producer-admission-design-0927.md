# 共享前缀生产者与兄弟准入：技术设计

2026-09-27，Codex。状态：设计提案，尚未实现完整机制。Fable 已将首次冷/暖分类冻结实现为 741f3eda、排入 130ez6z；本方案复用该工作，不另做一份。跨领域改动的接口与写入者需由参与者确认。当前没有改变 Pod 队列或引擎。

目标：对已到达、真实共享 token 前缀的一组请求，使共享前缀及时成为可复用检查点，并让就绪的短尾及时入批。保持单个 chunked request 的现有所有权模型，使用原生匹配、分配、COW、回载和回收路径。不能保证救回固定条数，收益须以同条件开场探针验证。

## 1. 现有调用链与缺口

源码和 CodeGraph 交叉核对：

```text
Scheduler.get_next_batch_to_run
  ├─ stash_chunked_request → maybe_cache_unfinished_req → cache_unfinished_req
  │    └─ MambaComponent.prepare_for_caching_req → insert → rematch/lock
  └─ _get_new_batch_prefill_raw
       ├─ calc_priority → match_prefix_for_req
       ├─ _ax_demand_limits → PrefillAdder
       ├─ _ax_admission_plan → _ax_family_plan → family_plan → tier_order
       ├─ add_chunked_req
       └─ init_next_round_input → add_one_req
```

现有128只给等待中的冷请求算家族分数；领头离开 waiting queue 后便不再由这份计划跟踪。原生 LPM held 先于家族分数生效；兄弟缓存增长后又可能变成短 warm、预算从30秒变3秒。即使排到前面，冷续块先占满预算仍可挡住其准入。[已复现反例与旧运行证据](dcp-chain-plan-review-0927.md)

缓存侧的重要契约：

- `MambaComponent.create_match_validator` 要求匹配节点有 device 或 host 的 Mamba 状态；`full_kv_hit_length` 可以比真正可复用匹配长。
- `cache_unfinished_req` 使用实际 `mamba_last_track_seqlen` 发布状态，可能只发布到本次执行末尾之前。生产者的 `prefix_indices` 还可拼接其私有KV，不能用它的长度判定兄弟已就绪。
- `match_prefix_for_req(..., cow_mamba=False)` 使用真实 `RadixKey(token_ids, extra_key, cache_salt)` 得到可复用长度；这个匹配可能拆分树、触碰LRU，不能称完全无副作用。复用本轮已发生的匹配，不在估计器里反复调用。
- host-only 匹配可以省计算，但仍需回载和设备资源。首版的立即搭车只接受 device-ready；host 路径继续交给原生机制。

## 2. 最少新增的状态

请求保存原始接收时刻及稳定的 deadline 类别；依赖账本只存CPU元数据：

```text
FamilyPlan:
  epoch                     # flush / request 生命周期隔离
  cache_domain              # 与原生缓存兼容的 extra_key、cache_salt 等
  producer_rid, generation
  target_prefix_depth       # 实际对齐网格上的共享位置
  dependent_rids
  state                     # PLANNED / PRODUCING / REUSABLE / RETIRED
  last_progress_depth, last_progress_round
  expires_at                # 有限等待，不是服务时限保证

Dependent:
  producer_rid, generation, target_prefix_depth
  original_deadline_at
  state                     # WAIT_PREFIX / READY / ADMITTED / FALLBACK
```

不把GPU地址、KDA槽或裸树节点指针长期藏进账本；实际准入仍使用当轮Req的匹配结果。先把候选数沿用128的64上限、同一请求最多一个依赖；超限退回现有排序。CPU前缀哈希增量缓存，避免每轮重扫整段prompt。

首版新增的是CPU元数据。更早入批仍会改变活跃KV/状态的峰值，必须走现有资源预算并实测；不为预测出来的兄弟提前分配KDA槽，也不长期额外锁住整条前缀。

## 3. 每个调度轮的流程

### A. 先更新真实匹配与截止时间

在原生stash/publish之后复用 `calc_priority` 的匹配结果。deadline的起点始终是原始接收时间，不能因停车、重排、回退或获得缓存而重新计时。

741f3eda 已实现首次冷/暖类别冻结，先独立测它。完整依赖方案对参与共享前缀等待的请求记录原来的绝对deadline；本轮更新剩余计算成本，不因命中增加而把30秒降成3秒。这个30秒仍是服务端启发式预算，不是读取harness的chain标签。是否把稳定deadline扩展到所有warm类别，另由实验决定。

### B. 选择真正能产生收益的前缀生产者

只比较已到达请求的真实token前缀，按缓存域隔离。哈希只用于筛选，关键共享区间按token确认；冻结的 `prefix_family_id` 仅可用于离线诊断，不参与调度。

为一个共享位置G建立候选组。优先复用正在生成该前缀、且路径兼容的现有partial；没有生产者时，在等待请求中选能最早发布G且资源可行的成员。账本跨越 waiting→partial 生命周期，不能因领头入批就丢掉依赖。

需要分清两个成本：

```text
C_publish = 生产者从当前可复用位置推进到G的预计时间
C_group   = C_publish + 能完成的兄弟尾巴预计时间 + 必要批次/干扰成本
benefit   = 在各自原deadline前预计可完成的请求数
```

家族与普通请求比较的是有边界的 `C_group / benefit`，并保留既有饥饿救援层。不能只用领头整条remaining除以兄弟数，也不能把赶不上deadline或无法准入的兄弟都算作收益。估计器使用实测块成本、当前partial最小不可中断片段、轮预算和已知资源；收益是估计，不作为保证。

首版可先沿用128候选与评分，只修真实生产者、held和ready链路；更复杂评分在日志证明有排序缺口后加入。深浅共享前缀不能用连通分量无差别合成一个家族，重复覆盖的兄弟只记一次收益。

### C. 协调两套held，保证至少一个生产者能竞争

只有确认兼容缓存域和真实前缀关系之后，才允许选定生产者解除相应的LPM暂缓；保持兄弟的有期限等待。已有partial时不再启动第二个生产者，不让两个等待请求互相hold。

不能简单清空LPM held：它可能代表另一个更早的生产者。若记录不够解释这条依赖，就保留原规则并记录原因。每组仅一个生产者；有更浅层生产者时，优先复用它已完成的前缀再重新评估。

### D. 以兄弟真实匹配确认REUSABLE

前缀目标G由运行时检查点/页/截断对齐约束决定，不硬编码256。首版沿用现有块边界和快照，不强迫新增检查点。

当某个兄弟在本轮匹配中得到足够长的device prefix、对应Mamba状态也有效、且不需要host回载，才进入READY。producer宣布“执行到G”只是进度提示，不是放行依据。同步与数据依赖继续使用原生forward/stash/COW路径，不新增 `cudaDeviceSynchronize` 或CPU轮询GPU。

如果完整共享LCP尚不可复用，但已有更浅的合法检查点，且从该点算到请求末尾已能一轮完成，也允许重新评估READY；不能死等一个不再必要的精确G。依赖失效或缓存被淘汰则重新匹配并回退。

### E. 为READY短尾落实批预算

READY优先参与按deadline与剩余成本的正常选择；首版只允许完整一轮短尾，不放开第二个partial。当前126关闭时，仅抬高排序不能保证有座位，因此必须把“选择谁”和“续块占多少”连接起来。

复用120/126的网格和124的资源预检，先挑本轮实际能完成的短尾，再决定续块预算。例如8192预算、一个约3000-token短尾，预留按页向上取整后的空间，续块取剩余并向下对齐；精确数值取决于真实page/grid，不能固定写成5192。

调度入口需要两阶段计划，解决现有 `_ax_demand_limits` 早于124最终排序的问题：

1. 当轮匹配 → rank0生成候选顺序/依赖与只读资源预览；
2. 选择可行READY短尾 → 定最终reserve/cold_cap → 广播最终计划 → 原生adder执行。

保持原来的总token、请求槽、KV、输出预留、Mamba额外收费和COW检查。preview不是承诺；实际rematch/分配失败要执行现有清理，释放无效预留，同轮尽量让生产者恢复执行，防止空转。不能直接复用“上轮没入批就一直停止留座”的规则而不区分失败原因；依赖/资源状态改变后要允许重试。

当短尾放不下但可独立一轮完成时，优先沿用现有124的有界停车条件。若日志显示这些条件本身挡住救援，再单独测试“可就绪家族短尾触发有界停车”，保留墙钟/轮数上限和同轮恢复；首版不把停车扩成任意多轮抢占。

## 4. 故障和多卡一致性

- 生产者取消、失败、回退或离开可追踪阶段：依赖失效，兄弟恢复普通可竞争状态，原始deadline保留；可换生产者但不能形成环。
- 没有进度、预测等待已无收益、资源长期不足：到有界条件即FALLBACK，不能无限等共享；过deadline的请求继续正常服务。
- RID复用、flush、请求结束：按epoch/generation清除关系与预留，不能误认旧生产者。
- 所有含时钟/代价/依赖选择的决策在request-plane rank0完成，广播RID顺序、生产者、目标G、预留及释放结果。各rank核请求集合和计划版本；不各自按本地时钟重选。
- COW/锁/refcount、HiCache回载、DCP owner映射、MTP verify/decode语义保持原生路径；首版不改变缓存内容与模型数值算法。

## 5. 分阶段实现与验收

| 阶段 | 变化 | 判定 |
|---|---|---|
| A：已排130ez6z | 首次冷/暖类别冻结 | 看兄弟命中增长后是否不再因类别改变被降级，以及新增fast/turn坏例 |
| B：诊断与最小闭环 | 实际生产者、原生held协调、跨waiting/partial依赖生命周期、真实READY事件 | 查明识别→获选→可复用→兄弟入批中哪段仍拖延，证据不足时不扩大改动 |
| C：有界短尾预留 | 与已有块预算/停车整合 | READY→实际入批缩短；预留失败可恢复，冷通道不空转 |
| D：仅按瓶颈追加 | 若G长期缺KDA快照，考虑让生产者某块恰好在合法G结束；若排序仍浪费，再加组成本评分 | 单独算新增块固定开销、状态/锁驻留和净通过条数，不能把改变块边界当免费优化 |

阶段D的强制块尾还必须检查既有 `mamba_branching_seqlen`：当前单快照逻辑可能优先保存更早分叉点，不能仅改变extend_end就声称G一定发布。先确认实际track depth；首版避免为所有家族新增双快照或额外状态槽。

必要反例覆盖：缓存增加而预算不缩短；LPM-held领头与兄弟不形成互等；只有FULL KV而缺KDA状态不能READY；host-only不能假装device-ready；资源预检通过但实际分配失败能恢复；生产者取消/flush/RID复用无陈旧依赖；各rank接收不同本地时钟仍应用同一计划。测试使用真实调度/adder与缓存契约，不只测新评分函数。

TP8先用固定N34和同一底座的开场对照，按同ID列出chain修复/新增。记录四个时刻：生产者获选、兄弟首次获得合法prefix、兄弟获选、首token；报告实际新算token、重复前缀计算、无效预留、TPOT与turn/fast，以及KV/KDA峰值。短测只作机制诊断；有实测收益再到N38和完整cohort。既有130ez队列由Fable管理，本设计不自行插队。

## 6. 一个账本示例

五个请求各35K、共享32K、各有3K尾巴：组的计算账本约为32K＋5×3K＝47K。先选择合适生产者，32K形成合法KV/KDA检查点后，兄弟即可按短尾预算获选，不需要等生产者整段decode结束。47K是组成本估计；现有缓存已经节省的重复前缀不能再次记成新增收益。

本机制争取的是让这47K在各自deadline前完成更多请求。独有历史很长、没有可共享前缀的冷链首，仍依赖prefill执行速度；两条路线的收益不能直接相加。
