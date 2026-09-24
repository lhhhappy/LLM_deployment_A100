# 合成长链分布与重复审计

这是 CPU 分布诊断，不是完整性验收、线上代表性证明或 GPU 成绩。

生成集 `/Users/lhappy/workbench/Agentic_science_challenge/cache/s1-dev-longchain-build`；manifest `19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c`。

| 指标 | 公开 s1-dev | 生成集 |
|---|---:|---:|
| 请求 | 722 | 5601 |
| 链 | 311 | 311 |
| session | 136 | 311 |
| fast_intra | 328 | 4765 |
| overall_intra | 388 | 5010 |
| turn_start | 20 | 159 |
| chain_start | 314 | 432 |
| 链长 p50 / p95 / max | 1 / 6 / 13 | 8 / 82 / 240 |
| prompt p50 / p95 / max | 36576 / 114376 / 256733 | 56610 / 161004 / 266208 |
| 冻结新增输入 p50 / p95 / max | 2867 / 61471 / 256733 | 538 / 13487 / 256733 |
| 输出预算 p50 / p95 / max | 198 / 915 / 5644 | 571 / 1836 / 9713 |
| gap ms p50 / p95 / max | 1547 / 16291 / 300782 | 2942 / 27110 / 300782 |
| cap后gap ms p50 / p95 / max | 1547 / 16291 / 300782 | 2942 / 27110 / 300782 |

源摘要 / 计划 / 实际事件：

```json
{
  "source_all": {
    "session_start": 104,
    "intra": 5125,
    "turn_start": 231,
    "context_reset": 141
  },
  "actual_all": {
    "session_start": 104,
    "intra": 5125,
    "turn_start": 231,
    "context_reset": 141
  },
  "planned_synthetic": {
    "turn_start": 139,
    "intra": 4622,
    "context_reset": 118
  },
  "actual_synthetic": {
    "turn_start": 139,
    "intra": 4622,
    "context_reset": 118
  },
  "displaced_planned": {},
  "context_pressure_events": 0,
  "replaced_behavior_references": 106
}
```

生成集估计gap 4921 条；0 链触发3600s cap。
Phoenix end-to-start proxy 不等于工具并集与用户思考分解；无法据此宣称恢复了正式到达节奏。

| 重复口径 | 公开 s1-dev 跨session组 | 生成集跨session组 |
|---|---:|---:|
| complete_prompt_excluding_req_id | 1 | 1 |
| introduced_blocks_exact | 8 | 110 |
| introduced_blocks_call_ids_normalized | 8 | 1929 |

GLM链首比较 48205 对；较对应source新增共享历史的对数 0；来源原本同session的分支对数 2229。

口径：公开集是链前缀样本，链长/事件量不能直接作为完整线上分布。新增输入主表比较双方冻结uncached_expected；生成集另报provenance中的实际LCP新增，二者不混用。
重复统计先剔除req_id；完整正文包含system/tools/messages。新增块是相邻快照完整消息/工具组多重集合的正差，保留历史与重建保留尾部不重复计数；每条链的入场历史单列，不当新增素材。归一化只改实际调用ID及其引用，保留参数与正文。短通用消息的重复不是自动失败。
生成session独立后，源中同session不同chain的相同入场状态会变成跨session重复；必须对照源head，不能全部当作新克隆。链首LCP扣除两侧system/tools渲染前缀后，再减对应源head同口径值。此处只检测链首，不能证明中后段不存在跨链共享；新增块重复是补充指标。
完整指纹组、来源素材引入次数、分位数及top重复组见同名JSON；没有以原self-check PASS代替独立完整验收。
