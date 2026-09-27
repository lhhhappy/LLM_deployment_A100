# 回放链首、源会话起点与冷请求的区别

2026-09-27，Codex。公开数据核实完成，已发 Fable 请求独立复核；不修改数据或评分。

用户指出，开发集的链首可能来自源业务会话中段。**这一点成立；但不能通过第一条 message 是否为 system 判定。**
本次只读本地公开开发集的元数据与正文结构，没有访问数据库、下载新正文或输出正文内容。

## 实测范围与结果

数据为 `s1-dev/data/dev-combined-v1`：311 条链、722 个公开请求。使用原 `s1_common.load_index()`
选择服务请求、排序并确定每条链的第一条；不把用户新提供的源数据库 5601 请求统计混入本次核实。

| 原始标注 | 链首数量 | 正文的独立 system 字段非空 | messages[0] 为 user | 历史缓存读取 > 0 |
| --- | ---: | ---: | ---: | ---: |
| session_start 且 chain_index=0 | 104 | 104 | 104 | 81（10 条缺计数） |
| 非 session_start 且 chain_index>0 | 207 | 207 | 207 | 196（1 条缺计数） |

207 条中段切片占回放链首的 66.6%，原 `phase` 为 intra 115、turn_start 72、context_reset 20。
311 条链首的 `edge_type` 为 chain-head 104、system-tools-changed 148、append-only 53、compact-rebuild 6。
这些字段说明源数据中的阶段和边类型；本核查没有重建完整源业务会话，也不把它们当作实测缓存命中。

实际正文格式是 `{system, tools, messages}`，system 与 messages 分开保存。原 `Renderer.render()`
（`s1-dev/harness/s1_common.py:356`）先用独立 system 字段构造系统消息，再拼接 messages，最后套用
GLM chat template。因此仅检查 messages[0] 会把全部 311 条都看到成 user，无法识别源会话起点。
中段切片也可能保留完整 system 和历史；这里 207 条中有 135 条含 assistant 消息，不能反过来把另外
72 条判成源首轮。正的源缓存读取也不是阶段判据：104 条原标注首轮中就有 81 条带正计数。

## 对 chain 结论的修正

必须分开以下三个概念：

1. **源会话起点**：由源 session、chain_index 和 phase 标注判断。
2. **回放链首**：原 harness 排序后的 idx_in_chain=0，不要求是源会话首轮。
3. **实际冷请求**：取决于该次引擎运行中可复用的 KV、混合状态与实际命中，不能仅由前两个标签确定。

原 `phase_gate()`（`s1-dev/harness/s1_common.py:107`）将 idx_in_chain=0 或 phase=context_reset
归入 chain 的 30 秒门。本次 311 条回放链首调用原函数全部得到 30 秒；公开集另有 3 条链内 context_reset，
总 chain 门为 314 个请求。中段切片仍属于正式定义允许的 chain 压力，不能删掉、重新分桶或提前加载其历史缓存。

用户已确认新统计来自**源业务历史会话**。源业务当时的缓存读取与当前 GLM 评测的缓存不是同一个状态；
`task.md` 的 `/flush_cache` 合同要求每档正式测量从清空前缀 KV 开始。回放期间允许自然形成共享前缀，
因此也不能声称所有回放链首始终零命中。源数据库的 13.3% 与 v4 预期未缓存代理的 81.0% 可以提示口径差异，
不能直接证明正式负载比 v4 更暖，更不能据此换算 N@SLO。

“本地大链首正文来自公开数据”与“本地复制了正式的压力”是两个命题。前者不保证后者：正式链顺序、
新链入场时机、链内重算尾部、长输出和缓存驻留仍需校准。不能只凭正式 chain p95 倒推出超时全在开场。

执行层优化仍针对实际 prefill 工作：开场同时入场、稳态加入的中段切片、链内重建都要验证。
后续逐请求归因应同时记录原始 phase/边类型、回放 idx、到达时间、实际 cached_tokens、新算 token、
首次入批等待和执行耗时；开场与稳态分层报告，最终保持原 harness 全量评分。

## 复现与证据

[核实脚本](../../scripts/analysis/audit_chain_origins.py) 使用原 harness，只导出字段与正文结构计数：

```bash
python3 scripts/analysis/audit_chain_origins.py \
  --s1-dev /workspace/Agentic_science_challenge/s1-dev \
  --out evidence/chain-origin-audit-0927
```

[汇总](../../evidence/chain-origin-audit-0927/summary.json) 与
[逐链 CSV](../../evidence/chain-origin-audit-0927/heads.csv) 可由上述只读输入复算。
评分定义以 [task.md](../../llm-challenge-arena-v1/task.md) 为准；本次不是服务实验或正式成绩。
