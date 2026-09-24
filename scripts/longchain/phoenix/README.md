# Phoenix结构观测

本目录只负责只读采集和结构分析；生产正文不进入长程测试集。连接/数据库背景见仓库[`phoenix_trace_analysis/`](../../../phoenix_trace_analysis/)。这些脚本只有明确执行采集命令时才联网；`--help`不会访问服务。

| 脚本 | 职责 |
|---|---|
| `phoenix_longchain_observe.py` | 指定session与时间窗的只读导出、结构审查；也提供共用解析函数 |
| `phoenix_collect_workload.py` | 按冻结数据库抽样框批量采集，可续传 |
| `phoenix_behavior_profile.py` | 从结构记录生成连续联合事件profile |
| `phoenix_event_windows.py` | 检查选定长gap/压缩边附近的工具与压缩span |
| `phoenix_workload_catalog.py` / `.html` | 用结构摘要生成可浏览的目录 |

查看参数示例：

```bash
python3 -B scripts/longchain/phoenix/phoenix_collect_workload.py --help
python3 -B scripts/longchain/phoenix/phoenix_behavior_profile.py --help
```

实际采集需要`arize-phoenix-client`和`pandas`，鉴权从`PHOENIX_API_KEY`环境变量读取，不把key写入脚本。原始正文是可删除的临时cache；目前已按用户要求清除，保留的生成输入为[`evidence/phoenix-longchain-20260924/expanded/joint-events.jsonl`](../../../evidence/phoenix-longchain-20260924/expanded/joint-events.jsonl)。抽样权重、窗口截断、工具等待分解与模型token口径的限制见各结构报告。
