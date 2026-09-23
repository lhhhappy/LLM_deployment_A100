# T56 — 026 N18 逐条归因证据

报告：[R20](../../research/codex/R20_true_lcp_attribution.md)。所有分析均为 CPU；脚本不调用网络/引擎，输出仅写本目录。完整原始日志通过 GPU SSH → `scripts/pod/pexec_codex` 读取、gzip/base64 回传并校验 SHA；没有推送脚本、操作 GPU/队列/引擎或修改生产工具。

## 复现

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/t42-venv/bin/python evidence/T56/attribute.py
```

可使用其它安装了 manifest 中 numpy/transformers/tokenizers 版本的 Python 环境。原 harness Renderer 以 `local_files_only=True` 读取仓库 tokenizer，不加载模型。`--render-only` 只产出 token/LCP 分类；`--reuse-render` 在输入 SHA 未变时复用本目录的渲染结果。完整复现使用默认命令。`reproduction.log` 为实际完整运行输出。

## 文件

| 文件 | 含义 |
|---|---|
| `attribute.py` | 完整分析与输入/输出SHA记录；约束不成立立即断言失败 |
| `026_server.log` | pod 完整日志原字节，LF行号；84个CR进度条不算新行 |
| `log_fetch_receipt.json`、`fetch_verbose.stdout` | 远端byte count/SHA、压缩传输证据；其它fetch/probe日志记录SSH中断及重试 |
| `manifest.json`、`render_inputs.json` | raw/dev/cohort/Renderer/tokenizer/补丁/计时源码SHA、依赖版本与输出SHA |
| `prompt_metadata.json`、`pairs_rendered.json` | 722个prompt的长度/渲染与token哈希/角色点；411个真实LCP及冻结差异 |
| `pairs_attributed.csv/json` | 全部411条逐条分类；CSV便于筛选，JSON含历史前驱和完整候选位置 |
| `classification_summary.json`、`render_summary.json` | 分类计数/token合计；正差与64网格上限分开 |
| `time_alignment.json` | 346个唯一形状/时间锚点、offset=0、亚秒报告插值分段 |
| `prefill_batches.json` | 全日志1733个prefill；仅1234个带raw_members的批次属于测量期 |
| `request_batch_map.json`、`batch_mapping_validation.json` | 重建请求分块与逐批计数/token闭合收据 |
| `fast_details.md`、`fast_10.json` | 10条fast的raw时间拆分、到达/等待/执行的全部批次 |
| `decode_overlaps.md`、`short_tpot_22.json` | 22条短输出的全部相交prefill原日志字段、每批停顿估计、窗口并集和±50ms敏感性 |
| `decode_baseline.json`、`pause_by_batch_size.json` | 无prefill报告段的decode步估计；按批预算分组的停顿分布 |
| `validation.json` | T56-01至04的完整运行断言收据 |

## 口径

- VERIFIED：重渲染长度、真实LCP、原始时间/token、harness名单、批计数约束与SHA。
- INFERRED：哪个历史节点提供/丢失状态、根据约束解出的请求批成员、亚秒时刻及停顿。
- c 是“零命中/共享区深回退”操作性类别，不是已经观测到某池的淘汰事件。
- 冻结高估是独立字段：仅高估便能解释的3条归d；另3条保留b与真实残余损失。
- `prior_kv_overlap_gap` 不能证明状态丢失；只有当前 LCP 包含前驱整个 cached 前缀，`verified_prior_hit_regression` 才能证明已用状态深度退化。
- `new/throughput` 是报告间隔，不是kernel耗时。停顿采用原始exec_start/first约束或续块间隔扣decode步，属于估计；±50ms扰动不是置信区间。
- 所有同链收益上限仅相对于本轮已发送的前驱prompt；不包括无法核对的生成输出重用、其它跨链潜力，也不保证可实现相同比例的时间收益。
