# T128：共享前缀生产者与兄弟准入的开发验证

2026-09-27，Codex。**DIAGNOSTIC，全部指定请求已完成；不是能力门或 N@SLO 成绩。** 底座 `741f3eda`，新机制与本目录同一提交，分支 `codex/128-prefix-producer`。结论及 TP8 交接见[报告](../../notes/reports/prefix-producer-admission-0927.md)，机制见[128p 文档](../../engine/docs/128-prefix-producer.md)。

## 原始记录与范围

| 文件 | 含义 |
| --- | --- |
| [cpu-tests.log](cpu-tests.log) | 144 项 CPU 检查的完整输出，全部通过 |
| [cpu-planner.json](cpu-planner.json) | 34/64 个 201024-token 请求的首次与稳态 CPU 规划成本，各三组样本 |
| [on-candidate.log](on-candidate.log)、[输出](on-candidate/responses.json)、[收据](on-candidate/summary.json) | 最终生产代码 ON，TP2/DCP2/MTP/graph/HiCache，21 请求完成 |
| [off-final.log](off-final.log)、[输出](off-final/responses.json)、[收据](off-final/summary.json) | 同一配置 OFF，21 请求完成；后续生产代码改动均在 ON 路径内 |
| [comparison.json](comparison.json) | 最终 ON 对 OFF 的 21 条逐请求输出比较：21/21 token 相同，输出 logprob 最大绝对差 0 |
| [决策摘要](on-candidate/analysis/summary.json)、[逐兄弟 CSV](on-candidate/analysis/dependents.csv) | 17 条决策、9 次预留和实际准入、12 条 prefill 完成；READY 到准入 6.15–6.85 ms |
| [on-final.log](on-final.log)、[输出](on-final/responses.json)、[比较](comparison-initial.json) | 较早的完整观测版本：尚未改等收益生产者的决胜顺序，也没有 received 日志字段；只作开发过程证据，最终验收引用 on-candidate |

最终 ON/OFF 服务日志没有 Traceback、OOM、rank_match_disagreement。最早一次实机试跑发现完成日志接线遗漏，已修正并加回归；那次试跑未列为完整观测验收，原文件仍在开发机本次工作目录的 `on.log`/`on/`，未作删除或迁移。

## 配置与负载身份

开发机工作目录 `/sjtu/linhang/arena/runs/prefix-producer-20260927`，GPU0/1。既有模型路径 `/sjtu/linhang/arena/runs/dcp-prefill-local-kv-20260926/model-mtp-h16`：缩小的 8+1 层、H16 模型，沿用 [dcp_check.py](../../scripts/analysis/dcp_check.py) 的确定性缩放 dummy 初始化。真实引擎执行调度、COW、缓存发布、前向与回收；没有替换这些执行路径。模型无正式能力含义。

TP2、DCP2、NEXTN steps/top-k/draft-tokens 为 3/1/4；CUDA graph decode 上限 8（原生 MTP 三阶段捕获），running 12，bf16 KV，Triton KDA、TileLang DSA、Humming，HiCache 4 GB、write-through、page-first、kernel I/O。max_total_tokens 65536，Mamba 槽 96，chunk 8192，max_prefill_tokens 16384，cold cap 8192，short 4096，PDI 1，124 与 freeze 开启、旧 128/125/126 关闭。完整引擎参数和环境以[生成脚本](../../scripts/analysis/prefix_producer_dev.py)、[启动脚本](../../scripts/analysis/run_prefix_producer_dev.sh)与日志为准。

这是脚本生成的合成机制负载，不使用评测 cohort：Python `random.Random(128)`，token 在 `[0,10000)`；四条共享 16384-token 前缀，尾部分别 3072/512/1024/1536；另有独立 14080-token 请求。先以五条预热，再两次真 `/flush_cache`，每次运行五条冷请求、三条暖重复，共 21 条。每条 greedy、16 个输出 token、ignore_eos，脚本检查输出完整及 logprob 有限。返回成功的 summary 说明两次 flush 及全部 generate 完成；原生 shutdown 后结束，GPU0/1 已释放。

## 复算

从 worktree 根目录执行；这些命令读取本目录已保留的记录，不启动 GPU 任务。

```bash
python3 scripts/analysis/prefix_producer_report.py \
  evidence/T128-prefix-producer-20260927/on-candidate.log \
  --output /tmp/prefix-producer-analysis
```

输出一致性复算：

```python
import json
import math
from pathlib import Path

root = Path('evidence/T128-prefix-producer-20260927')

def rows(arm):
    result = {}
    for case in json.loads((root / arm / 'responses.json').read_text()):
        for index, row in enumerate(case['result']):
            key = case['case'], index
            assert key not in result
            result[key] = row
    return result

on, off = rows('on-candidate'), rows('off-final')
assert on.keys() == off.keys() and len(on) == 21
same, delta = 0, 0.0
for key in on:
    a, b = on[key], off[key]
    assert len(a['output_ids']) == len(b['output_ids']) == 16
    same += a['output_ids'] == b['output_ids']
    pa = a['meta_info']['output_token_logprobs']
    pb = b['meta_info']['output_token_logprobs']
    assert len(pa) == len(pb) == 16
    assert all(math.isfinite(p[0]) for p in pa + pb)
    delta = max(delta, max(abs(x[0] - y[0]) for x, y in zip(pa, pb)))
print(dict(requests=len(on), same_tokens=same, max_logprob_abs=delta))
```

需要重新执行开发机诊断时，先确认 GPU0/1 可用，将本分支 `engine/sglang` 放到工作目录 `engine/sglang`，将 `prefix_producer_dev.py`、`dcp_check.py` 放到其 `engine/`。然后由启动脚本运行，ARM 必须是新的目录名：

```bash
bash scripts/analysis/run_prefix_producer_dev.sh \
  /sjtu/linhang/arena/runs/prefix-producer-20260927 on-new 1 1 1
# OFF 同参数，把 arm 改为 off-new、feature 改为 0。
```

运行时完整 stdout/stderr 要另存 arm 日志；脚本不替代 Pod harness。TP8/S1 实际组合、真实权重、N34 开场 chain 改善和 KV/KDA 峰值均待 Fable 独立复核及实验，不能由两卡的输出一致性外推。
