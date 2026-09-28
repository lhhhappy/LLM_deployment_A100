# KDA TP8 闭合对照：ezno 对 eznn

2026-09-28，Fable 运行与首次汇报，Codex 独立复算。范围：N30、2400 秒派发后排空的诊断窗口；不是全量档位成绩。

- [eznn.tgz](eznn.tgz)、[ezno.tgz](ezno.tgz) 保留两轮完整已获取的 `N30/`：raw、run、summary、flush、服务日志、任务日志、metrics、GPU 记录以及原 fetch/verdict。
- [source-manifest.json](source-manifest.json) 记录原目录文件和复算使用的原 harness/分析工具 SHA256。[paired-summary.json](paired-summary.json)、[paired.csv](paired.csv) 是从共同 ID 生成的派生物。
- 两轮各自闭合行数为 1844/1869，交集 1844。归档 `level_verdict.json` 对错数据根 `dev-combined-v1` 做完整性检查而返回 INVALID；这里保留原件，仅依据真实 run/flush 与 raw 合同确认窗口排空，不将短测改称整档通过。`window/` 下旧的运行中快照未参与比较。
- [submission.json](submission.json) 是独立准备的候选配置副本；[image-build.json](image-build.json) 是成功构建的原收据。没有发起正式上传。

在两个单独目录解开 tgz 后，在具备 `s1-dev/harness` 的仓库中执行：

```bash
python3 recompute.py /path/to/repo /path/to/eznn/N30 /path/to/ezno/N30 /path/to/output
```

[recompute.py](recompute.py) 复用仓库 `compare_window.py`、`window_gates.py` 和原 harness 分桶/分位数；复算前可用 source-manifest 核对版本。调用不写输入目录，不输出完整 cohort 或正式成绩。结论和归因边界只维护在 [R37](../../../research/codex/R37_kda_prefill_sm80_0928.md)。
