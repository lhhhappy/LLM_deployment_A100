# 46676 / 46677 后续方向的复算证据

2026-09-26，Codex。状态为 DIAGNOSTIC，不是正式成绩或完整 cohort 判分。

- `submission-scope.json`：上传的 S1/S2 配置逐字段差异、引擎身份、DCP 现有任务与 S1 的差异，以及标为算术的容量估计。本次正式状态查询仍为 queued，状态可能在之后变化。
- `guard-window.json`：112 / 130b 原始请求和完整 TP0 日志的时间对齐；输入路径、SHA256、日志行号、原 harness 分桶、共同请求对照、护栏前后事件。原始文件留在既有位置，没有生成另一份原始数据副本。

复算命令（`--root` 指向保有这两次运行的原工作区）：

```sh
python3 -B scripts/analysis/audit_125_measurement_window.py \
  --root /workspace/Agentic_science_challenge \
  --output evidence/post-submission-46676-46677/guard-window.json
```

分析与工程判断见 [报告](../../notes/reports/post-submission-46676-46677-0926.md)。本次没有新跑 GPU 实验，也没有归档操作。
