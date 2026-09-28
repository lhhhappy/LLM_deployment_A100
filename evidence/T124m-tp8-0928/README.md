# 124m TP8 证据

结论见 [报告](../../notes/reports/124m-tp8-attribution-0928.md)。`on/off/receipt.json` 为短探针原始 JSON（只重排 JSON 格式，字段保留）；`source.json` 记录 Pod 路径。两臂状态探针均为 STATE_PASS；`numeric-comparison.json` 是未通过严格 OFF 区间检查的数值诊断，不能称数值一致。

`on/off/timed_verdict.json` 来自 N30 的另一对 40 分钟负载实验，均为 DRAINED，不是上述短探针的判分。raw 与完整服务日志沿用共享 checkout 已下载的原始证据，不复制大日志。Pod 原件为 `/tmp/ax/runs/130ezne-tail_rot150_n30_124m_on_40m/N30/` 和 `130eznf-tail_rot150_n30_124m_off_40m/N30/`；服务日志在各自运行根目录。

`n30/` 是从原始 raw 和服务器日志生成的派生对照，使用原 harness 的门分类与分位数，不输出正式通过判定。命令（在本 worktree）：

```bash
python3 -B scripts/analysis/compare_chain_windows.py \
  --before /workspace/Agentic_science_challenge/evidence/L130eznf-tail_rot150_n30_124m_off_40m/N30 \
  --after /workspace/Agentic_science_challenge/evidence/L130ezne-tail_rot150_n30_124m_on_40m/N30 \
  --before-verdict evidence/T124m-tp8-0928/off/timed_verdict.json \
  --after-verdict evidence/T124m-tp8-0928/on/timed_verdict.json \
  --harness-dir /workspace/Agentic_science_challenge/s1-dev/harness \
  --out evidence/T124m-tp8-0928/n30
```
