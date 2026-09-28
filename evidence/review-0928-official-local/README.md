# 09-28 正式与本地复盘的派生证据

解释与采用判断见[综合复盘](../../notes/reports/0928-official-local-synthesis.md)。正式原件和提交收据分别在 `evidence/official/` 与 `evidence/submission-0928-kda/`；本目录不代替官方成绩。

- `local-today/audit.py`：调用仓库原 `compare_window`、`window_gates`、原 harness，复算 N30 各臂与三方共同 1687 个 ID。`audit.json` 保留源文件路径/SHA、有效性与原始判分收据；每对的 CSV 保留修好/新坏/持续请求。九份闭合 raw 属 DRAINED；j/k/p 仅旧 OPEN 快照。
- `n34-paired-summary.json`、`n34-paired.csv`：相同全候选配置，ezno N30 与 eznq N34 的共同 1850 个 ID，另存各自全 raw、按派发时间分桶和采样峰值。`candidate_completed` 在该配对摘要中是共同集大小，不是 N34 的全部 1960 条。
- `n34-new-fast-memory.json`：49 条新坏 fast 的缓存差值和等待区间附近的 KV 采样。包含相邻采样，只有相关意义，不能据此证明准入失败或某一核阻塞。
- `recompute_n34.py`：从原始 raw 和 metrics 重新计算并断言归档的门统计、时间分桶、显存峰值与坏例计时；N34 另核 timed receipt 的原始 SHA 与派发 ID 集。相邻 KV 采样归因不作为机制断言。
- N34 原始完整包为 [closed-N34.tgz](../L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/closed-N34.tgz)，SHA256 `c4b9b0e8c66a30a2ecc52bebaa897f11ed9be8fd8b70bf8e3abca11dfb08ce21`；解包目录保留 raw、run、server、timed verdict、派发 ledger、flush、metrics 和传输收据。

在仓库根目录复算（仅 CPU；第一条重写自身派生 JSON/CSV，第二条只核验）：

```bash
python3 evidence/review-0928-official-local/local-today/audit.py
python3 evidence/review-0928-official-local/recompute_n34.py
```

逐文件校验清单见 `manifest-sha256.json`；本目录中的报告数字不另造判门逻辑。完整两份独立审计见[正式历史](../../notes/reports/official-history-audit-0928.md)、[今日 N30](../../notes/reports/local-today-audit-0928.md)。
