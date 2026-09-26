# DCP local prefill 开发证据

2026-09-26，Codex；分支 `codex/dcp-prefill-local-kv`。本目录包含两类独立证据：130ed 的服务日志分析，以及新补丁的两卡开发验证。130ed **没有运行新补丁**。结论、失败解释与待验证边界统一见[报告](../../notes/reports/dcp-chain-20260926.md)。未做归档、清理或重复 SHA 核验。

## 130ed：已排空的限定窗口

`130ed-drained/` 保存原始 raw、dispatch、metrics、完整 server/job log；原始 Pod 路径和读取时间见 [snapshot.json](130ed-drained/snapshot.json)。配置为 b261cbd9、W2、N30、MTP 3/1/4、HiCache host64、60 分钟派发，启动/预热/flush/机制记录在 job/server log。3123 个派发 ID 完整闭合，零 raw 请求错误；**DRAINED DIAGNOSTIC，不是完整 cohort 的 N@SLO 结果**。

派生文件位于 `130ed-drained-audit/`。在此 worktree 根目录复算（输出另选目录，不覆盖原证据）：

```bash
python3 -B scripts/analysis/dcp_run_audit.py \
  evidence/dcp-local-extend-20260926/130ed-drained \
  --reference /workspace/Agentic_science_challenge/evidence/L130-v3_open_A_n30/N30/raw_s1-dev-longchain-v3_N30_1790420510.jsonl \
  --output /tmp/dcp-local-130ed-review --drained
```

脚本调用原 harness selector 和 quantile；API dispatch 与首次选入 batch 的时间含义见报告，不当作纯调度或 GPU 时间。`130ed-snapshot/` 与 `130ed-audit/` 是更早的 49 分钟只读快照，仍保留在开发 worktree，不替代闭合窗口。

## 两卡：数值与时间

原始模型配置、运行日志、`.pt` 敏感输出和 `.inputs` 选键捕获保存在开发机 `GPU:/sjtu/linhang/arena/runs/dcp-prefill-local-kv-20260926/`。GPU0/1 使用用户授权的两张 A100；TP2/W2，8 层缩小 dummy 模型，H8 是每 rank attention 形状，不能冒充真实 TP8 模型。环境与启动参数由 [run_dcp_devbox.sh](../../scripts/analysis/run_dcp_devbox.sh) 固定；基线关闭新开关，候选开启主开关，2K/8K 另显式开启 `LARGE_MAX`。主开关及大块选项在生产中均默认关闭。

| 证据 | 含义 |
|---|---|
| `gpu-results/paired_short.pt*.json` | 27 个 extend token、P=14336、H8，同进程 40 次交错完整前向 |
| `gpu-results/local_large2048.pt*.json` | T=2048、P=14336，完整前向交错计时与峰值 |
| `gpu-results/local_large8192.pt*.json` | T=8192、P=14336，完整前向交错计时与峰值 |
| `gpu-results/local_large8192_final.pt*.json` | 就地清理修订后的再次计时；数值失败单独保留 |
| `gpu-results/step-bench-*.rank*.json` | 两 rank 真实集合通信、索引变换、partial attention、LSE 合并；含全样本和内存 |
| [numerical-summary.json](numerical-summary.json) | MTP/HiCache/实际接受跨页各组敏感 trace 汇总 |
| `gpu-results/*-comparison.json` | MTP 每条 trace 对照、逐行误差和覆盖阶段 |
| `gpu-results/{base,local}_*/` | MTP 请求响应、阶段 trace 元数据与验证器观测（不含二进制 tensor） |
| [codegraph-callers.txt](codegraph-callers.txt) | 调用图查询；机制说明另有流程图 |

计时通过 `AX_PAIRED_TIMING=1` 在同一进程交错切换执行 policy；敏感数值快照先于计时，反复修改 KDA state 的计时结果不充当数值证据。整步曲线入口为 [dcp_local_extend_bench.py](../../scripts/analysis/dcp_local_extend_bench.py)，大块矩阵使用 `AX_BENCH_LARGE=1`。MTP 使用 [dcp_mtp_probe.py](../../scripts/analysis/dcp_mtp_probe.py) 和 [dcp_mtp_compare.py](../../scripts/analysis/dcp_mtp_compare.py)；自然输出与 controlled proposal 分开，后者仍由真实 verifier 判接受。

## 必须保留的失败

原生 8K 的 [repeat2](gpu-results/local_8k_repeat2-compare.txt) 复现第二个 attention 层第 3950 行超出 1% 容差。不得只看 greedy token 相同就算通过。`AX_CAPTURE_INPUTS=1` 保存该行 Q/K/V/top-k；`AX_TOPK_ROW_ORACLE` 只在诊断脚本中固定选键。两次固定选键对照恢复通过，且替换前再次出现原生选键集合变化，见 [topk-isolation.json](gpu-results/topk-isolation.json)。后续已经补齐两臂各 10 次原生重跑和该行 indexer logits，定位为底包 A/A 也会出现的精确平局换组；保留原失败记录，不能改记为全路径数值 PASS。

## 独立审查后的修正证据

`review-fixes/` 保存修订后两 rank 的 543 形状 JIT/oracle 实测、各臂 10 次 indexer 逐字段对照、8K 完整前向 40 对交错计时、MTP 数值与实际机制/路由观测。具体结果、命令与待验边界见[修订报告](../../notes/reports/dcp-local-extend-review-fixes-0926.md)。二进制捕获留在原开发机目录，不归档或清理。

二进制 tensor 和完整 GPU 日志留在上述开发机运行目录，不重复纳入 Git；本目录提交可审阅的数值/时间记录、服务原始文本和派生表。真实 TP8 权重、质量、峰值显存及整档收益仍待验证。
