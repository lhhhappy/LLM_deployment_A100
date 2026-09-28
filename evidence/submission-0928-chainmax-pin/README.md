# FINAL + Mamba400：独立提交包

2026-09-28，Codex。包名 `submit_0928_CHAINMAX_PIN`，配置审计、check-submission 和 CLI dry-run 已完成，**尚未创建新 attempt**。按约等 47043 出结果，再由 fable 复核并协调上传；本次查询 47043 仍为 queued。

相对已上传 47043 的 [FINAL 配置](../submission-0927-chainmax-final/submission.json)，[本配置](submission.json)逐字节只增加 ` --max-mamba-cache-size 400`。镜像仍为 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0927a`，引擎仍为 `20a58da99737c70f21c801ff587e94d8f54022be`。复用原构建收据，未重打镜像。124m 及其新日志代码不在此镜像中。

保留：无 MTP，DCP 默认1（不显式传参），running/graph=48，16k 块，128p/131/132，SHORT_TOKENS=2048、load=1.05，cold/warm 饥饿上限600/120秒，原生 hold 阈值4096，118/126关闭。env 和其余命令不变；2048 是准入阈值，不是保证的预留容量。

包位置（本 worktree）：`build/submit_0928_CHAINMAX_PIN/`、`build/submit_0928_CHAINMAX_PIN.zip`。配置另存 `submission/official-0928-chainmax-pin.json`。历史 FINAL/400/NOPIN 包均未覆盖。

## 实测支持与边界

[eznc 运行收据](runtime-receipt.json)与[原始 job.log](eznc-job.log)：同性能配置在 TP8/N30 启动成功，机制核对通过、能力冒烟12/12；[8 rank 的状态池日志](eznc-pool-startup.txt)均为400槽，KV池1810112 token（无钉池1260160）。2400秒派发后排空1726请求，零错误。正式只关闭逐请求决策日志并调低HTTP日志级别，差异见 [配置审计](config-audit.json)。本地启动收据不是正式镜像启动的替代收据。

独立按原 harness 对 eznb/eznc 的 **1687 个共同 ID** 复算：

| 指标 | FINAL 无钉池 | FINAL +400 |
|---|---:|---:|
| chain >30 秒 | 12 | 7（改善5、新漏0） |
| chain p95 | 30.305 s | 24.969 s |
| turn >15 秒 | 11 | 6 |
| overall >5 秒 | 209 | 21 |
| fast >3 秒 | 248 | 40 |
| TPOT >0.10 秒 | 83 | 118 |
| TPOT 均值 / p95 | 54.93 / 98.39 ms | 60.00 / 120.94 ms |

chain 与稳态等待方向明显改善，但 **TPOT p95 在这个窗口超过0.10秒**。这是用户 chain 优先下的实测取舍，不能写成完整N30通过，更不能推断正式N34通过。两臂分别完成1689/1726条，共同1687；集合差异见 [同ID汇总](paired-n30/comparison.json)，[变化请求](paired-n30/changed-chain.csv)保留具体等待与TTFT。

原始窗口收据 [before](before-timed-verdict.json) / [after](after-timed-verdict.json)均为 DRAINED，派发集合与 raw 对齐。原始 raw/server.log 沿用共享 checkout `evidence/L130eznb-tail_rot150_n30_chainmax16k_fix_40m/N30/` 和 `evidence/L130eznc-tail_rot150_n30_chainmax16k_fix_mamba400_40m/N30/`，不复制大日志。run源路径也保存在比较JSON中。

## 复现与检查

- [逐项对照46676与FINAL](config-audit.md)：确认只加400，性能参数匹配 eznc 的[任务快照](n30-job-reviewed.sh)。
- [check-submission](check-submission.txt)：0错误、0警告，所有27个启动参数存在于真实底包。
- [CLI dry-run](bundle-dry-run.log)与[包内容检查](bundle-audit.json)：三个输出位置的 submission.json 一致；顶层 submission.json 是CLI元数据。未创建 attempt。
- [expected-mechanisms.txt](expected-mechanisms.txt)仅为期望，实测机制见 job.log。

```bash
python3 -B scripts/analysis/audit_chainmax_candidate.py \
  --evidence evidence/submission-0928-chainmax-pin \
  --pin-final-from evidence/submission-0927-chainmax-final
python3 -B scripts/check_submission.py submission/official-0928-chainmax-pin.json \
  --final --trace submission/stub-trace.jsonl \
  --sglang-src /workspace/Agentic_science_challenge/build/base_exact/sglang/srt
playground submit --challenge-id llm-challenge-arena-v1 \
  --outputs build/submit_0928_CHAINMAX_PIN --trace submission/stub-trace.jsonl \
  --model gpt-6 --harness codex --dry-run --bundle-out build/submit_0928_CHAINMAX_PIN.zip
python3 -B scripts/analysis/compare_chain_windows.py \
  --before /workspace/Agentic_science_challenge/evidence/L130eznb-tail_rot150_n30_chainmax16k_fix_40m/N30 \
  --after /workspace/Agentic_science_challenge/evidence/L130eznc-tail_rot150_n30_chainmax16k_fix_mamba400_40m/N30 \
  --before-verdict evidence/submission-0928-chainmax-pin/before-timed-verdict.json \
  --after-verdict evidence/submission-0928-chainmax-pin/after-timed-verdict.json \
  --harness-dir /workspace/Agentic_science_challenge/s1-dev/harness \
  --out evidence/submission-0928-chainmax-pin/paired-n30
```
