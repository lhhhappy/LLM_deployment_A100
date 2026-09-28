# Chain-max FINAL：配置与提交包

2026-09-28 UTC，Codex。离线检查和 dry-run 打包完成，未创建或上传 attempt。fable 已同步 eznb 的性能配置；TP8 运行收据仍待取得。

当前配置：[submission.json](submission.json)，同内容在 `submission/official-0927-chainmax-final.json`。引擎仍为 `20a58da9`，镜像仍为 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0927a`；复用[构建收据](../submission-0927-chainmax400/build-receipt.json)与[构建日志](../submission-0927-chainmax400/build-log-retry.json)，未重建镜像。

| 相对 NOPIN 的改动 | 最终值 | 含义 |
|---|---|---|
| `SGLANG_AX_DEADLINE_MAX_WAIT_S` | `600` | 延后冷请求饥饿提级；不是修改 30 秒 SLO |
| `SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S` | `120` | 显式保持 warm 上限；不设置会继承 cold=600 |
| `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` | `4096` | 避免为浅共享前缀降级请求；128p 依赖发现保留 |
| `--max-running-requests` / `--cuda-graph-max-bs` | `48` / `48` | 扩展准入与 decode graph 捕获上限 |
| `--dcp-size` | 不传 | 固定引擎默认 1，解析器不要求显式传入 |

其他配置保持 NOPIN：无 MTP、无钉池、16k 冷块，128p+131+132，SHORT_TOKENS=2048、load=1.05，118/126 关闭。2048 是准入阈值，不是保证预留；请求正文、thinking 和输出预算不变。

running 扩展的直接准入收益出现在实际在途数超过 32 时（下一评测档 N34）；但 48 档 graph 捕获和静态缓冲在启动时生效，不能把整个改动说成“仅影响 N34 以上”。eznb 已同步 48/48，可验证相同启动配置；N30 结果仍不能代替 N34 的性能验证。此次为组合配置，不单独归因于某个旋钮；多轮停车缺口仍存在。

- [逐项对照 46676](config-audit.md) / [机器收据](config-audit.json)：确认相对 NOPIN 只有上表改动，性能配置与[最新 eznb 快照](n30-job-reviewed.sh)一致；日志差异沿用已审约定。
- [实际配置解析](effective-defaults.json)：CPU 调用真实 `deadline_config()` 确认 cold=600、warm=120；源码确认 DCP 默认 1。
- [check-submission](check-submission.txt)：0 错误、0 警告。[dry-run 日志](bundle-dry-run.log)与[包内核对](bundle-audit.json)：三个输出目录的配置一致，未创建 attempt。
- [机制期望](expected-mechanisms.txt)不是实测收据。`124=on` 不证明 600/120 的具体取值，held4096 也无独立 token；取值须结合任务环境与启动收据核对。运行与队列由 fable 跟进，见共享 checkout 的 `notes/queue.md`。

当前包为本 worktree 的 `build/submit_0927_CHAINMAX_FINAL.zip`，配置目录为 `build/submit_0927_CHAINMAX_FINAL/`。400 与 NOPIN 包保留为历史材料。正式上传仍按约由用户确认。

复现审计：`python3 scripts/analysis/audit_chainmax_candidate.py --evidence evidence/submission-0927-chainmax-final --final-from evidence/submission-0927-chainmax-nopin`。
