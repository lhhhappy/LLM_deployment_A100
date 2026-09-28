# Chain-max 16k：不钉池候选，已打包，未上传

2026-09-28 UTC，Codex。按 fable 的 chain 优先决定，从 400 版仅删除 ` --max-mamba-cache-size 400`，其他字节不变。引擎 `20a58da9` 与镜像 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0927a` 不变；没有重建镜像。

- 当前配置：[submission.json](submission.json)，同内容在 `submission/official-0927-chainmax-nopin.json`。去 MTP、DCP=1、running 32、128p+131+132、16k 块、SHORT_TOKENS=2048、load 1.05；118/126 关闭，Mamba 槽数由引擎按资源计算。
- [逐项对照 46676](config-audit.md) / [机器收据](config-audit.json)：只移除钉池参数，性能配置与 fable 的 [eznb 任务快照](n30-job-reviewed.sh) 一致。正式日志设置仍沿用已审版本。
- [配置检查](check-submission.txt)：0 错误、0 警告。[打包收据](bundle-audit.json)：三个输出目录的配置与当前 JSON 一致；只做 dry-run，没有创建 attempt。
- 共用[原镜像构建证据](../submission-0927-chainmax400/build-receipt.json)与[实际构建日志](../submission-0927-chainmax400/build-log-retry.json)。400 版配置和 ZIP 保留为历史材料，不用于当前上传。
- [机制期望](expected-mechanisms.txt)尚不是运行收据。fable 负责 eznb 修后引擎 TP8 验证；状态以主 checkout 的 `notes/queue.md` 为准。最终上传仍由用户确认；开场健康不能写成 N30 整档通过。

本地包为本 worktree 的 `build/submit_0927_CHAINMAX_NOPIN.zip`，配置目录为 `build/submit_0927_CHAINMAX_NOPIN/`。原因见[浅前缀 held 坏例分析](../../notes/reports/chainmax-pin400-held-0928.md)。
