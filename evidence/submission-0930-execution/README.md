# 09-30 正式提交归档：47798 / 47800

2026-10-03，Codex 只读查询平台并核验。当前正式结果统一见 [提交记录](../../notes/submissions.md)；本目录保留当时的上传产物、选择记录及构建收据。

两包共用 `lh-img:0930a` 和引擎 `ca5d646c252688177480c7e67cec0a901e4b7069`。引擎 Git tree 为 `c00a0cc4bca19797289da029bf7ded06dc78489e`。正式源码、原始提交历史和审核记录已经合入默认分支；本次整理未修改引擎实现。

| 方案 | 冻结配置 | 实际上传 ZIP | 上传回执 | 官方终态 |
| --- | --- | --- | --- | --- |
| CAP / 47798 | [official-0930-CAP.json](../../submission/official-0930-CAP.json) | [CAP/submission.zip](CAP/submission.zip) | [CAP/upload-receipt.json](CAP/upload-receipt.json) | [attempt-47798](../official/attempt-47798-final-20261003.json) |
| TPOT / 47800 | [official-0930-TPOT.json](../../submission/official-0930-TPOT.json) | [TPOT/submission.zip](TPOT/submission.zip) | [TPOT/upload-receipt.json](TPOT/upload-receipt.json) | [attempt-47800](../official/attempt-47800-final-20261003.json) |

[manifest.json](manifest.json) 记录上述配置、ZIP、回执、原始终态和构建资料的 SHA256。两份 ZIP 与 09-30 选择记录中的哈希一致，三个服务配置副本逐字节等于对应冻结配置。ZIP 顶层 `submission.json` 是 Playground CLI 包元数据，实际服务配置位于 `outputs/submission.json`、`results/submission.json` 与 `execution/results/submission.json`。

- [image-build.json](image-build.json)、[Dockerfile](Dockerfile)：成功构建记录和实际增量镜像 Dockerfile。
- [decision.json](decision.json)：当时为何选 CAP / TPOT、已有的本地证据边界、未提交 BASE 和取消的后续任务。
- [preparation.json](preparation.json)：提交前的源码、构建与配置身份。
- 两个方案各自的 `dry-run.json`、`arm_manifest.json`、`bundle-audit.json`：离线预检及打包审计。`DRY_RUN_ONLY` 是当时该收据的阶段状态；后续上传与正式结果另由上述回执证明。

在仓库根目录运行 `python3 scripts/verify_release.py` 可核验发布资料，不需要 GPU 或平台凭据。服务运行及评测入口见 [复现说明](../../notes/reproduce.md)。
