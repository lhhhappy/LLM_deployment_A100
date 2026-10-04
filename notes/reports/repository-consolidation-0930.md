# A100 项目迁移与归档（2026-09-30）

项目根目录为 `/workspace/Agentic_science_challenge/LLM_deployment_A100/`。代码、`.git`、`scripts/`、`build/`、`evidence/`、数据和缓存全部位于项目内。父目录只保留系列入口与项目子目录，其他比赛使用各自的子目录。

仓库：[lhhhappy/LLM_deployment_A100](https://github.com/lhhhappy/LLM_deployment_A100)。69 个分支、22 个标签已推送并逐项核验对象 ID，其中包括 18 个 `archive/wip-20260930/*` 工作区快照。归档快照保留清理前的未提交修改；原分支不合并这些修改。所有清理前 checkout 的 HEAD 都仍能由已推送的分支或标签到达。

47 个关联 worktree 中已移除 46 个，本地保留主仓库和 `build/worktrees/final-execution-0930/`。最终候选仍为 `codex/final-execution-0930` / `ca5d646c252688177480c7e67cec0a901e4b7069`，工作区干净。两份正式 CAP / TPOT 提交 ZIP 的 SHA256 保持不变。

独有的非 Git 资料移动到 `build/archives/worktree-artifacts-20260930/`，按 SHA256 合并重复内容；与主目录文件相同的资料沿用主目录中的副本。归档保留原路径、权限、哈希和符号链接信息。可再生成的引擎导出树、代码索引与 Python 缓存随旧 worktree 清理。额外生成的全量 Git bundle 已在核验提交可恢复后删除。

这批整理按原目录、剩余归档和移除 bundle 的实际磁盘占用计算，估计净释放约 **20.10 GiB**。独有归档文件内容约 315.6 MiB；归档目录的实际磁盘占用约 330.8 MiB。主目录的必要数据、缓存和原始实验记录继续保留一份，由既有 `.gitignore` 规则管理；本地归档、工具和 worktree 目录也已加入忽略规则。

11 个本地监控已使用新项目路径重新启动，日志和状态文件沿用原文件。父目录的三个兼容链接已移除。Pod 服务未做生命周期操作。

## 恢复一个旧工作区

从已有分支或工作区快照创建 worktree。例如：

```bash
git worktree add build/worktrees/restore-chainmax archive/wip-20260930/fix-chainmax-0927
python3 build/archives/worktree-artifacts-20260930/restore_artifacts.py \
  fix-chainmax-0927 build/worktrees/restore-chainmax
```

第二步按本地 `manifest.json` 恢复非 Git 资料，并校验 SHA256；已存在的目标文件会导致停止，避免覆盖。忽略管理的资料保留在本机，不包含在 Git 克隆里。

## 核验收据

- `build/scratch/worktree-consolidation/remote-verification.json`：推送分支、标签和对象 ID。
- `build/scratch/worktree-consolidation/snapshot-manifest.json`：各工作区原 HEAD 与归档快照。
- `build/archives/worktree-artifacts-20260930/manifest.json`：保留资料及每个旧工作区的清理状态。
- `build/scratch/worktree-consolidation/storage-result.json`：磁盘占用计算。
- `build/scratch/worktree-consolidation/monitor-migration.json`：本地监控路径迁移记录。
