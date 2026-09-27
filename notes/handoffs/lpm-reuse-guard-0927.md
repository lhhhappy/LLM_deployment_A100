# 128g 单变量探针交接

2026-09-27，Codex → Fable。实现提交 `873031fd`；分支 `codex/lpm-reuse-guard-0927`。代码与归因的唯一说明见[报告](../reports/lpm-reuse-guard-0927.md)和[机制文档](../../engine/docs/128-lpm-reuse-guard.md)。原归因已复核更正；代码独立审查和 TP8 待完成。运行状态由入队者在共享 `notes/queue.md` 登记。

已备好同引擎、同 S6、v5g-tail / N26 / running32 / 无 MTP / DCP1 / 40 分钟派发并排空的两个入口：

- [OFF 控制](../../scripts/pod/jobs/tail_n26_S6_lpm_guard_off_40m.sh)
- [ON 处理](../../scripts/pod/jobs/tail_n26_S6_lpm_guard_on_40m.sh)

文件唯一差异是 `SGLANG_AX_LPM_REUSE_GUARD=0/1` 与 `G_EXPECT` 的 `128g=off/on`。两臂均显式关闭 118 prefill、132 chain-first 与 DCP local extend，其他 S6 参数沿用现有 `tail_n26_S6_40m.sh`；输入/cohort 校验、冒烟和原 harness 都保留。数据已发布在原 RAM 路径，不复制正文。调度 trace 的预算保持原 120 秒／2048 轮。若要观察稳态 held，需在两臂同时固定更长 trace 预算，并先估算日志量。

注意非运行期的配置修正：DCP local 关闭时此引擎的有效 token 是 `dcp_local_max=0`，两个入口已经使用 0；旧 S6 文件中的期望 512 与初始化对象不符。该问题已单独通知 Fable，由他核对自己的发布任务。

引擎导出在本 worktree `build/engine/873031fdb4c975854eff63ca8b0d0a1fd2e9e762.diff`（约 444 KiB）。只生成本地导出和任务，尚未上传安装、入队或改动运行中的 130ezmb。请先独立 review，再按已约定顺序安排；若复用 ezn2 的旧引擎对照，应标明不同引擎，严格单变量比较优先用这里的同提交 OFF。

应收集：实际机制 token 与 grid、`lpm_hold.release_zero_gain` 的候选/轮次、深前缀 READY/复用是否保留、同请求修好/新坏集合、chain/turn/fast/overall/TPOT、KV/KDA 峰值、CPU 调度时间。短测只报 DRAINED/DIAGNOSTIC；TPOT/fast 的探索性超限不改正式判分。不能由放行某个冷头推断全部 chain 已修复。
