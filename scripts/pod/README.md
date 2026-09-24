# 8 卡实验队列

本机通过 `ssh GPU` 到开发机 `/sjtu/linhang/arena/repo`，再由 `bohr trisol inference exec` 进入已有 Trisol pod。`scripts/gssh` 为短命令提供连接探测和一次执行；`scripts/gjob` 在开发机的 tmux 里运行长任务。SSH 的 `GPU` 主机、端口和密钥使用本机 `~/.ssh/config`；`scripts/ssh_config.GPU` 是可选配置示例。当前连接直接走 SSH；需要 HTTP CONNECT 代理时可配置 `scripts/ssh_httpconnect.py` 的绝对路径。不要把密钥放进仓库。

`scripts/pod/qpush <队列名>=scripts/pod/jobs/<作业>.sh ...` 从任务的 `G_COMMIT` 导出底包到目标提交的差分，同时部署 Pod 实际使用的 `lib.sh`、任务模板和验证/定时工具，再发布任务；不再同步补丁列表。`G_EXPECT` 与启动日志核对机制开关。

更新运行库前先等待当前任务结束（或按用户要求用 `stopjob` 停指定任务），通过GPU机的 `scripts/pod/podq pause` 暂停取任务。`qpush` 只向空闲且暂停的队列部署，返回远程 `gjob` 名称；用 `scripts/gjob tail <名称>` 确认 `RUNTIME_DEPLOYED` 和 `DONE rc=0` 后，再通过GPU机的 `podq resume` 恢复。SSH断连先查状态，不重复提交。传输只含指定源码差分，不含源码缓存树。禁止覆盖已有结果目录。

当前067跑完整N30，不设70分钟截止；短预热与真实flush保留。首次15分钟、随后每30分钟作诊断，完整结束后判11门。见[任务入口](jobs/README.md)与[评估协议](../../notes/evaluation.md)。旧timed工具只供历史取证，不是当前任务入口。

单档使用 `verify_kit/run_dev_checked.py --runner $S1/run_dev.py -- ...` 执行原主办方 runner，仅替换 flush hook：清缓存失败在测量前停止，成功信息写入 `flush_evidence.json`。`level_verdict.py` 按 summary 定位测量 raw/run，核对 N、完整 cohort、runner 状态和本次清缓存日志，再调用原 harness 与 `score_formal.py` 判分。退出码 0=有效且估计通过，1=有效但估计失败，2=无效。主统计口径保持 CP；Wilson/Wald 仅作边界诊断，不能拿来挑通过算法。

只读取证用 `scripts/analysis/fetch_level.sh <完整run目录名> <N> [参考raw]`，结果在 `evidence/L<完整run目录名>/N<N>/`，与旧 `evidence/L035/` 等目录并存，不自动迁移旧证据。它取回 summary 指定的测量文件和 run_dev/server/job 日志，以及存在的 metrics、GPU、receipt 和退出码；不使用 `raw_*` 的第一个文件。有效失败照常分析并返回 1，无效停止分析并返回 2。`--archive FILE.tgz` 可离线复核已经导出的归档。

旧运行没有 receipt 时，需要补齐其真实 run_dev.log 和测量时间窗内的服务端清缓存日志；缺证据标 INVALID，不表示已证明旧引擎没清缓存。复用引擎时 `lib.sh` 关联持续更新的实际日志，`engine_current.log` 仅是启动快照。新逻辑本地已测试，pod 须由运行负责人协调同步；旧引擎缺 `engine_log_path` 时会明确报错，须核对来源后接入，不自动猜测或重启。详见 [审计修复](../../notes/fable-审计-2026-09-24.md)。

CPU 回归：`python3 -B -m unittest discover -s tests -p test_eval_tools.py`。

独立合成长链集由[通用生成器](../longchain/longchain.py)构建并冻结。新job可显式设置 `G_DATA_ROOT`、`G_DATA_SET`、`G_COHORT`，模板将同一root用于回放和判定；自定义root必须三项齐全。`LADDER_UP="22"` 或 `"26"` 可分别安排固定N研究，现有爬坡仍在首次FAIL后停。取证时传 `fetch_level.sh <run> <N> --data-root <本地对应数据目录>`，可另传 `--harness-dir`；不能用dev的root或LCP账本分析新集合。此项是本地脚本接入，不表示新数据或工具已同步pod，也没有新增入队。

只读审阅用 `scripts/pod/pread status|ls|tail|head|cat|grep|analyze`。`analyze` 只显示已有的单档判定 JSON 和服务日志末尾，不重跑评分。允许的 CPU 分析命令用 `scripts/pod/pexec_codex`，输出只写 `/tmp/ax/codex/`。这些审阅入口不会改变队列或服务。

分析已有profiler trace用`python3 -B scripts/pod/verify/prof_ledger.py TRACE.json.gz --json OUT.json`。一份trace只含一个GPU；工具按External id合并CUDA graph多个stream对同一次forward的标记，跨forward重叠无法按时间归因时返回错误。kernel分类靠名称，未命名算子需调用栈核对；no-kernel gap包含尚未归因的copy/依赖/host等待，不能直接认定CPU饥饿。组件duration份额不是关键路径占比，profile数据不用于宣称SLO成绩。CPU回归：`python3 -B -m unittest discover -s tests -p test_prof_ledger.py`。本地改动须经运行负责人同步kit后才作用于pod。
