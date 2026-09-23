# 8 卡实验队列

本机通过 `ssh GPU` 到开发机 `/sjtu/linhang/arena/repo`，再由 `bohr trisol inference exec` 进入已有 Trisol pod。`scripts/gssh` 为短命令提供连接探测和一次执行；`scripts/gjob` 在开发机的 tmux 里运行长任务。SSH 的 `GPU` 主机、端口和密钥使用本机 `~/.ssh/config`；`scripts/ssh_config.GPU` 是可选配置示例。当前连接直接走 SSH；需要 HTTP CONNECT 代理时可配置 `scripts/ssh_httpconnect.py` 的绝对路径。不要把密钥放进仓库。

Claude 当前使用 `scripts/pod/qpush <队列名>=scripts/pod/jobs/<作业>.sh` 从本机同步补丁、验证 kit 并按给定名字入队。作业模板在 `scripts/pod/jobs/`，公共启动和源码准备逻辑在 `lib.sh`。`scripts/pod/podq` 在开发机上查看或管理 pod 队列；`pstatus` 查看服务、GPU 和队列。pod 中的运行记录在 `/tmp/ax/runs/`，待运行、运行中和已结束的作业在 `/tmp/ax/queue/`。当前服务和 worker 持续运行；提交新作业前先看现有队列，避免重名与资源冲突。

单档结论以 `verify_kit/level_verdict.py` 为准：它从本档 `summary.json` 定位测量文件，核对 dev 集每条请求恰好一次，然后调用原 harness 与 `score_formal.py` 判定 11 道门。退出码 `0` 是有效且通过，`1` 是有效但未通过，`2` 是数据无效。旧 `analyze_run.py` 的 `formal_est_all_pass` 只供诊断，不能作为档位结论。原始日志仍需记录清缓存调用是否成功；未确认时不能称该档有效。

只读审阅用 `scripts/pod/pread status|ls|tail|head|cat|grep|analyze`。允许的 CPU 分析命令用 `scripts/pod/pexec_codex`，输出只写 `/tmp/ax/codex/`。这些审阅入口不会改变队列或服务。`collect_run_logs.py` 可通过已有 SSH 连接读取一个运行目录的日志快照。
