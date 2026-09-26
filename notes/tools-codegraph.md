# codegraph（代码知识图谱）在本仓库的用法

安装：`npm i -g @colbymchenry/codegraph`（容器内已装 1.6.0，`/usr/local/bin/codegraph`）。沙盒里不开文件监视：每条命令前加 `CODEGRAPH_NO_DAEMON=1`。`.codegraph/` 已加入 `.gitignore`。

索引按项目根建，且默认排除任何 `build/` 路径：主签出（master）的索引里没有只在分支上的机制代码（124/125/126/128 的 `ax_deadline.py`），所以要分析 stack2 的调度代码，在 worktree 里单独建：

```sh
cd build/worktrees/129-warm-seat && CODEGRAPH_NO_DAEMON=1 codegraph init .   # 约 1 分钟；提交后 codegraph sync .
```

有用的查询（在对应根目录下执行）：
- `codegraph callers <symbol>`：谁调用它。例：`_ax_short_hit_reserve` → `_ax_demand_limits`（126）与 `_ax_pace_limits`（122），正是 Codex 发现 flush 后未重置的共享留位状态的两个使用者。
- `codegraph impact <symbol> --depth 2`：改它会波及谁。例：`tier_order` → `_ax_admission_plan`、`_get_new_batch_prefill_raw`。
- `codegraph node <symbol|file>`：源码与调用者一屏看完。
- `codegraph explore "<问题>"`：一次给出相关符号、调用路径与影响面；范围偏大，适合陌生模块入门。
- `codegraph affected <files>`：受影响的测试文件；对 `scheduler.py` 这种被全仓库引用的文件会给出几百个，不如直接跑四个调度套件（`PYTHONPATH=tests python3 -B -m unittest test_ax_deadline test_ax_admission_scheduler test_demand_cap test_sched_protect_chain`）。

用在两处：改调度代码前先看 `impact`/`callers`，确认共享状态（留位、护栏、家族缓存、停车计数）的所有读写点；审同伴提交时用 `callers` 核对新状态有没有在 flush 与重试路径上清理。
