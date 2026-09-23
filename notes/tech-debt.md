# tech-debt.md — 技术债 / 未决项（暂不阻塞，但要跟踪）

2026-09-22 重新定基线：v0.5.20 线（001/002/004）的条目已随决策 29 关闭，历史见 git/归档。

| 日期 | 区域 | 描述 | 为什么存在 | 后续动作 | 状态 |
|---|---|---|---|---|---|
| 09-22 | 000 | ~~000 照 v0.5.20 写，只证明"打得上"；审计发现底包 `/flush_cache` 原样返回纯文本、DP 扇出只取 `[0]`（http_server.py:980–994、tokenizer_control_mixin.py:304–310），000 是否在底包上真正修好未实测 | 当时没有底包源码 | 对照 base_exact 复核 000；L2 01 项 IF 检查 | closed 09-23：8 卡 flush 返回 JSON、时间戳经 harness 服务端口径 100% 覆盖（F59/F76）~~ |
| 09-22 | 101 | ~~尾段切分只对齐 checkpoint grid，可能带着 004 的对齐缺陷（未确认） | 由 v0.5.20 设计移植 | 对照 base_exact 复核 add_chunked_req 路径 | closed 09-23：真实缺陷是双 partial 崩溃，105 修复（F62）；由 140 取代方向~~ |
| 09-22 | D1/102 | 中途状态取自 bf16 的 h；数值门 D1-04 未过（F45、F53） | 底包 kernel 设计 | 决定 fp32 快照或保留拆分；建数值验收 | open |
| 09-22 | D1 | `ROLE_BOUNDARY_STATS` 只在进程内累加，没有接到日志或指标 | 先求正确性 | 加定期日志或 Prometheus 计数 | open |
| 09-22 | L1 | L1 替身跑的是 v0.5.20，不是底包代码 | 历史原因 | 在 GPU 机上以 base_exact 装一套 L1 环境 | open |
| 09-22 | 工具 | `check_submission.py` 按 v0.5.20 源码校验启动参数，应改按底包 `server_args.py` | 历史原因 | 改指向 base_exact | open |
| 09-22 | 提交 | ~~`/mnt/models` 挂载、`glm45` 答案是否落在 content，还没在底包上实测 | 没有 8 卡 | 45734/45735 成绩 + L2 | closed 09-23：pod 内 /mnt/models 正常，能力冒烟答案在 content（12/12）~~ |
| 09-22 | 评分 | 正式统计余量的估计方法未经主办方确认 | 题面没给估计器 | score_formal 标注"估计" | open |
| 09-22 | 协作 | 多个 agent 同时编辑 notes/board 会互相覆盖（发生过） | 共享 markdown | 实例表只由 Claude 维护；`check_records.py` 检查 | mitigated |
| 09-22 | 环境 | GPU 机 SGLang 0.5.20 需要 CUDA 13，靠向前兼容包（驱动 535） | 驱动不能改 | 只影响本地 L1 | accepted |
| 09-23 | 评估 | analyze_run 的"正式规则估算"（95% 单侧下界）是自实现，可能与主办方实现有出入 | 题面只给规则描述 | 以 harness 与正式成绩对照校准 | open |
| 09-23 | 容量 | `--max-mamba-cache-size 200` 减少 KDA 槽，可能增加状态淘汰、抵消 KV 增益 | 用状态槽换 KV | 梯子对比 DCP 路线；读缓存效率 | open |
| 09-23 | 114 | indexer 行切分只做了预填充 kpool plan 路径；decode 仍 8 卡冗余 | 先做收益大的路径 | 按 batch 行切 decode（需 all-gather） | open |
| 09-23 | 150/160 | 启动预热、MTP 尚未在 8 卡上验证 | 优先级在容量/调度之后 | 梯子稳定后安排探针 | open |
| 09-23 | 通信 | allreduce 占预填充 11%（64MB 消息超过 custom allreduce 8MB 上限） | NCCL 默认 ring LL | NCCL_PROTO 探针（021 已跑，待分析）；分块或 reduce-scatter 布局 | open |
| 09-23 | 权限 | pexec_codex 的拒绝规则是正则黑名单，非沙箱 | 需要给审阅者灵活分析能力 | 规则 + 审计日志；不给 bohr 凭据路径 | accepted |
| 09-23 | 协作 | 旧 Codex 主会话 01a0c731 队列里残留一条未处理的 T49 消息（改由 W23 完成） | 主会话未运行 | 下次恢复主会话时忽略该消息 | open |
