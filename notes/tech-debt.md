# tech-debt.md — 技术债 / 未决项（暂不阻塞，但要跟踪）

2026-09-22 重新定基线：v0.5.20 线（001/002/004）的条目已随决策 29 关闭，历史见 git/归档。

| 日期 | 区域 | 描述 | 为什么存在 | 后续动作 | 状态 |
|---|---|---|---|---|---|
| 09-22 | 000 | 000 照 v0.5.20 写，只证明"打得上"；审计发现底包 `/flush_cache` 原样返回纯文本、DP 扇出只取 `[0]`（http_server.py:980–994、tokenizer_control_mixin.py:304–310），000 是否在底包上真正修好未实测 | 当时没有底包源码 | 对照 base_exact 复核 000；L2 01 项 IF 检查 | open |
| 09-22 | 101 | 尾段切分只对齐 checkpoint grid，可能带着 004 的对齐缺陷（未确认） | 由 v0.5.20 设计移植 | 对照 base_exact 复核 add_chunked_req 路径 | open |
| 09-22 | D1/102 | 中途状态取自 bf16 的 h；数值门 D1-04 未过（F45、F53） | 底包 kernel 设计 | 决定 fp32 快照或保留拆分；建数值验收 | open |
| 09-22 | D1 | `ROLE_BOUNDARY_STATS` 只在进程内累加，没有接到日志或指标 | 先求正确性 | 加定期日志或 Prometheus 计数 | open |
| 09-22 | L1 | L1 替身跑的是 v0.5.20，不是底包代码 | 历史原因 | 在 GPU 机上以 base_exact 装一套 L1 环境 | open |
| 09-22 | 工具 | `check_submission.py` 按 v0.5.20 源码校验启动参数，应改按底包 `server_args.py` | 历史原因 | 改指向 base_exact | open |
| 09-22 | 提交 | `/mnt/models` 挂载、`glm45` 答案是否落在 content，还没在底包上实测 | 没有 8 卡 | 45734/45735 成绩 + L2 | open |
| 09-22 | 评分 | 正式统计余量的估计方法未经主办方确认 | 题面没给估计器 | score_formal 标注"估计" | open |
| 09-22 | 协作 | 多个 agent 同时编辑 notes/board 会互相覆盖（发生过） | 共享 markdown | 实例表只由 Claude 维护；`check_records.py` 检查 | mitigated |
| 09-22 | 环境 | GPU 机 SGLang 0.5.20 需要 CUDA 13，靠向前兼容包（驱动 535） | 驱动不能改 | 只影响本地 L1 | accepted |
