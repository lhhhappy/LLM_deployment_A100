# 仓库入口

先读 [README.md](README.md) 与 [notes/queue.md](notes/queue.md)。正式赛规只认 [task.md](llm-challenge-arena-v1/task.md)；当前事实和待验证边界见 [notes/knowledge.md](notes/knowledge.md)。

每次实验记录配置相对基线的唯一变化、完整数据的判定和原始证据。下一次实验从失败原因出发。补丁照 `build/base_exact/` 写；`s1-dev/`、赛题、底包和上游参考只读。

8 卡服务不可停删；整理和审阅只读访问 pod，实验入队与正式提交由负责运行者统一协调。GPU 开发机只在 `/sjtu/linhang/arena/` 下操作。赛题的输出质量、诚实计时/计数、真清缓存与保密要求不可变。
