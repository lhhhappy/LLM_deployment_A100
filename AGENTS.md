# 仓库入口

先读 [README.md](README.md)，再读 [notes/queue.md](notes/queue.md) 认领问题。赛规以 [llm-challenge-arena-v1/task.md](llm-challenge-arena-v1/task.md) 为准；实验依据见 [notes/knowledge.md](notes/knowledge.md)。

- 工作循环：提问 → 单变量实验 → 用原 harness 与完整数据判分 → 分析原始记录 → 改进 → 记入 `notes/experiments.md`。
- `llm-challenge-arena-v1/`、`s1-dev/`、`build/base_exact/`、`refs/sglang-fe236ea6c3/` 只读；补丁照 `build/base_exact/` 写。
- 8 卡服务和现有队列不能停、删、杀进程。整理与审阅只用 `scripts/pod/pread`；CPU 分析可用 `scripts/pod/pexec_codex` 且只写 `/tmp/ax/codex`。入队、打镜像、正式提交须与当前运行负责人协调。
- GPU 开发机只在 `/sjtu/linhang/arena/` 下工作；不碰其他选手或评测平台。
- 不关 thinking、不压输出、不截历史、不删 tools；时间戳和 token 计数如实；`/flush_cache` 真清。对外可见的 Trisol 镜像、服务和启动元数据保持中性。
- 代码改动同步更新受影响文档。原始证据放 `evidence/`，结论注明来源与仍待验证的边界。任务无需编号；哈希只用于核对必要的源码等价性，不设成额外的迭代门槛。
