# Agentic Science Challenge：GLM-5.3-Flash 推理服务

本地脚本已完成第二轮清理（2026-09-24）；GPU 机仓库镜像待现有队列结束且 SSH 恢复后同步。

这是 8×A100-80GB 上的推理服务部署赛。唯一赛规是 [task.md](llm-challenge-arena-v1/task.md)：能力评测 AIME26 与 GPQA Diamond 都须严格高于 90 分；压测先比通过全部硬门的最大并发档 `n_at_slo`，同档再比越小越好的 `tpot_mean`。TPM 只作诊断。平台从 N=10 开始，成功加 4、失败减 4；单档约 4 小时。每档有完整性、错误率、四道 TTFT 和 `tpot_p95 ≤ 0.10 s/token` 等 11 道硬门；TTFT 按题面规定的统计余量判定，TPOT p95 没有余量。开发集只能比较我们自己的 A/B 和回归，不能预测正式 N@SLO（task.md「开发集」与「压测」节）。

## 现在做什么

截至 2026-09-24 已复核的 8 卡完整开发集 N22 结果：S0（035）只挂解码门，`tpot_p95=0.296`；S0+114 的 S1（036）改善 TTFT 与均值 TPOT，`tpot_p95=0.253` 仍失败；S0+固定 16 轮 decode（037）把 `tpot_p95` 降到 0.0775，却使 overall、turn、chain 三道 TTFT 门失败。当前要找同时保住 prefill 与 decode 的调度点。后续作业状态看 [任务队列](notes/queue.md)和 pod 实时状态，完整结果见 [实验记录](notes/experiments.md)。补丁与 S0/S1 的精确定义见 [patches/README.md](patches/README.md)；当前 7 个 S0 补丁已按 [等价性记录](evidence/T57/equivalence.log)验证与 026/035 源码树一致。

每轮按这个循环：从 [任务队列](notes/queue.md) 取一个明确的问题 → 复制现行 job，优先只改一个变量 → 入队跑完整开发集 → 用原 harness 评分器和题面规则核对完整性及 11 门 → 看原始记录解释瓶颈 → 更新补丁和 [实验记录](notes/experiments.md) → 再跑。多变量组合只评价组合，后续再拆分归因。失败或数据不完整就记失败或 INVALID。12 题能力测试只用于冒烟，不能证明能力门通过。

## 操作入口

日常直接使用的入口约十个，集中在下表；`scripts/` 其余主要是当前队列的 job、pod 运行与验证逻辑，不需要逐个作为任务入口阅读。

| 要做的事 | 入口 |
|---|---|
| 查看 8 卡队列和日志 | `scripts/pod/pread status`；队列说明见 [scripts/pod/README.md](scripts/pod/README.md) |
| 准备下一项 8 卡实验 | 先看 [任务队列](notes/queue.md) 和已入队[原始 job 快照](evidence/jobs-0924/)。当前旧名队列（037b–042）冻结，**不要用本地 job 重新 `qpush`**；这批任务全部开跑、GPU 镜像同步后，再按 [pod 工具说明](scripts/pod/README.md)入队新实验 |
| 判定单档 | job 的 `LEVEL` 行由 `scripts/pod/verify/level_verdict.py` 生成：先查 cohort、runner 与原始记录，再调用 `scripts/score_formal.py`（harness 评分器）并补上题面 TPOT 门 |
| 分析原因 | 保留 `raw_*.jsonl`、run/report、服务日志；`python3 -B scripts/analysis/review_raw.py <raw.jsonl>` 审计 cohort 与缓存账本，其余可复用分析见 `scripts/analysis/` 和 [research/README.md](research/README.md) |
| 修改引擎 | 补丁照只读的 `build/base_exact/` 写；一个机制保留一个可用版本，说明写在同名 `.md`；不要照旧的 v0.5.20 源码写 |
| 构建与正式提交 | `scripts/build_image.sh`、`scripts/submit_official.sh`，提交事实记在 [notes/submissions.md](notes/submissions.md)；只在明确安排正式提交时使用 |

GPU 开发机通过 `scripts/gssh` / `scripts/gjob` 连接，**只在 `/sjtu/linhang/arena/` 下工作**；仓库镜像位于 `/sjtu/linhang/arena/repo`。8 卡 Trisol 服务与正在运行的队列任务不能停、删或杀进程。整理者对 pod 只使用 `scripts/pod/pread` 只读查看，或 `scripts/pod/pexec_codex` 在 `/tmp/ax/codex` 做 CPU 分析；不要改队列、运行目录或向引擎发请求。实验入队和提交由当前负责运行的协作者协调，避免碰撞。

## 仓库地图

| 路径 | 内容 |
|---|---|
| `llm-challenge-arena-v1/`、`s1-dev/` | 赛题原文、公开开发集与 harness；只读 |
| `build/base_exact/`、`refs/sglang-fe236ea6c3/` | 底包副本与上游参考；只读 |
| `patches/` | 引擎补丁、精确基线、机制开关 |
| `scripts/pod/` | 8 卡队列、job 模板、判定与只读访问 |
| `research/` | 源码地图、仍有效的分析；入口见 `research/README.md` |
| `notes/queue.md` | 下一步问题与实验顺序；一条任务只写问题、判据、状态 |
| `notes/knowledge.md` | 当前成立的事实、已纠正的误区与未决问题 |
| `notes/experiments.md` | 完整实验的配置差异、结果、结论、证据 |
| `notes/submissions.md` | 正式提交及官方结果 |
| `evidence/` | 原始日志、JSON、复算脚本；文档只链接需要的证据 |

赛题合规底线：不关 thinking、不压输出、不截历史、不删 tools；`meta_info` 时间戳与 token 计数如实；`/flush_cache` 必须真清；不探测评测平台或其他选手。Trisol 镜像和服务的可见名称、标签、描述、command、env 保持中性，技术路线放镜像内部。服务 `command` 是 argv，不是 shell；A100 SGLang 使用 `SGLANG_OPT_USE_TOPK_V2=0`。具体依据见 task.md 与 [notes/knowledge.md](notes/knowledge.md)。
