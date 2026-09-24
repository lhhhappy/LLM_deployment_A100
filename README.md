# Agentic Science Challenge：GLM-5.3-Flash 推理服务

2026-09-24：当前两个任务与Pod运行库已统一到按源码提交号部署的流程。

这是 8×A100-80GB 上的推理服务部署赛。唯一赛规是 [task.md](llm-challenge-arena-v1/task.md)：能力评测 AIME26 与 GPQA Diamond 都须严格高于 90 分；压测排名（主办方确认）逐级比较：通过全部硬门的最大并发档 `n_at_slo` 越大越好 → `tpot_mean` 越小越好 → TPM 越大越好 → 先提交者靠前。平台从 N=10 开始，成功加 4、失败减 4；单档约 4 小时。每档有完整性、错误率、四道 TTFT 和 `tpot_p95 ≤ 0.10 s/token` 等 11 道硬门；TTFT 按题面规定的统计余量判定，TPOT p95 没有余量。开发集只能比较我们自己的 A/B 和回归，不能预测正式 N@SLO（task.md「开发集」与「压测」节）。

## 现在做什么

方向见 [notes/roadmap.md](notes/roadmap.md)，实时实验安排见 [notes/queue.md](notes/queue.md)。今天按用户决定只比较两项：**A+122 与 A+122+新版180**，同一个引擎源码提交、全量长链集311链/5601请求、N30；每项70分钟停止新准入后排空。061r已启动，063r排队。预热后真实清KV；不要求重复生成相同。

引擎已迁移到 [engine/sglang](engine/README.md)，按提交号部署和构建。源码含正式A全部13项改动；任务通过 `G_EXPECT` 核对调度与HiCache确实生效。旧28份任务入口已清理并留档，实验原始记录保留。

70分钟测量可能只覆盖全量集的一部分，发送与完成台账须闭合；结果用于候选间比较，不冒充完整集VALID或预测正式N。已有058/059与dev校准的完整记录见 [experiments.md](notes/experiments.md)。

## 操作入口

日常直接使用的入口约十个，集中在下表；`scripts/` 其余主要是当前队列的 job、pod 运行与验证逻辑，不需要逐个作为任务入口阅读。

| 要做的事 | 入口 |
|---|---|
| 查看 8 卡队列和日志 | `scripts/pod/pread status`；队列说明见 [scripts/pod/README.md](scripts/pod/README.md) |
| 准备下一项 8 卡实验 | 先看 [任务队列](notes/queue.md) 与 pod 实时状态，由当前负责人按 [pod 工具说明](scripts/pod/README.md)安排。037b–042 的[原始 job 快照](evidence/jobs-0924/)仅供追溯，不能当新实验重复推送；本地工具同步需协调 |
| 判定单档 | `level_verdict.py` 核对 N、完整 cohort、runner 与本次清缓存证据，再调用原 harness 和题面补充门；CP 为估计口径，其他区间仅诊断。修复及验证见[审计](notes/fable-审计-2026-09-24.md)，本地改动待负责人同步 pod |
| 取回并复核单档 | `scripts/analysis/fetch_level.sh <完整run目录名> <N>`；输出在 `evidence/L<完整run目录名>/N<N>/`，按 summary 选文件，返回 0=有效通过、1=有效失败、2=无效/工具失败 |
| 分析原因 | 保留 `raw_*.jsonl`、run/report、服务日志；`python3 -B scripts/analysis/review_raw.py <raw.jsonl>` 审计 cohort 与缓存账本，其余可复用分析见 `scripts/analysis/` 和 [research/README.md](research/README.md) |
| 修改引擎 | 在 `engine/sglang/` 里改，按机制提交（`engine NNN:`）；见 [engine/README.md](engine/README.md) |
| 构建与正式提交 | `scripts/build_image.sh`、`scripts/submit_official.sh`，提交事实记在 [notes/submissions.md](notes/submissions.md)，官方结果用 `scripts/official_status.sh <attempt_id>` 查；只在明确安排正式提交时使用 |

GPU 开发机通过 `scripts/gssh` / `scripts/gjob` 连接，**只在 `/sjtu/linhang/arena/` 下工作**；仓库镜像位于 `/sjtu/linhang/arena/repo`。8 卡 Trisol 服务与正在运行的队列任务不能停、删或杀进程。整理者对 pod 只使用 `scripts/pod/pread` 只读查看，或 `scripts/pod/pexec_codex` 在 `/tmp/ax/codex` 做 CPU 分析；不要改队列、运行目录或向引擎发请求。实验入队和提交由当前负责运行的协作者协调，避免碰撞。

## 仓库地图

| 路径 | 内容 |
|---|---|
| `llm-challenge-arena-v1/`、`s1-dev/` | 赛题原文、公开开发集与 harness；只读 |
| `build/base_exact/`、`refs/sglang-fe236ea6c3/` | 底包副本与上游参考；只读 |
| `engine/` | 引擎源码（git 管理）、机制说明与开关；旧补丁文件已删除，需要时从 git 历史取 |
| `scripts/pod/` | 8 卡队列、job 模板、判定与只读访问 |
| `research/` | 源码地图、仍有效的分析；入口见 `research/README.md` |
| `notes/queue.md` | 下一步问题与实验顺序；一条任务只写问题、判据、状态 |
| `notes/knowledge.md` | 当前成立的事实、已纠正的误区与未决问题 |
| `notes/experiments.md` | 完整实验的配置差异、结果、结论、证据 |
| `notes/submissions.md` | 正式提交及官方结果 |
| `evidence/` | 原始日志、JSON、复算脚本；文档只链接需要的证据 |

赛题合规底线：不关 thinking、不压输出、不截历史、不删 tools；`meta_info` 时间戳与 token 计数如实；`/flush_cache` 必须真清；不探测评测平台或其他选手。Trisol 镜像和服务的可见名称、标签、描述、command、env 保持中性，技术路线放镜像内部。服务 `command` 是 argv，不是 shell；A100 SGLang 使用 `SGLANG_OPT_USE_TOPK_V2=0`。具体依据见 task.md 与 [notes/knowledge.md](notes/knowledge.md)。
