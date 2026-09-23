# HANDOFF — 仓库整理（交给 Codex；2026-09-24，Claude 起草，依据用户当天的要求）

分工：**Codex 整理仓库**（本地仓库 + GPU 机上的镜像仓库与脚本）；**Claude 继续跑 8 卡实验、提交任务、改补丁**。

## 1. 用户要什么（原话摘录）
- "我觉得这些东西太乱了，太影响你的上下文和迭代了……做一次大的整理，把 harness-template 完全移除，文档重写。"
- "保留调研/过去的必要的 patch 以及尝试和经验。"
- "任务管理机制……层层硬编码，匹配……比对哈希，这些都和我们想要做的任务无关，我想变得干净清晰：简单的任务排队，然后分析各种脚本，保留可复用的脚本，维护整个仓库干净迭代，全部清理。"
- "以后就是：跑实验，分析问题，改进，记录，跑实验。"
- "这个 T 什么的、数字什么的，我觉得太乱了！" "我不想要层层审批。" "notes/findings.md 已经那么长了！"
- "过时的……清理干净，我很怕误导了后面的迭代和智能，能不能只维护正确的文档！" "那就直接删除吧，反正以后不再需要读到！"
- 评测只要求**准确**：按 task.md 的规则、用 harness 自己的评分器、在完整数据上算。"不希望你老是纠结我们自己给自己设置的硬门槛。"

据此：**工作区只留正确、现行、有用的东西；过时或错误的直接删（git 保留历史），不建归档目录；不要编号体系、不要审批流程、不要自设门槛。**

## 2. 红线（整理过程中任何时候都不能破）
1. **8 卡 Trisol 服务绝不能停、删、暂停**：不运行 `bohr trisol inference delete|stop`，不写 STOP 文件，不动 pod 里的进程。pod 只能用 `scripts/pod/pread` 只读。
2. **不要执行 `scripts/pod/qpush`、`podq init`、`ppush`**：它们会往 pod 推补丁和脚本，而 Claude 正在用队列（037–039 在跑/排队）。
3. **队列工具链每次提交后都必须仍然可用**。路径可以调整，但 Claude 要能照 README 继续入队、判分。涉及的文件：
   - `scripts/pod/` 下的 `common.sh lib.sh qpush podq podq_worker.sh ppush pexec pexec_codex pread pstatus stopjob stopjob_inpod.sh`；
   - `scripts/pod/jobs/dev_ladder_template.sh`；
   - `scripts/pod/verify/` 下的 `level_verdict.py make_kit.sh cap_smoke_body.sh metrics_sampler.py logstat.py coldprobe.py interference.py chunkcost.py`；
   - `scripts/score_formal.py`（会被拷进 kit）；
   - `scripts/gssh`、`scripts/gjob`、`scripts/ssh_*`。
4. **只读输入不删**：`llm-challenge-arena-v1/`、`s1-dev/`、`build/base_exact/`、`refs/sglang-fe236ea6c3/` 及仍被引用的 `refs/*.diff`。`src/sglang/`（v0.5.20，不是底包）先从文档里去掉引用；目录本身删不删问用户（它可能不在 git 里，删了不可恢复）。
5. **`patches/` 不改补丁文件**：今天已整理成每个机制一个版本，并证明源码树逐字节不变，见 `patches/README.md` 和 `evidence/T57/`。Claude 可能继续新增候选补丁。
6. GPU 机只在 `/sjtu/linhang/arena/` 下操作；`/sjtu/linhang` 下其他目录是用户别的项目，不要碰。
7. 每一步都是一个 git 提交，提交信息写清删了什么、为什么删。

## 3. 目标结构（建议；以"干净、能直接用"为准，可调整）
```
README.md        唯一入口（内容见 §4）
AGENTS.md        ≤20 行：先读 README；红线
CLAUDE.md        ≤20 行：先读 README；红线
llm-challenge-arena-v1/ s1-dev/ build/base_exact/ refs/   只读输入
patches/         引擎补丁 + README（栈、开关、耦合）——保持现状
scripts/
  pod/           8 卡队列与访问（保持可用）
  analysis/      可复用离线分析：level_verdict、score_formal、patch_stack、raw/日志分析
  submit/        build_image.sh、submit_official.sh、check_submission.py（按 B4 修好，它现在会误报）
  gpu/           gssh、gjob、ssh 配置
  devbox/        仍有用的开发机算子/数值测试
tests/           只留测试现行补丁/工具的 CPU 测试（如 test_sched_protect_chain.py，27/27 通过）
notes/
  experiments.md 实验记录：每次 8 卡运行一条（相对基线改了什么、正确判分的结果、结论）；
                 开头一段"09-24 之前的历史"，把过去的尝试和教训压缩成要点
  knowledge.md   当前已核实的事实与经验教训（把 findings/decisions/research 里仍正确的部分压缩，不带编号）
  submissions.md 正式提交与成绩（按 B1 更正）
research/        只留仍正确、仍有用的调研（按 B2 判定），其余删除
evidence/        只留仍被保留文档引用的原始证据，例如 L035、T53 的 026 raw、T56、T57；其余删除
data/            只留最新的榜单快照
```

建议整个删除的（具体逐项以四份审查为准）：
- 根目录：旧的 `HANDOFF.md`（本文件用完后也删）、`rule.md`、`board.md`；
- 整个目录：`plans/`、`docs/`、`cases/`、`logs/`；
- `notes/` 里除上面三份之外的全部；
- `tests/` 里的 `TEST_PLAN.md`、`TIERS.md`、`L2.md`、`queue*`；
- `scripts/` 里的 `archive/`、`session_a/`、`l2.py`、`test_status.py`、`check_records.py`、`next_id.py`、`index_notes.py`，以及 L1/v0.5.20 时期和一次性任务的脚本；
- `research/` 里的 `archive/`、`codex/archive/`，以及过时的报告；
- `refs/harness-template-cn/`；
- `build/` 下可再生成的目录。

## 4. README 必须写清楚的内容（新会话只读它就能干活）
1. **赛题要什么**（以 task.md 为准）：
   - 排名：`n_at_slo` → `tpot_mean`；TPM 只回报不排名。
   - 每档 11 道硬门：TTFT 四门带统计余量；tpot_p95 ≤0.10，无余量。
   - 爬坡：从 N10 起 +4/−4，单档约 4 小时。
   - 能力门：两科 >90。本地用 12 题冒烟即可，这是用户的决定。
   - 开发集只能做 A/B 相对比较（task.md:354）。
   - 诚实与 flush 要求。
2. **目标**：榜首 CalvinCao，N26、tpot_mean 0.0551。
3. **现状**：
   - S0 基线的定义（见 `patches/README.md`）；
   - dev N22 结果：11 门里过 10 门，只挂 tpot_p95 0.296；
   - 正在跑的实验；
   - 当前主要假设：重 prefill 时 decode 被饿住；缓存可修上限 8%。
4. **循环与操作**：
   - 复制一个 job 文件，只改一处；
   - `scripts/pod/qpush NNN-名字.sh=路径` 入队；
   - job 日志的 `LEVEL` 行就是判定（`level_verdict.py`：完整性 + harness 评分器 + task.md 规则）；
   - 用 `scripts/analysis/*` 分析，结果记进 `notes/experiments.md`。
5. 仓库地图（短）；GPU 机与 pod 的访问方式和安全规则。

## 5. GPU 机（`/sjtu/linhang/arena/`）
- `repo/` 是本地仓库的镜像，由 qpush 用 tar 同步。本地整理完之后，把镜像同步成同样的内容（多余文件也删掉），不要碰 `runs/`、`models/`、`env/`、`cache/`。
- tmux 会话 `arena`、`arena-daemons`（窗口 `submit`、`boot`）：先列出里面在跑什么，特别是有没有任何能删、停服务或自动提交的代码路径。列出后交给用户或 Claude 决定，不要自己杀进程。
- 其它目录：列清单、提建议，由用户决定。

## 6. 输入材料
- 逐文件的判定清单：
  - `evidence/T55/B1-notes.md`：notes；
  - `B2-docs-research.md`：入口文档与调研；
  - `B3-patches-tests-scripts.md`：补丁、测试、脚本、build、data、evidence、logs；
  - `B4-rules.md`：与 task.md 不一致的 38 处，含工具实现问题。
- 已核实事实：`plans/prompts/_context-0924.md`、`research/codex/R19_*`、`R20_*`、`evidence/L035/`、`patches/README.md`。

## 7. 与 Claude 并行时的约定
- Claude 在整理期间把新实验结果**追加**到 `notes/runs-0924.md`，新 job 文件放在 `scripts/pod/jobs/`（`s0_*.sh`）。这些文件保留；最后把 `runs-0924.md` 并进 `notes/experiments.md`。
- Claude 可能新增 `patches/12x-*.patch` 候选；不要动 `patches/`。
- 整理完成后，在 README 顶部写一行"整理完成（日期、提交号）"，并通知用户。
