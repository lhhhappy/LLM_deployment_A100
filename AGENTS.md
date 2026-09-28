# 仓库入口（Claude 与 Codex 共用）

先读 [README.md](README.md) 和 [notes/queue.md](notes/queue.md)。赛规只认 [task.md](llm-challenge-arena-v1/task.md)；当前成立的事实与待验证边界见 [notes/knowledge.md](notes/knowledge.md)，完整实验结果见 [notes/experiments.md](notes/experiments.md)。

文档的唯一归处、实测与推断的写法见 [notes/README.md](notes/README.md)；原始证据的状态与归档规则见 [evidence/README.md](evidence/README.md)。队列只记当前任务，闭合结果写入实验记录，避免维护两份动态状态。

## 目标与工作循环

目标以 [task.md](llm-challenge-arena-v1/task.md) 的正式排名为准：先比 `n_at_slo`（越大越好），再比 `tpot_mean`（越小越好）；TPM 只回报、不排名。最终评测回放 341 条会话链、5150 个请求，整轮约 8–10 小时。正式 A（attempt 45979，N14，保留 MTP）是部署基线。

每轮按同一个循环：理解明确的问题 → 在基线上实现最小修复并补必要观测 → 代码审查与按需验证 → 8 卡实验 → 用原 harness 与完整数据判分 → 看原始记录找原因 → 改进 → 记录。按正常工程开发选择验证方式，不强制先做 CPU/开发机探针或新增回归测试；Pod空闲且实验已授权时可直接实机验证。开发完成可请其他参与者review代码、查问题。开发探针按覆盖目标短测，确认错误留证后立即结束自己的测试，不必等70分钟；完整成绩仍要求全量闭合。多变量组合只评价组合本身。开发集是链前缀抽样，只用于比较我们自己的配置；与正式的差异用校准运行（正式 A/B 原样复现）来量。

## 判定优先级：chain 第一（用户多次强调，2026-09-26 / 09-28）

- 线上失败档只有 `chain_start` 门在挂（p95 37–46 s 对 30 s），fast / overall / turn 有 1.5–3 倍余量，tpot 均值有余量。所以任何实验、任何候选**先看 chain**：同一批请求 ID 上的 chain 超标条数，分开场与稳态、按链首类型（session_start / 切段 intra / turn_start / reset）列出修好与新坏。
- **chain 变差的改动一律否决**，不管它把 fast、overall、TPOT 均值改善了多少；fast / overall 只作为代价记录，不作为采用理由。TPOT 只盯 `tpot_p95 ≤ 0.10` 这道无余量的硬门（本地 p95 约为均值 1.75 倍，均值上限约 0.055）。
- 汇报和看板的第一句写 chain 的结果；"fast 大幅回落"这类话只能放在代价里。上传候选的取舍同样只按 chain 与硬门。

## 分工

没有领导，所有参与者平等合作；参与者不固定（Codex、Claude 或其他智能体会话都可能加入）。分工按问题领域分，不按上下级分；现在谁在做什么、怎么登记、怎么联系见 [collaboration.md](notes/collaboration.md)。

| | 执行层 | 负载与服务层 | 数据 |
|---|---|---|---|
| 核心问题 | 同样的计算为什么这么贵，怎样变便宜 | 这批请求为什么过不了门，整个服务怎样更好 | 测试负载像不像正式赛 |
| 研究与代码 | 模型前向、MoE 大块路径、prefill CUDA graph（170）、kernel、通信、张量布局、KV/KDA 状态契约 | 请求到达与 batch 形成、准入与排序、块预算、缓存复用与淘汰（含 HiCache）、状态池（`scheduler.py`、`schedule_policy.py`、`mem_cache/`）；本地与正式的校准；实验与评分工具 | 长链合成数据的生成、冻结、token/LCP 账本与偏差说明 |
| 交付 | 正确的补丁，外加同条件下省多少时间、多占多少显存的实测 | 调度与缓存补丁；完整对照结果、失败请求归因；局部加速能否兑现为整档收益的判断 | 冻结集及哈希、复现命令、已知偏差 |

所有人都可以深入任何源码；跨领域改动先约定接口和谁写。接口：执行层提供实测单块成本曲线（块长 × 上下文 × batch），调度侧直接使用；服务层提供真实请求的失败归因，执行层据此判断是调度缺口还是执行粒度问题；任何新缓冲先报显存，按 KV/状态池损失算账。任何结论进入决策前，由另一位参与者对照原始数据复核。8 卡队列共用：谁入队谁在 [queue.md](notes/queue.md) 登记并盯到结果。

## 怎么调研

0. **先读调研，再动手。** [research/README.md](research/README.md) 是入口：底包源码地图（请求入口、调度、混合缓存、模型与算子）、prefill 固定开销（R10）、MTP（R17）、缓存与显存（R18/R20）、N22 TPOT 坏例（R21）、上游候选（R9）。选方向和设计补丁前先查对应篇目；新的可复用结论写回 research/ 相应文件，被推翻的删掉。
1. **先算产能账，再选方向。** 按三个杠杆思考：活变少（缓存复用）、活变便宜（每 token 与每次前向的成本、每步 decode 产出）、活排得更好（谁等、谁被打断）。三个杠杆都会影响 N@SLO：调度即使不改变孤立算子产能，也可通过批形成、减少空转和控制尾延迟提高通过档；不能把排队直接归因调度，或把三者收益相加。先看正式结果里哪道门最紧，再对准它。
2. **先查事实。** 读真实源码（`build/base_exact/`）和原始数据后再设计；每个数字标明实测还是推断，不用两点外推承诺第三点。新分析工具先在真实数据上校准再用。
3. **按问题选择验证方式，不设固定的 CPU 前置门槛。** 下列是可用手段，不是必须依次通过的流程：
   - CPU：调度与缓存逻辑（真实调度器代码加假请求）、离线 LCP 账本、判分与分析脚本。
   - 开发机：按 TP8 每卡形状测单算子的数值与速度、host 开销与 CUDA graph、单块成本曲线。
   - 8 卡：TP8 切分、通信与补齐边界、真实权重输出、完整回放、混合负载时间账和最终确认。开发机结论都是临时的。
4. **分析坏例。** 先确认数据完整；同一请求跨运行比较；按 raw 行追到服务日志；逐个量化其他解释；指出日志证明不了什么；留下可复用脚本与逐请求 CSV。

## 规则

- 判分只用原 harness（`scripts/score_formal.py`，加 task.md 统计余量与 tpot_p95 门），数据须完整（每条请求恰好一次），缺数据即 INVALID。差异小于同配置重跑的噪声不算结果。
- SGLang在`engine/sglang/`改，底包`engine-base`、正式A `official-A-0923a`，沿用`engine NNN:`提交和`engine/docs/`说明。用户已授权Claude Code在`engine/vllm/`独立推进vLLM：先固定真实底包，使用独立tag、`engine vllm NNN:`提交和`engine/docs/vllm/`说明，不能复用SGLang导出/启动器假装已兼容。两路修正归所属机制，默认关闭等于各自固定底包，不许静默绕开；记录提交与有效机制并核对运行期望。详见[engine/README.md](engine/README.md)和[vLLM交接](notes/handoffs/vllm-claude-code.md)。
- `llm-challenge-arena-v1/`、`s1-dev/`、`build/base_exact/`、`refs/` 只读。只保留正确的现行文档，过时内容直接删除（git 留历史）。
- 改共享文件（`notes/knowledge.md`、`queue.md`、`experiments.md`）前先读最新内容，只提交自己的改动；各自的长篇分析放在自己的文件里。
- 8 卡服务不可停、删、释放。停单个测试 job 用 GPU 机上的 `scripts/pod/stopjob <job.sh>`。审阅只用 `scripts/pod/pread`；CPU 分析用 `scripts/pod/pexec_codex`，只写 `/tmp/ax/codex`。入队规则见 collaboration.md，正式提交由参与者一起判断、用户定；官方结果用 `scripts/official_status.sh <attempt_id>` 查。
- **Pod 临时存储限额仍为 20Gi（2026-09-25 现行 revision 2 已核实），070 曾因此被驱逐。** `df` 显示的底层磁盘空闲不是 Pod 额度，禁止据此安装第二套引擎或堆积数据。上传/安装/启动前核实际挂载与峰值预算；数据、源码、日志和可迁移缓存使用已核验的 RAM/专用挂载，不能只设环境变量却仍写根盘。缓存/临时目录导出为真实绝对路径（readlink -f）；已发现/tmp/ax符号链接会触发底包JIT重复编译，见[根因](notes/reports/sglang-cold-jit-0925.md)。`/dev/shm` 的 754GiB 计入 1509GiB 总内存，须给模型、host 缓存与临时缓冲留余量；安装缓存、JIT、临时文件和容器日志另计。新增诊断须有输出预算，但不截断影响评测或分析的完整记录；按用户要求，结果优先归档到 GPU 开发机 `/sjtu/linhang/arena/archives/pod-runs`，核验后清理 Pod，运行中证据不得误删。容量收据见 [pod-capacity-0925.md](notes/reports/pod-capacity-0925.md)。
- GPU 开发机只在 `/sjtu/linhang/arena/` 下工作；不探测评测平台或其他选手。
- 不关 thinking、不压输出、不截历史、不删 tools；时间戳和 token 计数如实；`/flush_cache` 真清。对外可见的镜像、服务和启动元数据保持中性。
