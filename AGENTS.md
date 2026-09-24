# 仓库入口（Claude 与 Codex 共用）

先读 [README.md](README.md) 和 [notes/queue.md](notes/queue.md)。赛规只认 [task.md](llm-challenge-arena-v1/task.md)；当前成立的事实与待验证边界见 [notes/knowledge.md](notes/knowledge.md)，完整实验结果见 [notes/experiments.md](notes/experiments.md)。

## 目标与工作循环

目标是正式 `n_at_slo` 更高，其次 `tpot_mean` 更低。正式 A（attempt 45979，N14，保留 MTP）是部署基线。

每轮按同一个循环：提出一个明确问题 → 在基线上只改一处 → CPU/开发机探针筛选 → 8 卡完整开发集确认 → 用原 harness 与完整数据判分 → 看原始记录找原因 → 改进 → 记录。多变量组合只评价组合本身。开发集是链前缀抽样，只用于比较我们自己的配置；与正式的差异用校准运行（正式 A/B 原样复现）来量。

## 分工

两边都深入源码，也都改核心代码；区别在于各自回答的问题。

| | Codex：执行层 | Claude：负载与服务层，兼统筹 |
|---|---|---|
| 核心问题 | 同样的计算为什么这么贵，怎样变便宜 | 这批请求为什么过不了门，整个服务怎样更好 |
| 研究与代码 | 模型前向、MoE 大块路径、prefill CUDA graph（170）、kernel、通信、张量布局、KV/KDA 状态契约 | 请求到达与 batch 形成、准入与排序、块预算、缓存复用与淘汰、状态池（`scheduler.py`、`schedule_policy.py`、`mem_cache/`）；本地与正式的校准；实验与评分工具 |
| 交付 | 正确的补丁，外加同条件下省多少时间、多占多少显存的实测 | 调度与缓存补丁；完整对照结果、失败请求归因；局部加速能否兑现为整档收益的判断 |
| 8 卡 | 准备补丁、数值检查、针对性探针 | 统一排队、采集证据、完整验证 |

接口：Codex 提供实测单块成本曲线（块长 × 上下文 × batch），调度侧直接使用；Claude 提供真实请求的失败归因，执行侧据此判断是调度缺口还是执行粒度问题；任何新缓冲先报显存，按 KV/状态池损失算账。跨两层的改动先约定接口和负责人。每方的结论进入决策前，由对方对照原始数据复核。

## 怎么调研

0. **先读调研，再动手。** [research/README.md](research/README.md) 是入口：底包源码地图（请求入口、调度、混合缓存、模型与算子）、prefill 固定开销（R10）、MTP（R17）、缓存与显存（R18/R20）、N22 TPOT 坏例（R21）、上游候选（R9）。选方向和设计补丁前先查对应篇目；新的可复用结论写回 research/ 相应文件，被推翻的删掉。
1. **先算产能账，再选方向。** 按三个杠杆思考：活变少（缓存复用）、活变便宜（每 token 与每次前向的成本、每步 decode 产出）、活排得更好（谁等、谁被打断）。只有前两个能推高 N；调度决定到达上限前谁超时。先看正式结果里哪道门最紧，再对准它。
2. **先查事实。** 读真实源码（`build/base_exact/`）和原始数据后再设计；每个数字标明实测还是推断，不用两点外推承诺第三点。新分析工具先在真实数据上校准再用。
3. **按层级验证，8 卡只做确认。**
   - CPU：调度与缓存逻辑（真实调度器代码加假请求）、离线 LCP 账本、判分与分析脚本。
   - 开发机：按 TP8 每卡形状测单算子的数值与速度、host 开销与 CUDA graph、单块成本曲线。
   - 8 卡：TP8 切分、通信与补齐边界、真实权重输出、完整回放、混合负载时间账和最终确认。开发机结论都是临时的。
4. **分析坏例。** 先确认数据完整；同一请求跨运行比较；按 raw 行追到服务日志；逐个量化其他解释；指出日志证明不了什么；留下可复用脚本与逐请求 CSV。

## 规则

- 判分只用原 harness（`scripts/score_formal.py`，加 task.md 统计余量与 tpot_p95 门），数据须完整（每条请求恰好一次），缺数据即 INVALID。差异小于同配置重跑的噪声不算结果。
- 补丁照只读的 `build/base_exact/` 写，每个机制只保留一个版本和同名 `.md`；每次运行打印所用补丁哈希；里程碑即提交。
- `llm-challenge-arena-v1/`、`s1-dev/`、`build/base_exact/`、`refs/` 只读。只保留正确的现行文档，过时内容直接删除（git 留历史）。
- 改共享文件（`notes/knowledge.md`、`queue.md`、`experiments.md`）前先读最新内容，只提交自己的改动；各自的长篇分析放在自己的文件里。
- 8 卡服务不可停、删、释放。停单个测试 job 用 GPU 机上的 `scripts/pod/stopjob <job.sh>`。审阅只用 `scripts/pod/pread`；CPU 分析用 `scripts/pod/pexec_codex`，只写 `/tmp/ax/codex`。入队由 Claude 统一安排，正式提交时机由用户决定；官方结果用 `scripts/official_status.sh <attempt_id>` 查。
- GPU 开发机只在 `/sjtu/linhang/arena/` 下工作；不探测评测平台或其他选手。
- 不关 thinking、不压输出、不截历史、不删 tools；时间戳和 token 计数如实；`/flush_cache` 真清。对外可见的镜像、服务和启动元数据保持中性。
