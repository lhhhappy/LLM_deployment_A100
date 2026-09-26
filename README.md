# GLM-5.3-Flash 推理服务赛：仓库入口

本仓库在 8×A100 上开发推理服务。**赛规以 [task.md](llm-challenge-arena-v1/task.md) 为准**；本地开发集只用于比较我们自己的配置，不能直接换算正式 N@SLO。协作与代码边界见 [AGENTS.md](AGENTS.md)。

## 从哪里看

| 需要知道什么 | 唯一现行入口 |
| --- | --- |
| Pod 正在跑什么、接下来排什么 | [实验队列](notes/queue.md)；实时状态用 `scripts/pod/pread status` |
| 当前阶段为什么这样设计 | [v3/N30 阶段计划](notes/program-n30-v3.md) |
| 已核实的事实与撤回的解释 | [知识库](notes/knowledge.md) |
| 本地运行怎样判有效、怎样比较 | [评估合同](notes/evaluation.md) |
| 历次结果和原始证据 | [实验记录](notes/experiments.md) |
| 正式提交和平台回报 | [提交记录](notes/submissions.md) |
| 源码地图、性能与缓存研究 | [研究索引](research/README.md) |
| 引擎分支、提交与机制 | [引擎说明](engine/README.md) |
| Pod 队列、取证与部署命令 | [Pod 操作说明](scripts/pod/README.md) |

写作与迭代记录统一按 [Notes 规范](notes/README.md)；原始材料的保留、状态和归档按 [证据约定](evidence/README.md)。这里仅维护导航，不复制运行中的分数或第二份计划。历史结论留在 Git 和原始证据中，旧交接页不是现行状态。

## 证据放在哪里

一次运行使用固定编号：任务入口在 `scripts/pod/jobs/`，本地证据在 `evidence/L<编号>-<名称>/`，状态写在[队列](notes/queue.md)，结果和解释写在[实验记录](notes/experiments.md)。短时派发并排空的对照看该目录的 `opening/analysis.json`、`opening/paired.csv`、`opening/brief.txt` 和 `window/raw.jsonl`；完整档看原 harness 报告、判分收据与 raw。每次记录引擎提交、启动开关、数据哈希、预热与 flush、错误和服务日志。窗口未闭合时只报诊断。

原始 raw、服务日志和收据不能为了减少目录数而删除。Pod 上已结束的运行先归档到 GPU 开发机 `/sjtu/linhang/arena/archives/pod-runs` 并校验哈希，再按 [Pod 归档流程](scripts/pod/README.md)清理副本。`build/scratch/` 是临时工作区，不是最终证据。

## 操作边界

`llm-challenge-arena-v1/`、`s1-dev/`、`build/base_exact/` 和 `refs/` 只读。SGLang 修改在 `engine/sglang/`，vLLM 在 `engine/vllm/`；固定提交、有效机制和差异由任务脚本与日志核对。8 卡服务保持运行；停单个测试任务仅用 `scripts/pod/stopjob`。准备新实验先看队列，判断结果先看原 harness 与完整性收据。
