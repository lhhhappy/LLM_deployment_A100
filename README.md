# GLM-5.3-Flash on 8×A100：正式 N=42 与探索记录

本仓库记录固定模型、固定 8×A100 条件下的推理服务优化：接口适配、缓存与调度、A100 算子实现，以及有效和失败的实验。排名先比较 N@SLO，再比较 TPOT 均值；规则见 [task.md](llm-challenge-arena-v1/task.md)。

**最新有效正式提交是 2026-09-30 的 CAP / attempt 47798。** 引擎为 `ca5d646c`，镜像为 `lh-img:0930a`；默认分支中的 `engine/sglang/` 与该提交逐文件一致。完整成绩与同日另一方案的结果只维护在 [正式提交记录](notes/submissions.md)。

## 第一次访问，从这里开始

| 想了解什么 | 阅读入口 |
| --- | --- |
| 最新交了什么、正式结果如何 | [提交记录](notes/submissions.md)、[冻结配置](submission/official-0930-CAP.json)、[原始包与回执](evidence/submission-0930-execution/README.md) |
| 怎样一步一步理解每个 commit | [探索历程与 65 个引擎提交的阅读索引](notes/read-history.md) |
| 请求怎样经过调度、缓存和模型 | [最新源码地图](notes/architecture.md) |
| 怎样克隆、核验、启动和比较 | [复现说明](notes/reproduce.md) |
| 本地诊断数据怎样生成、怎样校验 | [生成配方](scripts/longchain/recipes/README.md)、[参数 JSON](scripts/longchain/recipes/0930.json)；数据本体留在本地 |
| 某条路线为什么采用或放弃 | [实验记录](notes/experiments.md)、[研究索引](research/README.md) |

无需 GPU 的第一步：

```bash
git clone https://github.com/lhhhappy/LLM_deployment_A100.git
cd LLM_deployment_A100
python3 scripts/verify_release.py
git show ca5d646c -- engine/sglang engine/docs
```

核验脚本检查正式源码树、归档哈希、ZIP 内配置和官方终态。真实服务需使用记录中的 A100 镜像与模型挂载，见复现说明。本地短测用于机制比较，正式分数来自平台回报。

## 从哪里看

| 需要知道什么 | 唯一现行入口 |
| --- | --- |
| Pod 正在跑什么、接下来排什么 | [实验队列](notes/queue.md)；实时状态用 `scripts/pod/pread status` |
| 历史 v3/N30 阶段为什么这样设计 | [阶段计划与推理账本](notes/program-n30-v3.md) |
| 已核实的事实与撤回的解释 | [知识库](notes/knowledge.md) |
| 本地运行怎样判有效、怎样比较 | [评估合同](notes/evaluation.md) |
| 历次结果和原始证据 | [实验记录](notes/experiments.md) |
| 正式提交和平台回报 | [提交记录](notes/submissions.md) |
| 源码地图、性能与缓存研究 | [研究索引](research/README.md) |
| 引擎分支、提交与机制 | [引擎说明](engine/README.md) |
| 项目迁移、分支归档和恢复 | [仓库整理记录](notes/reports/repository-consolidation-0930.md) |
| Pod 队列、取证与部署命令 | [Pod 操作说明](scripts/pod/README.md) |

写作与迭代记录统一按 [Notes 规范](notes/README.md)；原始材料的保留、状态和归档按 [证据约定](evidence/README.md)。这里仅维护导航，不复制运行中的分数或第二份计划。历史结论留在 Git 和原始证据中，旧交接页不是现行状态。

## 证据放在哪里

一次运行使用固定编号：任务入口在 `scripts/pod/jobs/`，本地证据在 `evidence/L<编号>-<名称>/`，状态写在[队列](notes/queue.md)，结果和解释写在[实验记录](notes/experiments.md)。短时派发并排空的对照看该目录的 `opening/analysis.json`、`opening/paired.csv`、`opening/brief.txt` 和 `window/raw.jsonl`；完整档看原 harness 报告、判分收据与 raw。每次记录引擎提交、启动开关、数据哈希、预热与 flush、错误和服务日志。窗口未闭合时只报诊断。

原始 raw、服务日志和收据不能为了减少目录数而删除。Pod 上已结束的运行先归档到 GPU 开发机 `/sjtu/linhang/arena/archives/pod-runs` 并校验哈希，再按 [Pod 归档流程](scripts/pod/README.md)清理副本。`build/scratch/` 是临时工作区，不是最终证据。

## 操作边界

`llm-challenge-arena-v1/`、`s1-dev/`、`build/base_exact/` 和 `refs/` 只读。SGLang 修改在 `engine/sglang/`，vLLM 在 `engine/vllm/`；固定提交、有效机制和差异由任务脚本与日志核对。8 卡服务保持运行；停单个测试任务仅用 `scripts/pod/stopjob`。准备新实验先看队列，判断结果先看原 harness 与完整性收据。
