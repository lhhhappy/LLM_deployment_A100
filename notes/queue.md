# 当前实验队列

## 已核实的收尾状态

09-30 最后两包已上传；10-03 查询确认两包均已出正式终态，结果只维护在 [submissions.md](submissions.md)。当天用户结束工作，拟追加的 cold12k + relief1 试验未生成、未入队；依据见 [收尾记录](../evidence/submission-0930-execution/decision.json)。

本次任务只整理 Git、发布资料和文档，没有派发新的 GPU 任务。没有在本次整理中重新查询 Pod 实时状态；实时状态使用 `scripts/pod/pread status`。09-28 / 09-29 的旧排队表已退出当前队列页，结果见 [experiments.md](experiments.md)，当时的发布顺序仍保留在 Git 历史。

当前没有本次任务登记的待运行实验。后续实验需记录问题、对照、代码/配置、数据与派发范围；操作规则见 [Pod 说明](../scripts/pod/README.md) 和 [协作约定](collaboration.md)。

最新冻结引擎为 `codex/final-execution-0930` / `ca5d646c`；默认分支源码已经与其一致。研究计划、失败路线和逐提交阅读入口见 [read-history.md](read-history.md)。
