# 协作约定

参与者平等协作，按问题领域分工；同一处代码同时由一人修改，跨领域改动先约定接口。当前任务与 8 卡顺序只记在[实验队列](queue.md)，长期有效事实记在[知识库](knowledge.md)，完整结果记在[实验记录](experiments.md)。本文件不维护人员名单或会话状态。

当前可联系会话在忽略目录 `build/scratch/coordination/sessions.json` 登记；联络工具是 [`scripts/agent_message.py`](../scripts/agent_message.py)。终端重启后须更新登记，投递后以对方确认收信为准。

入队者在[队列](queue.md)登记基线、唯一改动、问题和状态，并负责跟到结果闭合、保留原始证据及更新实验记录。队列状态以 `scripts/pod/pread status` 为准；发布流程见 [`scripts/pod/README.md`](../scripts/pod/README.md)。

结论进入决策前由另一位参与者对照原始数据复核。开始工作前查队列和相关报告，避免重复实验；共享事实文件改动前先读最新版，只提交自己负责的修改。其余实验、代码、证据及安全边界统一遵守仓库根目录 [`AGENTS.md`](../AGENTS.md)，不在此重复维护。
