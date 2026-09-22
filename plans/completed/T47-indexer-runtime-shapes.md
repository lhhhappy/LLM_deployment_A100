# T47 — 112/113 v2 runtime shapes（W21）

## 目标
移除精确长度/stride Triton constexpr，任意新长度在预热有限特化类别后不再JIT；保留110语义与原性能。

## 范围
四个kernel参数最小修改，112/113原名重新生成，v1保留drafts；全部证据在evidence/T47。仅开发机arena算子，无8卡/服务/镜像/提交。

## 验证方式
- P112-01…05、P113-01…03原数值/graph/大矩阵对110，全过原误差/topk门。
- P47-01：50个不同NQ/NK/P/batch，预热后JIT miss/实际编译/磁盘命中全部0；记录有限类别预热计数。
- P47-02：112/113分别v1/v2同输入交替计时，六prefill+两decode graph，退步≤5%，否则分析并调tile。
- P47-03：11补丁全栈fuzz0、py_compile、确定生成、反向字节还原、base未改。

## 风险与回滚
runtime除法/stride可能增加指令或降低向量化，需实测；恢复drafts中的112/113 v1配套补丁即可回滚，但恢复精确长度JIT问题。

## 进度记录
- accepted → in-progress：v1归档，参数去constexpr。
- 112全部88组/12graph，113全部222组/30graph通过；16行配对性能最大+3.38%，无需调tile。11补丁fuzz0/3623+6编译/反向还原通过。
- done：634有限类别预热后200随机+8回归调用零JIT/编译/磁盘命中；286key审计通过。PTX/summary/SHA收据全通过，0spill，两卡空闲；F68及交付节已落盘。

## 决策记录
沿用原tile优先，以实测决定是否调优；不修改110 oracle与原验收阈值。不覆盖T43/T44历史证据。
