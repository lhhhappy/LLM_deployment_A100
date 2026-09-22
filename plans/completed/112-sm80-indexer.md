# T43 sm80 indexer融合kernel

- 状态：completed　负责人：Codex W17　创建：2026-09-22
- 关联：dispatch T43；F56–F63

## 目标
在base_exact+000→101→105→110→111之上交付112，两个入口保持110语义，A100算子实测通过。

## 范围
- 包含：软件fp8解码、bf16 MMA融合decode/prefill、生成器、数值/边界/图/性能与补丁栈证据。
- 不包含：8卡服务、镜像、Trisol、提交。

## 前置条件与约束
- 用户已授权GPU开发机算子测试，只在/sjtu/linhang/arena工作，用前查占用。
- base_exact等只读；110参考不改；测试数据是随机合成的真实形状，不冒充真实模型激活。

## 风险与缓解
| 风险 | 缓解 | 回滚 |
|---|---|---|
| fp8/舍入/边界导致topk变化 | 与110原实现逐项对照；保留每头bf16舍入 | 反向撤112 |
| clean=False区间外语义冲突 | 按110全宽计算；clean=True才剪枝 | 撤112 |
| 稀疏长上下文/图动态长度 | GPU分支跳过无效页并写零，图重放改长度测试 | 撤112 |

## 里程碑
1. 源码语义核对、环境接通。
2. 融合kernel与复现脚本。
3. GPU矩阵、性能、图验证；完整补丁链。
4. 文档与交付账本。

## 验证方式
- scripts/make_112.py；scripts/test_sm80_indexer_112.py；scripts/verify_112.py。
- logits相对误差<1e-2、topk集合≥99.5%，边界逐项一致、图可捕获且动态重放一致；完整栈fuzz=0、py_compile。
- 开发机算子耗时仅代表微基准，8卡SLO交Claude验证。

## 进度记录
- [x] 已读110与现有tilelang kernel；两卡空闲。
- [x] kernel与测试：两个入口、软件解码、数值/graph/benchmark脚本。
- [x] GPU与完整栈验证：最终88组通过，4种graph各3次动态重放；8补丁栈fuzz0、3623编译、反向逐字节还原。
- [x] 交付：112说明与evidence/T43完整证据；F64、D33、TEST_PLAN已回填，L2交Claude。

## 决策记录
- 2026-09-22：用Triton uint8软件解码+bf16 dot；现有tilelang仍依赖FP8 GEMM且只支持N=1，改写范围并不更小。保留110的bf16点积输出舍入、-1页映射到0与prefill clean=False全宽语义。

- 2026-09-22 完成：CUDA graph decode 32k/190k为4.75×/4.54×；8192-query prefill为2.01–2.95×。最终按D33选直接bf16位编码与小tile；无生产autotune。两卡测试进程退出，gpu_final_idle.log记录空闲。
