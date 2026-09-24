# 171 — FP8 模型内免量化 KDA 投影融合（候选）

目的：减少首 token 前每个 KDA 层的投影调用成本。底包已经有融合模块与 checkpoint 加载映射，但 `quant_config is None` 排除了本模型的 FP8 配置；该配置中的 KDA 投影实际全部免量化。正式 A 保留 MTP，是后续部署验证的对照。

## 开关与范围

`SGLANG_AX_KDA_FUSE_PROJ=1` 显式启用，默认关闭。只扩展 `quant_config.get_name()=="fp8"` 且原 qkv/b/f_a/g_a/f_b/g_b 均经底包 `is_layer_skipped` 判定免量化的层。qkv 的三个分片沿用底包一致性检查。head shard size 必须等于 TP size；其他配置沿用原路径。

仅给这两个融合模块使用 `quant_config=None`，不修改全模型配置、其他 KDA 投影或 MoE。沿用原来的无量化模型融合路径。没有新增通信、状态池或持久权重副本。

## 计算与加载契约

输入侧当前已有合并 qkv，因此是 **4 次 linear → 1 次 linear**：

| 部分 | checkpoint 全局形状 | TP8 每卡形状/归属 |
|---|---|---|
| q、k、v | 各 `[8192,4096]` | 各 `[1024,4096]`，连续头分片 |
| beta | `[64,4096]` | `[8,4096]`，连续头分片 |
| f_a、g_a | 各 `[128,4096]` | 原样复制，不除以 TP |
| f_b、g_b | 各 `[8192,128]` | 各 `[1024,128]`，连续头分片 |

一级融合权重为 `[3336,4096]`，输出按 `[3072,8,256]` 切开；后 256 列按 f_a、g_a 分成两个 128 向量。二级权重为 `[2,1024,128]`，用原 `ColumnParallelBatchedLinear` 的一次 bmm，代替原两次 linear。合计 6 次矩阵调用 → 2 次；不是输入侧 6→1。

原 `Glm5NextForConditionalGeneration.load_weights` 将 q/k/v/b/f_a/g_a 映射到一级 shard 0–5，将 f_b/g_b 映射到二级 shard 0–1。原模块直接把 checkpoint 分片写进融合参数；不先加载一份再复制。171 显式传入 head shard rank/size，并让二级参数 dtype 跟随一级的实际 dtype，避免 `--dtype` 覆盖后与 checkpoint dtype 不同。

融合 qkv、beta 是非连续行视图。不得仅由 GEMM 输出正确推出卷积、KDA recurrence、MTP verify 和 graph 正确。

## 验证入口与当前边界

CPU：`python3 -B -m unittest discover -s tests -p test_kda_fusion_171.py -v`。真实 checkpoint 配置的 34 个 KDA 层、默认关闭、部分免量化、未知量化方式与不匹配 TP 条件均检查。补丁在只读底包副本、S0、正式 A 的完整补丁栈上均以 `--fuzz=0` 应用。

GPU 算子入口：[test_kda_fusion_171.py](../tests/gpu/test_kda_fusion_171.py)。从当前候选 `PYTHONPATH` 导入真实模型类与加载器；在一张 A100 上依次模拟 TP8 各 rank 的权重切片。使用真实模型维度和随机 BF16 权重，检查加载顺序、投影、prefill→decode→后续请求状态、重复 graph 输入，以及独立计时。

**这不是实际 TP8 通信或真实权重模型验证。** 投影没有历史依赖，成本记录中 P、服务 batch、KV/KDA 容量均为 null；不能把它当 `T(c,P,B)` 服务成本。完整模型成本应扩展通用 [extend_check.py](../scripts/analysis/extend_check.py)，使用真实缓存历史。

开发机 A100 实测已通过：8 个模拟 rank 的加载/投影、非整齐行、冷 prefill→decode→后续 prefill、MTP 非融合 verify 及接受 1–4 token 的 SSM/conv 提交、11 种形状的投影 graph 重放、FP16 dtype 覆盖。逐位相等并非通用要求；完整误差、边界与原始记录见 [R22](../research/codex/R22_kda_projection_fusion.md) 和 [final.jsonl](../evidence/kda171/final.jsonl)。

单层投影调用实测 6 linear → 1 linear + 1 bmm。非 profile 的重复计时：256 行 eager 墙钟中位数约 141→87µs；4096 行约 588→477µs；16384 行约 2082→1872µs。1024 行 eager 两组存在明显 host 波动，不使用混合中位数宣称 2×；同形状投影 graph 约 160→119µs。这里的 graph 是算子图，不是 170 的模型 BCG。

**显存：** 每层完整 KDA 模块参数均为 36,294,944 bytes/rank，持久权重/状态增量为零；独立 eager 探针在 c=4096/8192/16384 的峰值 allocated 增量相对基线约 +1/+2/+4 MiB。图池及工作区的服务端累计增量、实际 KV/KDA 容量仍未测，不能称“零显存代价”。没有整模型或 SLO 收益结论。

## 下一步真实模型验收

由 8 卡负责人安排正式 A 原路径的 flag=0/1 对照，其他参数固定。先同配置重复建立误差参照；覆盖 33/37/63/65、图桶/补齐、chunk 续算、长前缀命中、多请求与 MTP verify/accept。保存逐层输出、SSM/conv、实际融合与 graph 路径、KV/KDA 容量及峰值显存。

数值通过后，同负载测 256/1024/4096/8192/16384 的模型成本，随后完整回放检查四道 TTFT、TPOT 及全部其余硬门。只改善算子或某一门不记为并发档晋级。170 的 prefill BCG 先独立验证，再测组合；正式提交由用户安排。
