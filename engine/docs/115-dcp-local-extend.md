# 115：DCP eager prefill 使用本地 KV 分片

2026-09-26，Codex。候选分支 `codex/dcp-prefill-local-kv`。默认关闭；两卡开发验证，TP8 真实权重与服务成绩待验。实验与日志结论统一见[本轮报告](../../notes/reports/dcp-chain-20260926.md)。

## 问题与实现

现有 DCP 把每次 ordinary extend 所需的历史 KV 全量 gather，再按虚拟位置生成 gather-buffer 行号。短尾接长前缀时，为少量 Q 重搬整个 prefix；冷链分块执行时，后续块反复搬越来越长的历史。新路径 gather 同一 DCP 组的 Q，每个 rank 在本地 owner-striped KV 上算 partial attention，再用原有 LSE 加权与 reduce-scatter 恢复各自 head 的输出。token、上下文、输出预算及 MTP 验证规则不变。

调用关系由 code graph 查询和整段函数阅读确认：

```mermaid
flowchart TD
  M[ModelRunner：padding 与 eager 分发] --> E[EagerRunner：按复制的 CPU 长度选择一次]
  E -->|原路径| P[planner：prefix indices / gathered KV buffer]
  E -->|新路径| D[metadata.dcp_local_extend = true]
  D --> Q[MLA：Q all-gather]
  Q --> S[DSA：写本地 KV / 生成 owner-local indices]
  S --> A[TileLang：partial attention 与 LSE]
  A --> L[原有 LSE 校正 / output reduce-scatter]
  L --> O[本 rank V 与输出投影]
```

选择发生在旧 planner 分配之前，所以被选择的 batch 不分配 gathered KV buffer，也不启动 prefix-index planner。此前 cold batch 已创建的持久 `loc2row` 表不会因此释放；不能把这部分也算作节省。

新 Triton kernel 把 owner 过滤、虚拟转物理、KPool 列压缩、64 列 padding 合成一次写入。KPool 展开列的 residue 与页对齐保证 `rank % gcd(W,KPool)` 的选列；保留 `-1`、尾部不足一组的合法 token，并支持非连续输入。它仅接入新 extend 路径，旧 decode/verify/draft 的选列函数保持原状。

空 shard 的 partial 输出就地清理 NaN/Inf，LSE 决定其零权重。就地写入仅用于新 eager 路径独占的输出，不改图捕获中的旧路径。没有跨层缓存可变 tensor，也不引入新持久 GPU buffer。

## 开关与支持范围

- `SGLANG_AX_DCP_LOCAL_EXTEND=1`：开启短尾选择。默认 token 上限 512，prefix 至少 4096。
- `SGLANG_AX_DCP_LOCAL_EXTEND_MAX_TOKENS`：短尾上限，允许 1–1024；默认 512。用 padding 后的 token 数判定。
- `SGLANG_AX_DCP_LOCAL_EXTEND_LARGE_MAX=2048|8192`：另行开启大块实验；默认 0。选择 `2048 <= padded_Q <= LARGE_MAX`，不沿用短尾的通信量条件。只支持每 rank 8 heads、D=512、KPool=4、topk=2048。8192 以下的中间形状没有逐点性能保证。

共同范围：GLM5Next target/NextN、DCP W2、CP/DP/PP 均 1、BF16 KV、rope 维度 0、SM8x CUDA、TileLang prefill/decode、ag_rs、未开启 Q projection replication、HiSparse 或 118 Triton override。不支持的已请求组合启动时报错。prefill CUDA graph 不使用这条 eager 路径；decode/MTP 图仍走原路径。

短尾条件比较每 rank 集合通信量，P 是 prefix token 总数，T 是 padding 后的 Q 行数，H 是原本每 rank head 数，D 是 latent width，W 是 DCP 宽度：

- 原 KV gather：`P*D*2*(W-1)/W` 字节。
- Q gather + FP32 output reduce-scatter + FP32 LSE gather：`T*H*(W-1)*(6*D+4*W)` 字节。

这个条件是保守的通信量筛选，不是性能预测器。大块实测中，KPool 压缩与 head tile 利用率可能使第二条路线更快；不能仅按通信字节数否定它，也不能据此承诺 TP8 加速。

启动日志打印完整 policy；第一次选中新路径打印 prefix、padded Q 与 heads。关闭主开关时 policy 为 None，旧 planner、old decode predicate、gather、MTP 路径保留。

## 显存与验证边界

单 attention step 的大块路径会增加临时 Q/O 内存；完整小模型的峰值又可能由另一层决定。报告同时保留单步与整段前向测量，不能把“没有新持久 buffer”说成“没有显存代价”。TP8 接入需记录实际可用内存和 KV/KDA 池容量，并与同一引擎、开关关闭的对照比较。

数值检查包含两个 rank 的敏感 attention 输出、真实 decode graph replay、非连续高虚拟位置、KPool 尾列、全遮蔽行与空 owner shard、MTP 各阶段、真实 verifier 的接受长度 1–4、HiCache device/host restore 与真 flush。缩小 dummy 模型的最终 logits 对错误不够敏感，因此通过判断不能只看 greedy token 一致。

CPU 合同：`python3 -B -m unittest discover -s scripts/tests -p 'test_dcp_local_extend.py'`。
开发机入口：[run_dcp_devbox.sh](../../scripts/analysis/run_dcp_devbox.sh)、[dcp_check.py](../../scripts/analysis/dcp_check.py)、[整步曲线](../../scripts/analysis/dcp_local_extend_bench.py)。这些工具只在显式指定的开发机运行目录工作，不部署 Pod。
