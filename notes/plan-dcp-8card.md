# DCP（解码上下文并行）上 8 卡：计划、风险与难点（fable，2026-09-26）

**已过时（2026-09-28）：** 本文是 8 卡验证前的计划（目标 DCP8、估算 ×3.45–4.45），实际做出并反复验证、两次正式提交的是 `--dcp-size 2`（不是本文设想的 DCP8），KV 池实测增益约 ×1.7（1.40M→2.38M，N30），且本地测到的容量收益不对应线上的墙，没有带来 chain 判定提升；MTP 兼容性也已解决（不是本文说的"未上过 8 卡"）。现行结论见 [knowledge.md](knowledge.md)、[program-n30-v3.md](program-n30-v3.md)、[R34 §3](../research/claude/R34_kv_capacity_options.md)。

目标：拆掉 N34+ 的显存墙。MLA 潜向量 KV 在 TP8 下每卡各存一份完整拷贝（12,716 B/token/卡）；DCP 按 token 把它分到 8 张卡，每卡只存 1/8，KV 池等效 ×3.45（MTP 草稿层仍复制）到 ×4.45（无 MTP）。现成代码：115/116（`engine/docs/115-dcp-sm80.md`），开发机 TP2+DCP2 数值验证通过（相对 L∞ ≤ 2e-3/5.2e-3，含 CUDA graph decode），未上过 8 卡。业界证据见 [R34 §3](../research/claude/R34_kv_capacity_options.md)、[R35](../research/claude/R35_kv_capacity_survey.md)。

## 收益（估算）
- 线上 N34 按第一名的形态约 16 路同时解码：锁住的 KV ≈ 16 × 85k ≈ 1.36M，与现在的 1.40M 池相当；DCP8 后池等效 4.8–6.2M token/卡，这堵墙推到 N100 以外。剩下的限制只有开场 prefill 吞吐与解码步时间。
- 通信：每个解码步每层 DSA 两次集合通信（AllGather Q、LSE 合并），11 层共 22 次；我们的卡是 A100-SXM4（NVSwitch），单次几十微秒，每步约 0.5–1 ms（估算），在 55 ms 一步里可忽略。Hopper 上外部报告 ITL +25% 是大批量下的数字。

## 风险与难点（按可能性 × 代价排序）
1. **8 卡真机数值**：只验过 TP2+DCP2。8 路分片的地址换算（`mla_buffer.py` 的 DCP_RANK/WORLD_SIZE 掩码、`dsa_backend.py` extend/decode 的换算、indexer 池对齐）在 8 卡上第一次跑；025 曾因高水位越界崩溃。对策：先 12 题能力冒烟，再 20k/60k/190k 冷探针对照 TP8 非 DCP 的 logprob，再 CUDA graph decode 重放一致性。
2. **MTP 不兼容**：投机解码接受 token 后用 `move_kv_cache`（`memory_pool.py:4514/4967`）把 KV 从验证位置搬到最终位置，是纯 `kv[tgt]=kv[src]`，没有 DCP 感知；上游 #39638 同样卡在这里。没修好之前只能关 MTP：TPOT 从 .023 涨到约 .045（仍低于第一名 .050；同档 tie-break 才比 TPOT）。修法：按 rank 掩码只搬本卡持有的 token 行、跨 rank 不搬；难点是写出能暴露"跨 rank 错位"的用例（现有 r5/r7 用例只覆盖 decode/verify 数值）。3–5 天。
3. **草稿层的 KV 复制与 HiCache**：DCP 下目标层分片、草稿层复制（115 的设计）；180 的主机层按行宽/组所有权做声明式校验，分片后行身份变了，主机层的 D2H/H2D 与 256 组所有权要重新核对（未验证）。
4. **性能未知**：decode 每步的通信、extend 阶段每层多一次前缀 AllGather；A100 无实测。对策：60 分钟 N34 窗口对 136 底座，看 TPOT p95、四门与完成量。
5. **调度侧账目**：KV 池按 `dcp_size` 变大后，120/122/124/125/126 的预算与 `kv_room`、126 的留位、180 的池大小日志都要重新核对；`max-running-requests`/graph bs 需一并放开（否则容量放大了并发上限还是 32）。
6. **官方约束**：DCP 官方只验证 GB300 且不支持配 MTP（R24）；我们是自己蹚的路，没有上游先例可抄。

## 顺序
1. 明天：115/116 在 8 卡上关 MTP 跑能力冒烟 + 三种长度冷探针 + 60 分钟 N34 窗口（对 136）。1 天，主要是验证。
2. 数值与性能过关后：`move_kv_cache` 的 DCP 感知 + 跨 rank 错位用例，让 MTP 一起用。3–5 天。
3. 与今晚容量六组（139–144）的结果合看：便宜杠杆若已够 N30，DCP 就是 N34–N38 的准备，不抢在提交前。
4. FP8 KV 只在 DCP 走不通时再立项（×1.75、5–10 天、A100 反量化 kernel 数值风险最高）。

分工建议：8 卡验证由 115/116 的作者（Codex，执行层）主导；`move_kv_cache` 与用例由 fable 或用户指定的最强会话做；fable 负责调度侧账目核对与实验设计。
