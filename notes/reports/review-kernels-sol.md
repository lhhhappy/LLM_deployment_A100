# 171/172 独立只读审查（2026-09-24）

- 171：融合资格、head TP rank/size、六路加载映射和参数 dtype 与底包契约相符；单卡随机权重证据覆盖各模拟 rank、状态、MTP 与投影图，未发现阻断性错误。
- 171 证据边界：`evidence/kda171/final.jsonl` 对应提交版 patch SHA `807856…`；当前工作树 SHA `08eee8…` 只新增路径日志。真实 TP8 通信、真实权重整模型输出、图池/KV 容量与 SLO 仍待验证。
- 172：BF16 `silu` 后舍入与输出舍入均保留；零行跳过 launch，行 stride 被传入，非 BF16、列 stride、其他 limit 走原路径。生产 Marlin 分支调用条件相符。
- 测试盲点：`test_moe_swiglu_172.py:compare` 只比较 NaN/Inf 掩码，不比较无穷大的符号；参考值非有限的位置也跳过逐值比较。建议比较正负 Inf 掩码，防止特殊值假阳性。
- 172 `full_moe` 的随机 289 专家、TP8 每卡形状可筛选输出/延迟；单 GPU、无真实权重/collective，`peak_reserved` 与 `extra_allocated` 是共存图和缓存分配器下的探针值，不能当服务显存收益。
- `numcheck.py` 新增的 flush 耗尽报错、Future 异常传播、48 token 完整性检查均合理；65536 前缀会形成最长 66048 token 请求，需用目标服务实际长度上限验收。原脚本无对应 CPU mock；这项改动逻辑简单，未增镜像测试。

结论：171 可进入真实 TP8 数值筛选；172 单卡通过后仍须真实 TP8/整模型对照。修正 172 特殊值断言并记录新旧 171 哈希后，局部证据可用于下一轮决策。
