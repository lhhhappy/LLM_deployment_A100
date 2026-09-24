# 171/172 独立只读审查（2026-09-24）

- 171：融合资格、head TP rank/size、六路加载映射和参数 dtype 与底包契约相符；单卡随机权重证据覆盖模拟 rank、状态、MTP 与投影图。`evidence/kda171/final.jsonl` 的提交版 patch SHA 为 `807856…`；新增路径日志使当前 SHA 为 `08eee8…`。
- 172：BF16 `silu` 后舍入与输出舍入保留；零行跳过 launch，行 stride 被传入，非 BF16、列 stride、其他 limit 回退。修正后的特殊值比较已核对正负 Inf；全 BF16 编码、stride、fallback、图变输入测试通过（`evidence/moe172/numeric.jsonl`）。
- 172 因果审计：旧 `full-audit.jsonl` 的 wrapper 两臂都写融合输出，只证明同 GEMM 输入的激活逐位一致。现行脚本 SHA `fd23f69a…` 在原版臂补 `out.copy_(ref)`；`arm-audit.jsonl` 核对 M256 三次同输入激活均 0 差，原版重复差 224/1048576、候选对原版差 140/1048576，两者最大绝对差均 .00390625。
- M256 固定首次路由后原版重复及候选对原版均 0 差。AOT/JIT expert sort 用 atomicAdd；开放路由差异与此相符，但此实验只锁定排序结果，未证明每个差异的精确算术来源，也不把 140 个差异当允许误差。
- 172 完整 MoE **测速**在恢复原函数后执行，两臂有效：随机 289 专家、单卡 TP8 形状下，M33–16384 墙钟中位数约快 3–5%；图池、真实权重、TP8 collective 与整模型收益未测。`peak_reserved`/`extra_allocated` 不能用作服务显存差额。
- 上述定向补测已完成；旧 full-audit 脚本单独保存在 `evidence/moe172/full-audit-test.py`，其成本测量在审计 wrapper 恢复后运行，无需因本次修复重测。
- `numcheck.py` 的 flush、并发失败和短输出边界已由 `tests/test_numcheck_probe.py` 四项 CPU 回归通过；65536 前缀需按目标服务实际长度上限验收。

结论：171 仍需真实 TP8/权重验收。172 已满足进入真实 TP8 数值筛选的单卡门槛；需在真实权重、通信、完整模型图及负载下确认，不能称全模型通过。
