# SM80 Humming 大块 MoE 的可复用执行约束

2026-09-28，Codex。来源为 Humming 0.1.12 实际源码和单张 A100 探针，
只支持已测合成权重的每卡形状；chain/TP8 尚未验证。
实验数字与完整失败历史统一见[报告](../../notes/reports/moe-mhc-sm80-0928.md)，
原始证据见[目录](../../evidence/mhc-moe-sm80-0928/README.md)。

## 表驱动调优必须确认实际生效

- Humming indexed 表使用 routed rows，即 tokens×9；区间为 `(lo, hi]`。
  8192–16384 token 的下切点是 `8192*9-1`，不能直接写 token 数。
- 原生 Python shape 是 tuple，JSON 是 list。直接和 list fixture 比较曾
  让优化静默成为 no-op。只规范化 shape 容器，仍严格比较键集与其他值；
  回归必须截获实际 kernel 调用参数，不能只测试自造配置表。
- W13/W2 共用 expert block 索引，M tile 高度必须一致；本次保持 M128，
  仅调整 W2 的 N tile、stages、CTA 参数和 W13 的 stream-K。
- 117 已把原 checkpoint 的 128 行 block scale 展开至 N 行，因此转换后
  metadata 为 K group128、N group0、BF16 scale，不能用原文件的 N128
  去判断已转换权重是否支持。
- 原 117 首 token warmup 编译表中的全部 variant；测试在其后禁止 compiler
  调用，并执行真实大形状，才能证明没有把首次 JIT 留给服务请求。

## Stream-K 的数值语义

Humming 0.1.12 的 `epilogue/smem_writer.cuh` 把每个分片的累加结果转为
输出 dtype；`epilogue/gmem_writer.cuh` 对 BF16 分片执行 BF16 加法/原子
合并。关闭 W13 stream-K 保留完整 K 维 FP32 累加并在末尾舍入。这没有
删计算，却会改变数值，不能以“同为 FP32 accumulate”推导 bit exact。
需要原 checkpoint 独立参考；直接归并/W2 的逐位检查与完整层误差分别验收。

## 性能与内存账

16k 的 `[M,9,4096]` BF16 down-output 是 1152 MiB；写一次再读一次的
名义流量为 2304 MiB。它是逐层复用的瞬时工作区，不应乘模型层数当作
常驻 KV 损失。本次主候选不移除该张量，也未增加临时峰值。

归并单核约 2% 的改进在完整层上可能小于噪声；W13/W2 联合则在最终
均匀/偏斜对照中减少完整 MoE 时间约 6–8%。单个 kernel、完整 MoE、
整段 prefill 和 chain 是不同验收范围，不能互换。graph、随机顺序、
同轮配对、频率记录、边界 no-op 控制与实际配置打印共同约束测量解释。

Fusion 也不保证快：本次原子归并、行锁归并、最后写入者归并及 mHC
post-prenorm 原型都更慢。是否由争用、寄存器或 occupancy 主导需要
额外硬件计数器才能定论；不能从慢了就直接宣布某个硬件瓶颈已被证实。

部署前仍需真实权重 TP8 与同 ID chain/TPOT 验证。启用 attention-TP
输入分片后先检查 MoE 实际每卡行数，不直接套用未分片的 8k/16k 成本。
