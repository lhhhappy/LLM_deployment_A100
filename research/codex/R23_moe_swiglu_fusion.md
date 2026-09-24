# R23 — Marlin MoE 的 clamped SwiGLU 融合

2026-09-24。开发机 A100-SXM4-80GB，真实生产 Marlin 函数、随机权重、TP8 每卡形状；实际 TP=1。补丁 [172](../../patches/172-moe-clamped-swiglu.md) 默认关闭，不新增持久权重、KV、SSM 或通信。

## 选择依据与数值契约

046r 的 S1 时间账只有7个prefill样本，不能估计正式A成绩；大块样本的kernel累计时间中MoE约占38%，足以支持先检查这条实际路径。基线111把FP8专家转换为Marlin W8A16；激活仍是多次clamp、SiLU、multiply、copy。合为一次kernel可减少发射与临时读写，无需改变专家权重格式或路由。

BF16的SiLU结果必须先舍入再乘up。全FP32表达式负对照确实改变结果，因此172显式保留该舍入边界。比较式clamp保留NaN；穷举全部65536个BF16 gate编码与7个up取值，再交换两者角色。不是穷举所有BF16值对。另测正负Inf、非整齐行、行stride、FP16/列stride/其他limit回退、图捕获后改变输入。

## 数值审计的修正与边界

- 首轮 `screen.jsonl` 在M256出现6个差异并停止，未隐去失败。
- `full-audit.jsonl` 每次在相同真实GEMM输出上比较原/新激活，M33–16384均逐位相同。但wrapper两臂都写融合输出，其全MoE“reference”标签无效；只采信同输入激活检查和恢复wrapper后的测速。
- `arm-audit.jsonl` 对应现行测试SHA `fd23f69a2834d1d9d29bf4b78a7ec2b8f0cc23b1cbedb4d4c4ed28a0ef8e4de0`。wrapper按原flag写回，M256原版重复224/1048576个差异、候选对原版140个，max_abs均0.00390625。
- 缓存第一次expert排序结果，再运行未包裹的原版重复/候选，两项逐位一致。底包expert排序使用atomicAdd，实验支持排序波动这一解释；没有证明所有差异的精确算术来源，也没有由此定义宽松容差。

旧脚本按首行hash恢复为 `evidence/moe172/full-audit-test.py`，仅作历史证据。新旧测试使用同一172源代码SHA `c33fc53e551572f27a142412c18566deae2186a2a6aff4f9a37a81f9b740d999`，补丁SHA `2752e70a32a732294b6025bd13ad6226c269c885c25fb96800885da5b669d606`。

## 成本与部署边界

完整MoE使用hidden4096、中间维度256、288路由专家+1共享专家、top8+shared、FP8 block128权重。每臂预热5次，每组5次样本、每样本5次调用，两组顺序互换。原始wall/event样本均保留；该eager event区间也可能含host发射间隙。M33–16384墙钟减少3.1–4.9%，详见patch表及 [summary.json](../../evidence/moe172/summary.json)。复算命令：`python3 evidence/moe172/summarize.py`。

没有新增持久缓冲。4k/8k/16k整MoE调用的额外allocated峰值分别少40/80/160MiB；这是单进程缓存分配器与临时张量生命周期的测量，不是服务KV池收益或graph池收益。激活本身的图检查通过，完整MoE未单独捕获graph；尚无真实TP8 collective、模型权重、MTP、逐层状态、质量或整档证据。

下一步只在正式A上单独加172做真实TP8输出/logprob筛选，再决定完整回放。171、122与172的局部收益不能相加。若需正式A时间账，另开明确诊断窗口，不能把profile期间延迟作为SLO对照。

开发机复现（候选源码为exact底包+111+172）：

```bash
cd /sjtu/linhang/arena
source code/e1_env.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/sjtu/linhang/arena/runs/moe172/src:/sjtu/linhang/arena/cache/T48/deps \
  env/m0/bin/python runs/moe172/kit/tests/gpu/test_moe_swiglu_172.py --full
```

`--full-audit-only` 只运行M256的臂/路由补测，不复测成本；`--numeric-only` 只测激活数值。最终同原harness的完整负载对照决定能否改善并发档。
