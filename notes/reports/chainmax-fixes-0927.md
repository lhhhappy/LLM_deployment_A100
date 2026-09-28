# Chain-max 修复与候选交付

2026-09-28 UTC，Codex；引擎已获 fable 独立审查通过。分支 `codex/fix-chainmax-0927`，引擎 `20a58da9`。前次问题见 [审查报告](chainmax-review-0927.md)。

## 已修复

1. **131 rank 分歧**：最终 decode interval 由请求组 rank0 决定，经既有 CPU group 广播；无风险请求也共同进入。其他 rank 不读取本地时钟/冻结分类来决定间隔。OFF 不增加 collective。
2. **131 成本口径**：默认不再固定 8k；使用本批经过 125/126/READY 后的计划冷块上限，并计入当前尚未执行 batch 的一次固定成本和全部新 token。正数 CHAIN_RISK_CHUNK 仍可显式覆盖。未来到达/预算变化无法预知，仍是估计；124 前期排序使用计划当时的预算，不宣称完美预测。
3. **132 配置可见性**：新增有效 `132=on`，开关开而缺 124 立即报错。131 同时报 `131_sync=rank0 131_chunk=auto`。
4. **普通暖请求预算**：确认 SHORT_TOKENS 只是资格阈值；16k 冷块占满 16k 总预算时它没有座位。复用 126 的单独探针已准备：floor12288/max16384，无需求仍满16k，有需求才让空间。**按用户最终决定，126 不混入上传候选**；两个脚本未入队。

修改只涉及 ax_deadline.py/scheduler.py，无内核或额外 GPU 缓冲。多轮长头抢占不是这四项修复的一部分，仍未实现；没有“所有 chain 已解决”的结论。

## 证据

- [验证收据](../../evidence/chainmax-fixes-0927/validation.json)：129 项相关 CPU 测试通过；实际 8 进程 Gloo 的 6 种时钟/分类/cadence 场景通过，未使用 GPU，不替代 TP8 模型验证。
- [Gloo 原始结果](../../evidence/chainmax-fixes-0927/gloo.json)、[控制流图](../../evidence/chainmax-fixes-0927/codegraph.json)：get_next_batch → prefill plan/adder → 本批冷块 → arming → rank0_decide → defer。
- fable 已独立读引擎差异并复测相关测试；完整 test_sched_protect_chain 的基线环境有 sglang 导入错误，未把整个仓库测试描述为通过。
- 131 的通信和成本口径同时修正，后续 N30 比较只能评价修后组合；不能把所有差异归给钉池。

## 候选

[镜像、配置与 46676 差异](../../evidence/submission-0927-chainmax-nopin/README.md)。16k、无 MTP/DCP，128p+131+132，118/126 关；按 chain 第一的决定去掉 Mamba400，其余配置不变，使用原镜像 0927a。fable 负责 eznb 修后引擎 TP8 验证；正式尚未上传，用户确认后方可提交。钉池臂的两个新增 chain 坏例见[逐决策归因](chainmax-pin400-held-0928.md)，多轮抢占缺口仍未解决。
