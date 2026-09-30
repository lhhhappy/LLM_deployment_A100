# 47266：保持调度节奏的执行优化与 N46 路线

审查对象是正式 47266 的冻结配置、镜像 0928c 和引擎 `bf6b66faf3ffbe10e593e858d3bbedd1c48cefb2`，没有用主工作区正在修改的 scheduler 代替已提交代码。关键源码已按该提交导出并记 SHA256，见[审查收据](../../evidence/review-47266-0929/review-receipt.json)。本次完成代码和既有证据审查，未新增内核测速、负载运行或正式提交。

## 线上余量

以下均为[47266 官方终态](../../evidence/official/attempt-47266-final-20260929.json)的 N42，目标取原 task.md。

| 指标 | 实测 | 点目标或硬门 | 能说明什么 |
| --- | ---: | ---: | --- |
| chain p95 | 45.075 s | 30 s | 已超过点目标；官方统计门仍 PASS，未返回超标条数及下一失败档明细 |
| fast p95 | 1.639 s | 3 s | 低于点目标 1.361 s |
| overall p95 | 2.056 s | 5 s | 低于点目标 2.944 s |
| turn p95 | 3.457 s | 15 s | 低于点目标 11.543 s |
| TPOT p95 | 71.276 ms | 100 ms | 距硬门 28.724 ms；与排名均值是两个指标 |
| TPOT mean | 43.266 ms | 同档越小越好 | 对用户提供的 37.352 ms，需再少 5.914 ms，约 13.7% |
| AIME / GPQA | 44/44、151/156 | 两科严格 >90% | 本次能力通过；不能替未来内核的数值/能力验收 |

这些秒数差是同档点目标距离，不是还能容忍多少次超标、多少并发或多少质量下降。错误率/覆盖等本档通过，但当前回包未给出各项完整分母及下一档门，因此不能量化它们的余量，也不能断言 N46 只卡 chain。

## 保持节奏、降低 TPOT：三个具体候选

### 1. KDA packed decode 的接线缺口

代码事实：[KDA backend](../../evidence/review-47266-0929/source/srt/layers/attention/linear/kda_backend.py) 的 `forward_decode` 在约 718 行要求 `lower_bound is None` 才调用 packed decode。GLM 的 safe gate 设置 lower_bound，因此走通用 recurrence；已有日志里的 `packed_decode=True` 只表示 dispatcher 能力，不证明 GLM 每层实际进入快路径。

与此同时，[Triton dispatcher](../../evidence/review-47266-0929/source/srt/layers/attention/linear/kernels/kda_triton.py)、[packed recurrence](../../evidence/review-47266-0929/source/kernels/ops/attention/fla/fused_recurrent.py)及[CUDA packed wrapper](../../evidence/review-47266-0929/source/kernels/ops/attention/kda_packed_decode.py)都接受 lower_bound，底层实现了同一 `lower_bound * sigmoid(exp(A_log) * (a + dt_bias))` 公式。CUDA 路径对 K=V=128、BF16 输入、FP32 状态、batch>=8 等有覆盖检查，小 batch 回退 Triton packed。

推断：这是值得验证的性能路由缺口，不是已证明的正确性 bug。packed CUDA 用另一种状态访存/归约布局；A100 是否更快、长 decode 是否数值稳定必须实测，不能把源码注释中其他 GPU 的 TB/s 数字当 A100 收益。

最小实验：保持 47266 的 interval、cold cap、状态池与精度；只给满足完整布局/stride、T=1、设备和 dtype 条件的普通 decode 增加可回退路由。保留 conv update 与状态快照跟踪，不把 MTP verify、多 token、ReplaySSM、padding 或槽迁移混进去。比较真实 shape 的单步输出及多步 FP32 状态，含 -1 图补齐槽与非连续物理槽；再看纯 decode 和混合负载。

### 2. 减少 decode CUDA graph 的补齐行

实测日志：[启动路由](../../evidence/review-47266-0929/runtime-routing.txt)中 decode capture 档位为 `1,2,4,8,12,16,24,32,40,48`。取 N42 A1 在 17:20:31–17:41:56 UTC 的前 600 个匹配日志记录，真实 running batch 多在 25–34；见[样本](../../evidence/review-47266-0929/decode-log-sample.txt)和[可复算账](../../evidence/review-47266-0929/review-receipt.json)。这是有界日志样本，不是全程每个 decode step 的分布。

28 条会补到 32，33 条会补到 40。对这 600 条记录，真实行数合计 18,208，当前捕获行数 20,080。候选仅补 28/36/44 三个档位，捕获行数变为 19,112，虚行从 1,872 减到 904。**这只是补齐行算术，不能称 GPU 时间降低 4.8%。** 权重读取、通信、无效槽掩码及 occupancy 都可能改变兑现比例。

这是成本很小的现有配置实验，调度节奏不变。必须量额外图池/工作区、实际 KV/状态容量和纯 decode 步耗时；不能用“多捕获几个图”换回显存墙。逻辑 N42 不是 GPU batch42，也不宜只测 M42。

### 3. mHC 小 batch post→pre 边界融合

代码事实：[MHCState.attn_to_mlp](../../evidence/review-47266-0929/source/srt/layers/communicator_mhc.py)先 post、再 pre。GLM 的模型接线没有调用已有的 [mhc_fused_post_pre](../../evidence/review-47266-0929/source/kernels/ops/layernorm/mhc.py)。该 helper 对 <=32 tokens 已有小 batch FMA 路径，保留 BF16 residual 写出和后面的 mixing/Sinkhorn/norm；大 batch 沿用另一分解。

推断：真实 capture M24/32 上，减少边界 kernel 与重复读写有机会；M40/48 不属于这条小 batch 快路径。先比较完整 mHC 边界，而非只测 GEMM。融合不是自动更快，原小 batch split-K tensor-core 路径也有优势。

此前 8k/16k 自写 mHC 融合更慢，这次关注已有 small-batch 路径，是不同问题。即使启用，也需验证 BF16 舍入、FP32 mixing、输出 norm 和 communication buffer 合同。

## 冲 N46：优先减少必须执行的 prefill 成本

47266 已包含 118 prefill-only attention、attention 输入 scatter、8k–16k MoE up/down tune，以及三项 KDA prefill 优化。不能把它们再算一轮新增收益。[MoE 表](../../evidence/review-47266-0929/source/srt/layers/quantization/fp8_humming_tuning.py)明确只改 8192–16384 token 范围，decode 用原表；attention scatter 也不覆盖普通 decode。这里尚有执行优化空间，但没有当前完整 decode profile 支持宣布哪个是最大热点。

第一项值得继续的工程工作是 **KDA hybrid 的 prefill graph 兼容性**。A1 运行日志明确自动关闭 prefill graph；已有 170 opt-in 代码不能当未经验证的开关。历史 T52b v1 的 TP8+scatter 输出曾错误，v2 只留下 TP1/TP2 对照；[历史说明](../../engine/docs/170-glm-bcg-prefill.md)也记录了 decode 小幅变慢。应先验证当前栈的 live metadata、padding、KDA 状态/检查点与 TP8 scatter，再测小 prefill 的完整墙钟成本和图池。小块省时可减少打断 decode 的成本及排队，但大冷块可能 GPU 计算主导，不能由小块推算 N46。

第二项是 **检查 deadline 成本模型是否跟上执行层提速**。[ax_deadline](../../evidence/review-47266-0929/source/srt/managers/ax_deadline.py)仍用 fixed=0.13s、per_token=68us，47266 load=1.05；16k 估价约 1.306s，250k 估价约 20.034s。它用于可救/无望分层、producer 保留和 131 风险判断。旧的 118 测速快于这个估价，但不是当前混合负载的校准，不能直接把 per_token 改成另一常数。

最小验证是用当前真实 batch/context/cold-warm 成本对照决策日志，找“模型判无望、实际执行仍能赶上”的请求。确有误判再修估计或资格判断；没有证据就不改。避免把前面的排队时间重复算进服务成本，也不写请求 ID 例外。

冷块24k、短阈值1024、park2 的 N42 五组刚闭合，[收据](../../evidence/chain-night-0929/verdicts/)没有显示超出两次对照波动的 chain 净收益；尚待共同 raw ID 配对，不把汇总改善当提交依据。

## 既有候选的去重判断

- 171 投影融合：在镜像代码里但关闭。单卡投影有收益，旧 056 TP8 完整 N22 中 TPOT mean 61.976→62.802ms，未证明服务净收益；需要当前栈复核及显存账，作为备选，不称新发现。
- 172 SwiGLU：旧 Marlin 的局部/服务证据不能直接用于当前 117 Humming。Humming 的 clamped activation 已调用独立 Triton 实现；仅翻旧开关不能保证命中当前 runner。
- MTP、池320、间隔和本轮冷块改动已有结果，不重复把它们列成未测的新路线。Humming decode GEMM 的独立调优值得在新 profile 确认热点后做，不能照搬 prefill M128 配置或假定原 decode 表低效。

建议实操顺序：小成本 graph 档位对照与 KDA packed 路由数值/成本探针；有完整步收益后进入 N42 相同 ID 混合负载。mHC 小 batch 融合作为第二个执行补丁。N46 的工程主线放在 prefill graph 兼容性和成本误判核查；候选 N46 与 47266 N46 同档对照，不能与线上 N42 直接兑换。只有 chain/其他门均成立、完整性及能力通过后，才整理成正式提交。
