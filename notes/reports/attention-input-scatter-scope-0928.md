# 当前 chain-max 的 attention-TP 输入 scatter 为什么关闭

2026-09-28，Codex，按实际启动配置、已闭合实验与源码复核。

**当前不是引擎自动关掉 scatter，而是启动命令没有启用这个默认关闭的参数。旧笔记的“当前 S1 已开”指9月24日的042，不能沿用到后来的 S1/chain-max。**

## 配置与历史

- `server_args.py` 的 `enable_attn_tp_input_scattered` 默认 False。`AttnTpContext.init_context()` 的第一项条件就是这个参数；没传开关，后面不可能自动打开。47043 FINAL 与 eznc 的启动行没有此参数，eznc 实际 `server_args` 也为 False。见 [FINAL 配置](../../evidence/submission-0927-chainmax-final/submission.json)、[当前 PIN 配置](../../evidence/submission-0928-chainmax-pin/submission.json)、[eznc 启动收据](../../evidence/submission-0928-chainmax-pin/eznc-job.log)。
- 旧 `notes/codex-分析-2026-09-24.md` 第10节明确引用 **L042** 的各 rank enabled 行。该轮被称为 S1；其运行事实没有因此变错，但“当前”的适用范围已经结束。
- 后来081基线没有启用；085只加这一个flag，TP8各rank生效、冒烟12/12。共同3349条：chain超时23→21，fast238→257，overall190→195，turn7→7，TPOT均值33.23→32.31ms。它不是当前无MTP/16k/Humming组合的测量，也不是完整整档判分。
- 当时计划的087是117加119，保留小块原路径、仅大块scatter；`notes/experiments.md` 085条目记录因负载校准优先而暂缓。此后已审启动配置持续没带scatter；现有记录支持“未晋级且沿用关闭基线”，不支持“当前chain-max已经实测证明关闭更优”。

历史来源：共享 checkout `notes/experiments.md` 085段与 `evidence/L085-cap6144_scatter_n30_60m/window/analysis/20260925T211458Z-97aba312/analysis.json`。后者比较数值已核对，但顶层 scope 是旧分析工具的开放窗口描述；是否排空应看原始派发/排空收据，不能只读该标签。

## 调用路径与优化范围

固定引擎的实际路径：

```text
启动参数 enable_attn_tp_input_scattered（默认 False）
  → AttnTpContext.init_context(): allow_input_scattered
  → use_input_scattered(forward_batch)
  → Glm5NextForConditionalGeneration.forward()
       maybe_input_scattered() 设置本次 forward 的布局
  → MHCCommunicator / LayerCommunicator
       按本地 token 做 mHC/norm
       按层需要执行 all-gather / reduce-scatter
  → attention 与 MoE 所需布局
```

代码入口：[配置](../../engine/sglang/srt/server_args.py)、[条件与通信](../../engine/sglang/srt/layers/communicator.py)、[MHC 通信](../../engine/sglang/srt/layers/communicator_mhc.py)、[模型 forward](../../engine/sglang/srt/models/glm5_next.py)。这是沿真实调用点核对的图，不以文件名或机制编号推断路径。

打开还要求 CUDA/NPU、q_lora_rank、GLM的mHC条件、TP>1、非DP attention、兼容的MoE/graph路径等；逐批只用于符合条件的extend，verify与decode不走这一路。119增加最低token数门槛，主flag没开时119本身不能启用scatter。当前关闭的直接原因在第一项启动参数，不是去MTP、DCP=1或131调度触发了自动禁用。

`SGLANG_AX_INDEXER_ROW_SHARD=1` 则是114的 **DSA indexer查询行分片**，不是这个 attention 输入布局开关。114保持开启不代表mHC已经按token分片，也不能再次宣称所有indexer工作可省八倍。

分片减少的是部分重复的mHC/norm与布局相关工作，并没有让完整attention/MoE计算都除以8。源码的 `_tp_all_reduce_with_scattered_residual()` 会 reduce-scatter → 本地mHC/norm → all-gather；DSA等后继阶段仍可能需要完整token维。全块净耗时必须连通信一起量，不能仅用mHC核占比估算已兑现收益。

## 对当前开发顺序的影响

1. 在现行 chain-max16k、无MTP、同池大小与同kernel下做scatter OFF/ON单变量剖析。先看16k主块，再看1k/4k暖尾块，确认8rank实际路径和数值；固定输入比较整块GPU时间、mHC核时间与collective时间。
2. 若只有大块获益，再验证已有119阈值机制；旧1025阈值只是历史方案，不能直接当成现行最优值。运行日志的119=on只证明阈值env开启，需额外核对主flag和实际分支。
3. 据实测剩余成本决定mHC融合/通信优化优先级。现成分片若有效，应先消除重复计算，再优化剩余核。若通信抵消收益，再研究局部融合与搬运；旧实验fast变差本身不能证明“小块通信就是原因”。

本次只修正文档适用范围，不更改已打好的 PIN 包、不启动新队列任务，不承诺本地或线上 chain 净减少。
