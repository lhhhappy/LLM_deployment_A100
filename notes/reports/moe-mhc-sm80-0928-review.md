# SM80 MoE up+down 候选独立审查

**chain 尚无结果，不能据本报告采用线上配置或声称 chain 改善。** 2026-09-28，Codex `kernel_correctness_review`。状态：已完成单卡合成负载的独立代码与证据审查；没有发现阻止 `up+down` 继续进行真实权重、TP8 验证的正确性问题。本报告没有运行新的 GPU 任务，没有修改执行代码或提交。

主要候选为 117 加 `SGLANG_AX_SM80_MOE_UP_TUNE=1`、`SGLANG_AX_SM80_MOE_DOWN_TUNE=1`，归并开关保持关闭。三个开关仍默认关闭。这个选择依据完整层组合实测，不能把各独立 kernel 的收益相加。

审查入口是 [D 摘要](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-d.summary.json)和 [E 摘要](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-e.summary.json)，结论逐项回查了 [D 原始记录](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-d.jsonl)、[E 原始记录](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-e.jsonl)、两份 [D manifest](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-d.manifest.json)/[E manifest](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-e.manifest.json)及其中封存的执行源码。

**证据身份与完整性（OBSERVED）。** D/E 均为 A100-SXM4-80GB、SM80、108 SM；PyTorch 2.13.0+cu130、CUDA 13.0、Triton 3.7.1、Humming 0.1.12。两次运行各有 123 条合法 JSON 记录、唯一末尾 `complete`、exit 0、空 stderr。两份 manifest 合计 36 个文件的 SHA256 和字节数全部匹配；每份摘要的 83 条记录均与 raw 中相应种类的记录逐条同序一致。所有时间中位数、逐轮配对收益及 bootstrap 区间均从 `timing_round` 重算，与封存 probe 的 `paired_summary` 结果完全一致。

| 身份 | SHA256 |
| --- | --- |
| D raw | `94b2b95e504f491c9afefabf913c8eed037b417e255919f724ff8f1381b8b652` |
| E raw | `cb8a807550e4a290a3089e080955185d25e1fa0ad351de04a70b46c0c9c8934d` |
| benchmark | `6eb1770b5c2617ca8436282d9515e5ca54806bbf75b9386f774ea6789123f778` |
| FP32 reference fixture | `9f3160fcbbd39fb5f7df8f9545704078259594dc609f6039faba07a61f05f78f` |
| production fp8 method | `155e916fafd86a96bebb18e12c79add42a2493231c7e7cf341726d574c83333e` |
| production tuning helper | `ef1d15fb6c9dd34f96a2d379071f2d48ed622c23100f7ba65283068571a4b008` |
| production reduce | `82766f32d74390c4195a41f0b218a33e60568bd8e3c277bd5c6cee39ff6e84ab` |
| production runner | `da8b5d99a6a07205e6ca8318821ea5e2938914288bacd8d3adbbdf0fee043a04` |
| Humming 56 个 `.cuh` 文件聚合 | `1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102` |

上述六个本仓库执行/验证源码的当前 worktree 内容均与 raw 内 SHA 及封存源码相同。Humming `layer.py`、`forward.py` 的封存内容也分别匹配 raw 的 SHA；56 个 `.cuh` 的聚合 SHA 用本地原始 Humming 源码独立重算一致，确认没有把 research epilogue 头文件混入本次候选。远端副本的 `git_head` 为 null，因此这里按完整文件哈希确定身份，不把本地分支名当作远端提交证明。

**生产路径（代码审查与集成实测）。** [fp8 method](../../engine/sglang/srt/layers/quantization/fp8_humming_moe.py)只在 117 indexed、H4096/I256/E289/TK9、BF16 参数、对应投影的 BF16/FP8/BF16 及 BF16 K128/N0 scale 元数据、显式权重设备 SM80 下考虑修改。[helper](../../engine/sglang/srt/layers/quantization/fp8_humming_tuning.py)还要求完整 native 配置相等，含 108 SM、FP32 accumulation；tuple/list 仅对两个 shape 字段归一化，未知键不会被丢弃。输入表不被修改，JSON 与表同步更新，范围为 8192–16384 tokens（routed rows=`tokens*9`，区间下界排除、上界包含）。W13 仅关闭 stream-K；W2 仅把 N256/stages4/CTA1 改为 N128/stages3/CTA2；M/K 与共享 expert block 索引保持不变。

此前独立复核的 [production-c 原始记录](../../evidence/mhc-moe-sm80-0928/moe-up-down-production-c.jsonl)使用同一 method/helper SHA：四个真实 off/down/up/both layer 与缓存独立；48 项 effective table、40 项实际 forwarded table 覆盖 4096/8191/8192/8193/16384/16385；四个 core 的第一个 1-token warmup 后禁止调用 `Compiler.compile`，后续 warmup 与真实 prefill 均通过。26 个 early fallback、两个投影各自的 wrong-SM/unknown-config/独立 metadata 检查通过；15 个 converted tensor 比较及 18 个同实际 activation/alignment 的 W2 原始 BF16 位比较一致，原表始终不变。另有 5 项 CPU 表测试、独立抽取实际 method 的 72 个 gate/boundary 检查，以及 benchmark 七臂的 42 个配置/恢复检查通过。

**数值语义与独立参考（OBSERVED）。** Humming native stream-K 会把各 slice 的 FP32 accumulator 先转 BF16，再以 BF16 加法/原子加合并；关闭 stream-K 改变分片、累加顺序和中间舍入。因此 W13 以及完整层不作 bit-exact 声明。W2 和归并的直接 raw-bit 身份检查不能替代 W13 的独立参考。

封存 [reference fixture](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-e.source-reference_helper.py)从原合成 FP8 checkpoint 和原 FP32 block scales 解量化，使用 FP32 matmul，TF32 关闭；包含 gate 上限 10、up 区间 [-10,10]、SiLU、W2、八个 routed expert 加一个 shared expert、top-k weights 和一次 routed scale 2.5。参考不读取候选中间结果或调优后的权重。每个 shape/input 检查 16 个分布于首尾及内部的 token，覆盖这些 token 的全部 4096 个输出列；完整输出另检查 finite。这里的“完整层参考”指整个 MoE 计算链，并非对全部 8192/16384 token 计算 FP32 oracle。

所有七臂、D/E 两个尺寸合计 28 条 `layer_numerics` 与 56 条 `graph_numerics` 均通过原有硬门：参考相对 L2 < 0.02，且不超过同输入 native 参考误差的 `1.05x + 1e-4`；全量 finite。native 的重复波动只校准 graph/eager 一致性，不替代这两道独立参考门。

| 运行/输入 | Tokens | 初始 native 参考 L2 | 初始 up+down 参考 L2 | 两次 fresh graph/eager 的 up+down 最大参考 L2 |
| --- | ---: | ---: | ---: | ---: |
| D uniform，x_scale=1 | 8192 | 0.00597352 | 0.00581809 | 0.00576741 |
| D uniform，x_scale=1 | 16384 | 0.00579574 | 0.00567137 | 0.00579545 |
| E skew，x_scale=4 | 8192 | 0.00568630 | 0.00564400 | 0.00576305 |
| E skew，x_scale=4 | 16384 | 0.00596887 | 0.00591628 | 0.00585011 |

E 的 [launch](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-e.launch.json)、环境记录及实际 `case.arm_configs` 一致：8192/16384 均启用两个 production helper，无 research override。源码实际将 `x_scale=4` 传入 BF16 输入生成，skew 实际改写 routed top-k IDs；首 32 个 expert 承接初始输入的 99.539%/99.564% routed assignments，而 D 对应约 11%。执行路径与参考均保留 clamp=10，故本次放大输入、集中路由的 clamp-heavy 场景确实覆盖新开关的 active range。没有记录 clipping 元素比例或本轮 no-clamp 负对照，不能声称已量化实际饱和比例。

**CUDA graph（OBSERVED）。** 每臂单独 capture/pool，串行 replay；两个 trial 把新 hidden states、top-k IDs、weights、logits复制到被捕获的地址。源码硬断言 hidden states 与 routes 改变；raw 记录了新 seed、独立 oracle 及 stale 输出拒绝。`up_down` 的八个 fresh trial，graph replay 与 eager 的全输出相对 L2 均为 0；这是数值差异为零，不是原始位身份断言。旧 capture 输出对新参考的 L2 为 1.346–1.432，远高于 stale 拒绝门 0.1。其余六臂也通过全部 fresh/finite/reference 门。native graph/eager envelope 最大约 0.00236，没有用于放宽独立 oracle 门。

**作为主要候选的依据（OBSERVED / INFERRED）。** 以下是同一 case 内九轮随机臂顺序、每轮 16 次 replay 的完整层 graph 实测；时间和区间已从 raw 重算。D/E 在不同 GPU 同时运行，SM 时钟随轮次变化，不比较两张 GPU 的绝对耗时，也不外推 prefill 或服务收益。

| 运行 | Tokens | native 中位 ms | up+down 中位 ms | 配对收益中位数 | 配对 bootstrap 95% 区间 |
| --- | ---: | ---: | ---: | ---: | --- |
| D | 8192 | 3.481870 | 3.199162 | 8.16% | 7.53%–9.57% |
| D | 16384 | 6.916238 | 6.499830 | 6.12% | 5.80%–6.19% |
| E | 8192 | 4.065394 | 3.749558 | 8.12% | 7.36%–8.97% |
| E | 16384 | 7.395510 | 6.941980 | 6.15% | 5.90%–6.99% |

本次 eager 也观察到约 6.26%–8.64% 的配对收益。加入 reduce 的 `all` 通过正确性门，但相对 `up_down` 的小幅差异方向/逐轮胜负不一致，D/E 没有证明叠加 reduce 后获得一致的额外收益。因此以 up+down 作为主要后续候选合理；不是判定 reduce 错误，也不排除它在其他负载有效。

native/up+down/all 的 eager transient 分别在 8k 为 640.427 MiB、16k 为 1280.711 MiB，完全相同；graph retained 与 capture peak 也相同。不能声称工作区或 KV 容量节省；所有独立 graph pool 同时常驻的总量也不是单臂显存成本。

**已知边界与未验证项。**

- 已证实：当前哈希代码的默认关闭与独立 gate/cache 行为、六个边界尺寸的真实生产 dispatch、首次 1-token warmup 后复用已初始化 Humming kernel、W2 同输入 raw-bit 身份，以及 D/E active-range 完整层抽样 FP32 和 fresh graph 检查。
- 尚未验证：真实模型 checkpoint、真实请求的激活/路由分布、TP8 通信与跨卡归并、全模型累计误差、服务启动与正式 CUDA graph 集成、同 ID chain 超标和 TPOT p95。以上均不能从单卡合成层收益推导。
- 数值覆盖限于 checkpoint seed 0、每个运行一个初始输入 seed 加两个 fresh trial、两个输入尺度及两种路由模式；没有逐 token 全量 FP32 oracle、穷举 active range 的每个 M、专门的极端抵消/非有限输入测试。8193 等边界已有生产 wiring/finite/W2 身份检查，D/E 的完整层独立 oracle 仅在 8192/16384。
- D/E graph 独立分配但串行 replay；没有验证多个完整 MoE graph 的并发 replay。调用方仍须遵守原 Humming 的当前 CUDA device/stream 契约；本轮每进程只暴露一张 GPU。
- 117 现有方法拒绝 EP，优化不扩展 expert-map/EP、其他模型尺寸、SM 或 native 配置支持范围。开关按启动时读取，不能把已缓存调优表的 core 临时关 flag 当作独立 native 基线。
- research epilogue 原型仍不具备可部署 ABI，未纳入候选；其 M256 同步修正 sanity 通过，不给大尺寸正确性、速度或生产可用性预支结论。
