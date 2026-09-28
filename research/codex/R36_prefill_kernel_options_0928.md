# R36：SM80 prefill 内核候选的测量边界与收尾

2026-09-28，Codex 离线归档；依据已保存的独立源码审查和原始运行记录。**本轮没有新增 chain、真实权重 TP8 或完整 prefill 成绩。** MoE scale-fold、mHC split-K、DSA scorer tile 有局部开发机读数；N 分条与整 FFN 同 CTA 只有源码算账；这些候选暂不生产化。本报告不撤销已有117/113/114机制，也不把研究脚本当成服务补丁。

工作树为 `prefill-kernels-0928`，起点 `be392558`。远端是逐文件上传环境，不能声称整个远端源码等于该 HEAD；各次加载的引擎/探针 SHA 以 raw environment、identity 和 [归档来源清单](../../notes/reports/prefill-kernels-0928-review/sources.json) 为准。本次整理未运行 GPU、未修改生产或共享索引、未删除失败证据。

| 候选 | 已测边界 | 本次主要观察 | 当前决定 |
| --- | --- | --- | --- |
| MoE FP8 固定指数前移到 scale | 单A100、实际Humming转换的合成完整MoE层，8192/16384行 | 配对graph降幅2.846% / 1.967%；已测域raw bits相同 | 收益不足，不生产化、不继续扫shape |
| W2按N条带计算后立即归并 | 源码/逻辑流量账，无GPU实验 | N128的额外activation扫描最多抵消96.875%的理想中间输出流量节省 | 不实现 |
| 同CTA W13→SiLU→W2 | 源码/SMEM及寄存器账，无GPU实验 | 当前流水加gate stash需197KiB；改流水可容纳，但大W2输出仍保留 | 不实现 |
| mHC large-M split-K | 单A100、合成完整mHC边界，8192/16384行 | 最佳被测点配对降幅9.031% / 5.123%；额外scratch | 保留研究证据，不接大M生产dispatch |
| DSA scorer tile | 114切行后的每卡1024/2048 query，65536 pooled keys | scorer+top-k配对降幅5.655% / 6.093% | 尚无完整DSA/TP8门，不接生产 |
| 上游PR筛选 | 9月28日冻结的源码/PR审查 | 已有机制、硬件不适用项和无配对证据项分开 | 不以PR标题或上游倍率决定移植 |

下文百分比均为逐轮 `(native-candidate)/native` 的中位数，可能与显示的两个中位数之比不同。各边界和输入分布不同，不能相加、乘层数，或据此预测服务档位。

## MoE：scale-fold已闭合，另外两个结构问题止于算账

`moe-scale-fold-b` 比较当前117的up+down tuning、reduce off，与完全相同配置加私有fold header。H=4096、I=256、E=289、top-k=9，BF16输入/输出、原FP8 E4M3权重、BF16 K128 group scale、FP32累加。两份header只改`mainloop_arith.cuh`：把每个解码权重的固定 `2^120` 修正前移到每轮新加载的scale寄存器；tile、raster、K累加顺序与舍入点保持不变。

| 完整MoE graph，9轮×16 replay | native median | fold median | 配对降幅 |
| --- | ---: | ---: | ---: |
| 8192行 | 3.318620ms | 3.229828ms | 2.846% |
| 16384行 | 6.563152ms | 6.421220ms | 1.967% |

两形状均9/9配对胜出，但只有一次开发机运行，无独立复跑或频率轨迹。单独W13/W2降幅为8k的4.690%/5.728%、16k的3.726%/3.044%；这些阶段数不能相加替代完整层结果，也不能与之前另一轮3.199/6.500ms绝对数直接比较。[完整raw](../../evidence/moe-scale-fold-0928/results/moe-scale-fold-b.jsonl)、[证据说明](../../evidence/moe-scale-fold-0928/README.md)。

CPU整数dyadic模型和实际SM80 `__hmul2` 均穷举254个有限FP8 patterns ×17280个sign-bit-zero BF16 scale patterns（+0到+255，含subnormal），4,389,120组合零差异；256折叠溢出，实际stage变异测试选择native。真实Humming转换后的合成fixture还通过16条W13、同activation的W2、完整eager及fresh-input graph raw-bit记录。私有include路径、编译stamp、dispatch key隔离，四个stage cubin的ID/路径/SHA不同，解析配置相同，准备后禁止JIT；scale SHA与原配置cache在末尾不变。没有SASS，不能声称测得HMUL2条数或寄存器变化。[独立review](../../notes/reports/prefill-kernels-0928-review/scale-fold-review.md)。

**数值域仍有明确边界。** 研究guard的数值`>=0`也接受未枚举的BF16 -0；权重NaN编码`0x7f/0xff`排除在证明之外，但guard只检查scales。当前有限合成权重和正scale的结论成立，不能宣称全域生产安全。首次`a`通过intrinsic检查后因脚本误写`run_activation`失败；失败收据与当时脚本保留，不计性能。候选不生产化，也不为这些边界另开实验。

W2 N分条保持全部E/M行，理论上可由scheduler N-range、全宽B/BS物理stride和compact C writer实现，不必重复整个投影。但原launcher要求连续且exact shape的B/BS/C，原reduce也把`size`同时作为输入宽度和最终输出stride；现有strided view不能直接使用。N128要32次GEMM+reduce，N64要64次；还需新输出offset/stride契约。

| N分条逻辑流量账；不是DRAM counter | 8192行 | 16384行 |
| --- | ---: | ---: |
| 全activation `[9M,256]` BF16 | 36MiB | 72MiB |
| N128单条scratch | 18MiB | 36MiB |
| 理想中最多可删的W2 output写＋reduce读 | 1152MiB | 2304MiB |
| N128额外31次activation全扫描 | 1116MiB | 2232MiB |
| N64额外63次activation全扫描 | 2268MiB | 4536MiB |

原indexed scheduler以N为内层，相邻CTA本已有A行块的L2复用机会。上表假设native只需一次HBM A扫描、分条A大量miss；该不利场景下N128重读等于最大输出节省的96.875%，N64为196.875%。16k的36MiB scratch还与72MiB activation、每条约9MiB权重争用约40MB级L2；不能只由scratch大小推断驻留。额外launch/startup、dirty输出是否在覆写前被逐出也未知。这个容量/流量账足以拒绝本轮探针，并非实测其一定变慢。

整FFN同CTA中，BM128的gate或activation各64KiB。Humming原W13 `[128,256,64]` s4共有133KiB shared，加gate stash达197KiB，超过源码SM80门163KiB；s3加stash为164KiB，s2为131KiB才可容纳。一次N512又需至少65536个FP32 accumulator registers。可行的重写需顺序处理gate/up、精确保留BF16舍入、resident-A loader和SMEM布局，再逐个算32个W2 N128块；静态shared资源限制一CTA/SM，失去当前W2两CTA/SM的空间。完整W2 FP32 tile `[128,4096]` 为2MiB，无法一次片内保留。

即使全部实现，最多删除gate/up及activation的一次写读合计216/432MiB，释放workspace108/216MiB；W2 output写＋reduce读1152/2304MiB仍在。前者只占上述三种中间张量逻辑流量的15.79%，不是时间加速上限；SiLU工作也只是移动，不能将旧activation时间再全额相加。资源和N并行损失可抵消流量节省，因此不实现。完整源码行依据及CPU容量计算见 [MoE架构审计](../../notes/reports/prefill-kernels-0928-review/moe-architecture-audit.md)。

## mHC：最佳split-K点只有局部收益

`mhc-splitk-a`在单A100上测BF16 residual、hc=4、H=4096、FP32 Fn `[24,16384]`。边界包括`post → prenorm GEMM/square-sum → mix/Sinkhorn/weighted sum/fused RMSNorm`；原生比较分支的`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`与相关Pod模板和开发runner相符，不能仅看通用环境默认值判断native分支。候选复用原有小M split-K stage0并直接喂同一个finalizer，未接入大M生产dispatch。[raw](../../evidence/prefill-kernels-0928/mhc-splitk-a.jsonl)、[identity](../../evidence/prefill-kernels-0928/mhc-splitk-a.identity.json)。

| 7轮×16 graph replay，`split4_m32_k128` | native median | candidate median | 配对降幅 | native→candidate scratch |
| --- | ---: | ---: | ---: | ---: |
| 8192行 | 0.844414ms | 0.769278ms | 9.031% | 0.781→4.125MiB |
| 16384行 | 1.579682ms | 1.499028ms | 5.123% | 1.562→8.250MiB |

这是6个候选中最佳被测点，其他点不一致获益，不能推广成“split-K普遍更快”。raw telemetry显示8k前几轮时钟由1155/1245爬到1410MHz，因此配对随机顺序和保留全部轮次很重要；没有独立整模型复跑。split-K改变FP32求和分组，不以native逐位相同作为结论。独立FP64公式参考只抽32行、使用相对L2；最佳点norm输出误差8k约`5.60e-4`、16k约`5.40e-4`，fresh-input graph也通过同类门，尚非全行真实权重/模型质量证明。

运行exit0，但stderr保留TileLang“data race”警告。独立review检查六种生成CUDA并枚举实际global partial-store地址，M32/M64的写地址均唯一，支持这些已生成global stores上的警告为lowering后的误报；它不等于所有shared/pipeline操作、版本或shape都race-free，也未做race sanitizer。本轮保留研究，不屏蔽检查器、不接生产。源/生成store推理见 [mHC review](../../notes/reports/prefill-kernels-0928-review/mhc-splitk-review.md)，原 [stderr](../../evidence/prefill-kernels-0928/mhc-splitk-a.stderr)完整保留。

## DSA：scorer小幅改善，完整注意力门仍缺失

`indexer-tiles-b`用实际113 FP8 unpack/scorer及pooled top-k，在114切行后的每卡1024/2048 query上测65536 pooled keys、32 heads×128。只有在整块8k/16k均匀TP8切行时这些query行数才对应；pool-size4意味着约262k历史token，是长上下文尾块替身，不是冷8k/16k prefill。测量不含query投影/旋转、KPool gather、cache准备、TP collectives和下游sparse attention。[raw](../../evidence/prefill-kernels-0928/indexer-tiles-b.jsonl)。

| 7轮×8 graph replay | native median | `[BQ2,BK128,GROUP16,LOOP16,warps4]` | 配对降幅 |
| --- | ---: | ---: | ---: |
| 1024 query，unpack+scorer | 4.189112ms | 3.922160ms | 6.354% |
| 1024 query，再含top-k | 4.717396ms | 4.451068ms | 5.655% |
| 2048 query，unpack+scorer | 8.116508ms | 7.564228ms | 6.800% |
| 2048 query，再含top-k | 9.098452ms | 8.544172ms | 6.093% |

native参数为`[2,128,32,4,4]`；winner只改GROUP/LOOP，raw报告同为166 registers、0 spills、49152B shared。六种BQ2配置在两形状的全部score原始FP32 bits和排序后的selected-token multiset相同。fresh replay实际更新Q/K FP8 bytes、weight、history end及tail长度，12条fresh记录全部过门。

首次`indexer-tiles-a`的exact index顺序门连native自身都拒绝，未形成timing；其中BQ4/8又改变score bits，按该数值门排除。`b`先证实native重复top-k顺序会变、selected multiset不变，再使用score bits＋排序multiset门。原top-k使用atomic reservations，顺序无保证；集合相同仍不能证明下游sparse-attention累加逐位相同，也未检验真实权重TP8输出。失败 [a raw](../../evidence/prefill-kernels-0928/indexer-tiles-a.jsonl)和 [stderr](../../evidence/prefill-kernels-0928/indexer-tiles-a.stderr)保留。

单个logits就有256/512MiB；实验同时保留六组graph/score和reference，raw peak allocated为2,336,829,440 / 4,631,453,696 bytes，不是单次服务调用的增量。早期 [方法review](../../notes/reports/prefill-kernels-0928-review/probe-review.md)提出fresh Q/K与内存读数缺失，这两项已由 [b源码快照](../../evidence/prefill-kernels-0928/indexer-tiles-b.source.py) 的132–134、179–184行及raw补齐；保留原审查快照以交代修正。仍因局部收益和完整链路数值/容量门缺失而暂不生产化。

## 上游筛选：只保留适用机制，不搬上游倍率

以下是9月28日 [Sol PR审计快照](../../notes/reports/prefill-kernels-0928-review/sol-pr-audit.md)的筛选结果，本次离线整理未重新查询GitHub状态。上游硬件、backend与测量范围均不可换成本栈成绩。

| PR / 机制 | 本栈适用性及决定 |
| --- | --- |
| [#39695](https://github.com/sgl-project/sglang/pull/39695)：KPool CPU request-slot mirror、显式`repeat_interleave(output_size)`、准备阶段overlap | 普通SM80 eager路径可移植；planner同步每forward一次，不能按11个DSA层乘。真实overlap仅compress/cache-write与quant/head-gate，并非所有query工作。上游无配对性能；本地CPU mirror原型通过合同检查但未接公共元数据，随后撤回到scratch patch，无生产遗留接口。 |
| [#40854](https://github.com/sgl-project/sglang/pull/40854)：query-row分块 | 上游主要是OOM/峰值内存修复，未显示TTFT改善；可另作为容量修复审查，不列本轮加速结果。 |
| [#38469](https://github.com/sgl-project/sglang/pull/38469)：按请求切K列再切Q | 审查时尚未完成且超预算时才走；公开warm wall未改善。若改变默认路径还需重新核本地114行shard和绝对索引映射，暂缓。 |
| [#40461](https://github.com/sgl-project/sglang/pull/40461)：LiteTopK | 审查实现面向`dsa_indexer.py`和SM90 WGMMA；本模型实际KPool、硬件SM80。不能直接移植或引用H100倍率。 |
| [#41400](https://github.com/sgl-project/sglang/pull/41400)、[#34299](https://github.com/sgl-project/sglang/pull/34299)、[#39816](https://github.com/sgl-project/sglang/pull/39816) | 相关kernel或已报收益分别依赖SM100/103、GB300或decode；不作为本轮SM80大块prefill的现成方案。 |

**KDA 的最终实现和证据见 [R37](R37_kda_prefill_sm80_0928.md)。** [#39688](https://github.com/sgl-project/sglang/pull/39688)同时含投影、元数据和β融合，本地171已经覆盖投影，不能重复算它及[#39350](https://github.com/sgl-project/sglang/pull/39350)/[#39248](https://github.com/sgl-project/sglang/pull/39248)的收益；[#41105](https://github.com/sgl-project/sglang/pull/41105)则涉及静态metadata、状态快照和整段breakable graph生命周期，不能等同于一个局部KDA图。CPU长度、局部图、recurrence并行度及packing+norm研究的最终实现、数值和性能归 R37；本报告不合并不同范围的收益。

## 归档与复核入口

- [review目录说明](../../notes/reports/prefill-kernels-0928-review/README.md)与 [sources.json](../../notes/reports/prefill-kernels-0928-review/sources.json)记录原scratch审计SHA、归档副本SHA、三份Humming源码/许可证、被引用raw和探针快照SHA。复制时仅增加快照提示、改本地链接；未静默改写原技术审查。
- MoE [receipts.tar.gz](../../evidence/moe-scale-fold-0928/receipts.tar.gz)为331161B，SHA256 `21354aaca4f31ecaa1785d6597d244e78fe6799cf96180b273e3c8edd5ef8d09`；156项内部manifest已全部核验，另一个tar成员为manifest自身。native/fold headers、五份cubin源码/命令/签名和a/b原始收据都在。
- mHC [source](../../evidence/prefill-kernels-0928/mhc-splitk-a.source.py) SHA256 `e52a6018a7a676844506331f051f6c4353f7611336b0c5a42db196f9c79afae5`，生产源SHA `29de652497a4524f69a0d2b13c09ed84cac25de44791161a3face8dff07ba25c`；DSA b [source](../../evidence/prefill-kernels-0928/indexer-tiles-b.source.py) SHA `f89230298c752296ff7ccfd81b61b49ede3b95c4d4ffd8b0c4d955987df59ce2`，scorer源SHA `2c6302f7e9c5403d973bf2ee2bf6834926bb399679d802ef3ffaba39c23822d3`，均与raw environment核对。
- mHC、DSA的jsonl保留每轮次、median、配置、数值、telemetry或memory，不只保留summary；退出码、stderr和起止时间均在相邻同名前缀文件。文档链接和归档哈希做离线检查，不新增GPU结论。

本轮局部测量支持“有小幅执行层机会”，尚不支持“完整prefill或chain明显改善”。因此收束上述候选，不继续tile扫描、不据此修改服务配置。
