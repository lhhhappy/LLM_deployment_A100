# KDA prepare C：独立源码与raw复核

2026-09-28，Codex只读源码/离线复算；未运行GPU、未修改生产代码或其他报告。**C支持四种合成完整KDA core形状中的原型组合收益，不是生产接线验收。** 后续真实production-arm D仍需单独核对；本报告不预判D、完整模型、TP8或chain结果。

复核对象：[C raw](../../../evidence/prefill-kernels-0928/kda-prepare-c.jsonl)、[执行源码快照](../../../evidence/prefill-kernels-0928/kda-prepare-c.source.py)、[summary](../../../evidence/prefill-kernels-0928/kda-prepare-c.summary.json)、[exit](../../../evidence/prefill-kernels-0928/kda-prepare-c.exit)、[stderr](../../../evidence/prefill-kernels-0928/kda-prepare-c.stderr)，以及A/B同名前缀收据。C为A100-SXM4-80GB、Torch `2.13.0+cu130`、Triton `3.7.1`，11:18:43Z–11:19:10Z，exit0；stderr只有`tl.make_block_ptr`弃用提示，不能写成stderr为空。

## 身份与A/B/C关系

| run | 执行探针SHA-256 | 状态及用途 |
| --- | --- | --- |
| A | `3e8fd4035b126e6bb726e9c5d4940d9833fd810b819d7436a8d6d7bbd0de4e1c` | exit0；native BV32下筛选三种prepare配置，有限合成数据的早期graph结果。不是C的四臂组合/特殊V编码证明。 |
| B | `39ea5be95b2cedc43963b94624ae4874af89ac821955ad808f88645d2a0f70e7` | exit1；只有environment，无数值或timing记录，保留为编译失败。 |
| C | `9c7ca5943d529b53abf8ca488c2993dd343dfb1182375701fb5b8867746feced` | exit0；本报告采用的四臂graph/eager、V完整bits域及strided输入结果。 |

三份`.source.py`实际SHA均与各自raw environment相等。共同加载的backend SHA为`557315df0e3371338ad358282828f7e274cbee0314548455b8f7a527d46bb2b0`，可由已保存的 [backend快照](../../../evidence/prefill-kernels-0928/kda-local-graph-b.source-kda_backend.py)独立重算，不能拿之后生产修改的当前文件代替执行身份。

B的 [stderr](../../../evidence/prefill-kernels-0928/kda-prepare-b.stderr)明确是Triton编译变量类型错误：动态分支之前`x`为BF16，Q/K分支将同名`x`重绑定为FP32，而V分支仍要求BF16。B→C源码diff只把Q/K支路的FP32临时改为`xf`和`y`，保留BF16 `x`直接写V，未换tile、算法或计时方法。B没有可用性能数据，C的成功也不是吞掉该异常后的继续运行。

C raw共372条，summary的84条与raw去掉`timing_round`/`eager_timing_round`后的序列逐项完全相同。C raw SHA为`35063e72125d1d9942ef3d5360aa44a26ab141e5aad075e3b2961a4ac9c29594`；summary SHA为`8146f2b3b417c0d593f33cd37817e51897e8db0c869c503c9b620aeb51137fdc`。

## 数值门及真实覆盖范围

四种形状是M8192/M16384各一条序列，以及各三条ragged序列。三序列长度分别为`[4095,2049,2048]`、`[8191,4097,4096]`，各另有7行physical padding；padding输入注入NaN。单序列raw stride为`[3072,1]`，ragged在宽3080的envelope里切取前3072列，实际raw stride为`[3080,1]`。这两个stride在每一条eager timing raw里有记录，不只是源码注释。它模拟171投影返回split view的行间隙，**没有实际执行171投影**。[输入构造](../../../evidence/prefill-kernels-0928/kda-prepare-c.source.py#L165)。

测试调用真实`KDAAttnBackend.forward_extend`，包含packed causal conv及KDA状态更新。ragged分支包含零/非零prefix和140式两个snapshot目的槽；conv与FP32 SSM都取envelope的`[:,0]`视图，实际slot pitch大于紧凑张量。比较对象是完整conv/SSM envelope与native结果，覆盖有效槽、snapshot、未参与区域及间隙；这不是独立于native的状态oracle。backend先按逻辑长度trim，再在返回值恢复零padding。普通`has_mamba_track_mask`分支、MIXED/CP/TBO/verify等模式不在C的四形状内。[已加载backend调用链](../../../evidence/prefill-kernels-0928/kda-local-graph-b.source-kda_backend.py#L762)。

四臂及实现如下：

- `native`：原Q/K/V contiguous复制、原Q/K l2norm、BV32 recurrence。
- `prepare`：私有`prepare_qkv` BT64/8 warps/3 stages，一次从conv后的BF16 packed views写三个连续平面，Q/K融合l2norm，BV32不变。
- `state16`：prepare关闭，只通过原Autotuner的`.fn`固定BV16，保留原warps/stages/CTA参数；普通和snapshot recurrence均替换。
- `prepare_state16`：两项组合。context manager在每臂结束和异常退出时恢复原符号；未修改生产dispatch。

所有臂都强制同一个CPU-length原型为开，并预热原tensor-identity metadata cache；因此这里的`native`是相同CPU长度路径下的原kernel比较项，不是生产默认全off配置。源码断言初始普通/snapshot BV均为32，防止把已经BV16的环境误当native。[override](../../../evidence/prefill-kernels-0928/kda-prepare-c.source.py#L72)。

离线核对了两个完整16项矩阵：`4 shapes × 4 arms`的`core_numerics`全部output/conv envelope/FP32 SSM envelope raw bits为true；同样16项`fresh_graph_numerics`也全部为true，源码同时断言输出有限。fresh阶段修改raw QKV、gate、beta，并翻转活动slot映射，然后从相同初始state各跑一次native eager和各自graph。序列长度、chunk数、prefix及snapshot拓扑保持固定；C不能代替任意动态metadata重放测试。[fresh gate](../../../evidence/prefill-kernels-0928/kda-prepare-c.source.py#L241)。

独立prepare测试对两种M均比较Q/K/V的全部BF16 bits。Q/K覆盖随机、零向量和小幅值；V在前64个token写入`arange(65536).to(int16).view(64,8,128)`，恰好覆盖全部65536种BF16编码，包括signaling/quiet NaN payload、Inf、subnormal和负零。V分支保持`tl.load`所得BF16直接`tl.store`，不经过FP32转换；两条`prepare_numerics`的三平面结果全true。**完整BF16枚举只证明V搬运，不是Q/K归一化的全域证明，也不是让整个KDA core处理任意NaN的证明。** [编码构造](../../../evidence/prefill-kernels-0928/kda-prepare-c.source.py#L143)、[V直接写回](../../../evidence/prefill-kernels-0928/kda-prepare-c.source.py#L27)。

## 重算的完整core graph与eager计时

每形状/模式9轮随机臂顺序，每条样本8次调用。warmup后在计时外reset一次，计时内8次调用连续修改state；不是每次replay都恢复初始状态。两种模式都排除state reset、输入/池创建、JIT和graph capture。graph不计动态输入staging；eager的CUDA event span包含Python发射造成的GPU空隙，另记host enqueue时间，不能把二者相加。两者均不含模型投影、TP通信或全模型其他层。

下表为median ms；括号为逐轮配对降幅中位数。已从288条原始round重算32条timing/eager summaries，逐轮samples、paired gains、medians以及eager host medians均相符。

| Graph | native | prepare | BV16 | prepare+BV16 |
| --- | ---: | ---: | ---: | ---: |
| 8192，一序列 | 0.893864 | 0.814368（8.873%） | 0.848772（5.051%） | 0.770020（13.867%） |
| 8192，三序列+140 | 0.805752 | 0.726436（9.854%） | 0.769552（4.504%） | 0.693944（13.865%） |
| 16384，一序列 | 1.710808 | 1.542428（9.832%） | 1.624056（5.071%） | 1.458220（14.788%） |
| 16384，三序列+140 | 1.533760 | 1.366264（10.916%） | 1.469900（4.107%） | 1.300984（15.152%） |

| Eager CUDA-event span | native | prepare | BV16 | prepare+BV16 |
| --- | ---: | ---: | ---: | ---: |
| 8192，一序列 | 0.954796 | 0.871288（8.729%） | 0.913784（4.464%） | 0.829904（13.080%） |
| 8192，三序列+140 | 1.010244 | 0.936584（7.366%） | 1.008620（0.469%） | 0.929716（8.173%） |
| 16384，一序列 | 1.761044 | 1.574992（10.133%） | 1.671564（4.956%） | 1.502488（14.682%） |
| 16384，三序列+140 | 1.585884 | 1.407808（11.229%） | 1.521064（4.123%） | 1.343948（15.169%） |

graph中每个候选每形状均9/9配对胜出；eager并非如此。8k三序列combo为8/9胜，单轮配对降幅范围`-14.820%..8.963%`；该形状单BV16仅6/9胜、median0.469%，不支持把其graph降幅原样兑现到当前eager服务。8k单序列BV16也仅8/9胜。raw两种模式所有telemetry记录均为SM1410MHz、memory1593MHz，但这不消除host噪声，也不是独立复跑。16k单序列host样本存在明显前后段分布变化；host median不适合另当稳定收益承诺。组合以四臂同轮的实测为准，不能把两个单项百分比相加。

## memory口径与尚未完成的生产门

`eager_memory`是当前臂单次调用的`max_memory_allocated - starting memory_allocated`；已有graph/输入/状态等live allocations计入starting并被相减。它不是reserved/private graph pool，也不是全服务峰值或每层持久增量。依次构建的graph仍在进程中，不能把此delta乘34层推断图池容量。[计量位置](../../../evidence/prefill-kernels-0928/kda-prepare-c.source.py#L226)。

| 形状 | native peak delta，bytes | prepare / BV16 / combo，bytes |
| --- | ---: | ---: |
| 8192，一序列 | 251920896 | 三臂均251920896 |
| 8192，三序列 | 252183040 | 三臂均252183040 |
| 16384，一序列 | 503841280 | 三臂均503869952 |
| 16384，三序列 | 504103424 | 三臂均504103424 |

原型没有测出峰值显存下降；16k单序列三个候选比native多28672B（28KiB）。减少Q/K/V复制和l2norm中间流量，不能自动表述成降低实测峰值或释放KV容量。

源码/收据支持C的原型数值门和局部计时，无需因B编译错误否定C。仍待D证明的是生产flag与guard确实命中、default-off/unsupported fallback保留、真实wrapper开销与cache/JIT契约、实际生产arm下相同完整core数值/性能；C的monkeypatch和assert不能替代这些门。C也没有真实权重/完整prefill/TP8/chain证据，故本审查只签认**限定形状的原型组合结果**。
