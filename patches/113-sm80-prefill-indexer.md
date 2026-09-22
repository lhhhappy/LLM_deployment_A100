# 113 — sm80 预填充 indexer（T44 / W18）

状态：T44交付，A100算子与完整补丁栈验收通过。**独立113，叠加112；不是112 v2**。不改RELEASE，待Claude交叉审阅。

## 实现与选择

112在每个小tile里重复做软件fp8解码。113对模型形状（H32、nq≥32、nk≥1024）先用两个线性kernel把q和k各解码一次到bf16临时缓冲，再用query-major MMA计算。每个tile为2query×128key，M=2×32头；每个program保留query，顺序处理4个key tile，32个相邻query tile分组共享L2内的K。4warps、stages=1，离线确定参数，生产不autotune。

更大query tile、8warps、key-major朝向在扫描中较慢；部分组合产生寄存器spill。128key比256key在190k档更稳；短key循环及分组带来进一步收益。stages=3没有稳定改善，保留1。原型与全部扫描（包括失败/提前中止）保留在 `evidence/T44/`。

其余head数/小形状调用112原预填充实现。112的decode入口、paged kernel、软件解码函数和原prefill fallback均保留；CPU验证逐函数字节一致（fallback仅重命名）。

## 语义与内存

- uint8接口软件解码e4m3fn，A100执行bf16 MMA、fp32累积。**每头dot先舍入bf16**，再relu、乘fp32头权重、归约、乘key scale；关闭epilogue FMA，保持110。
- `clean=True` 仅在区间与tile交集非空时做点积；所有输出都写，区间外`-inf`，含空/反向/越界区间。`clean=False`仍计算全宽；不能利用ks/ke剪枝。
- 最终输出仍fp32。8192×190000输出6.226GB，额外bf16 scratch为115,748,864字节（110.39MiB），每次调用/graph capture独立分配，无跨请求缓存。
- 输入、长度和边界不读回host，grid只依赖shape；重放时q、k、scale和ks/ke变化会重新解码与计算。输入strides仍受112的fp8 byte-view约束。
- 不改调度、metadata、topk、工具、输出预算、统计或flush。回滚只反向113，恢复112。

## 复现

CPU（只写私有代码副本 `build/p113/`，不改只读底包）：

```bash
python3 scripts/make_113.py
python3 scripts/verify_113.py
```

生成器以 `build/base_exact/sglang + 000→101→105→110→111→112` 为基线，将 `scripts/kernels/sm80_indexer_113.py` 写入候选代码。验证再打113→120→130，完整9补丁fuzz0、Python编译、确定生成、全栈反向逐字节还原以及base_exact不变。

GPU开发机（先source与检查占用；等待每个运行结束后再开始下一个）：

```bash
source env.sh
ssh GPU 'source /sjtu/linhang/arena/env.sh && cd /sjtu/linhang/arena && nvidia-smi && mkdir -p code/T44 runs/T44'
scp scripts/kernels/sm80_indexer_{112,113}.py scripts/test_sm80_indexer_{112,113}.py scripts/run_sm80_indexer_113.sh scripts/inspect_sm80_indexer_113.py scripts/profile_sm80_indexer_113.py build/p110/sm80_deep_gemm.py GPU:/sjtu/linhang/arena/code/T44/
ssh GPU 'cd /sjtu/linhang/arena/code/T44; nohup bash run_sm80_indexer_113.sh test_sm80_indexer_113.py --mode all > /sjtu/linhang/arena/runs/T44/final_all.log 2>&1 < /dev/null &'
ssh GPU 'cd /sjtu/linhang/arena/code/T44 && bash run_sm80_indexer_113.sh inspect_sm80_indexer_113.py /sjtu/linhang/arena/runs/T44/compiler'
ssh GPU 'cd /sjtu/linhang/arena/code/T44 && bash run_sm80_indexer_113.sh profile_sm80_indexer_113.py /sjtu/linhang/arena/runs/T44/profile'
```

runner只使用GPU0和已有m0环境、CUDA13兼容库；不安装软件。`--mode smoke/numeric/graph/bench`可分测。最终原始日志在GPU的 `runs/T44/`，复制回 `evidence/T44/` 后 `python3 scripts/summarize_113.py` 核验源码/oracle/测试/compiler/patch SHA并生成表。

## 验证与性能

**VERIFIED（P113-01…04）**：222组数值对照全部通过，最大逐行相对L∞/L2均为3.956824e-5，最低topk集合重合率99.951171875%。10组graph（原4种分别对110/112、另2种带q/K/scale变化）合计30次动态重放均与eager逐bit相等。最终 `final_all.log` 末行PASS，`summary.json`绑定全部SHA。

| 场景（nq=8192） | nk | 112 ms | 113 ms | 112 TFLOPS | 113 TFLOPS | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| causal | 32000 | 96.854 | 15.378 | 22.2 | 139.6 | 6.30× |
| ragged | 32000 | 76.112 | 11.662 | 28.2 | 184.1 | 6.53× |
| causal | 95000 | 314.727 | 47.073 | 20.3 | 135.4 | 6.69× |
| ragged | 95000 | 227.133 | 34.326 | 28.1 | 185.7 | 6.62× |
| causal | 190000 | 622.164 | 97.047 | 20.5 | 131.4 | 6.41× |
| ragged | 190000 | 425.674 | 68.814 | 30.0 | 185.3 | 6.19× |

六档全部超过≥100等效TFLOPS目标。完整样本/有效区间pair口径在 `evidence/T44/performance_table.md` 与 `summary.json`。

**decode保持112**：函数源码逐字节一致，代表形状编译PTX SHA与T43完全相同（128寄存器、0spill、8KB shared）。本轮graph 32k：112/113为0.102520/0.102427ms，190k：0.592206/0.592515ms（差0.05%，同PTX的测量波动）。历史112数据F64/T43保持原值，不将波动称为decode优化。

**profile与编译**：190k causal的112 kernel平均624.397ms；113主kernel96.102ms，两次预解码合计0.104790ms/调用，仅占GPU时间0.11%。113主kernel162寄存器、0spill、48KB shared，实际sm80 bf16 MMA，无FP8指令。这里是torch CUDA profiler证据，不是ncu硬件counter；没有声称达到312TFLOPS峰值。9补丁全栈fuzz0、3623 Python编译、确定生成、全栈反向字节还原、base_exact未改全部通过。

P113-01/02沿用112全部普通/边界/次正规数/大幅值/strides/head数/空维/动态graph用例，分别对照未改110与112；另加新tile边界、95k和双临时缓冲重放。P113-03六个8192×{32000,95000,190000} causal/ragged全部行对照两版，逐行相对L∞<1e-2、topk(min(2048,有限有效key数))集合≥99.5%。误差定义和topk实现沿用112。

计时为A100-SXM4-80GB单卡算子CUDA event，七轮交替顺序比较112/113，每轮预热2次后测1次，取中位数；包含fp8预解码、scratch及输出分配，排除JIT和输入构造。decode另测eager与20次调用/graph。额外CUDA profiler按kernel计时与编译收据放在 `profile/`、`compiler/`。

TFLOPS采用任务指定 `2*nq*nk*32*128 / 秒 / 1e12` 全宽等效口径；clean=True的剪枝会提高该指标，不能据此宣称实际MMA峰值利用率。JSON另报有效pair比例及对应吞吐。

## 开放问题与交叉审阅

- 输入是随机激活与真实模型形状，未采集真实模型q/K/weights；完整模型准确性、NEXTN集成、TP8、TTFT/TPOT/N@SLO仍待Claude的L2验证。
- 113减少算子时间，不提供8卡端到端性能结论。scratch与graph池占用、首次新形状JIT预热需在服务内核验。
- 完整fp32输出与后续topk仍分开；未融合topk，未更改输出精度。
- Claude交叉审阅：待审；建议检查临时缓冲的服务显存、清空/增长graph和同栈112/113 A/B。本任务未起服务/队列、未操作bohr/Trisol/pod、未打镜像或提交。
