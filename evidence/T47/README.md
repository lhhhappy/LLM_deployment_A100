# T47 / W21 — 112/113 v2 去除形状特化

生产源`scripts/kernels/sm80_indexer_{112,113}.py`；同名112/113补丁重新生成，旧补丁+说明在`patches/drafts/*-v1.*`，旧源在本目录`sm80_indexer_*_v1.py`。112/113需配套更新。

- P112-01…05：`112_all.log`，原测试完全未改，110 oracle对照。88组数值、4种graph×3动态重放，全256 FP8编码/空维断言；全部通过。
- P113-01…03：`113_all.log`，原测试未改，对110及112 v2。222组数值、10种graph×3动态重放，含六个8192×32k/95k/190k大矩阵，全通过。
- P47-02：`performance.log`，同机同输入两版各自v1/v2配对。prefill六档每版七轮交替顺序CUDA event；decode B6 N1两档各版三轮交替顺序、每图20调用×7 event样本。计时包含预解码/scratch/输出分配，不含JIT/输入构造。保留所有样本，退步门≤5%。随机激活，非模型/SLO。
- P47-01：`cache.log`，专用空cache启动，Triton3.7.1原生`jit_cache_hook`与`compilation.listener`分别统计JIT内存miss、实际编译、磁盘命中。预热有限的1/16整除/其它标量类别、CLEAN两模式、连续/转置布局、共享/逐token context、fallback/fast path；随后每轴50个不重复随机NQ∈[1,16384]、NK∈[1,200000]、P∈[1,3125]、batch∈[1,64]。每组在112/113各跑prefill/decode，真实全grid/full fp32输出，共200次调用；再600→601×两CLEAN×两版共8调用。每调用断言所有三类计数为0，非时间推断。预热不使用这50个随机值。
- P47-03：`112/`、`113/`下生成/全栈收据及`verify*log`。113 verifier以完整11补丁链fuzz0应用、全部Python编译、确定生成、reverse全部逐文件SHA还原、base_exact未改。AST断言仅固定模型/tile参数为constexpr，112/113共享实现逐函数相同。
- `112/compiler/`、`113/compiler/`：代表形状实际PTX、寄存器/spill/shared及sm80 bf16 MMA证据。
- `gpu_initial.log`、最终`gpu_final_idle.log`：开发机占用与自有进程退出。只用GPU0和既有环境，全部远端工作在arena目录。

## 复现

本地先顺序运行`make_112.py`、`make_113.py`、`verify_112.py`、`verify_113.py`（均在scripts/）。把两个kernel源、原112/113测试、`build/p110/sm80_deep_gemm.py`、本目录v1源与47测试/runner传到`/sjtu/linhang/arena/code/T47/`。
开发机先`source /sjtu/linhang/arena/env.sh`、`nvidia-smi`确认空闲，再从该目录按顺序运行：

```bash
bash run_sm80_indexer_47.sh test_sm80_indexer_112.py --mode all
bash run_sm80_indexer_47.sh test_sm80_indexer_113.py --mode all
bash run_sm80_indexer_47.sh bench_sm80_indexer_47.py
T47_CACHE=new-empty-cache bash run_sm80_indexer_47.sh test_sm80_indexer_cache_47.py
bash run_sm80_indexer_47.sh inspect_sm80_indexer_112.py /sjtu/linhang/arena/runs/T47/112/compiler
bash run_sm80_indexer_47.sh inspect_sm80_indexer_113.py /sjtu/linhang/arena/runs/T47/113/compiler
```

日志重定向至`/sjtu/linhang/arena/runs/T47/`，长任务nohup。回收后`python3 scripts/summarize_47.py`核验源码/oracle/测试/补丁/证据SHA与全部门，输出`summary.json`及`performance_table.md`。T43/T44证据原样保留，旧summarize脚本针对v1，不用于v2。

范围：只证明这些kernel在固定模型/tile/dtype与已热标量/布局类别下不因精确长度重编译；不保证其它服务kernel、未热head数/布局、整服务/TP8/NEXTN或真实模型输出/SLO。150预热形状计划本轮未改。

## 最终结果（VERIFIED / F68）

全部门通过。112最大逐行相对L∞1.11846e-5、113为3.95682e-5，最低topk均99.95117%。16行性能最大退步3.38%，保留原tile。冷预热634调用：286 JIT miss、153实际编译、133磁盘命中；随后200随机调用+8个600→601回归调用三类计数全部0。286实际key均通过精确值审计。12远端源码SHA与交付一致；GPU自有进程退出，两卡4MiB/0%。

生成器/verify本轮日志与收据写112/113子目录；`verify_113.log`为加入AST参数检查前的首次打包结果，`verify_113_final.log`为最终检查。开发机原日志名`112_all_initial.log`/`113_all_initial.log`/`performance_initial.log`/`cache_initial.log`均第一次执行即通过，回收时分别命名`112_all.log`/`113_all.log`/`performance.log`/`cache.log`，没有丢弃失败数值或调参结果。原112/113数值测试、oracle和生产tile未改。
