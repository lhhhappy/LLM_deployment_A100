# 112 — A100 DSA indexer 融合kernel（T43 / W17）

状态：T43交付；A100算子与完整补丁栈验证通过，待Claude交叉审阅及L2集成，未加入 RELEASE。

## 问题与实现

110 的两个 torch 入口在 F63 profile 中占主要 GPU 时间。112 保留原 dispatch、metadata 和缓存写入，仅替换两个 logits 入口；代码以 `build/base_exact/sglang + 000→101→105→110→111` 为基线。

- decode：每个 `(B*N行, 64-token页)` 一个 Triton program。在 GPU 上读 context length，只有有效页读取 query/KV；无效输出写0。通过 block_table 读 packed KV，软件解码 uint8 e4m3fn，在 A100 bf16 tensor core 上做点积并融合 relu、头权重、归约和key scale。
- prefill：将 query 的全部头放入同一MMA tile，H32时每个tile计算2 query×64 key，4warps。`clean_logits=True` 时跳过与所有query区间无交集的tile；边界逐元素写 `-inf`。不生成 `[nq,nk,H]` 中间张量。
- 现有 tilelang `fp8_paged_mqa_logits_kernel` 使用 FP8 GEMM，wrapper只接收N=1，而且不处理110的全部mask契约。Triton版本可直接复用110的uint8接口并覆盖N>1；本补丁未改现有tilelang kernel。

## 必须保留的110语义

| 项目 | 112行为 |
|---|---|
| 点积精度 | 软件fp8转bf16，MMA fp32累积；**每头结果先转bf16再转fp32**，对应110 `torch.mm/bmm(...).float()` 的bf16输出舍入 |
| 头归约 | relu×fp32权重后求和，最后乘key scale；fp32加法顺序允许微小差异，不融合epilogue FMA |
| packed KV | 每页先存64×128字节fp8，再存64×4字节f32 scale；不是逐token的132字节交错布局 |
| decode query | `[B,N,H,128]` 或 `[B,H,128]`；输出 `[B*N,max_context_len]` |
| context_lens | `[B]`、`[B,1]`或`[B,N]`；N>1共享或独立长度，不在host读取值 |
| decode clean_logits | 110忽略该参数；112同样忽略，超出context或表宽的位置都是0 |
| block_table负数 | 和110的`clamp(min=0)`一致，读物理页0；不是跳过负页 |
| prefill clean=True | 区间外全为`-inf`，含空/反向区间 |
| prefill clean=False | **全宽计算**，包括`[ks,ke)`之外；110确实有这个行为，不能为剪枝改变它 |
| 限制 | `indices is None`、`max_seqlen_k==0`；模型head_dim128，页64；支持非连续的query/weights/长度/表，fp8最后一维需能按字节view |
| graph | 所有data-dependent判断均在GPU；grid只依赖shape；所有输出每次重写，支持重放时长度增长/缩短/清零 |

正常路径仅分配最终fp32 logits；元数据reshape/view不转换整段KV。110的原fallback与环境分发开关保持原有作用：`SGLANG_AX_SM80_INDEXER=0`指向原DeepGEMM，在sm80上不可用，不应作为112回退方式。回退应在代码副本反向应用112，恢复110。

## 复现

本地：

```bash
python3 scripts/make_112.py
python3 scripts/verify_112.py
```

生成器的kernel源为 `scripts/kernels/sm80_indexer_112.py`，不是只读底包。创建 `build/p112/{baseline,candidate}`，补丁使用 `patch -p3 --fuzz=0`。verify核验确定性、完整000→101→105→110→111→112→120→130栈、编译、反向逐字节还原与底包未改。

开发机（只在授权目录，先检查占用；按顺序等待每条完成）：

```bash
ssh GPU 'source /sjtu/linhang/arena/env.sh && cd /sjtu/linhang/arena && nvidia-smi && mkdir -p code/T43 runs/T43'
scp scripts/kernels/sm80_indexer_112.py scripts/test_sm80_indexer_112.py scripts/run_sm80_indexer_112.sh build/p110/sm80_deep_gemm.py GPU:/sjtu/linhang/arena/code/T43/
ssh GPU 'cd /sjtu/linhang/arena/code/T43 && nohup bash run_sm80_indexer_112.sh --mode all > /sjtu/linhang/arena/runs/T43/all.log 2>&1 < /dev/null & wait'
scp GPU:/sjtu/linhang/arena/runs/T43/all.log evidence/T43/
```

可分别用 `--mode smoke/numeric/graph/bench`。测试直接加载**未修改的110**作为oracle；软件解码检查所有256种编码。测试内的同步/`.item()`只用于对照和计时，不在生产入口里。

## 验证与数据

**VERIFIED，P112-01…06**：最终 `final_all.log` 完整PASS，88组数值对照（72普通/边界/幅值/形状，12次动态graph，4个8192-query大矩阵）；fp8全256编码、空维断言另计。最大逐行相对L∞误差 **1.1185e-5**，最大逐行相对L2误差 **2.6658e-06**，最低topk集合重合率 **99.951171875%**（2048中最多差1个）。

误差定义：每行有限有效位置 `max(abs(new-old))/max(abs(old))`，零分母下限1e-12；另报相对L2。不是逐元素相对误差；近零抵消点的逐元素比值不稳定。topk取`min(2048,有限位置数)`；clean=True的-inf不当作有效key。随机种子43；标准、次正规数区域、scale30/100并截到±448，signed head weights；非真实模型激活。

4种graph：decode `[B,N]`、decode `[B]`共享长度，prefill clean=True/False，各3次更新输入重放，与eager逐bit一致。编译收据/PTX确认 `.target sm_80`、bf16 MMA、无FP8指令；代表形状paged/ragged分别128/205寄存器、0 spills、8KB共享内存。完整8补丁链fuzz0、3623文件py_compile、确定再生成、整栈反向字节还原与只读底包未改全部通过。

A100-SXM4-80GB，torch2.13.0+cu130、Triton3.7.1，系统driver535.129.03配现有CUDA13兼容库。最终测试保留torch默认 `allow_bf16_reduced_precision_reduction=True`。同输入两版CUDA event中位数；eager warm2后110采10/112采20次，prefill采3/5次；decode graph每次捕获20次调用，7次测量除20。eager计时含Python发射造成的设备空隙；graph更接近服务decode路径。编译和输入构造排除；不是8卡端到端性能。

| 场景 | key长度 | 110 ms | 112 ms | 加速 |
|---|---:|---:|---:|---:|
| Decode B6 N1 / eager | 32000 | 0.652 | 0.172 | 3.79× |
| Decode B6 N1 / CUDA graph | 32000 | 0.486 | 0.102 | 4.75× |
| Decode B6 N1 / eager | 190000 | 2.859 | 0.662 | 4.32× |
| Decode B6 N1 / CUDA graph | 190000 | 2.687 | 0.592 | 4.54× |
| Prefill 8192 / causal | 32000 | 207.927 | 96.382 | 2.16× |
| Prefill 8192 / ragged | 32000 | 207.715 | 76.112 | 2.73× |
| Prefill 8192 / causal | 190000 | 1248.861 | 622.451 | 2.01× |
| Prefill 8192 / ragged | 190000 | 1249.067 | 423.385 | 2.95× |

Prefill均clean=True；causal为长度`nk-nq+i+1`，ragged为每query不同ks/ke（含空/反向/越界边界）。190000 key输出8192×190000约6.23GB；全部行通过数值/topk门。clean=False行为与graph已测，但这里不宣称它拥有区间剪枝收益。

首版和调优原始结果全部保留。48组首轮tile扫描、后续布局/4/8/16warp比较指向小tile；最终选择bits解码、prefill BQ2/BK64/4warps、decode页64/4warps；不启用生产autotune。前两个CPU verify尝试分别发现GNU patch备份文件与未先重新生成的旧补丁，均为验证流程问题，修正后最终verify通过；布局attempt1是文件传输尚未完成导致FileNotFound，attempt2完整完成。

证据索引 `evidence/T43/README.md`；`python3 scripts/summarize_112.py` 核对最终源码/oracle/测试/补丁SHA绑定并生成 `summary.json`。

## 开放问题与适用边界

- 数据为随机激活和真实模型形状，未取得真实模型q/K/weights；不声称真实模型激活或最终输出语义通过。
- 单卡开发机的算子耗时不能换算8卡TPOT、TTFT、N@SLO；L2服务、实际MTP集成与端到端开发集由Claude审阅安排。
- cold compile不包含在微基准；服务期首次新形状编译仍需初始化预热策略。
- 必须保留输出矩阵（8192×190000约6.23GB），尚未融合topk；prefill clean=False全宽计算是110兼容代价。
