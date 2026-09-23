# T48 / W22 原始证据索引

任务限定 2×A100 开发机算子/随机小权重；没有完整 checkpoint、8 卡模型、服务或评分测量。源码路径审计与 INFERRED 容量/TPOT 模型见 [R17](../../research/codex/R17_nextn_sm80.md)，补丁和启动约束见 [160 说明](../../patches/160-nextn-sm80.md)。当时的用例 ID 与通过范围以下表及原始证据为准；旧测试总表已清理。

## 最终证据与通过范围

| 用例 | 日志 | 范围 |
|---|---|---|
| P160-01 | `cpu.log`、`gpu/resolve_v4.log` | 10项CPU：作用域、拒绝错误topk/path、接线和精确加权统计；开发机真实ServerArgs.resolve_once从NEXTN/fa3解析到EAGLE/tilelang，保留MR32；draft ModelConfig去KDA、单层架构、width2051、FP8 ignored-layer mapper。无模型构建/权重加载。 |
| P160-02 | `gpu/kda_final.log` | B/T/H=(1/2/8),(1/4/8),(6/4/8),(6/6/8),(1/4/64)，D128；真实conv+safe-gate recurrence与fused verify对照；所有接受长度的SSM/conv active/tracking回写，masked destination、单token续算；verify图3次动态输入重放，commit图3次重放；SSM与conv的active/tracking及masked槽均断言。 |
| P160-03 | `gpu/kpool_final.log`、`dsa_v1.log`、`indexer_v1.log`、`sharing_final.log` | kpool B6/T2,4,6，多pool/边界/partial/pad，软件FP8缓存与独立torch参考一致；DSA BF16 sparse M1/6/24/36、H8/D512/topk2051；112 verify B6×4与113 prefill64×2048/H32；真实IndexTopKShareState seed选择、publish/carry/finally清理。各自graph。 |
| P160-04 | `gpu/small_final.log`、`acceptance_v3.log`、`mhc_v1.log`、`marlin_v2.log`、`marlin_final.log` | EH norm H4096、draft argmax vocab154880；greedy与target-only采样（单热点概率夹具）、accept长度1–4、256边界commit prologue；普通topk2048及真实kpool topk512扩展；mHC4 pre/post M6/24；111 FP8 MoE E33/K4096/N256/top8及clip=10补测，原生dense FP8 Marlin投影；graph。 |
| P160-05 | `stack_receipt.json`、`full_stack_apply.log`、`gpu/remote_source_hashes.json`、`summary.json` | 12补丁fuzz0；确定再生成；3624源码+8工具py_compile；应用树=候选；整栈reverse逐字节还原；base未改；GPU实际源码/最终脚本与交付hash一致；两shell脚本bash -n。 |
| P160-06 | 尚无服务证据 | **todo**：实际checkpoint/draft weight loader、TP8 collective、三完整graph runner、overlap/metadata plan stream、cache压力/abort/retract/flush、输出与能力、接受长度与全部SLO。准备脚本没有执行。 |

精确数值摘要、文件哈希及通过标志见`summary.json`。随机激活/小权重不代表真实模型激活或能力。MoE仅缩小专家数为33（真实模型288 routed+1 shared），保留TP8每专家K4096/N256/top8；没有覆盖真实router分布或完整专家权重/显存。Marlin微秒/毫秒日志含首次编译和开发机环境影响，不用作TPOT外推。

## 环境与历史实验来源

`gpu/environment.json`记录Python、torch/Triton/TileLang、驱动、CUDA toolkit/compat、模型config与测试脚本hash。`gpu/remote_source_hashes.json`是完整候选文件清单。显式工作目录为`/sjtu/linhang/arena/code/T48`、`runs/T48`、`cache/T48`；早期C++ JIT隐式根盘缓存偏差与纠正见下。z3-solver4.15.4.0仅装入`cache/T48/deps`；没有全局安装。两个开发GPU只运行独立单卡算子，无TP进程组。

当时先打包、全栈校验和参数策略 CPU 测试，再把候选源码及 runner 传到开发机 `code/T48`。开发机只放 config，不放模型权重；按 resolve、KDA、小算子、kpool、DSA、acceptance、indexer、mHC、sharing 和 Marlin 分组测试。原一次性打包、runner 与策略脚本已清理，`dev_b160_mtp_n6.sh` 从未入队，**不能按历史命令直接运行**。保留的 GPU 数值测试位于 `tests/gpu/test_mtp_sm80_160.py`；本目录的日志与收据说明当时实际覆盖范围。

算子脚本的bootstrap仅跳过服务包初始化，并提供平台/分布式组/分配策略查询。被测数值kernel来自候选，没有替换运算。mHC禁DeepGEMM；DSA BF16路径的dtype谓词固定false。真实参数解析、ModelConfig/mapper及Marlin走完整正常导入。算子图不包含完整模型runner、TP通信、真实调度metadata或采样分布等价证明。普通BF16 embedding/head/EH投影/BMM/RMSNorm、完整router分布、全模型MLA cache写入与prefill KDA未在本轮逐一独立重测；已有110–113/F58/F59/T45证据仅作背景，不算本轮全覆盖。

## 失败史与限制（保留原始日志）

- `deps*.log`：原环境缺libz3.so.4.15；两次pip因开发机失效代理失败，随后限定T48目录且清代理安装成功。
- `small_v1…v4.log`：早期夹具的JIT helper替身、nvcc/ninja/CUDA兼容路径配置问题；最终调用真实JIT helper，`small_final.log`通过。不是修改计算kernel绕过错误。
- `kpool_v*.log`：早期夹具ring容量/调用参数不符，以及独立参考漏掉Hadamard前的BF16舍入；实际路径是压缩结果BF16→旋转→BF16→FP8，修正参考后全部一致。保留失败输出，不把错误参考当作生产缺陷。
- `acceptance_v1/v2.log`：通用AOT topk仅接受2048，pool实际调用另一个512路JIT kernel；原始unsorted topk次序不可逐元素比较，最终比较集合并验证pool扩展/tail。没有改变生产排序。
- `resolve_v1.log`未调用resolve_once；`resolve_final.log`未配置CUDA_HOME；`resolve_v3.log`误用不存在的模型目录。最终`resolve_v4.log`用T48 config-only目录通过。
- `marlin_v1.log`缺ninja路径，v2已用真实nvcc首次编译成功。`marlin_final.log`加clip=10与dense投影覆盖。
- `mhc_v1.log`保留TileLang对既有prenorm kernel的data-race分析警告；随机数值与graph通过不消除静态警告，也不能据此保证所有输入。L2需真实target输出检查。
- `sgl_kernel`是开发机已有AOT包，其安装元数据/版本可能不同于实际L3；环境收据保留路径/版本文件，L2需确认。没有用开发机依赖改写底包或打镜像。

统计口径：`accepted_tokens_including_target`是引擎spec接受计数（含target保证token），不是HTTP实际输出token数；EOS/stop截断可能使两者不同。meta_info仍由原服务逻辑报告。

## 已纠正的目录约束偏差

真实C++ JIT使用`SGLANG_JIT_CACHE_DIR`，而初版runner只设`SGLANG_CACHE_DIR`，因此7个本轮JIT build目录意外写到`/root/.cache/sglang/jit/sm80/`。发现后停掉自有dense编译进程树；按build.ninja/cuda.cu中的T48绝对源码路径确认归属，将这些build目录迁入`arena/cache/T48/sglang/jit/sm80/`，不移动/删除其他任务缓存。迁移清单见`gpu/cache_relocation.json`。最终runner显式设正确变量，并从arena缓存重新完成dense验证；被中断日志另保留。不能把早期运行表述成始终只写arena。
