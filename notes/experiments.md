# experiments.md — 实验台账

每个实验：编号 · 目的 · 环境（机器/镜像/commit/参数）· 命令 · 结果（附 runs 路径）· 结论 · 负责人。只追加。

| # | 状态 | 目的 | 负责 |
|---|---|---|---|
| E1 | 完成（功能基线，非性能成绩） | 本地替身（小 Kimi-Linear，KDA+MLA，SGLang v0.5.20，page 64，extra_buffer）回放开发集链，测 stock 的 cached_tokens，与 F13 模拟对账 | Codex |
| E2 | 已完成（v1.1验证；D1-02/04未过） | 三组缓存A/B完成；raw logits门失败，不能批准数值安全 | Claude（补丁）+ Codex（运行） |
| E2b | 已完成（T26；004缓存机制通过，未通过完整正确性资格） | 001+002/FCFS控制与+004对照；reminder fast uncached p95 8135→2439，143项0退步；004额外对齐major及旧D1-04未解除 | Codex main |

## E1 — Codex 接收与范围（2026-09-22 UTC）

- 本轮用户明确批准本地 2×A100 开发机验证。仅在 `/sjtu/linhang/arena/` 下创建环境、模型、代码副本和运行结果；不使用 Trisol、8 卡配额或提交。
- 基线固定 `src/sglang` commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，本地参考源码/harness 仍只读；补丁 E2 不混入本次 stock。
- 目标：随机权重小 Kimi-Linear（KDA+MLA，无 DSA）+ 原 dev tokenizer，page=64、extra_buffer；检查 `/generate`、原生 `/flush_cache`、缓存计数，回放选定真实 dev 链并与 F13/F24 的 stock token 模型逐请求对账。
- 小输出预算与 ignore_eos **仅用于随机替身功能实验**，不修改原 harness、真实比赛请求预算或候选提交。实验 TTFT 不代表完整模型性能/能力分。
- 当前状态：环境预检；尚无本轮运行结果。将追加完整命令、源码/模型指纹和 runs 路径。

### E1 环境搭建进展 1（Codex，非服务结果）

- 预检：2×A100-SXM4-80GB 均无运行进程、各 4 MiB；driver 535.129.03。根盘剩 3.1 GB，`/sjtu` 剩余约 366 GB（共享盘实时变化）。未改驱动、未占用其他项目目录。
- Conda 原配置的清华源不可达；改为 `conda create --offline --clone /sjtu/linhang/arena/env/ana -p /sjtu/linhang/arena/env/sgl -y`，成功得到独立 Python 3.12.14 环境。原 ana 未改动。
- 机器 pip 配置仍指向失效的 127.0.0.1:7890；进程级 `PIP_CONFIG_FILE=/dev/null` + 去代理解决。火山镜像缺少 `cuda-toolkit==13.0.3` 和 `sglang-kernel==0.4.7`，保留固定版本，以官方 PyPI 为缺包后备源；安装尚未完成。
- 仅解包 NVIDIA `cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb` 到 `arena/env/cuda-compat-13-0/`，通过 E1 专用 LD_LIBRARY_PATH 使用；未执行 apt/驱动安装。NVIDIA 的 [forward-compatibility 表](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) 列出 535 与 compat13.0 的支持；仍以 Torch/GPU 实测为准。
- 新脚本：`scripts/e1_env.sh`、`setup_e1_env.sh`、`make_e1_kimi_standin.py`、`launch_e1_standin.sh`、`replay_chains.py`、`test_replay_chains.py`。源码 copy 到 `arena/code/sglang-v0.5.20/`，不改本地 `src/sglang`。
- **已执行的 CPU 检查**：本地 7 项 unittest 通过；shell 语法检查通过。GPU 机 ana 环境运行 `replay_chains.py --plan-only --num-chains 3` 成功：3 条完整链、9 请求、5 个 fast_intra；选中子集 stock 预测 uncached p95=3391。这是渲染/预测结果，**不是服务命中或 TTFT**，也不代表全开发集。
- 远端日志根目录 `/sjtu/linhang/arena/runs/E1_20260922/`，保留 `conda-create.log`（失败）、`conda-clone.log`（成功）、`bootstrap.log`（镜像缺包失败）、`install-v2.log`（安装中）、`plan/`（仅预测）。

### E1 环境搭建进展 2（Codex，2026-09-22 06:34 UTC）

- **VERIFIED / GPU 最小检查通过**：独立 sgl 环境中 `torch==2.13.0+cu130`、`torch.version.cuda=13.0`，`cuda.is_available()=True`，`torch.ones(1, device="cuda")` 返回 `tensor([1.], device='cuda:0')`；识别 A100-SXM4-80GB。运行前再确认无其他 GPU 进程。仅此不足以证明整个 SGLang/JIT 路径兼容。
- 火山镜像缺少匹配版本的 cuBLAS；官方 CDN 在该机器极慢。增加阿里云 PyPI 镜像后备，下载的 `nvidia_cublas==13.1.1.3` / `sglang_kernel==0.4.7` SHA256 分别为 `37936a16db8fe4ac1f065c2139360608a543a09275cb1a1af612e08cfa065436` / `666f39de214a1558c5a98f43e6562e1032d8d08823b61f6a307830882f3241ce`。版本未降级；中止的下载日志保留为 `install-uv.log` / `install-uv2.log`，当前安装日志 `install-uv3.log`。
- nvcc/headers 来自 `cuda-toolkit[nvcc,cccl]==13.0.3`，位于独立 env 的 `site-packages/nvidia/cu13/`；仅 E1 shell 设置 CUDA_HOME。文本服务依赖仍在安装，未生成模型/启动服务/获取实际 cached_tokens。

### E1 环境搭建进展 3（Codex，2026-09-22 06:58 UTC）

- 模型已生成：`arena/models/e1-kimi-linear-4l/`，3 层 KDA + 1 层 MLA、hidden512/heads4/head_dim128、dense MLP、173,243,404 参数，目录约350MB；`model.safetensors` SHA256=`30967d99eae6f7f8a0a216464210e753724c7772fb38eb722a9cbe643e6d4618`。`E1_manifest.json` 保存形状及 tokenizer/配置/权重哈希。
- 源码 tar 上传和 GitHub shallow clone 均遇到断连。最终用 rsync 续传固定 commit 的 Git 对象包；包完整后用该仓库的 `git archive HEAD` 补齐**自有远端副本**，`git status --short` 无输出，HEAD 与本地 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` 一致。未改原 `src/sglang`。仅本地工具目录解包 rsync/libpopt，不做系统 apt install。
- SGLang v0.5.20 editable 安装完成，`SGLANG_BUILD_RUST_EXTS=none`、运行时 Python TreeCore。最终安装日志 `source-install-final.log` 使用完整 Git 元数据，**没有版本覆盖变量**；此前为了做 import 诊断的临时安装曾显式设置 0.5.20，已重新安装替换。
- 文本替身仅安装实际所需依赖，不包含 DeepEP/DeepGEMM/FA4/TokenSpeed 等未用路径和多数多模态组件；因此 `pip check` 仍报这些缺项，不能称全量官方环境。Torch2.13、tokenizers0.22.2、FlashInfer0.6.18、kernel0.4.7、numba0.65.1、cuda-tile1.6.0rc5、CutlassDSL4.6.2 等已固定；torchvision0.28.0 是 `srt.utils.common` 的直接 import 所需。
- `--help` 和 `from sglang.srt.managers.scheduler import Scheduler` 通过。CPU plan20 渲染 20条完整链/72请求/42个fast_intra，预测 stock uncached p95=4750；仍非服务结果。
- 已启动自有 tmux `arena-e1-server`，仅 GPU0、localhost:31000、page64/extra_buffer、Triton KDA+MLA、FP32 SSM，无补丁/HiCache/speculative。日志 `server-v1.log`。启动脚本增加 commit/源码无差异/GPU0空闲检查。当前启动中，**尚无 /generate 或缓存结果**。

### E1 启动排障（Codex，2026-09-22 07:11 UTC）

- `server-v1`：HTTP 入口直接 import `soundfile`，补源码固定的0.13.1。`server-v2/v3`：Kimi MLA 共用 DeepSeek forward → dsa_backend 有**导入期** `deep_gemm` 依赖，即使不运行DSA也需要 `sgl-deep-gemm==0.2.0`，已补（修正进展3的未装列表）。不启用该包的Hopper算子，也不修改源码来屏蔽导入。
- `server-v4`：权重加载成功（0.34GB），KDA池 conv0.01GB/SSM0.38GB、MLA KV0.14GB/131072tokens；实际默认缓存是 **UnifiedRadixCache + FULL/MAMBA components + Python UnifiedTreeCore**，KDA dispatcher 的 decode/extend/verify 均为 Triton。保留真实默认路径，不为迎合F13预测切换旧缓存实现。
- v4 首次 Triton autotune 后，MLP激活的 NVCC JIT 链接报 `cannot find -lcudart`：pip CUDA13 放 `lib/libcudart.so.13`，源码链接器固定查 `CUDA_HOME/lib64/-lcudart`。仅在自有 env 增加 `lib64 -> lib`、`lib/libcudart.so -> libcudart.so.13`，没有修改系统路径或SGLang源码。
- **缓存路径纠正**：该 commit 的 `SGLANG_JIT_CACHE_DIR` 不继承 `SGLANG_CACHE_DIR`，v4意外在 `/root/.cache/sglang/jit` 留下16KB失败构建元数据。现已显式设置前者为 `arena/cache/sglang/jit`；不触碰根目录其他项目/已有缓存。服务cwd和file-storage-path也固定在arena/runs。根盘仍剩3.1GB。
- `server-v5.log` 为上述环境修正后的重启，尚待接口 smoke。此前启动失败不计作功能通过。

### E1 替身配置修订（Codex，2026-09-22）

- `server-v5` 已越过上述 JIT 链接错误，但在 MLA 首次 prefill 的 `AttnTpContext.fetch_qkv_latent()` 断言失败。源码链：`kimi_linear.py:630` 直接调用共用 `DeepseekV2AttentionMLA`；`forward_mha.py:189` 在 q_lora_rank非空时要求 AttentionInputs，而该 Kimi 包装层未像 DeepSeek/GLM5 的 LayerCommunicator 那样设置它。**这不是 D1 或缓存命中的结果，也不据此断言完整 GLM5 会失败**（GLM5 有 LayerCommunicator）。
- 为保持 **stock 源码不动**，调整自有替身配置为 `q_lora_rank=null`（完整 Q 投影）；`kv_lora_rank=512`、MLA KV池、3层KDA、所有cache/scheduler参数与tokenizer不变。共用 MLA 代码对这个分支直接计算 q/kv，不依赖上述上下文。是否能完整服务仍以随后 smoke 为准。
- 新权重保存在 `models/e1-kimi-linear-4l-qfull/`（不覆盖原失败配置），173,472,652参数；权重SHA256=`16a4e0413d8a810f18666443b869c229ae70419c78e1c958aa91e59b461500eb`，配置SHA256=`030408d0e8810c3509fdecfee1694ad254ac537e5213f068b2082f7757233375`。脚本增加 `--q-lora-rank`，默认0代表null；`--q-lora-rank 128` 可复现先前失败配置。
- 新启动日志 `server-v6.log`。后续 E1/E2 对照必须使用同一 qfull 权重，不可把两种配置的时延混作补丁差异。

### E1 最终结果与 E2 交接（Codex，2026-09-22 UTC）

**VERIFIED / E1 完成**：最终服务是 stock commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，Python UnifiedTreeCore + UnifiedRadixCache（FULL/MAMBA），单张 A100；没有应用 D0/D1。`models/e1-kimi-linear-4l-qfull/` 的 3 KDA + 1 MLA 随机替身跑通；q_lora_rank=null 的配置修订见上一节。服务停止后两卡各4MiB、均0%利用率，自有 tmux 已退出，其他会话保留。根盘剩3.1GB。

| 实际运行 | 请求数 / 链数 | 请求错误 | stock 预测精确匹配 | fast_intra 实际 / stock预测未命中 p95 |
|---|---:|---:|---:|---:|
| stock_smoke3（表内不含3次接口smoke） | 9 / 3 | 0 | 8 / 9 | 3634 / 3391 |
| stock20 | 72 / 20 | 0 | 70 / 72 | 4750 / 4750 |

- **IF-07 / L2 PASS**：同一个18,678-token prompt，cold / repeat / flush后cold 的 `cached_tokens` 为 **0 / 18,624 / 0**，均生成4tokens。支持 `/generate` 与真实清缓存；不是把 HTTP 200 当作清空证明。原生 `/flush_cache` 仍返回文本，`json_contract=false`，所以 **IF-08 JSON接口合规未通过，需D0**；单worker不能验证IF-09全worker归约。
- **VERIFIED / stock20审计**：全部72请求的服务端prompt_tokens与原Renderer分词数相等，范围18,678–52,558；output_tokens全部为4（ignore_eos）；TTFT全部来自server，0个client_proxy；每条链前flush，20个链首cached_tokens均0。选择为可用多请求链按chain_id排序的前20条（整链≤65,536，不截prompt），N=1、无原gap、无跨链共享；非原cohort压测。fast_intra按原harness的链首优先分桶，共42个，而非只看phase标签。
- **VERIFIED / 原始证据**：GPU机 `/sjtu/linhang/arena/runs/E1_20260922/{stock_smoke3,stock20}/` 各含manifest、requests.jsonl、events.jsonl、summary.json；服务日志 `server-v6.log`。本地仅保存小摘要 `evidence/E1_stock/{summary.json,manifest.json,smoke_events.jsonl,model_manifest.json}`，全量逐请求记录仍在GPU机。
- **VERIFIED / 可复现版本**：两个实测均用同一个 `replay_snapshot.py`，SHA256=`ed3eb0d77fd6e9beb4988b1037f28aed58577899b45b171f3ef9df5accb1e278`，与env/launch快照一起保存在该runs根目录。W4随后给本地replay增加的`--case-file`没有混入这两次结果；E2若采用新用例，必须同时重跑相同用例的stock，不混用样本。远端与本地参考源码git状态干净，Renderer/loadgen/tokenizer哈希一致（manifest中有完整值）。
- **VERIFIED / 测试边界**：收尾重跑replay 7项 + W4工具21项，共28项CPU/mock测试通过；3个shell脚本语法检查通过，`python3 scripts/check_records.py`返回0 error / 0 warning。IF-03/04/05/12只获得部分证据，未逐SSE事件审计，也未做所有长度/时间戳入口检查，不标PASS。D1、logits、lazy、并发单partial、槽位归还、多卡、HiCache、MTP均未验证。

#### 与 F13/F24 对账的两个偏差

delta定义为actual_cached−predicted_stock_cached，两个均为负，不能用p95相等掩盖逐请求差异：

| req_id（均为biomaster:canon前缀） | prompt_tokens | 实际cached | F13 stock预测cached | delta |
|---|---:|---:|---:|---:|
| tn-7M_Qm9Vs78vbJBjiXu:llm:3 | 37362 | 33728 | 34816 | -1088 |
| KF4xNsDjdLQoOy88dN2Cu:llm:4 | 42653 | 33664 | 37376 | -3712 |

- **VERIFIED / 源码区别**：F13及replay中的预测从上一条完整prompt的LCP推branch；实际 `unified_cache/components/mamba.py:154` 用 `result.full_kv_hit_length`。`mamba.py:525` 在extra_buffer下返回`mamba_last_track_seqlen`作为可缓存长度；`unified_radix_cache.py:987`取各component最短长度，截掉token key与FULL-KV尾部。`schedule_batch.py:2957`只在branch落入当前extend时才替换该extend的end；也不等价于无条件“所有chunk末尾+一个branch”。
- **INFERRED / 最可能原因（未加运行时内部trace）**：第一条偏差链，req1的branch33728替换end后，仅该深度以内FULL-KV可复用。req2不能凭上一prompt的34868-token LCP得到34816branch；它可能记录end37056，req3在36681处分叉又不能复用该end，故仍命中33728。第二条偏差同型（33664→虚构37376）。外部计数与源码吻合，但尚未逐次观测内部树节点；不是已经证实的引擎bug。
- **交接建议**：保留F13/F24作为理想预测，后续应显式建模驻留FULL-KV/分chunk的branch选择；不要悄悄改写本次预测列，也不要把预测作为绝对测试oracle。D1收益以同配置实测A/B为准。
- **VERIFIED / 分桶复核**：直接调用原`in_ttft_gate(row,"fast_intra")`复核42个请求，冻结uncached_expected p95=2949，stock实测p95=4750、max=8989；role_conservative理想预测p95=3196，**后者尚无GPU实测**。只按phase=intra筛选会误纳入2个被回放为链首的请求，不能用该简化口径。
- **与F24全量值区分**：F13/F24全开发集模拟的fast_intra n=328、stock p95=7238；这里是所选20链中的42个fast_intra、同子集预测/实测均4750，**不是7238→4750的性能改善，也没有完成全量GPU复现**。F24全量role_conservative p95=3332、role_all=3326（2次branch冲突跳过），仅近似保留收益；这些模型值不能与E1子集或尚未进行的E2混为一谈。

#### 复现入口与注意事项

已有环境与权重，不需要重新安装或下载；先确认GPU0空闲，服务前台命令：

```bash
ssh GPU
source /sjtu/linhang/arena/env.sh
source /sjtu/linhang/arena/code/e1_env.sh
nvidia-smi
bash /sjtu/linhang/arena/code/launch_e1_standin.sh
```

另一终端，同样source两个env，选择**新的输出目录**（脚本拒绝覆盖）：

```bash
python /sjtu/linhang/arena/runs/E1_20260922/replay_snapshot.py \
  --dev-root /sjtu/linhang/arena/s1-dev \
  --output /sjtu/linhang/arena/runs/E1_repeat_smoke3 --num-chains 3 --smoke
python /sjtu/linhang/arena/runs/E1_20260922/replay_snapshot.py \
  --dev-root /sjtu/linhang/arena/s1-dev \
  --output /sjtu/linhang/arena/runs/E1_repeat_stock20 --num-chains 20
```

- 完整启动flags见 `scripts/launch_e1_standin.sh`：page64、extra_buffer、FP32 SSM、path cap=-1、chunk8192、track_interval256、KV131072tokens、mamba512slots、max-running8、CUDA graph关闭、metrics/增量stream打开。Python TreeCore显式固定；全部E2对照应保持相同。
- Torch2.13+CUDA13配合**arena内**compat库运行，未改系统driver535。包主要来自火山镜像，缺包/慢速fallback见前述阿里云与SHA记录。`pip-freeze-final.txt`及`pip-check-final.txt`保留在runs根目录；文本路径虽可运行，未安装所有可选依赖，**pip check不干净**，不能称为主办方底包环境。早期“未装DeepGEMM”的进度记录后来由v3启动的直接import需求修正。
- 首次长prefill含JIT/autotune，smoke TTFT约141秒；flush后同prompt冷算约0.519秒。不能把前者用于稳态推断，不能用二者比较缓存性能。随机小权重、4token输出、单并发和本地库均不是GLM-5.3的性能/能力证据，**没有N@SLO结论**。
- D0合规自检、D1 v1.1审阅和E2仍是后续任务；此次只完成E1，不启动Trisol、不使用8卡、不构建或提交镜像。

## E2 — D1 v1.1 开关 A/B 与数值检查（Codex main，2026-09-22）

- 已获本轮用户授权；T13，设计§10完成。只用GPU开发机arena目录与GPU0，不使用Trisol/8卡/提交。预检两卡均4MiB/0%且无compute进程，根盘3.1GB可用；E1源码/权重保留。
- 冻结D1 v1.1 SHA256=`60f98ced6d618a086bda7d49670d176fa75564168d594274bb4a0c63035663c2`；D0 SHA256=`e0d7924989ebb538e331110f4458ccd4e95a0bfa3cbd87dfb8324ac0328960f6`。在`arena/code/sglang-e2-v1.1/`从固定commit做独立clone，补丁只在该副本应用；没有修复审阅发现的host-miss缺陷，HiCache保持关闭。
- 三组：原stock；D0+D1但role IDs unset；D0+D1且IDs=`154827,154829`。先无trace回放，再单独做数字诊断；D0在off/on相同，不能归因到D1收益。CLI见`scripts/launch_e2_standin.sh`和`run_e2_variant.sh`。
- 同E1的qfull模型/权重、tokenizer、chunk8192/page64/extra_buffer/FP32 SSM/path cap=-1/KV131072/mamba512；**仅统一提高context上限到131072**以容纳reminder_heavy的92236与strict_append的66133token冻结长度，不截断。与E1旧时延不混算；原stock同子集缓存应可对照。
- 用例：D1-01 smoke原stock/off逐请求一致；D1-02 reminder_heavy收益与F24差异；D1-03 strict_append回归；追加stock20与E1对应。D1-04应检查真实全词表首tokenlogits及32步greedy，W6 API top-k logprobs只是补充，不冒充raw logits。
- CPU前置：原15项D1 helper + 新3项add_one/commit AST测试通过；其中host-miss测试是**复现缺陷**，不是HiCache安全通过。其他两项确认本轮已有partial时初始partial被拒、full可继续。GPU结果待追加；完整日志根`/sjtu/linhang/arena/runs/E2_20260922/`。

### E2 中间交接（08:23 UTC；数值检查仍在进行）

- **VERIFIED**：原stock vs D0+D1关闭，四集合215请求的ID/prompt哈希/缓存逐条相等；E2 stock20与原E1 stock20的72条也完全相同（尽管context上限增大）。
- **VERIFIED / D1-02 FAIL**：reminder_heavy 70条中D1开关=31改善/39相同/0退步，实际未命中总量700622→657678（少42944）；但fast_intra n=54的p95保持8135，理想role预测2439。误差≤64的仅45/70=64.3%，不满足≥95%且p95下降的既定门。
- **VERIFIED / D1-03 PASS（L2 N=1 extra_buffer限定）**：strict_append 67条全部与stock相同，fast n=49，p95=3747。
- smoke 6条全部相同，fast n=3 p95=6315，理想role预测562。长冷请求无role钩子是已确认的覆盖缺口；各尾部miss的逐项归因仍需结合trace，不能全归为一个原因。
- 比较工具`scripts/compare_e2.py`使用原harness的`q`秩定义，严格比对相同请求/文本/输出长度；结果不是N@SLO或完整GLM性能。完整stock20/on和raw logits结果后续追加。

### E2 最终结果（Codex main，2026-09-22 08:33 UTC）

**实验执行完成，不等于补丁验收通过。** F42缓存结果与F45数值结果；本地小证据`notes/e2_d1/`，全部请求/TTFT/服务日志、18份raw tensor和代码快照留在GPU机`/sjtu/linhang/arena/runs/E2_20260922/`。只用独立`code/sglang-e2-v1.1`，D0+D1应用均fuzz0；无002/003/004。原E1源码git干净，模型/配置SHA与E1不变。数值服务结束后两卡均4MiB/0%，自有三组tmux全部退出，未动其他会话。

| 内部集合 | 请求 / fast数 | 开启后改善/相同/退步 | fast未命中p95：stock=off → on | 同子集role理想预测 | on与role预测误差≤64 |
|---|---:|---:|---:|---:|---:|
| smoke | 6 / 3 | 0/6/0 | 6315 → 6315 | 562 | 3/6 |
| reminder_heavy | 70 / 54 | 31/39/0 | 8135 → 8135 | 2439 | 45/70 |
| strict_append | 67 / 49 | 0/67/0 | 3747 → 3747 | 3747 | 67/67 |
| stock20 | 72 / 42 | 6/66/0 | 4750 → 3962 | 3196 | 61/72 |

- **D1-01 PASS**：每组215项cache逐条off=stock、prompt哈希/长度一致、输出均4、0请求错误；集合之间有重叠，不能称215唯一请求。E2 stock20与E1 72条cache也相同。原stock flush文本；off/on严格验证JSON success=true。只覆盖空闲单worker，IF-08忙碌和IF-09多worker仍未验证。
- **D1-02 FAIL**：reminder总未命中少42944（6.13%），但p95没降、预测匹配率仅64.3%。其fast最差5条全部是idx_in_chain=1，前一条长冷请求没有role钩子；计数与§10.3缺口相符。后续其他miss还受branch/驻留FULL-KV等影响，不能全归为冷首缺口。
- **D1-03 PASS**：strict67无退步，仅证明本配置N=1；不推广到并发、lazy、MTP或HiCache。stock20 p95改善是同子集4750→3962，**不是全量F24的7238→3962**。

#### D1-04：全词表raw logits，严格门未通过

固定3条不同链的实际改善相邻请求对（均目标idx=3），从链首完整warm到上一请求，上一请求输出4、目标greedy输出32；cold0/cold1各flush，未截prompt。独立trace server只包装Python方法，不改模型tensor或参考源码；截取`LogitsProcessor.forward`返回、采样之前的`[1,154880]`FP32全词表logits。`splits_on.jsonl`、目标响应cache深度、实际batch RID三者一致，证明on组恢复的是上一请求的role快照。目标本身也可切role，冷/暖比较包含整个分块路径变化，不能把差异单独定位到state保存。

| 对 / 目标ID短名 | prompt长度 | stock/off恢复 → on角色恢复 | on冷/暖max abs | on greedy32 | off冷/暖max abs | off greedy32 |
|---|---:|---:|---:|---|---:|---|
| 0 / 8cWp9ey…:llm:4 | 91819 | 90176 → 90624 | 0.00390625 | 一致 | 0 | 一致 |
| 1 / NG-6e4…:llm:4 | 73176 | 66880 → 71488 | 0.0087890625 | 不同（第20token起） | 0.00390625 | 不同（第20token起） |
| 2 / hlpVa1…:llm:4 | 77223 | 71680 → 74176 | 0.00390625 | 一致 | 0.009765625 | 不同（首token起） |

所有cold/cold max abs=0，既定2×噪声容差因此为0；on三对全FAIL，off也有两对FAIL。**不能修改阈值宣布通过，不能仅凭此判定D1状态损坏；F43的“safe”只能限定为本次缓存计数不回退。** 需要预先制定能区分正常分块/恢复数值差异与错误状态的精度基线，并另做完整模型能力测试。随机权重上的greedy一致/不一致不是能力成绩。

跨组补查：三对on/off冷算raw logits逐元素数值相同（max abs=0）；暖算on/off max abs分别0.00390625、0.0087890625、0.009765625，greedy首次差异依次无/index22/index0（零起点）。新容差不可用这些候选结果倒推；后续应先选定同分块路径对照/高精度参考，再独立复测。

- 工具：`e2_serve_trace.py`、`e2_raw_logits.py`、`analyze_e2_numeric.py`、`compare_e2.py`；顺序编排`run_e2_numeric_pipeline.sh`。数值导出同步GPU，不能取该run的TTFT宣传性能。
- CPU收据：18项D1审阅测试（1项是HiCache缺陷witness）+ 5项E2工具在远端CPU全部通过；本机同23项有2项因无torch明确skip。shell语法、只读源码状态、模型哈希检查通过。`check_records.py`收尾必须再跑。
- **仍未验证**：D1-05并发/partial统计、D1-06关闭chunk、D1-08槽位无泄漏、D1-09 lazy、D1-10专门同prompt flush角色快照、D1-11 MTP，以及完整GLM/A100 DSA、HiCache、多卡、能力与N@SLO。本次没有Trisol、8卡配额、镜像或提交。

## Tools


### T23 W10 — Session A 整批 runner（CPU/dry-run，未执行会话）

- 测试配置收据：初次TL-01未设测试专用flush重试变量，1项因预期2次实际4次调用失败；按既有Tools规定设`ARENA_FLUSH_ATTEMPTS=1 ARENA_FLUSH_SERVER_WAIT_S=0`复跑29/29通过。两个日志和最终哈希见`evidence/T23/validation.json`；未改变production重试逻辑。

- **VERIFIED / 工具**：`scripts/session_a/run_session_a.sh` + `runner.py` / `remote.py` / `common.py`，详细用法 [`scripts/session_a/README.md`](../scripts/session_a/README.md)。已先读 R11，按用户修正为 D0-only FCFS/D1 unset 基线；原 stock 仅 CAP 对照；IF+TL-03+两科2–3题前置，N6/N10必测，随后±4，P/P+4 SPF→SPF+D1→D1矩阵，最佳组合CAP。未运行任何Trisol exec/start/modify；仅读取 CLI `exec --help` / `delete --help`。
- **传输/同步**：one-shot exec的stdin无文档保证，使用48KiB原始/64KiB base64 argv分片，每exec默认8片；约45MB bundle，911片/114批（数值随源码变动）。传全部请求输入及依赖、补丁、candidate、R4/R7/patch源文件对照；分片/包/文件SHA256验证。每步骤、每档（含失败）及长任务每120秒拉一致快照到`runs/session_a/<timestamp>/`，再带包SHA复制到`GPU:/sjtu/linhang/arena/runs/session_a/`。已有快照安全；服务被外部突然删除仍可能丢最后一次同步之后的数据，不承诺尚未写回的证据。
- **时间盒**：live调用必须显式 `--minutes <批准分钟数>`，10分钟final reserve；估时直接调用`plan_8gpu_session.py`，原35min/N6+21min启动假设乘1.25。完整默认示例的各步骤上限总和约550min（未含transfer/sync/最佳组合额外重启），不是实测耗时；240min可能不够。优先保留SPF与组合的两档及最后CAP；无法找到相邻档就显式incomplete，不凭推断安排“临界”矩阵。supervisor/引擎watchdog防止控制器消失后无界运行，final各动作独立；不删服务，打印Claude使用的精确delete命令。
- **检验与限制**：SA-01～07 26/26 CPU/mock通过；TL-01 29/29、T19/preflight38/38通过。`evidence/T23/`收据；`scripts/session_a/run_session_a.sh --dry-run --minutes 240`零网络/子进程/写入。真实CLI/底包/sm80/T8/能力/性能均未验证（SA-08 todo）；不把小CAP的single-server `passed=false`改成registry pass，配对只作smoke；未运行全量logits对照。数据与src只读参考未改。

```bash
# 无副作用：展示完整命令与自适应分支
scripts/session_a/run_session_a.sh --dry-run --minutes 240
# Claude 待服务running且已确定批准时间盒后使用；W10没有执行此命令
scripts/session_a/run_session_a.sh --minutes "$APPROVED_SESSION_MINUTES"
# CPU工具测试
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s scripts/session_a -p test_session_a.py -v
```

### T19 — IF/D1/CAP 测试缺口工具（Codex W6，2026-09-22 UTC）

- **VERIFIED / CPU 与 mock**：`scripts/if_checks.py` 覆盖 IF-02/04/05/06/08 忙碌路径/10/11/12；`preflight_8gpu.sh --extended-if` 可调用。IF-09 源码存在 `routed_dp_rank` + 返回 `dp_rank`，因此实现全 rank 轮流 busy 与 flush 后逐 rank 归零检查；路由不可控时保留手工步骤，不把未知 rank 当通过。三类 live 工具均有无 HTTP/下载/写入的 `--dry-run`，不启动服务。
- **VERIFIED / D1-07**：`test_role_boundary_split.py` AST 编译 `build/d1/b/.../PrefillAdder._maybe_role_boundary_split` 实际方法和实际 grid helper，验证与 patch AST 一致；15 项 CPU 测试通过，覆盖全部派发边缘情况与正常 split。无需导入 sglang/torch，无需 SSH/GPU。注册表只把本项标 pass，不能据此认定 scheduler 集成或数值恢复正确。
- **VERIFIED / D1-04 工具契约**：`logits_check.py` 参数按只读 `io_struct.py:226` 与 `tokenizer_manager.py:2759` 核对；`return_logprob`、`top_logprobs_num`、`token_ids_logprob`。冷两次标定、previous→next 命中验证、首 token top-k 并集统一 ID 比较、greedy 32 ID 一致性；支持一/两服务器和 calibration-only。比较的是归一化 logprobs，不是全词表原始 logits；默认容差=冷噪声×2。须另给预期 b 深度并核对 D1 taken/trace 才能确认是边界快照，不将普通命中冒充 D1 证据。
- **VERIFIED / CAP-01/02 工具契约与公开源**：`cap_spot_check.py` 默认 AIME 10/GPQA-Diamond 20，固定 seed/permutation、stock/candidate 相同题目、`model=default`，只从 content 抽答案，**不发 max_tokens/max_completion_tokens 或 thinking 开关**。要求答案可抽取、非 length 截断、candidate 正确数≥stock−1；单服务器/小样本/内置 tiny 均不能通过注册表。公开 [AIME 2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) API 30 行可达，hf-mirror API 308→HF 200（已兼容 Python3.10）；[GPQA 原仓库](https://huggingface.co/datasets/Idavidrein/gpqa) gated=auto，但 [OpenAI simple-evals 公开来源](https://github.com/openai/simple-evals/blob/main/gpqa_eval.py) 的 Diamond CSV HTTP200 可达。只在内存探查来源；正式运行才下载到 `/sjtu/linhang/arena/cache/t19`，记录摘要，无完整数据集落仓库。两道简短改述的 AIME 2024 I-2/II-11（25/601）作为明确标记的离线 smoke fallback，不能代替 GPQA。
- **验证收据**：`evidence/T19/tools_validation.log`：**53/53**（32 HTTP/mock 工具测试 + D1-07 15 + 原 preflight 6），含 real-socket 并发 flush/断连、错误成功/缓存泄漏/累计前缀/时间戳/rid 泄漏/数值漂移等反例、全部 CLI/dry-run；`evidence/T19/harness_integrity.json`：原 harness/cohort 7 文件 SHA 与 T16/T12 一致、无 harness bytecode。IF/D1-04/CAP 未连接 live 引擎，保留 todo；main 已记录的 E1 stock IF-08 fail 未覆盖掉。无 Trisol、镜像、提交；本轮未改只读目录或 board 实例表。
- 完整参数、前提、IF-09 人工步骤、数值工具限制、源 URL 与命令见 **`tests/T19_USAGE.md`**；三种直接 dry-run：`python3 -B scripts/{if_checks,logits_check,cap_spot_check}.py --dry-run`（分别展开执行）。CPU 复验：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts python3 -B -m unittest test_t19_tools test_role_boundary_split test_t16_tools.PreflightTests -v`。
- **公开源实读解析收据**：`evidence/T19/public_sources.json` 保存三次 HTTP200 的 URL/bytes/SHA256；实际 loader 在内存中解析 AIME10 / GPQA20，唯一 ID 数一致（只替换 fetch 的缓存写入层，不保存题库）。最终 `evidence/T19/records_check.log`：`check_records.py` 0 error / 0 warning。

```bash
# 已获授权且已在运行的服务；同机直连验证 IF-12。
python3 -B scripts/if_checks.py --base-url http://127.0.0.1:8000 --same-host \
  --flush-wait 300 --timeout 900 --out runs/if_checks.json
# D1-04：从 reminder_heavy 选 3 对真实相邻 prompt 分别运行；先 --calibrate。
python3 -B scripts/logits_check.py --base-url http://127.0.0.1:8000 \
  --chain-id CHAIN_ID --pair-index 0 --calibrate --out runs/d1_noise.json
# CAP-01；CAP-02 换成 --suite gpqa --count 20。
python3 -B scripts/cap_spot_check.py --base-url http://127.0.0.1:8000 \
  --candidate-url http://127.0.0.1:8001 --suite aime --count 10 --out runs/cap_aime.json
```

### 提交包工具（Claude，2026-09-22；仅本地，未构建/未提交）
- `scripts/build_image.sh`：把补丁打成确定性 tar.gz，再 base64 内嵌进 RUN，生成 `build/image/Dockerfile`（6319 B，上限 64 KiB）。RUN 里做四件事：sha256 自检；定位 sglang 包目录；`patch -p3 --fuzz=0 --force`（失败就中止构建）；py_compile 后 `import sglang`。**只打印** `bohr image build …` 命令，不执行。已验证：载荷往返 sha256 一致（000=3478d8ef…，001=f486737c…）；对 `build/scratch/python`（`src/sglang/python` 的副本）dry-run 并真实应用、py_compile 都 OK；在 sh 里模拟跑 RUN 块，第一次成功，重复应用时 FAILED 并返回 exit 1，说明不会悄悄跳过。前提未核实：底包里的 sglang 就是 v0.5.20。
- `scripts/check_submission.py`：离线检查 submission.json 是否符合 task.md（四个字段、argv 而非 shell、served-model-name 与 model_name 一致、env 为字符串、TOPK_V2=0、flag 是否在源码中存在），`--trace` 检查占位轨迹，`--final` 要求 digest 已钉。对 `candidate-01.json` 结果为 0 error、1 warn（image 是占位符）。
- `submission/candidate-01.json` 与 `.notes.md`、`submission/stub-trace.jsonl`（`playground trace validate` 显示 valid；注意 CLI 对只有 session_start 的文件也判 valid）。

### T12 — 临界 N 搜索与 formal-ish 离线评分（Codex W2，2026-09-22 UTC）

- **VERIFIED / 工具完成**：`scripts/ladder_search.py` 的 `official-climb` 从 10 开始，按 +4/-4，复用下降前已测失败档，直到相邻 pass/fail 或 N=2 失败；`fast --hint N` 先指数扩展找到区间，再按档位索引二分。不重复测档；可选 `--max-n` / `--max-levels` 到预算只报 incomplete 与观测界，不伪造临界 N。每档落盘 command、N、所有 dev/estimated gates、TPOT、flush 结果及 raw/run/report 路径到 `ledger.json`。
- **VERIFIED / F23 防护**：每档前由 wrapper 自己 POST `/flush_cache`，要求 2xx **且** JSON 严格布尔 `success: true`；失败则不启动该档。还通过 `S1_FLUSH_URL` 将原 runner 的预热后清缓存请求接到本机校验 guard，失败/超时杀掉 runner 进程组，防止 runner 忽略失败后继续测量；guard 总期限最多 20s，早于 runner 自带 30s 超时。仅首档 warmup，随后显式 `--skip-warmup`。token 只走环境/Authorization，不写 argv/ledger。
- **VERIFIED / 评分口径**：`scripts/score_formal.py` 直接调用只读 dev scorer，原 10 门完整保留；新增不带统计余量的 `tpot_p95<=0.10` 与逐请求均值。四 TTFT 门沿用 frozen 分桶/原 p95，输出 over_limit、pass_point、allowed_over、95% 单侧 **Clopper–Pearson** 下界（数值反演二项尾概率），只在下界 >5% 时估计失败；报告标 **estimated**，不是主办方成绩。coverage/错误率/空桶不会被余量覆盖；缺失多 token TPOT 阻断估计。缓存报告包含 actual-vs-frozen 差值、分位数/直方图、按门分布和 prompt-vs-frozen-token 差异，支持 D1 对账。方法参考：[R stats::binom.test](https://www.stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html)；主办方具体估计器未知，链内相关性也限制统计解释。
- **VERIFIED / 本地 CPU 验证**：`python3 -B -m unittest discover -s scripts -p test_ladder_search.py -v`：**29 tests / OK**。包含全部题面爬坡样例、330 组 fast 阈值/起点、二项分布端点/直接多项式对照、错误/空桶/TPOT/冻结缓存分桶、HTTP/JSON 失败、guard 异常/超时、dry-run 无副作用；另用临时 synthetic loadgen 调用**原封不动的 `s1-dev/run_dev.py` 子进程**，验证实际 CLI、评分文件选择和首档 warmup/后续复用。`s1-dev/run_dev.py` + harness 共 7 文件的清单与 SHA256 前后一致；证据与 dry-run 输出：`evidence/T12/tools_validation.log`。
- **INFERRED / 使用边界**：fast 模式依赖固定配置下 pass/fail 随 N 单调；实际性能噪声可能破坏这个假设。默认 `--gate-policy dev` 保持既有 dev 10 门定档；`dev+tpot` 增加解码门，`estimated` 才用统计 TTFT 估计。单档 exit=0 只代表执行成功，工具读取 gate verdict，不拿进程退出码当通过。未运行任何推理引擎、GPU 实验、Trisol 或提交；本记录不是完整模型容量实测。

仅预览（不发请求）：
```bash
python3 scripts/ladder_search.py --mode official-climb --base-url http://HOST:8000 --out /path/new-session --dry-run --dry-run-outcomes PASS,FAIL
```
未来已获授权且已有运行中引擎时：
```bash
python3 scripts/ladder_search.py --mode fast --hint 18 --gate-policy estimated --base-url http://HOST:8000 --out /path/new-session --max-levels 6
python3 scripts/score_formal.py --run-dir /path/new-session/level_001_N18 --out /path/estimated.json
```

### harness 自带测试（Claude，2026-09-22；只用 CPU）
- `s1-dev/harness/test_s1_harness.py` 在 GPU 机 `/sjtu/linhang/arena/env/ana`（Python 3.12，transformers 5.12.1）上：**29 项全部通过**（0.42s）。这说明 harness 与 tokenizer 环境可用。8 卡容器里要再跑一次，由 W4 的 `scripts/preflight_8gpu.sh` 自动执行。

### T16 — 真实前缀用例 / 加权诊断 / 8 卡规划 / 容器自检（Codex W4，2026-09-22 UTC）

- **VERIFIED / 仅 CPU 工具**：`scripts/make_case_sets.py` 生成 `cases/{formal_like,reminder_heavy,strict_append,cold_heavy,smoke}.json`（有序 chain_id/prefix_len/req_ids 列表）、每集 `.manifest.json`（每链原因、源文件 hash、门/产品/split/边类型/uncached 档占比、目标差距）及 `.predictions.jsonl`（逐请求 stock/role_conservative 未命中 tokens）。只用真实请求和从链头开始的连续前缀，不重排、不拼链、不复制请求。**内部 E1/E2 / simulator A/B，官方可比结果仍必须用原 dev cohort + `s1-dev/run_dev.py`。**
- **VERIFIED / F31 可实现性限制**：发放的 `requests.jsonl` 正好 722 请求，较长链只在 `chains.jsonl` 有原始长度元数据；可用 hidden 前缀最长 6 条。任意可用前缀 intra 占比最高 87.5%，chain_start 最低 11.11%，故不能真正复制正式约 8.16%/91.18%/0.66% 构成。生成器采用确定性 greedy gate/band/p95/fast-stratum 目标并偏好 hidden biomaster，明确保留至少一个 turn_start。没有声称全局最优或正式代表性。
- **INFERRED / population bands**：cohort 只给全人口 p95，不给 p50/p80/p90/p99 数值；这些 cutoff 用 sampling_weight 加权 dev 分位数作**标明的近似**，p95 固定为 population full，manifest 保留 cohort 原始 alignment 字段、band TVD/max-share 差及小样本标记。不能把这些近似当作人口分布匹配证明。
- **INFERRED / F24 预测**：复用 chunk8192/page64/branch-replaces-end/role_conservative 保末尾规则；使用冻结长度/相邻 LCP，当前 prompt 的 role 位置猜为 length−340（F3 中位尾长），没有偷看下一个请求设置 checkpoint。缺真实角色位置/旧分支路径，且无 decode/eviction/timing，**不是 F24 精确 tokenizer 复现**。JSONL 顶层 `stock` / `role_conservative` 是整数，兼容 W3 profile loader；`policy_details` 存缓存量/猜测边界。E1 `replay_chains.py --case-file` 会重新渲染并调用已有精确 token 模型。
- **VERIFIED / replay 兼容**：仅新增 `--case-file`；旧 CLI 默认不变。显式用例顺序/全部条目得到保留，不受默认 num-chains=3 截断；拒绝非前缀、漏行、重复链，超 prompt limit 报错而非删请求。需 tokenizer/Renderer 的 `--plan-only` 同样不发 HTTP；本轮用 CPU fake renderer 验证选链接口，未在 GPU 机重跑 tokenizer。
- **VERIFIED / 加权口径**：`score_formal.py --requests <requests.jsonl>`（默认原 dev index）按 req_id 加入冻结 sampling_weight，缺 ID 才回退 raw；权重缺失视为 0，负值/非有限值拒绝，报告来源/缺失/冲突数。逐 TTFT 门输出原 unweighted p95/exceed-rate/CI + weighted p95/exceed-rate；TPOT、缓存分布和每门 budget 同时加权报告。原 hard gates/二项 CI 完全不使用权重。源码真实 formal 仅以总体 `budget_attainment.weighted` 为诊断权威；80% tripwire 不是硬门；TTFT p95、coverage、错误率、TPM、分层 budget 不按 sampling_weight。等权分位数与 dev floor(p*n) 一致，零权重不进入加权分位。
- **INFERRED / 规划器**：`plan_8gpu_session.py` 调用原 `s1_loadgen.build_gap_plan(cap=3600s)`，按原 FIFO 整链闭环分槽，不把总 gap 简单除 N；用 output_budget + uncached/32 的工作代理标定 N6=35min，再给出 N2/6/10/14/18/22 的恒定服务时间/默认 sqrt(N/6) 变慢/线性争用情景。默认计每 policy block 启动21min、warmup5min，每档 preflight0.5min + flush总额0.5min，session自检2min；D1 默认无收益，15%预算余量，不含排队/拉镜像。reference 35min 被解释成 measure-only，可能保守重复计入未公开开销。240min 选定前6档 stock6/10/14/18+D1 14/18（约234.8min含余量），D1 22顺延；完整7档约223.1min未加余量。这是计划模型，不是测速/容量/置信区间。
- **VERIFIED / 自检 wrapper**：`preflight_8gpu.sh` 调 Python helper，条件安装 harness requirements（当前要求 transformers>=4.51,<6），pytest `-q -p no:cacheprovider` / 缺 pytest 时 unittest，禁止 harness bytecode；GET /v1/models 要求 default，/generate SSE 每事件计数/首末时间戳/ignore_eos精确长度，cold→repeat命中上涨→严格 JSON flush→cold归零，chat model=default/content非空/response model匹配，最后再 flush。只存安全 JSON 摘要，不打印 key、环境、响应文本、异常细节或 pip 输出。`S1_API_KEY` 仅环境传入。**原生 stock 文本 flush 会失败；须 D0 JSON 合约，不能把文本200假装通过。** 默认 chat path `/v1/chat/completions`，可显式选根路径 `/chat/completions`。
- **VERIFIED / 本轮验证**：原29项 ladder/scorer、21项新增 T16、原7项 replay（共57项）CPU检查；另原 harness 29项 unittest。覆盖真实前缀/顺序、加权不改判、分数权重/零权重/缺失/坏权重、预算/两次启动/尾链 gap 下界、HTTP stub 完整流程及8种失败、pip条件安装与fallback、无秘密回显、dry-run无写入。证据：`evidence/T16/tools_validation.log`、`evidence/T16/harness_self_tests.log`、`evidence/T16/harness_integrity.json`、`evidence/T16/preflight_dry_run.txt`、`notes/t16_session_plan.{json,txt}`、`evidence/T16/case_composition.json`。原 harness/cohort 7文件与T12 SHA256一致；直接调用 harness unittest 的子进程曾生成一个 s1_common.pyc，已删除该生成物恢复目录，并以 PYTHONDONTWRITEBYTECODE=1 重跑（wrapper 本身已设此环境）。最终无 s1-dev/__pycache__。没有实际 GPU/引擎 HTTP、Trisol、镜像构建或提交；live 容器自检尚未执行。

实际生成规模（gate percentages 为 unweighted）：

| case | chains | requests | chain_start / intra / turn_start |
|---|---:|---:|---|
| formal_like | 21 | 124 | 17.74% / 81.45% / 0.81% |
| reminder_heavy | 12 | 70 | 17.14% / 82.86% / 0.00% |
| strict_append | 12 | 67 | 17.91% / 82.09% / 0.00% |
| cold_heavy | 15 | 35 | 51.43% / 40.00% / 8.57% |
| smoke | 3 | 6 | 50.00% / 50.00% / 0.00% |

CPU生成与预算预览：
```bash
python3 -B scripts/make_case_sets.py
python3 -B scripts/plan_8gpu_session.py --budget-minutes 240 --out evidence/T16/session_plan.json
scripts/preflight_8gpu.sh --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts ARENA_FLUSH_ATTEMPTS=1 ARENA_FLUSH_SERVER_WAIT_S=0 python3 -B -m unittest test_ladder_search test_t16_tools test_replay_chains -v
```
已有授权/运行中服务时的命令（本轮未执行）：
```bash
scripts/preflight_8gpu.sh --base-url http://127.0.0.1:8000 --out runs/preflight.json
python3 -B scripts/replay_chains.py --dev-root s1-dev --case-file cases/smoke.json --output /path/new-E1-case --plan-only
python3 -B scripts/score_formal.py --run-dir /path/measured-level --out /path/weighted-score.json
```


### T20 — D2 SPF port and fixed R10 simulator cross-check（Codex W7，2026-09-22）

- VERIFIED / CPU: `patches/002-spf-scheduling.patch` ports #40024 onto `94602c9 + D1 v1.1` in `build/d2/{a,b}`. D2-01…07: 16 production-class tests pass; default FCFS has 56 paired / 112 executions with identical serialized admission/accounting; 001→002 fuzz0 and final-file parity pass. 002 alone fails three D1-context hunks and must not be forced. Shared partial eligibility covers ordinary truncation, host-miss re-selection and D1 splits; local DSA continuation alignment uses page/truncation LCM.
- VERIFIED / CPU: D2-08, D2-10: 47 simulator tests pass (41 previous + 6 new), including 500 randomized rounds matched against actual port admission, exact stock HRRN/LPM sort comparisons, and a request-slot counterexample distinguishing legacy SPF from #40024. Original 15 D1 tests pass; 12 full legacy FCFS/SPF results/traces are byte-identical to the pre-T20 simulator. Receipts: `evidence/T20_d2/*tests.log`, `d1_regression.log`, `legacy_sim_parity.log`, `patch_validation.json`.
- MODEL OUTPUT / D2-09: `PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_d2_sim_calibration.py` reruns R10 C6/C17/C18/C19/C31 plus C42/C43, five schedulers, stock/D1 profiles, N=2,6,10,14,18,22,26,30: **560 runs**, no timing refit. Main five stock SPF ceilings=14; C42/C43=18. All FCFS/HRRN ceilings=6; LPM=6 except C6 stock=2. All 14 legacy/new-SPF ceiling pairs agree; C42 stock N14 has some nonzero timing changes while verdict stays the same. These are conditional model results, not real serving capacity or D1-benefit evidence.
- Scope/commands/deviations/risks/table: `patches/002-spf-scheduling.md`; full compact model data `evidence/T20_d2/calibration/comparison.json`. Original SPF already budget-shares (not pure ordering); new `spf-upstream` reserves before slot admission and runs continuation first. HRRN uses processed-token aging; LPM absolute cache matches; neither stock policy reserves for waiters while a long chunk continues. All use ideal static cache counts in the model. D2-11/12 remain live todo. No GPU forward, Trisol, image build or submission by W7.

- T20 补充 VERIFIED：GPU机 CPU 完整导入 `/sjtu/linhang/arena/env/sgl` 后16/16生产单测也通过（43.006s，`evidence/T20_d2/full_import_tests.log`）；首次scp断线造成fixture缺失，重传后通过。全部远端工作只在arena/code/d2-worker-w7与runs/T20。直接 sim_closed_loop CLI C19/N14五策略结果与560-run wrapper逐项一致。

### T22 — Unattended submission queue and daemon (Codex W9, 2026-09-22 UTC)

**VERIFIED / CPU only**：`scripts/submit_daemon.py`、`scripts/queue_submission.sh` 与 `scripts/test_submit_daemon.py` 已交付。使用 Python 标准库；本轮未真实提交、未访问 Playground 网络、未启动后台 daemon、未创建任何 `APPROVED` 文件（测试与演示的审批只在内存中 mock）。正式 API/CLI 连通性与真实提交仍待获批运行。本节是无需聊天上下文的操作 README。

**入队与审批**：`scripts/queue_submission.sh CANDIDATE_JSON SLUG [TRACE_JSONL] [NOTES_MD]` 自动分配 `submission/queue/NN-slug/`，复制四字段候选、trace 与说明，生成的目录没有 `APPROVED`。默认 trace 为 `submission/stub-trace.jsonl`；手工队列也可用同名 symlink 指向它。只有用户或记录用户明确授权的协调方可以提供审批文件。格式、哈希绑定与示例见 [queue/README.md](../submission/queue/README.md)。`approved_by`、带时区 `approved_at`、`what`（精确目录名）、`submission_sha256`、`trace_sha256` 五行都必需；审批不匹配、内容变化或审批撤回都会停在 `awaiting-approval`。

**执行与限额**：每次先核对台账和 Playground API；默认上海自然日最多 2 次、最多 1 个在途。`--daily-limit 1` 可进一步降低额度（不能大于 2）；`--max-in-flight N` 仅在明确选择时改变并发上限。GET `/auth/me` 获取账号/关联 operator，分页 GET `https://play.bohrium.com/api/challenges/llm-challenge-arena-v1/attempts?limit=100&page=N`；只计该账号/关联 operator 的尝试，包括手工提交，已知 attempt ID 在 API/台账间去重。其他作者不占用我们的额度；未知状态、分页重复/缺页、API 失败都阻止新提交。不要并行手工提交：锁只协调同一工作区的 daemon，不能锁住外部 CLI 或其他工作区。

**预检**：在私有临时目录生成 `outputs/submission.json`（outputs 下恰好这一个文件）与单独 trace；执行 `check_submission.py <snapshot> --final --trace <snapshot-trace>`，再 `source env.sh && playground submit --challenge-id llm-challenge-arena-v1 --outputs <outputs> --trace <trace> --dry-run`。两步任一出错即终态 `failed-precheck`，不自动重试该项、不消耗提交额度，可继续检查其他已审批项。实际提交使用相同命令去掉 `--dry-run`。`candidate-01.json` 当前是 image placeholder，不能通过 `--final`；需要真实 digest 的候选重新入队及审批。

**安全与持久化**：真正调用 CLI 前先以 fsync + atomic replace 写 `submitting` 占位与 UTC `submitted_at`。解析返回的 `attempt_id` 后改为 `submitted`；超时、非零退出、响应无 ID 或进程在占位后中断，会进入 `submission-unknown`，跨天/重启也不自动重试，且暂停所有新提交。CLI 没有创建 attempt 的幂等键，不能假定一次报错就是没有提交。文件锁阻止并发 tick；台账错误不会回退为空表。CLI 调用每次 source 根目录 `env.sh` 并设置 `NODE_USE_ENV_PROXY=1`；凭据只从 `~/.config/playground/credentials.env` 的指定字段读入 `PLAYGROUND_TOKEN` 环境变量，REST 仅通过 Authorization header 使用。token 不进入 argv、日志、台账；不保存 CLI stdout/stderr 或完整 API 响应，错误日志只含固定诊断码。

**取分与输出**：每轮同时使用 `playground status --attempt-id ID` 和上述 REST 列表；两者终态一致后才完成。`pending_review + scoringState.failed` 是失败终态；`execStatus.failed + scoringState.evaluating` 仍视为在途。成功终态没有 scorecard 时继续等待。`data/submissions.json` 每项一条，包含审批人/审批内容、提交时间、attempt ID、状态历史及 final_scorecard：`aime26.points`、`gpqa.points`、`gate_passed`、`gate_reason`、`stress.{n_at_slo,tpot_mean,tpot_p95,chain_start_p95,turn_start_p95,overall_intra_p95,fast_intra_p95,evaluation_status,reason}`。缺失指标保留 null，不伪造 0 或 PASS。终态追加 `notes/submissions.md` 表格行与 `logs/submit_daemon.events` 的一行 `RESULT`；提交记 `SUBMITTED`，另有 `WAIT`、`FAILED_PRECHECK`、`SUBMISSION_UNKNOWN`、`POLL_ERROR`、`DAEMON_ERROR`。终态摘要/事件可在重启后从台账补写并去重。

离线演示（隔离临时目录、内置 CLI/API mock、无需 token、不读写正式队列/台账）：

```bash
cd /workspace/Agentic_science_challenge
python3 -B scripts/submit_daemon.py --dry-run
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s scripts -p test_submit_daemon.py -v
```

入队（仅创建待审批项；以下没有生成审批的命令）：

```bash
scripts/queue_submission.sh submission/candidate-01.json candidate-01
```

**后台启动命令（本轮没有执行）**：队列中每个可提交项必须先获得用户明确审批。默认 600 秒；`--once` 是一次真实工作周期，**不是演示**，会提交已审批且过检的项。

```bash
cd /workspace/Agentic_science_challenge
mkdir -p logs
nohup python3 -u -B scripts/submit_daemon.py --interval 600 >> logs/submit_daemon.log 2>&1 &
echo $! > logs/submit_daemon.pid
# 协调方可低成本观察：
tail -F logs/submit_daemon.events
```

**暂停、终止、恢复**（以下仅文档命令，本轮没有启动进程）：

```bash
cd /workspace/Agentic_science_challenge
touch submission/queue/STOP
# 若还要终止进程，确认此 PID 仍对应本 daemon 后：
ps -p "$(cat logs/submit_daemon.pid)" -o pid,args
kill -TERM "$(cat logs/submit_daemon.pid)"
# 恢复时删除 STOP；daemon 已终止则再执行上面的 nohup 命令。
rm submission/queue/STOP
```

STOP 在两次外部操作之间生效，包括预检之后、提交之前；已发出的 HTTP/已被 Playground 接收的提交无法靠 STOP 撤回。暂停时也不收集成绩；恢复后继续。默认外部调用超时为 180 秒，可用 `--command-timeout` 调整；SIGTERM 清理正在等待的 CLI 进程组，已写入的提交占位不会丢失。

**异常恢复**：看到 `submission-unknown` 时先 STOP 并终止 daemon，查自己的 Playground 尝试记录。若确认对应 attempt 已创建，把该项 `attempt_id` 补为字符串、`status` 改为 `submitted`，保留原 `submitted_at`/审批/历史并追加恢复依据，即可恢复轮询；不要另建同一提交或抹去占位。仅在确证没有创建 attempt 后，才可将该项标为 `failed-precheck`、清空 `submitted_at`、填写 `completed_at` 并追加对账依据；再次尝试用新队列项和新审批。台账损坏要恢复可靠备份，不得清空来绕过配额。API 连通性/身份或 schema 异常按事件码修复后下一周期继续。

**测试收据**：`evidence/T22/submit_daemon_tests.log`，**59/59 CPU tests pass**；注册为 SUB-06/07/08/09、TL-07。覆盖无审批、审批哈希/撤回、STOP、两实例锁、CLI/API 双在途检查、手工提交计数、Asia/Shanghai 00:00 切日、历史台账/API 去重、API 分页/失败/未知状态、真实离线 validator、mock CLI source env/安全 argv、两级预检失败、崩溃/超时占位、重启不重提、取分/终态补写、凭据不泄漏、queue helper 不创建审批、隔离 dry-run。另直接解析 `data/all_att.json` 的既有真实 scorecard 样本；此项验证解析器，不代表真实提交成功。

T22 收尾收据：`evidence/T22/submit_daemon_mock.log` 为纯 mock 演示；`evidence/T22/records_check.log`：0 error / 0 warning。生产 `data/submissions.json` 保持空数组，queue 下只有 README，无 APPROVED。

### Tools — T27 keep-alive 提测 daemon（Codex W13；CPU/mock）

- **VERIFIED / TQ-09…12**：55/55 `test_trisol_test_daemon.py` CPU/mock tests pass。覆盖默认 `daily_gpu_hours=0`（unlimited）、8 卡服务跨批准项复用、`lh-arena-sess-a` 优先接管、idle-hold 释放、单服务约束、服务丢失后的后续重建、每项 `score_formal.py` + simulator receipt、单次 `QUEUE_LOW`、forever STOP wrapper 与少于30行 status。证据命令：`PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s scripts -p 'test_trisol_test_daemon.py'`。
- **VERIFIED / scope**：未启动 daemon、未创建/进入/删除真实 Trisol 服务、未打镜像、未使用 GPU。每项结果仍标记 estimated；TQ-09…12 只证明本地状态机和 mock 协议，不证明 T8/live 服务性能。

### Tools — T25 / W12 I6 statistical analysis + CPU model + draft 003（2026-09-22）

- **VERIFIED / CPU, SLO-01…07**：16新增测试、47模拟回归、29 scorer/ladder、21加权/工具、16 D2与15 D1，共144项通过。命令/收据见 `research/codex/R12_slo_aware_scheduling.md`、`evidence/T25_slo/*regression.log` / `unit_tests.log`。T14评分测试须 `ARENA_FLUSH_ATTEMPTS=1 ARENA_FLUSH_SERVER_WAIT_S=0`；初次未设造成fixture与默认重试不匹配，按原验证环境重跑通过。F40刷新导致F35原测试计数失效，现按R10保存的原attempt IDs固定样本；R10参数与结果未改。
- 003源自W7已就绪`build/d2/b`，独立`build/d3/{a,b}`；clean→001→002→003及000叠加均fuzz=0。session/namespace/可信到达时间专用字段透传，实际cache miss推断3/5s；first-seen30s，无法可靠识别turn/reset如实标注。单partial复用002唯一guard，450随机模型/实际helper对照；其余Adder方法AST未变。**SLO-08/09 live仍todo**。
- **estimated** CP统计：n808允许51条>30s，第52条失败；758×20s+50×61s令p95=61s且L=.048529≤.05。F31构成不能替代某次实际观测桶数；不把统计条数余量换成固定秒数。
- **MODEL OUTPUT**：固定R10 7候选×dev/formal-mix×8臂×8档=896次；SPF主候选formal-mix严格首败18，EDF首败10（C17=14）、EDF+D1首败14（C6=10），绑定fast/overall；weight2与least-slack未稳定胜SPF。完整strict/estimated首败门表及每档数据 `evidence/T25_slo/calibration/comparison.json`；不能推断真实N或对方具体调度。
- 仅CPU工具/模型/补丁草稿，W12未启动推理服务/GPU/Trisol、未打镜像或提交；未修改`s1-dev`、`src/sglang`、题面。

### T24 — 自动提测队列与回收守护进程（Codex W11；CPU-only）

- VERIFIED / CPU：`scripts/trisol_test_daemon.py`、`scripts/queue_test.sh`、`tests/trisol_test_config.json`、空 `data/trisol_tests.json`/`logs/trisol_test.events` 已交付；用法及精确 nohup/暂停/停止命令见 `tests/TRISOL_TEST_DAEMON.md`。队列只读用户 APPROVED，单服务锁+本人全分页列表阻止第二个服务，运行候选按哈希冻结。TQ-01…08 共51项 CPU/mock通过，证据 `evidence/T24/trisol_test_daemon_tests.log`；合成 raw 调用真实 score_formal，不冒充性能结果。
- VERIFIED / CPU：12 GPU·h/UTC日默认限制为8卡90min；WaitingForAdmission 单列排队秒数；分配到删除确认期间如实记账；硬截止预留collect/delete。错误、SIGTERM、runner超时、控制进程意外退出走回收；独立watchdog只回收，不排新任务。delete accepted不是完成，cleanup未确认不接着创建。网络/平台删除失效时保持隔离并重试，不能保证云端资源强制释放。
- VERIFIED / CPU：逐档读取session_result合约，核对profile SHA和run.config.N，重新调用score_formal；RESULT/experiments引用PF-01/PF-02，均标estimated。满足N阈值、当前全部P0逐ID通过、精确被测候选与digest且删除已确认，才调用T22 create_queue_item生成未批准项。缺CAP/D1/接口P0证据不会把小样本smoke当通过。
- 集成限制已通知T23：当前runner仍未支持profiles/matrix/collect-only扩展，daemon以本地--help前置检查阻止实际执行；完整接口/manifest写在dispatch与使用文档。创建还需协调方从已有收据补齐gpu_product_id（CLI必需，board/plans未记录）；接管已有匹配服务不需此ID。实际Trisol JSON schema未live验证。PF-01/PF-02 T8状态保持todo；本轮没有服务操作、GPU实验、镜像构建、提交、APPROVED或daemon启动。

### T28 — E1/E2 测试 ID 与证据回填（Codex main，2026-09-22）

**VERIFIED / 既有证据复核，不是新实验**：本轮只读本地E1/E2摘要、原harness和回放编排，回填注册表。未SSH、未占GPU、未运行live检查。以下结论只适用于E1 stock及E2 **D0+D1 v1.1**、随机qfull替身、N=1/page64/chunk8192/extra_buffer；不覆盖004/v1.2、D2、HiCache、lazy、MTP或T8。T19已完成工具与CPU/mock验证，不代表这些live缺口已补齐。

#### 样本与运行对齐

- E1的`stock_smoke3`（9条链请求另加3次接口smoke）和`stock20`（72条）是历史选择，**不是**后来生成的`cases/smoke.json`。原E1保持不变；D1-01/03需要的新case stock基线已在E2重新跑，不能把旧E1样本直接改名。
- 本轮逐项核对`notes/e2_d1/comparison.json`的req_id列表：与`cases/smoke.json`、`reminder_heavy.json`、`strict_append.json`的**完整请求ID及顺序全部相等**，分别为3链6条、12链70条、12链67条。编排`scripts/run_e2_variant.sh`对stock/off/on均使用`--case-file`；三组都另跑历史stock20以衔接E1。每组共215项，集合间有重叠，不是215个唯一请求。
- `cases/cold_heavy.json`为15链35条，**没有对应E1/E2实测run**；特别是D1-05要求的cold/reminder混合N=4没有执行。不能用N=1 reminder中含长冷请求代替。
- 本地证据：E1在`evidence/E1_stock/{manifest.json,summary.json,smoke_events.jsonl}`；E2在`notes/e2_d1/{comparison.json,numeric_on.json,numeric_off.json,splits_on.jsonl,numeric_cross.json}`。完整GPU收据根为`/sjtu/linhang/arena/runs/E1_20260922/`和`E2_20260922/`，后者回放子目录`{stock,off,on}_{smoke,reminder_heavy,strict_append,stock20}`。沿用原证据位置，本轮不迁移或改写原始结果。

#### ID → 证据与尚缺的通过条件

| 注册表ID | E1/E2映射与既有证据 | 本次回填判定 |
|---|---|---|
| IF-01 | E1/E2启动与健康检查不能代替`GET /v1/models`响应及期望model_name校验；没有专门保存该检查收据 | todo，不能补标pass |
| IF-03 | E1 stock20有72条server TTFT及计数；E2也有逐请求汇总。原`s1_loadgen.py:137`忽略`[DONE]`，不强制它出现；只保留首/末meta而不验证所有事件 | todo；逐事件字段、completion单调性、首/末时间戳、DONE仍需专门检查 |
| IF-04 | E1/E2回放实际输出均4；数值检查目标输出32 | todo；这不是规定的1/2/240/4096长度矩阵 |
| IF-07 | E1 `smoke_events.jsonl`同一18678-token prompt，cold/repeat/flush后cold：cached=0/18624/0 | pass，仅L2 stock；不外推D0候选/T8 |
| IF-08 | E1 stock文本响应，json_contract=false；E2 off/on带`--require-json-flush`，空闲单worker的JSON success=true通过 | wip（部分覆盖）；保留E1 stock FAIL，D0完整用例仍未通过：busy timeout=0与等待成功路径未测，多worker另属IF-09 |
| D1-01 | E1提供环境/历史stock基线；正式case对照来自E2 `stock_smoke`与`off_smoke`，6/6 cache相等；扩展四集合215项全相等 | pass，L2 v1.1/N=1限定 |
| D1-02 | E2 `stock/off/on_reminder_heavy`；31改善/39相同/0退步；45/70在role预测±64内；fast实际未命中p95为8135→8135 | fail；不满足≥95%预测匹配和p95下降。stock20的4750→3962不能代替本case过门，更不是全量F24的7238→3962 |
| D1-03 | E1提供机制基线；同case stock已在E2补跑，`stock/off/on_strict_append`67条全部相同 | pass，仅N=1 extra_buffer，不代表并发/lazy/MTP通过 |
| D1-04 | E2 raw全词表3对；`numeric_on/off.json`、split trace确认on的角色恢复；冷算重复噪声均0，on冷暖全超既定容差，1对greedy32不同；off也2对失败 | fail；既定门未过，不能直接把差异归因到D1快照损坏，也不事后改阈值 |
| D1-05 | N=1回放没有N=4混合case，也未记录每轮partial计数和准入统计总和 | todo；CPU守卫测试、顺序回放无崩溃不替代并发验证 |
| D1-08 | E2没有服务存活期间的KV/Mamba空闲池启动值与flush后值配对 | todo；成功flush或停服后显存释放均不是槽位无泄漏证据 |
| D1-10 | E1的flush归零属于stock；E2数值顺序为cold0/cold1/warm再清理，未在已确认角色命中后flush并重打同一prompt取归零收据 | todo；链间flush不代替角色快照闭环 |

不重标历史结果，不改变通过标准；以上已同步`tests/TEST_PLAN.md`。本节是T7/T13证据的注册表索引，不增加一个E编号，也不把记录回填当作补丁验收通过。

### T29 — E2 v1.1 遗留用例补测（Codex main，2026-09-22）

- 响应再次派发T13/E2：既有D1-01/02/03/04与T17不重复执行；补D0代码审阅、IF-08忙碌/等待、D1-05并发、D1-08池回收、D1-10角色快照flush。D1-04采用的全词表raw证据强于T19的API top-k覆盖，仍按原门FAIL，不能改记pending。
- 新结果目录`GPU:/sjtu/linhang/arena/runs/E2_T29_20260922/`，原E1/E2目录不变。启动前两卡4MiB/0%、无compute进程；只占GPU0，专用tmux `arena-e2-t29`，2400秒watchdog。同qfull权重/配置SHA及远端000+001三个改动文件SHA相同，stock副本干净；没有应用002/003/004。
- `scripts/e2_completion_trace.py`只在诊断进程内包裹原方法：记录startup/flush池、new/continuation准入和每轮partial、实际split深度及计数。原函数返回值不改，模型tensor/预算不改；有诊断开销，不取该run时延作性能证据。`e2_completion_checks.py`复用W6 IF-08、原Renderer及`replay_chains.py --case-file --concurrency 4`；混合case保留两组所有完整前缀，重复链取较长前缀，不改prompt内容。
- **CPU VERIFIED**：6项D0实际AST测试+4项新工具检查全部通过，收据`evidence/T29/cpu_tests.log`；补丁代码审阅在000 §6。live结果随后追加，不提前将TEST_PLAN的todo/wip改为pass。

#### T29 第一轮结果与配置修正

- **VERIFIED**：IF-08已按W6工具完成真实4096-token在途请求：timeout=0返回400/false，等待约54.409s后200/true，再打同prompt cached=0。D1-10目标91819 tokens，先完整warm原链，上一请求trace在90624切分，目标确实命中90624；flush后原样重打cached=0。两项仅TP1/无HiCache。
- **未执行，不是引擎崩溃**：mixed_case含cold35+reminder70=27链105条，cold prompt达到256733；原context131072配置的replay在发请求前拒绝，0轮。保留`mixed_replay.log`，不能记D1-05已做并发。
- **撤销工具误判（未曾回填注册表）**：第一轮`completion_report.json`的D1-08写了pass，但混合回放根本未运行，只能证明空池相等，**此项pass无效**。生成器现要求回放完整/请求数一致/0error才有资格判断池回收，新增反例单测；11项CPU测试通过，`evidence/T29/cpu_tests_v2.log`。原JSON保留，不篡改历史收据。
- 决策24：同qfull模型max_position_embeddings本为262144；只为完整长prompt诊断，把新服务context设262144、KV容量524288，其余不变，独立`E2_T29_20260922_large`。旧服务已发正常中断退出；不重跑已通过IF-08/角色flush，不与原E2矩阵混作缓存收益或性能对照。新run只补N4与池回收。

#### T29 最终收据（F47；补测完成，不代表D1验收全过）

实际成功的长配置目录是`E2_T29_20260922_large_v2`：首次Ctrl-C未让旧服务退出，`large`启动脚本发现GPU仍忙后拒绝，无请求运行；核对旧自有进程组后TERM退出、确认两卡空闲，再启新服务。一次SCP握手失败已重传并核对SHA；以上失败日志保留。旧E1/E2数据与只读源文件不变。

| ID | 实际结果 | 判定 / 收据 |
|---|---|---|
| IF-08 | 原E2配置，空闲成功；4096-token流在途timeout=0→400/false；等待54.409s→200/true；flush后同prompt cached=0 | PASS，仅TP1；`evidence/T29/first/if08.json` |
| IF-03（附带覆盖） | W6 IF-08调用`events`强制DONE、`BackgroundStream.join→validate_stream`逐事件审计计数及首/末时间戳，均通过 | PASS，仅该L2 4096-token流；不替代IF-04全长度矩阵或IF-12时钟精度 |
| D1-10 | 目标91819 tokens，经完整链warm；上一请求trace split90624，目标cached90624；flush后同prompt cached0 | PASS；`first/completion_report.json`对应子项及previous_splits |
| D1-05 | 大配置、客户端N4，cold35+reminder70共105条/27链全完成，0错误；306调度轮partial最大1，最大prefill batch3；全部batch/commit RID吻合 | 单partial/稳定性子项PASS；统计总门仍未通过，详下。`large/completion_report.json`及`scheduler_trace.jsonl` |
| D1-08 | 完整混合回放后，flush前空闲KV8320/Mamba378/request8；flush后524288/512/8，等于startup及回放前基线 | PASS，仅context262144/KV524288、extra_buffer；`large`完整收据。第一轮空跑报告的此项pass仍无效 |

**D1-05计数分母明确交接给Claude**：ROLE_BOUNDARY_STATS共105=taken45+already_chunked34+branch20+active5+short1，正好等于105次**新请求成功准入**；不等于215次准入尝试，也不等于加229次continuation后的334次commit。原补丁没有给续跑/早拒绝记此计数。诊断工具按包含续跑的严格总commit口径判fail，注册表记wip（部分通过/口径待明确）；**不是发现第二partial或崩溃**。请先明确注册表意图：若要覆盖所有调度准入，应补独立计数；若只验新请求准入，应显式写出分母，再审定结果，不默默用有利分母替换。

- 本地证据`evidence/T29/{first,large}/`含JSON/manifest/逐轮trace；远端保存全部请求/服务日志及`large_v2/code-snapshot`。trace完整308轮含2轮启动暖机，混合run统计严格只取306轮，batch/commit RID错配0。IF-08/10在原131072配置，N4/池回收在大配置，不混称一个完全相同profile。
- 新增工具/AST11项，加原replay7项及W6 HTTP/mock32项，共**50项CPU测试通过**，收据`evidence/T29/cpu_regression.log`。未改生产补丁，只加诊断工具；没有重跑原stock/off/on矩阵或改数值门。
- 原D1-01/03 PASS、D1-02/04 FAIL保持；T17计划审阅已在原计划决策记录完成。以上不证明完整GLM、DP/HiCache/MTP/lazy、数值安全或N@SLO；无Trisol、镜像、提交动作。
- **清理与交接**：已仅TERM本次核实的自有服务进程组；最终GPU0/1均4MiB/0%、无compute进程，自有两个服务/检查tmux已退出，保留其他项目及协调方`arena-daemons`会话。本地`src/sglang`与远端原stock git状态干净；E2三个补丁文件SHA前后相同。T29 done，D1-05分母问题交Claude，根README仍由Claude维护。索引刷新，`check_records.py`收尾0 error/0 warning。

## E2b — 004最终chunk切分：v1.1 / 001+002控制 / v1.2（T26，Codex main，2026-09-22）

**结论：本L2缓存机制门通过；不代表数值安全、GLM性能或镜像B可直接启用。** reminder_heavy实际未命中token p95由8135降至2439，正好等于F24在该子集的预测；三套143个不同请求逐项无退步、且候选cached_tokens全部精确符合role_conservative预测。004审阅另发现未透传额外truncation alignment的major，未私自修补；旧D1-04数值FAIL保持。

### 环境、控制与命令

- 用户今日wrap-up仅恢复004审阅/E2b，不开新任务。开发机GPU0，另一卡空闲；无Trisol/镜像构建/提交/daemon操作。仅`/sjtu/linhang/arena`内操作，先source env.sh及code/e1_env.sh。
- 与E2同一random qfull Kimi-Linear（3KDA+1MLA、无DSA），权重SHA `16a4e0413d8a810f18666443b869c229ae70419c78e1c958aa91e59b461500eb`、config SHA `030408d0e8810c3509fdecfee1694ad254ac537e5213f068b2082f7757233375`已复核。
- SGLang `94602c9`；独立`code/sglang-e2b-control`=000+001+002，`code/sglang-e2b-candidate`=000+001+002+004。全部fuzz0应用、py_compile通过，源指纹见004审阅和`evidence/T26/remote_prepare.log`。没有003，没有改参考src或旧E2快照。
- 固定FCFS、TP1、BF16/SSM FP32、Triton MLA/KDA、page64/chunk8192、extra_buffer、track_interval256、max_states_per_path=-1、context131072/KV131072、Mamba512、max_running8、mem_fraction0.25、disable_cuda_graph、enable_metrics、incremental_streaming_output。实际startup的truncation_align均为null，故未覆盖审阅发现的额外alignment缺陷。
- 原Renderer完整prompt，N1/flush-per-chain，输出4token/ignore_eos仅替身实验。控制与候选各跑smoke6/reminder70/strict67；004栈unset再跑smoke6。共292次case请求、0错误；两条栈均用同一只读startup/flush/tail trace，**本轮不解释TTFT/SLO性能**。
- 批次脚本`scripts/run_e2b_batch.py`；回放命令由脚本冻结在run内。最终对账：`python scripts/compare_e2b.py /sjtu/linhang/arena/runs/E2b_T26_20260922_v3 --control-root /sjtu/linhang/arena/runs/E2b_T26_20260922_v2`，旧stock/v1.1收据来自`E2_20260922`。逐项核对ID及顺序、prompt SHA、链位置/phase、渲染与服务token数、输出预算和预测；不是比较不同子集。

### 结果与TEST_PLAN映射

“好/同/差”按cached_tokens越高越好；p95为fast_intra实际**未命中token数**，不是秒。

| 用例 | 请求/fast数 | 001+002控制 vs 旧v1.1 | 004 vs 控制：好/同/差 | fast未命中p95：v1.1→004 | role预测p95 | 候选逐项精确符合role预测 |
|---|---:|---|---|---:|---:|---:|
| smoke | 6/3 | 6/6相同 | 3/3/0 | 6315→562 | 562 | 6/6 |
| reminder_heavy | 70/54 | 70/70相同 | 27/43/0 | 8135→2439 | 2439 | 70/70 |
| strict_append | 67/49 | 67/67相同 | 0/67/0 | 3747→3747 | 3747 | 67/67 |

相对原stock，reminder候选58好/12同/0差（原v1.1为31好/39同/0差）。候选trace实际记录27次tail split，控制0次。三case控制与旧v1.1逐项完全一致，因而本限定配置的新增收益可归因于004，而非混称002收益。

| TEST_PLAN ID | 本轮结果 | 范围限制 |
|---|---|---|
| D1-01 | PASS：004栈unset smoke6/6等于原stock，fast p95仍6315，tail split=0 | 只新增smoke；旧v1.1四组记录保留 |
| D1-02 | PASS（v1.2）：70/70精确吻合预测，0退步，fast p95 8135→2439 | 原v1.1 FAIL保留；不外推完整GLM、N@SLO |
| D1-03 | PASS：strict67/67不变；扩展143配对项全0退步 | TP1/N1/extra_buffer |
| D1-07 | 原001 15/15 pass；004扩展10pass+1已知alignment缺陷expectedFailure | **expectedFailure不是安全门通过**，见004审阅§2 |
| D1-08 | PASS，限定本N1：候选28次成功flush全恢复KV131072/Mamba512/request8；控制28次、off4次同样恢复 | 不替代N4、取消、DP、HiCache/lazy测试 |
| IF-08 | 仅空闲JSON/真实清池子项通过；布尔success=true允许附message | 本轮未重跑busy/timeout和all-worker，T29旧TP1证明仍保留 |
| D1-04 / D1-05 / D1-10 | 本轮未重测数值、N4并发或角色恢复后同prompt再flush专例；原状态分别FAIL / wip / v1.1 PASS | 不用本轮缓存收益解除旧数值门，不用N1替代并发 |
| D2-04 / D2-05 | 002原16项CPU回归通过，新增004路径实测方法也拒绝host-miss第二partial | 是指定准入漏洞已覆盖，不是HiCache组合live验收 |

### 审阅、失败保留与清理

- 审阅`patches/004-role-boundary-final-chunk.md`：新建Adder先add_chunked_req后add_one_req，额外alignment字段尚未写入；page64/要求512的反例产生768长度。当前qfull对齐=null可限定测试，不能宣传通用DSA/deterministic安全。原001 §10.2 host-miss问题确认由002共享guard覆盖。
- CPU最终55项：**54通过、1已知对齐缺陷expectedFailure**（00411、原00216、原00115、比较器/flush检查6、replay7），`evidence/T26/cpu_final.log`。第一次短尾fixture写错的失败已修，保留原日志，不把它当生产缺陷。
- `E2b_T26_20260922/`首次遗漏make_case_sets依赖，尚未发送case请求便退出；服务自动停。`..._v2/`控制三case全部成功，但批次最终flush检查误拒绝合法的message字段；实际flush成功且池恢复。修正检查器并补正反例测试后，`..._v3/`**只续跑candidate/off**，不重跑或丢弃有效控制。所有失败/中断记录原样保留；一次SSH握手失败重试后成功，未改变引擎配置。
- 完整GPU artifacts：`/sjtu/linhang/arena/runs/E2b_T26_20260922{,_v2,_v3}/`（服务/回放日志、trace、manifest、请求、补丁与脚本冻结副本）；本地小证据`evidence/T26/comparison.json`、CPU/prepare/input/cleanup receipts。结果JSON记录各来源root，便于重算。
- **10:11:56 UTC批次完成**，自动停止自己创建的进程组。再次核实GPU0/1各4MiB/0%、无compute进程、31000无监听；`evidence/T26/final_host.log`。未操作既有daemons/其他项目，root README交协调方维护。T26完成后idle，不启动新任务。
