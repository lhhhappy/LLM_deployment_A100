# HANDOFF：交接给下一个 session（2026-09-23 凌晨）

先读：`CLAUDE.md` → `AGENTS.md` → **本文件** → `research/README.md` → `research/claude/base/00-summary-mainline.md`。

## 0. 一句话现状
底包 SGLang 在 A100 上原样根本起不来（DeepGEMM / fp8 Triton 不支持 sm80）。我们在写**补丁 110** 让它在 A100 上启动；
L2 的 8×A100 服务已上线，pod 内排队机制已跑通，005 因 fa3 仅支持 Hopper 失败，已改用 tilelang 后端（F57），**任务 006 正在 8 卡上验证启动，007（开发集 N6）已排队**。
**用户指示：先把测评跑通并记录，暂不正式提交。**

## 1. 必守规则（用户明确要求）
- **8 卡服务不停、不删、不释放，停必须用户同意**（守护进程 idle_hold 已改 1e9）。尽量不让它空闲。
- 调研优先、大胆读源码改源码；**调参放最后**；每次压缩/新会话后先重读 `research/`。
- 工程细节要认真：先把最短链路在真实环境跑通，再加功能；不要应付式修补。
- 正式提交每天 2 次（平台消息写 3，按用户的 2）；自测不限。审批已授权 Claude，每次提交向用户汇报。
- 路线保密：Trisol 上的镜像名/服务名/描述/command/env 中性；token 不打印不落盘。
- 用户要中文沟通。

## 2. 关键事实（都有证据，见 notes/findings.md）
- **底包 = GitHub 公开 sglang@fe236ea6c3 + 两处多模态修复**；逐字节副本 `build/base_exact/`（4686 文件指纹全对，F53/F54）。
  L3 实际代码：A=`build/l3_0922e`（+000），B=`build/l3_0922f`（+000+101）。**写补丁一律照 base_exact，不照 src/sglang（v0.5.20）。**
- **原版在 A100 起不来**（F56，2 卡替身 + 真实 8 卡都复现）：DSA indexer 调 DeepGEMM → `Unsupported architecture`；
  还有 Triton `fp8e4nv not supported`。所以已交的 A/B 不可能跑出分。
- 正式提交 image 字段**只能写 `名字:tag`**，`tag@sha256` 会被平台拒（F55）。45734/45735 部署失败（不计额度）；
  45766/45767（格式已修）在队列，但因 F56 预计也起不来（部署失败不计额度）。
- 排行榜（`data/all_att_2026-09-22b.json`）：3 人 N=22，LewyM tpot 0.0273 同分第一；夺冠需 N=26 或 N=22 且 tpot<0.0273。
- 源码地图（主线依据）：`research/claude/base/01–04`；最大性能瓶颈：长冷启动分块预填充独占 GPU（预填充优先于 decode，分块续算吃满预算）。

## 3. 补丁 110（进行中）— `scripts/make_110.py` 生成 `patches/110-sm80-dsa-indexer.patch`
- 源：`build/p110/sm80_deep_gemm.py`（DeepGEMM 三个 MQA-logits 入口的 torch 实现，装成 deep_gemm 模块上的惰性分发）、
  `build/p110/ax_soft_fp8.py`（sm80 软件 e4m3 编码 + `_ax_store_fp8`）。
- 改的文件：dsa_backend.py、dsa/dsa_indexer.py、dsa/dsa_indexer_kpool.py（包 deep_gemm）；dsa/kpool_fp8_index.py（4 个启动器 + 5 处 fp8 写入）；
  kernels/ops/attention/dsa/triton_kernel.py（act_quant）。生成器用行锚定正则 + 命中数断言 + 残留检查。
- 单测（GPU 机 A100 上已过）：`build/p110/test_sm80_deep_gemm.py`（误差<0.5%、CUDA graph 可捕获）、
  `test_soft_fp8.py`（22 万值逐比特一致）、`test_act_quant.py`（100% 字节一致）。
- 启动参数另需：`--dsa-prefill-backend tilelang --dsa-decode-backend tilelang`（index_kpool=4 时默认 flashmla_sparse 不被接受；fa3 在 A100 报 Only Hopper，F57），
  env `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`（mHC 走 tilelang）。都已写进 pod 任务脚本。
- 8 卡迭代历史：002 DeepGEMM(kpool_plan 未覆盖) → 003 fp8e4nv(kpool_fp8_index) → 004 fp8e4nv(act_quant) → 005 fa3 仅 Hopper → **006 tilelang 进行中**（pod 单卡核测试已过，F57）。
  若 005 仍失败：看 `scripts/pod/podq log 005-b110_start_probe`，按调用栈找下一个 sm80 不兼容点，**先在 GPU 机 A100 上单测再上 8 卡**。

## 4. 8 卡 pod 工作方式（`scripts/pod/`，在 GPU 机上运行；README 在该目录）
- 服务：`lh-arena-sess-a`，service id `2102309548588015616`，team arena。pod 内底包 `/sgl-workspace/sglang`（只读用），模型 `/mnt/models`。
- `pexec '<cmd>'` 进 pod 执行；`ppush <dest> <paths>` 传文件（bohr exec 不转发 stdin，base64 分块 8 路并行，41MB≈1 分钟）；
  `pstatus` 一眼状态；`podq init|submit <job>|ls|log <job>|pause|resume|cancel`（pod 内 worker 顺序执行 `/tmp/ax/queue`）。
- 版本管理：`lib.sh:prepare_src <名> <补丁...>` 从原版复制到 `/tmp/ax/src/<名>` 再打仓库补丁（按补丁内容哈希自动重建）；
  `ensure_engine` 同代码同参数则复用正在跑的服务（避免重载）。模型冷加载约 21 分钟，页缓存热时 2–7 分钟。
- 任务脚本：`scripts/pod/jobs/`：`b110_start_probe.sh`（启动+500..20000 token 探测+flush）；`dev_b110_n{6,10,14,18,22}.sh`（官方 run_dev.py 原样，
  tokenizer 用 /mnt/models）。开发集已在 pod `/tmp/ax/s1/s1-dev`。
- 连接：本地 → `scripts/gssh`（带重试的 ssh GPU）→ GPU 机 → pod。SSH 隧道偶尔断，gssh 会重试。干活放 GPU 机 tmux（会话 `arena-daemons`）。
- 旧守护进程（`scripts/trisol_test_daemon.py`，tmux `arena-daemons:l2`）现在只负责**保住服务**；旧 L2 队列 10 项都因 stdin 上传失败，已无意义，实验一律走 podq。

## 5. 下一步（按用户认可的顺序）
1. 让 B+110 在 8 卡上启动并通过探测（任务 005 或后续修复）。
2. 在 pod 里跑开发集压测：`podq submit scripts/pod/jobs/dev_b110_n6.sh`，再 10/14/18/22；把 summary（各门 TTFT p95、tpm、n_at_slo）记进 `notes/experiments.md` 和 findings。
3. 记录清楚后再考虑：用 000+101+110 打新镜像（`scripts/build_image.sh`，tag 唯一）→ `scripts/submit_official.sh`（command 要加 fa3 两个参数与 env）。
4. 之后主线（决策 30）：M1 调度保护链中间请求（源码主瓶颈）、M2 KDA 双点 fp32 快照（参考 vLLM #56960）、M3 分词移出事件循环 + 路由键、M4 MTP。

## 6. 记录位置
决策 `notes/decisions.md`（最新 30）；事实 `notes/findings.md`（最新 F56，顶部有"当前事实基线"）；派发 `notes/dispatch.md`；
提交 `notes/submissions.md`；计划 `plans/active/`；调研索引 `research/README.md`。编号先 `python3 scripts/next_id.py F|T|D`，写完 `python3 scripts/check_records.py`。
GPU 机仓库镜像 `/sjtu/linhang/arena/repo`（改完本地后 tar 同步过去）；GPU 机只在 `/sjtu/linhang/arena` 下工作，其他 tmux 会话属用户别的项目，别动。
