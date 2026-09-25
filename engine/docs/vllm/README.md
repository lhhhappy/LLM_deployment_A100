# engine/vllm — vLLM 路线的底包与开发规则

2026-09-25，Claude 负责。按用户决定，vLLM 路线从**官方 vLLM** 源码开发，不从主办方镜像取源码；镜像以后再合并。
SGLang 路线（`engine/sglang/`、`engine/docs/NNN-*.md`、tag `engine-base`/`official-A-0923a`）与这里互不共用编号和 tag。

## 底包

| 项目 | 值 |
|---|---|
| 来源 | 官方 [vllm-project/vllm](https://github.com/vllm-project/vllm) `main` |
| 提交 | `a811738a6051b5b12a7fcf800f465f2dd2df0a0e`（提交时间 2026-09-24T17:17:10-07:00，#57743） |
| 源码包 | `https://github.com/vllm-project/vllm/archive/<提交>.tar.gz`，SHA256 `63ddb29bf744de1dfeaa6efa1351f4fcf8457a32036974bed4e31612f173f2d0` |
| 官方预编译包 | `https://wheels.vllm.ai/<提交>/vllm-0.30.1rc1.dev114%2Bga811738a6-cp38-abi3-manylinux_2_28_x86_64.whl`（CUDA 13.0，依赖 torch 2.13.0） |
| 本仓库 tag | `vllm-base-a811738a6` = 导入提交，之后不移动 |

导入时去掉了与运行和测试无关的 `docs/`、`.buildkite/`、`.github/`；其余文件逐字节等于源码包。
上游 `.gitignore` 会挡住 5 个上游本身跟踪的文件（2 个 benchmark JSON、2 个 HIP 测试、`vllm/vllm_flash_attn/.gitkeep`），导入时用 `git add -f` 保留。
核对方法：解开源码包，`diff -r --exclude=docs --exclude=.buildkite --exclude=.github <解包目录> <git archive vllm-base-a811738a6:engine/vllm>` 应无输出。

### 为什么选这个提交

- GLM-5.3-Flash 在官方 `main` 于 2026-09-04（#53906）加入，第一个包含它的正式版是 0.30.0（分叉点 2026-09-15）。
  分叉点之后 `main` 又合入了与本负载直接相关的修复，0.30.0 没有：混合模型状态块估算错误导致请求无法准入（#57050）、
  MTP 下 prompt 末尾的前缀缓存命中（#58368）、GLM 稀疏索引器 decode 工作区（约省 3 GiB/卡，#57701）、kpool 索引器若干正确性修复。
- 官方对每个 `main` 提交发布预编译包，选"有官方预编译包的最新提交"，就能用官方二进制加我们的 Python 源码开发，不必自己编 CUDA 扩展。
- 官方 vLLM 本身**不能**在 A100 上跑这个模型：稀疏 MLA 注意力没有 sm80 后端，稀疏索引器要求 DeepGEMM（Hopper 及以上）。
  这部分 A100 移植是我们自己的提交（`engine vllm` 编号见下），参考公开分支 [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)；
  该分支只作参考实现，不是底包，也不声称等于主办方 `vllm-backport:260918-sm80` 镜像。

## 机制

| 编号 | 机制 | 开关 | 状态 |
|---|---|---|---|
| 000 | `/generate`（SGLang 形 SSE）与 `/flush_cache` 端点插件，启动打印 `[ax] vllm mechanisms:` | `VLLM_PLUGINS` 含 `generate_compat` | CPU 测试通过；两卡替身接口探针见 research/claude/vllm |
| 010 | GLM-5.3-Flash 稀疏注意力层在 A100 上运行（fp8 软件编码、Triton 索引器打分、Triton 稀疏 MLA 后端） | 按硬件：仅 SM8x | A100 单卡内核测试通过；两卡替身（截 8 层 + MTP、dummy 权重）启动、CUDA graph 捕获、接口探针通过；TP8 未验证 |
| 101 | 在提示词最后一个开轮 token 处保留 KDA 检查点，供下一轮续算（候选，不在冻结基线） | `VLLM_AX_MAMBA_ROLE_CHECKPOINT_TOKEN_IDS`，空 = 关 | CPU 测试通过；上游调度器/前缀缓存测试与底包对照无新增失败；GPU 未验证 |

说明见 [000](000-generate-compat.md)、[010](010-sm80-glm5next.md)、[101](101-role-boundary-checkpoint.md)。

## 开发规则

- 只改 `engine/vllm/`。每个机制一个或一组 `engine vllm NNN:` 提交，说明写 `engine/docs/vllm/NNN-*.md`；修正并入所属机制，不另开编号。
- 编号：`0NN` 为跑通所需（接口、A100 移植）；`1NN` 以后为优化机制。与 SGLang 同号（如 120）只表示针对同一类问题，不代表实现等价，文档里写清对应关系。
- A100 移植按平台门控：只在 compute capability 8.x 上生效，Hopper 及以上仍走底包路径。其余机制默认关闭，关闭时等于底包；不支持的组合启动时拒绝，不静默绕开。
- 启动时打印一行 `[ax] vllm mechanisms:`（底包提交、各机制状态），任务写明期望值并在测量前核对。
- 共享工作区有他人未提交的改动：只按路径暂存自己的文件，不 `git add -A`。

## 开发与验证环境

- 开发机：`/sjtu/linhang/arena/vllm/`（`env/` 独立虚拟环境，`src/` 源码副本，`wheels/` 官方预编译包，`runs/` 探针输出）。
  安装方式为官方"Python-only build"：`VLLM_USE_PRECOMPILED=1` + `VLLM_PRECOMPILED_WHEEL_LOCATION=<本地官方包>`，editable 安装 `src/`。
- 开发机只有 2 张 A100，装不下完整 FP8 模型；两卡上用真实 config 截短层数、dummy 权重做接口与算子探针，结论标"两卡/替身"。TP8 与真实权重另排。
- 现有 `scripts/engine/*`、`scripts/build_image.sh`、`scripts/pod/lib.sh` 只适配 SGLang，不能直接用于 vLLM。
