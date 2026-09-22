# candidate-01 — SGLang + D0/D1 patches, GLM-5.3-Flash, 8×A100（Claude，2026-09-22，草稿，**未提交、未构建**）

状态：**只是提交包的准备**。镜像还没打（`image` 是占位符），没起过服务，也没压测。提交前要做三件事：用户批准；双方确认（rule.md §1）；`scripts/check_submission.py submission/candidate-01.json --final --trace submission/stub-trace.jsonl` 零 error。

## 文件
- `candidate-01.json`：四个字段。提交时复制为 `<outputs>/submission.json`（task.md：`--outputs` 下必须能读到 `submission.json`）。
- `stub-trace.jsonl`：按 task.md 原样写的占位轨迹。`playground trace validate` 返回 valid。
- 镜像：`scripts/build_image.sh` 生成 `build/image/Dockerfile`（6.3 KB，远低于 64 KiB 上限），只生成、不构建。补丁用 `patch -p3 -d <sglang 包目录>` 应用，`--fuzz=0`，失败就让构建失败。

## command 逐项说明（每个 flag 都在 src/sglang v0.5.20 @94602c9 里核对过）
flag 由 `arg_groups/arg_utils.py:353` 从字段名生成（`"--" + name.replace("_","-")`），bool 字段一律是 `store_true`（`arg_utils.py:456`）。

| flag | 值 | 出处（字段定义） | 为什么 |
|---|---|---|---|
| `python3 -m sglang.launch_server` | | | argv 的第一个 token 是可执行文件（task.md「command 不是 shell」） |
| `--model-path` | `/mnt/models` | `fields/model.py:52` | 平台挂载点（task.md 示例；I：以 Trisol `--model glm-5-3-flash:2` 的实际挂载为准） |
| `--host` / `--port` | `0.0.0.0` / `8000` | `fields/serving.py:72-73` | 平台按固定端口接管 |
| `--tp-size` | `8` | `fields/parallel.py:52` | 8×A100 |
| `--served-model-name` | `default` | `fields/serving.py:163` | 必须等于 `model_name`，能力评测才能命中 |
| `--enable-metrics` | | `fields/observability.py:77` | 自测时用 Prometheus 指标（不影响计分） |
| `--incremental-streaming-output` | | `fields/serving.py:265` | `/generate` SSE 按增量分段输出（D0 接口合规；是否必需以 task.md「压测口」为准） |
| `--page-size` | `64` | `fields/schedule.py:145` | 与 E1/F13 的前提一致；`mamba_track_interval`（默认 256，`fields/exec_.py:385`）必须是它的整数倍（`mamba_hook.py` 的 assert），满足 |
| `--mamba-radix-cache-strategy` | `extra_buffer` | `fields/exec_.py:367`（choices auto/no_buffer/extra_buffer/extra_buffer_lazy） | KDA 前缀命中需要快照；D1 补丁建立在 extra_buffer 的 track/donation 路径上 |
| `--reasoning-parser` | `glm45` | `fields/serving.py:199`；choices 来自 `parser/reasoning_parser.py` DetectorMap（有 `glm45`，**没有** glm5/glm53 专用项） | 和主办方 vLLM 示例一致；把 `<think>…</think>` 拆成 `reasoning_content` |

`env`：
- `SGLANG_OPT_USE_TOPK_V2=0`：A100 硬性要求（task.md），否则 CUDA graph 捕获时 JIT 会挂。
- `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829`：D1 补丁的开关（`<|user|>`、`<|observation|>`；`patches/001-*.patch` 第 19–23 行按逗号解析；不设置就是 stock 行为）。**只有镜像里打了 001 补丁时才有意义**。

静态核对：`scripts/check_submission.py` 从源码字段里抽出 500 个已知 flag，本 command 用到的 10 个全部存在。**没能做到的**：用 sglang 自己的 argparse 真正解析一遍。本地容器没装 numpy，`import sglang` 失败。所以 choices 校验、`resolvable` 字段的解析逻辑、GLM5 的模型级 override（`cuda_graph_hook.py:404/422` 的 glm5 chunked-prefill / prefill cuda graph 默认值）都没有实际跑过。

## 未决问题
1. **底包是否已经设了这些**：`arena-sglang-glm53:260918` 的 `ENV`/入口脚本可能已经设了 `SGLANG_OPT_USE_TOPK_V2`、chat template 渲染（task.md 提到"用镜像里的脚本 `--exec` 接真正的 server 命令"），也可能已经打了接口补丁。`bohr image get/dockerfile` 返回 403（README 阻塞项 3），所以还没法确认。如果底包自带 `/generate` 增强或渲染脚本，command 可能要改成经那个脚本 `--exec`。
2. **底包里的 sglang 版本**：补丁是对 v0.5.20 @94602c9 核对的（在 scratch 副本上 dry-run 加真实应用、py_compile 都通过）。底包如果不是这个版本，构建会在 `patch` 那一步**明确失败**，不会悄悄跳过。
3. **`--model-path /mnt/models`**：task.md 的示例和 vLLM 说明都用 `/mnt/models`，但 SGLang 路线下 Trisol 是否同样校验这个 flag，还没实测。tokenizer/模型 revision 由平台 `glm-5-3-flash:2` 钉住（I）。
4. **reasoning parser**：`glm45` 会不会影响能力评测的答案抽取（评测读 `content` 还是也读 `reasoning_content`），不知道。vLLM 示例用的也是 glm45，暂时照用。没加 `--tool-call-parser glm47`（这个 flag 存在，`server_args.py:374`）；能力评测和压测看起来不需要 tool 解析，待定。
5. **MTP / 投机解码暂缓**：模型有 1 层 MTP（F1）。`--speculative-*` 和 extra_buffer 的约束（`mamba_track_interval >= speculative_num_draft_tokens`）、D1 与 verify 路径的交互都还没评估，所以本候选不开。
6. 容量和调度相关的 flag 都没调：`--mem-fraction-static`、`--max-running-requests`、`--chunked-prefill-size`、cuda graph 的 bs、DP-attention（D2/D3）。这些要等 8 卡自测才能定。本候选只是"接口合规 + D1"的最小形态。
7. `--enable-metrics` 在正式评测里用不上，开销应该可以忽略，但没实测。
8. `playground trace validate` 0.1.39 在本地对"只有 session_start"的文件也返回 valid，而 task.md 说服务端会 400。所以轨迹格式以 `check_submission.py --trace` 的检查为准。

## 复现步骤（全部本地，不花配额）
```bash
scripts/build_image.sh                       # 生成 Dockerfile + 载荷往返校验 + scratch dry-run
scripts/check_submission.py submission/candidate-01.json --trace submission/stub-trace.jsonl
source env.sh && playground trace validate --trace submission/stub-trace.jsonl
```
下面这些要用户批准：`bohr image build --dockerfile build/image/Dockerfile --name <name:tag> --project-id <ID> --wait -o json`，然后把 digest 钉进 `image`，再做 8 卡自测，最后提交。


## 2026-09-22 更新（Claude，依据 W8 R11）
candidate-01 默认开着 D1，不适合作为基线，已被拆成四个显式 profile：`candidate-b0-d0-baseline`（只有 D0，D1/D2 都关）、`-b1-d1`、`-b2-spf`、`-b3-spf-d1`。SPF 的策略名要等 patches/002 确定后再替换占位符 `SPF_NAME_FROM_PATCH_002`。四个 profile 共用同一个镜像（000+001+002），只靠开关区分。
