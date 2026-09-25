# vLLM 路线（Claude）

2026-09-25 接手，交接见 [vllm-claude-code.md](../../../notes/handoffs/vllm-claude-code.md)。本页顶部是现状摘要，明细链接到证据。

## 现状（2026-09-25 06:00 UTC，阶段交接）

- **冻结基线**：`engine/vllm` @ `fb18e488` = 官方 vLLM `main` a811738a6（tag `vllm-base-a811738a6`）+ 000 + 010。
  部署包由 `scripts/vllm/build_wheel.sh <commit>` 从提交打出（已为 `8e289cf4` 打过：320 MB；`fb18e488` 需重打）。
- **000 接口**：`/generate`（SGLang 形 SSE）与 `/flush_cache` 端点插件；收到时刻取自 ASGI 入口（与 SGLang 000 同层），flush 返回结构化收据（`level_verdict` 已接）。CPU 20 项。
- **010 A100 移植**：稀疏注意力层三处 sm80 缺口（后端、索引器打分、fp8 写入）已补，按硬件门控，H100+ 路径不变。A100 单卡内核测试 48 + 123 + 8 项通过。
- **两卡替身（TP2、真实配置截 8 层 + MTP、dummy 权重）**：启动、CUDA graph 全部捕获、接口探针全部通过。
- **token 一致性（CPU）**：长链集 5601 条提示词经 vLLM 渲染器分词后与冻结 `glm_tokens` 逐条相等。
- **契约复核**：[contract-review-0925](contract-review-0925.md)——prompt、`ignore_eos`、首 token 与收到时刻所在层、TPOT、`cached_tokens`、flush、默认关闭，逐项给依据与未验证边界。
- **101 角色边界 KDA 检查点**：独立候选、默认关闭；CPU 17 项通过（含 Codex 复核发现的"解码填满角色块后键丢失"回归）；上游测试与底包对照见契约复核；GPU 数值与 TTFT 代价未验证。[说明](../../../engine/docs/vllm/101-role-boundary-checkpoint.md)
- **比较原则（用户 2026-09-25 澄清）**：同一评测下比较两路谁更好，不要求生成文本、缓存命中、调度顺序或内部参数相同；已证明有效的 SGLang 设计能用就复用，vLLM 原生更好就用原生；不把实现等价作为评测前置。细则见[契约复核](contract-review-0925.md)末节。
- **调度对照 R27**：[R28](R28_vllm_scheduler_vs_R27.md)。vLLM 同样有"队首放不下即停止扫描""长块占满一步"，没有单 partial 限制；最大的 vLLM 特有损失是 KDA 检查点保留位置（两卡实测，原生开关可降一半）。
- **TP8**：未做。072（真实权重冒烟）预留在 071 之后，未入队；新 GPU 实验先与用户讨论配置、预算、判据。

## 冻结基线的开关默认值

| 机制 | 开关 | 默认 |
|---|---|---|
| 000 | `VLLM_PLUGINS` 含 `generate_compat`（`scripts/vllm/serve.sh` 设置） | 仅在点名时加载 |
| 010 | 按硬件（compute capability 8.x） | A100 上开，H100+ 不生效 |
| 101 | `VLLM_AX_MAMBA_ROLE_CHECKPOINT_TOKEN_IDS` | 空 = 关（不在冻结基线里，候选） |
| 原生缓存开关 | `--prefix-cache-retention-interval None --prefix-match-unit 64` | 底包默认 0 / 块大小；两卡实测重算 −48%，未上八卡 |

## TP8 待验证

真实权重数值与能力（12 题冒烟、与 SGLang 正式 A 的贪心输出对照、长上下文检索）、MTP 接受率、实际块大小 B 与 KV/KDA 容量、graph 显存、
TTFT 两端时间戳与客户端时间的分布对照、真实运行中的 flush 收据、性能。

## 证据

| 路径 | 内容 |
|---|---|
| [evidence/vllm-a0-20260925/](../../../evidence/vllm-a0-20260925/README.md) | token 一致性、两卡接口探针、两卡多轮复用回放收据 |
| [engine/docs/vllm/](../../../engine/docs/vllm/README.md) | 底包、000/010/101 说明与验证层级 |
| `scripts/vllm/` | 开发机环境、同步/安装/测试、服务启动、打包、探针、复用模型 |
