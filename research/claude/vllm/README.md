# vLLM 路线（Claude）

2026-09-25 接手，交接见 [vllm-claude-code.md](../../../notes/handoffs/vllm-claude-code.md)。本页顶部是现状摘要，明细链接到证据。

## 现状（2026-09-25 04:10 UTC）

- **底包**：官方 vLLM `main` a811738a6（tag `vllm-base-a811738a6`），官方预编译包 + 我们的 Python 源码。理由与导入方式见 [engine/docs/vllm](../../../engine/docs/vllm/README.md)。
- **000 接口**（`05dca72d`）：`/generate`（SGLang 形 SSE）与 `/flush_cache` 作为官方端点插件实现，不改核心。CPU 测试 19 项通过。
- **010 A100 移植**（`8e289cf4`）：稀疏注意力层的三处 sm80 缺口（后端、索引器打分、fp8 写入）已补；H100 及以上路径不变。A100 单卡内核测试全部通过。
- **两卡替身验证（实测，开发机 2×A100，TP2，真实配置截成 8 层 + MTP，dummy 权重）**：启动成功、CUDA graph 全部捕获、
  接口探针全部通过（token 数、`ignore_eos`、逐事件累计计数、首 token 时刻、命中后上涨、flush 后归零、并发、`/chat/completions`）。
- **token 一致性（实测，CPU）**：长链集全部 5601 条提示词经 vLLM 渲染器分词后与冻结 `glm_tokens` 逐条相等（3.94 亿 token，0 条不一致）。
- **TP8**：未做。共享八卡目前无运行中的 Pod；vLLM 在 Pod 上需要另建运行环境，待与 Codex 协调排队。
- **调度对照 R27**：[R28](R28_vllm_scheduler_vs_R27.md)。结论：vLLM 同样有"队首放不下即停止扫描"和"长块占满一步"，但没有单 partial 限制；换引擎不会自动解决等待。

## 证据

| 路径 | 内容 |
|---|---|
| [evidence/vllm-a0-20260925/](../../../evidence/vllm-a0-20260925/README.md) | token 一致性收据、两卡接口探针收据 |
| [engine/docs/vllm/000](../../../engine/docs/vllm/000-generate-compat.md)、[010](../../../engine/docs/vllm/010-sm80-glm5next.md) | 机制说明与验证层级 |

## 下一步（按依赖）

1. TP8 真实权重：启动、12 题冒烟、与 SGLang 正式 A 的贪心输出对照、长上下文检索、MTP 接受率；记录实际块大小与 KV/KDA 容量。
2. 原 harness 全流程短测（preflight→短预热→flush→测量→判分）在 vLLM 上跑通，再做同负载完整对照。
3. 首轮优化候选（单变量）：显式设 `--max-num-batched-tokens`、`--long-prefill-token-threshold`；主机层 `--kv-offloading-size`（GLM 上先验正确性）。
