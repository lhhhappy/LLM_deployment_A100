# vLLM 路线（Claude）

2026-09-25 接手，交接见 [vllm-claude-code.md](../../../notes/handoffs/vllm-claude-code.md)。本页顶部是现状摘要，明细链接到证据。

## 现状

- **底包**：官方 vLLM `main` a811738a6（tag `vllm-base-a811738a6`），用官方预编译包 + 我们的 Python 源码开发。选择理由与导入方式见 [engine/docs/vllm](../../../engine/docs/vllm/README.md)。
- **A100**：官方 vLLM 不支持在 sm80 上跑 GLM-5.3-Flash（稀疏 MLA 无 sm80 后端，索引器要求 DeepGEMM）。移植进行中，参考公开分支 wtdcode/vllm-backport。
- **接口**：官方 `main` 已有 `vllm.endpoint_plugins` 扩展点；`/generate` 与 `/flush_cache` 插件待实现。
- **GPU**：开发机 2×A100 环境搭建中（`/sjtu/linhang/arena/vllm/`）；共享 8 卡当前无运行中的 Pod，vLLM 尚无 TP8 证据。
- **调度对照 R27**：进行中。

## 文件

| 文件 | 内容 |
|---|---|
| [engine/docs/vllm/README.md](../../../engine/docs/vllm/README.md) | 底包、开发规则、环境 |
