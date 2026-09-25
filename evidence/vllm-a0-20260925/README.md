# vLLM 路线 A0/A5 收据（2026-09-25，Claude）

引擎：`engine/vllm` @ `8e289cf4`（底包 a811738a6 + 000 + 010），开发机 `/sjtu/linhang/arena/vllm`，官方包 vllm-0.30.1rc1.dev114+ga811738a6（SHA256 见开发机 `wheels/*.sha256`）。

| 文件 | 内容 | 层级 |
|---|---|---|
| `prompt-tokens-full-5601.json` | `scripts/vllm/check_prompt_tokens.py`：长链集（manifest SHA256 19a7e5a6…）5601 条提示词经 harness 渲染、vLLM 补全渲染器分词，与冻结 `glm_tokens` 比对：0 条不一致，共 393,887,274 token | CPU |
| `probe-interface-tp2-8l-dummy.json` | `scripts/vllm/probe_interface.py` 对两卡服务（TP2、8 层截短配置 + MTP、dummy 权重、`max_model_len` 131072）的接口探针：7 项全部通过 | 两卡替身 |

两卡探针的 TTFT 与 MTP 事件数来自 dummy 权重的 8 层替身，不代表真实模型性能或 MTP 接受率。
缓存命中数与模型权重无关：87,995 token 的提示词二次命中 85,248 = floor(L/B)·B − B，B=2304（启动日志 "Setting attention block size to 2304 tokens"）。
