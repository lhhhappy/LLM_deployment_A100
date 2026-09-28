# 2026-09-28 prefill 内核审查归档

这是离线归档，不是新一轮实验。现行汇总是 [R36](../../../research/codex/R36_prefill_kernel_options_0928.md) 和 [KDA R37](../../../research/codex/R37_kda_prefill_sm80_0928.md)；原始GPU证据留在 [prefill-kernels-0928](../../../evidence/prefill-kernels-0928/) 与 [moe-scale-fold-0928](../../../evidence/moe-scale-fold-0928/README.md)，没有改写或删除。

| 快照 | 用途与阅读边界 |
| --- | --- |
| [sol-pr-audit.md](sol-pr-audit.md) | 冻结的上游PR适用性审查；早期建议是当时计划，最后段落已记录KPool mirror撤回。PR状态未在本次离线归档重新查询。 |
| [moe-architecture-audit.md](moe-architecture-audit.md) | MoE源码、scale-fold已闭合结果、N条带与整FFN容量/流量拒绝理由。 |
| [scale-fold-review.md](scale-fold-review.md) | 独立raw/二进制隔离核验，有限FP8和scale负零证明边界。 |
| [mhc-splitk-review.md](mhc-splitk-review.md) | mHC测量范围、原生分支命中和生成global-store地址审查；不等于全面race-free或真实权重验收。 |
| [probe-review.md](probe-review.md) | KDA局部图与DSA探针的早期方法审查。DSA b后续已补fresh FP8 Q/K和memory读数，见R36；KDA最终结论由另文负责。 |

原件在 `/workspace/Agentic_science_challenge/build/scratch/kernel-next-0928/`。归档副本只增加快照提示并将本地Markdown链接转为相对路径，技术正文未重写。Humming header链接指向MoE证据中已冻结的native树；该树没有包含的 `tensor.h`、`smem.py`、`sm8x.py` 连同许可证保存在 [sources/humming-0.1.12](sources/humming-0.1.12/)，仅用于复核容量/布局源码。原worktree的引擎/既有报告链接在确认内容与当前副本相同后本地化。

[sources.json](sources.json)记录每份原审计与归档件的独立SHA-256、支持源码SHA、R36/本README SHA，以及本次引用的raw、identity、source snapshot和MoE证据README SHA。归档文件因增加提示/修改链接，哈希与scratch原件不同是预期变换；raw/source快照未改动。MoE tar内的原manifest继续保持原值。

归档复算包括：MoE 156项内部manifest、16条raw-bit记录、native/fold同stage配置相同且cubin不同；mHC/DSA逐轮median与配对降幅；probe/engine SHA与raw environment；本地Markdown链接。本次不做GPU或生产回归。文档后续若更新，应同步本目录对应文档SHA，不能据新的README哈希否认原始运行身份。

KDA 最终审查另见 [prepare](kda-prepare-review.md)、[state/F](kda-state-production-review.md)、[CPU length](kda-cpu-length-review.md)、[C 原型 raw](kda-prepare-c-raw-review.md)。[execution-audit](execution-audit.md) 保留局部图/BV8 等未采用路线的原始研究边界。后加 KDA 文件哈希见 `kda-review-sha256.json`；旧来源清单的技术原件保持不变。
