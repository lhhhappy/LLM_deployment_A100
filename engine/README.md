# 引擎源码、机制与版本

当前发布的 SGLang 源码是正式 09-30 CAP / 47798 使用的 `ca5d646c`，与 `engine/sglang/` 逐文件一致。最终成绩和两份配置见 [提交记录](../notes/submissions.md)；首次阅读先看 [源码地图](../notes/architecture.md) 和 [逐提交索引](../notes/read-history.md)。机制文档记录实现时的验证范围，后续组合成绩不能分摊为各机制的独立收益。

## 版本身份

| Git ref | 含义 |
| --- | --- |
| `engine-base` | 主办方固定底包，作为源码 diff 的起点 |
| `official-A-0923a` | 早期正式 A，保留 MTP 的历史校准基线 |
| `image-lh-img-0928c` | 47266 的 KDA prefill 组合，源码 `bf6b66fa` |
| `image-lh-img-0930a` | 47798 / 47800 共用源码 `ca5d646c`，通过配置区分两份提交 |
| `official-47798-n42` | 本次完整发布：原始源码、冻结配置、官方终态、造数方法与阅读文档 |

当前运行开关以 [official-0930-CAP.json](../submission/official-0930-CAP.json) 为准。源码包含启用路径、关闭候选和历史探索；“代码存在”并不表示本次运行使用了它。

## 按问题找机制

| 问题 | 实现与说明 |
| --- | --- |
| 接口、真实时间戳与 flush | [000](docs/000-interface-compliance.md) |
| A100 DSA / MoE 基础执行 | [110](docs/110-sm80-dsa-indexer.md)、[111](docs/111-sm80-fp8-moe-marlin.md)、[114](docs/114-indexer-row-shard.md) |
| KDA 边界状态、快照与主机层 | [101](docs/101-role-boundary-split.md)、[140](docs/140-kda-dual-snapshot.md)、[180](docs/180-hicache-glm-dsa.md) |
| 冷块、短命中与检查点对齐 | [106](docs/106-defer-chunk-on-no-kv.md)、[120](docs/120-sched-protect-chain.md)、[对齐修复](docs/120-chunk-alignment.md)、[121](docs/121-sched-cap-while-decoding.md) |
| 冷/热准入、块间停车与积压 | [124](docs/124-deadline-admission.md)、[停车](docs/124-multiround-parking.md)、[125](docs/125-opening-mode.md) |
| 共享前缀与 chain 优先 | [128 prefix producer](docs/128-prefix-producer.md)、[131](docs/131-chain-risk-interval.md)、[132](docs/132-chain-first.md) |
| prefill 执行成本 | [117 Humming](docs/117-sm80-fp8-moe-humming.md)、[up](docs/117-sm80-moe-up-followup.md)、[down](docs/117-sm80-moe-down-followup.md)、[reduce](docs/117-sm80-moe-reduce-followup.md)、[118](docs/118-dsa-sparse-triton.md)、[KDA prefill](docs/170-kda-sm80-prefill.md) |
| 09-30 小 M、decode 与元数据 | [175](docs/175-moe-small-m-down-tuning.md)、[176](docs/176-dsa-sm80-decode-page-loop.md)、[178](docs/178-kda-shared-prefix-mask.json)、[179](docs/179-mamba-prefill-track-gpu.json)、[181](docs/181-dsa-query-view.json)、[182](docs/182-mamba-alloc-gpu-index.md)、[184](docs/184-hicache-index-alias.json) |
| 关闭或未采用的探索 | [115 DCP](docs/115-dcp-sm80.md)、[local extend](docs/115-dcp-local-extend.md)、[122 pace](docs/122-tpot-paced-prefill.md)、[123 SRPT](docs/123-srpt-admission.md)、[126 demand](docs/126-demand-cold-cap.md)、[128 family](docs/128-family-leader.md)、[174 packed](docs/174-kda-safe-packed-decode.md)、[177 graph](docs/177-metadata-glue-startup.json) |
| 其他历史执行候选 | [119 scatter 门控](docs/119-scatter-min-tokens.md)、[130 async tokenize](docs/130-async-tokenize.md)、[150 warmup](docs/150-startup-warmup.md)、[160 MTP](docs/160-nextn-sm80.md)、[170 breakable graph](docs/170-glm-bcg-prefill.md)、[171](docs/171-kda-bf16-proj-fusion.md)、[172](docs/172-moe-clamped-swiglu.md) |

170 的早期 breakable prefill graph 与后续 KDA prefill 优化是两项不同工作。最新配置启用三个 KDA prefill 开关，未请求 breakable graph。124 也曾在迁移前用于一个未采用的短命中预留方案；当前 124 指 deadline-tiered admission，应以提交与文档内容识别。

## 阅读和开发工具

```bash
python3 scripts/verify_release.py
git diff engine-base ca5d646c -- engine/sglang
git show 6b629210 -- engine/sglang engine/docs tests
python3 scripts/engine/tree.py ca5d646c
```

`tree.py` 支持固定 ref、`HEAD`、`mech:NNN` 和 `before:NNN`，历史树缓存到忽略的 `build/engine/trees/`。`scripts/engine/export.sh <ref>` 导出底包到指定版本的 diff。既有 Pod 任务以 `G_COMMIT`、`G_ARGS`、`G_ENV` 固定身份，并用 `G_EXPECT` 核对实际生效的机制；具体操作见 [Pod 说明](../scripts/pod/README.md)。服务启动和增量镜像构建见 [复现入口](../notes/reproduce.md)。

开发约束见 [AGENTS.md](../AGENTS.md)：在所属源码目录改动；一项机制用一条或一组可理解的 `engine NNN:` commit；修复沿用所属机制编号；默认关闭路径、形状/数值、资源所有权与不支持组合的拒绝条件须明确。未修改正式源码的文档或 CPU 测试修复独立提交。

## vLLM 路线

`engine/vllm/` 是独立探索，固定底包为官方 vLLM `a811738a6`，使用 `vllm-base-a811738a6`、`engine vllm NNN:` 和 [docs/vllm/](docs/vllm/README.md)。上述 tree、export、镜像和 Pod 工具面向 SGLang，不自动兼容 vLLM。它不属于当前 N42 正式提交。
