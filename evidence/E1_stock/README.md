# E1 stock：本地小摘要

实际运行：2026-09-22，GPU 开发机单张 A100，SGLang commit `94602c9`。

- `summary.json`：stock20 的原始汇总，72请求/20链，0错误，70个stock预测精确匹配。
- `manifest.json`：相同运行的完整选择列表、参数、harness/tokenizer/replay 哈希。
- `smoke_events.jsonl`：stock_smoke3 的接口 smoke 及链间 flush，原始小日志。
- `model_manifest.json`：实际运行的 qfull 随机替身参数形状、配置/权重/分词器哈希。

全量请求记录、源码/启动快照、环境包清单和服务日志保留在 GPU 机
`/sjtu/linhang/arena/runs/E1_20260922/`。模型在
`/sjtu/linhang/arena/models/e1-kimi-linear-4l-qfull/`，不要使用较早失败的无-qfull目录。

解释、两个预测偏差及 E2 复现入口见 `notes/experiments.md` 的 E1 最终结果。
这是小模型缓存功能验证，不是正式压测或 N@SLO 成绩。原生 flush 文本也不满足 D0 JSON 合约。
