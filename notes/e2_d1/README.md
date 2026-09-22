# E2 — D1 v1.1 缓存与数值结果

这是随机小Kimi-Linear（3 KDA + 1 MLA，无DSA）、单A100的功能验证，不是完整GLM或正式成绩。

- `comparison.json`：四个相同集合stock/off/on逐请求join，含prompt哈希、actual与F24理想预测。off=stock全部215项；reminder70项31改善0退步，但fast未命中p95仍8135；stock20为4750→3962。
- `numeric_on.json` / `numeric_off.json`：三对完整链前缀cold/cold/warm测量；raw全词表logits摘要及greedy32。on在既定容差下全失败；off也有两对失败，尚不能据此断言快照损坏。
- `splits_on.jsonl`：实际D1 split RID/深度，与暖请求命中的90624/71488/74176对应。
- `numeric_cross.json`：on/off冷算和暖算raw logits交叉比较，含首次greedy分歧位置。

冻结D1 SHA256 `60f98ced6d618a086bda7d49670d176fa75564168d594274bb4a0c63035663c2`，D0 `e0d7924989ebb538e331110f4458ccd4e95a0bfa3cbd87dfb8324ac0328960f6`。基线94602c9；模型与E1 qfull相同。page64/chunk8192/extra_buffer、SSM float32、cap=-1、context131072；N=1，每链flush。

完整日志、manifest、请求/TTFT、raw .pt、补丁和code-snapshot保留在GPU机`/sjtu/linhang/arena/runs/E2_20260922/`，未复制大tensor到本地。`code/sglang-v0.5.20`不变，补丁仅应用`code/sglang-e2-v1.1`。自有服务已停。

分析入口：`scripts/compare_e2.py RUN_ROOT`、`scripts/analyze_e2_numeric.py RUN_ROOT`（后者需torch）。启动/编排脚本为本轮归档入口，写死E2目录且拒绝覆写；若重测必须使用新run根并重跑所有对照，不复用旧输出冒充新版本。完整解释见`notes/experiments.md` E2、D1设计§10、F42/F45。
