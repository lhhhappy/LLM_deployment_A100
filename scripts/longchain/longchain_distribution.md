# 长链分布和重复诊断

`longchain_distribution.py` 只读标准数据文件，使用原 harness 的索引、四门与 gap cap 规则，独立对比公开 s1-dev 与合成集。它不修改生成器、checker 或回放协议，也不代替完整验收。

```bash
uv run --offline --with-requirements scripts/longchain/requirements-longchain.txt python -B scripts/longchain/longchain_distribution.py --root data/s1-dev-longchain --out evidence/longchain-design-20260924/pilot-v2b-distribution.json
```

完整集在 `cache/s1-dev-longchain-build` 完成后可只换 `--root` 与输出文件名；正式替换当前成品后继续使用 `data/s1-dev-longchain`。输出 JSON 和同名 Markdown，必须放在数据根目录外。生成中 `BUILDING` 的目录拒绝审计。`--skip-head-tokens` 可先作结构分布诊断，但会明确省略 token 级链首审计。

- 四门、chain/phase、prompt、冻结新增输入、decode预算与 gap 同口径比较；额外列生成 provenance 的真实LCP新增，避免与源冻结标注混用。
- source 摘要、计划合成事件、实际合成事件分别列示；统计压力触发重建、被置换事件、参考观测替换。
- 完整正文重复排除 req_id；新增块按相邻快照的完整消息/工具组多重集合正差计算，保留历史不重复计数。另报调用ID归一化后的重复，防止换ID掩盖克隆。
- 来源 query/answer/donor 的引入次数与覆盖session按 provenance统计，不把历史在后续每个快照的重复出现当作新引入。
- 链首使用原 GLM Renderer；逐对比较合成与对应源链首，扣除system/tools前缀。来源原本同session的分支共享单列，不算新制造的克隆。仅检查链首，不能据此证明全部中间快照没有跨链前缀共享。

正文流式处理，SQLite临时表仅保存指纹和完整块边界摘要，结束自动清理；常驻内存为请求元数据、来源账和链首token数组，不保留完整正文。真实token头部审计需要原tokenizer依赖。统计短消息重复、合法工具重复不是自动拒绝依据；结果只支持诊断，不宣称线上代表性或正式 N@SLO。
