# 长链数据独立 CPU 复核（诊断归档已验收）

- 状态：冻结诊断归档的哈希、大小、收据已独立验收；代码修复通过最小反例，账本算术一致。成品代表性不足，不作新代表集或 GPU 入队依据。
- 范围：`notes/reports/codex-data.md`、`evidence/longchain-audit/workload-ledger.csv`、所选来源链摘要和成品 row/provenance；未重渲染全量 token，未碰 GPU/队列。
- 逐链独立重算 96/96 行、全部 13 个数值列与 CSV 相同；无算术失配。所选来源摘要为 `s1-dev/data/dev-combined-v1/chains.jsonl`，其完整链总量是预算口径，非 1718 条可见正文。
- 来源冻结 uncached `8,969,783 = 4,073,420`（公开 247 条）`+ 4,896,363`（未公开 1471 轮的剩余预算）；缺失轮可能在可见起点之前，不能全称未来工作。
- 合成 1471 条的实际可见前驱 LCP 新增 `1,934,529`；比来源剩余预算少 `2,961,834`，即 `60.4905%`。这仅比较两种逻辑账，不能当 GPU 节省率。
- 成品冻结 `6,008,325 = 4,073,420 + 1,934,529 + 376`；末项是 20 条 adapted_original 相对原冻结标签的重算差，报告若用“合成+公开”直接相加会少 376。
- 成品可见前驱 LCP 新增 `7,608,798`，比混合冻结标签多 `1,600,473`；链首贡献 `1,592,034`，其他请求 `8,439`。混合冻结标签保留 227 条原请求的来源前驱约定；可见 LCP 每链首为零。
- 来源/成品 prompt 总量 `147,875,921 / 148,080,545`；输出总量 `1,267,114 / 1,268,003`，差 889 全在一条已披露的来源聚合冲突链。不能以输出总量“完全相等”描述。
- 成品分类为 227 原样、20 adapted_original、1471 synthetic；与报告计数一致。现行校验器核文件/hash/cohort、body_ref、预算/gap/context、渲染/LCP/provenance 与合成阶段；此前旧 `VALID` 不作冻结证明。
- **代表性判断：不适合替代 dev 或作 N22/N26 代表性长压测。** 主要是 2.962M 的缺失轮预算差、1.592M 转为链首冷入场、合成无重建且仅 8 条新 turn、gap p95 为 9.23 秒且输出/gap 联合关系为估计；这些改变各 TTFT 门与缓存/MTP 压力。可作明确标注的诊断候选；2–3 小时窗口仍需实际回放测量。
- 已验证的修复反例：`build/scratch/review-data-medium/repair-clean.json` 为 VALID；单独旧 hash、负 gap、零预算、旧 body_ref 各为 INVALID/1。
- 冻结验收：`frozen-candidate/SHA256SUMS` 共 36 项全通过；`freeze.json` 的 35 项各自 SHA/字节数正确。manifest/cohort 副本逐字节一致；数据根 7 份产物及 cohort 顺序短哈希 `cd106a80519548d4` 均匹配，来源和 tokenizer 文件哈希也匹配。
- 全量收据 `validation/final-check.json` 为 VALID、1718/1718、0 errors、20 条 adapted 来源核对；当前请求索引与其 1718 条逐请求 token、LCP、预算、gap、phase 字段无失配。冻结时结构复查为 STRUCTURAL_OK、1718/1718、0 errors；其未渲染正文，不能单独当完整 PASS。原 harness 自检 1718/1718 PASS。
- manifest 记录历史 polish `39fa4d46…`、metadata export `096b685c…`；归档冻结代码 `0609df4b…`（生成器）、`7f1da766…`（checker）各自核实，不混称历史执行代码。历史源码字节未归档，不能从当前快照反证历史执行过程。机器可读验收：`evidence/longchain-audit/independent-freeze-acceptance.json`，状态 `ACCEPTED_DIAGNOSTIC_CPU_ARCHIVE`、0 errors；未重渲染全量 token。
