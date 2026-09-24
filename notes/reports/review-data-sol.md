# 长链数据独立审查（CPU，只读）

- 状态：原发现的 2 项缺陷已在当前代码修复；最终重建数据尚待独立验收。
- 范围：`longchain.py` build/polish 与 `longchain_check.py`；未用 GPU/远程进程，未重复全量渲染。
- 原始复现：`build/scratch/review-data-medium/hash-fixture/`；独立复核：同目录 `repair-*.json`。
- 分布偏差已由数据作者披露，不列为代码缺陷。

## 修复复核

1. **冻结与工作量字段。** `longchain_check.py:168-198` 逐文件核 `manifest.artifacts`，`:265-273` 核 cohort 的有序链摘要，`:279-299` 核正整数输出预算、非负有限 gap 与 prompt+budget 上界。原篡改 fixture 现输出 `INVALID: 1 expected requests, 1 bodies, 5 errors`（`independent-recheck.json`）；其中 4 项是旧 fixture 故意缺的 manifest/cohort 元数据，另 1 项为负 gap。为隔离原因，我修齐元数据后重新跑：`repair-clean.json` 为 `VALID/0 errors`，`repair-stale_hash.json` 只报 `requests.jsonl` SHA 失配，`repair-bad_gap.json` 只报负 gap，`repair-bad_budget.json` 只报零预算；后三者均 `INVALID/1 error`。这证实各拒绝条件独立生效。
2. **`body_ref`。** `longchain.py:456-457` 在 build 原请求写新 shard 引用，`:708-709` 在 polish 对每行统一写新引用，同时把源路径留在 provenance。`longchain_check.py:333-338` 对每条请求核引用与实际 shard；隔离样本 `repair-bad_body_ref.json`（同步更新 manifest hash）报 `INVALID/1 error`，明确指向旧 shard。最终目录仍需检查重建产物，不沿用 raw 早期 `VALID` 结论。
3. **phase/gate。** `longchain.py:185-194,498,516` 由实际追加消息判 `turn_start/intra`，源 donor 阶段存 provenance；polish 在 `:713-719` 重算 synthetic 阶段，checker 在 `:417-427` 根据实际新增消息复核。原来两条借用 `context_reset` 的 synthetic 不会继续凭 donor 标签进入 chain_start 门。原 donor 本身为高 LCP append-only，不能单靠 LCP 改标签；现规则按生成事件而非阈值。

独立复核命令：`.venv-longchain/bin/python -B scripts/analysis/longchain_check.py --root build/scratch/review-data-medium/repair-clean --harness-dir s1-dev/harness --tok-dir s1-dev/glm_tok --cohort build/scratch/review-data-medium/repair-clean/cohort.json --out-json build/scratch/review-data-medium/repair-clean.json`；其他三组只替换目录后缀。
