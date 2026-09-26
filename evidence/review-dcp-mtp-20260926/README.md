# DCP/S2 独立审阅证据

日期：2026-09-26。对象：`codex/dcp-mtp-stack2-n34` / `e4d7ca680bfc0efd5a7f5121a814cc93279eaa90`，加本轮独立分支的 NextN 默认 topk 守卫修正。结论见 [报告](../../notes/reports/review-dcp-mtp-stack2-20260926.md)。

状态：代码合同及保存观测的 DIAGNOSTIC，不是正式成绩，没有 harness cohort、派发窗口或 N@SLO 结果。本轮不运行 GPU 模型、不占用 GPU、不操作 Pod。CPU 测试构造新的池，包含已有 flush/在途写穿用例；这不等于完整服务模型的恢复/flush 验证。

- `scope.json`：最终两项提交、13 个源码文件、原始函数范围、源码哈希与开发版/移植版增删行对照。函数清单是范围索引，实际阅读合同见报告表格。
- `codegraph-final/`：最终工作树同步后的六组查询。早期开发版查询保存在 `codegraph/`，区分版本；图中的缺边、误边见报告。
- `guard-before.log` / `guard-after.log`：7 项实际生产函数 AST 测试，前者六个默认参数子例被误拒绝，后者通过。
- `test-source-manifest.json` / `source-verification.json`：4713 个运行源码/测试文件及逐项核验。运行目录 `/sjtu/linhang/arena/runs/review-dcp-mtp-stack2-20260926`。复用作者只读 `fix4` 导出的共同文件，再覆盖最终 S2 差异及本轮修正；所有目标文件随后按本地清单重新验证，零不符。
- `host_cpu.log`：55 项通过，包括新增 W1/2/4/8 的实际 host backing 预算与传输字节核算，以及原有六种 DCP 高逻辑地址的毒化/异址往返。测试只替换底层 CUDA mover，生产 controller/assembler/pool/tree 原样执行。
- `*-v1.*`：第一轮新 accounting 测试误把 MLA token-major 首维当页，期望值少乘 64。保留当时源码清单、测试和日志；这是测试错误。修正后重跑全部 55 项通过。
- `author-snapshot/`：作者旧 GPU 证据的只读副本与复制 SHA256。明确区分旧低地址模型结果、局部 sparse kernel 结果和 host copy 合同；不代表最终分支新跑了 GPU。
- `mtp_dcp4-recheck.json` / `mtp_graph4-recheck.json`：独立 CPU 重读原始 `.pt`，各 504 条观测通过；每个采样行相对 L∞ <= 0.01，实际最大 0.0075757578。输出 token、元信息和文件集合一致，graph 臂三种阶段均有 replay 标识。每个原文件路径和 SHA256 保存在 JSON。自然接受长度仍只有 1；没有高地址、HiCache 完整模型或 TP8 结论。
- `cpu-review-receipts.json`：实际命令、运行路径、`CUDA_VISIBLE_DEVICES=""`、`HC180_DEVICE=cpu` 和退出状态。
- `document-link-check.json`：新增报告无坏链接；固定候选原有 DCP 文档有指向未提交计划/证据的坏链接，属于报告 D2。

归档位置与传输哈希见交付时的 archive-transfer-receipt；这里的源日志不因归档而被删改。原始模型 `.pt` 留在作者记录的开发机目录，独立复算读取后未修改。
