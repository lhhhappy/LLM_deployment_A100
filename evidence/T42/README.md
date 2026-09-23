# T42 / W16 CPU evidence

交付：[130 补丁](../../patches/130-async-tokenize.patch)及[说明](../../patches/130-async-tokenize.md)。当时的 CPU 一次性测试脚本已清理，本目录保留原日志与结果。
CPU 依赖 transformers 5.12.1 / tokenizers 0.22.2 / jinja2；完整版本见 requirements-lock.txt。

- `cpu_tests.log`：14 单测 + 722 真实对话全量三路径逐 token 对照 + 7 边界/21 并发/3 batch-pair 对照。
- `token_comparison.jsonl` / `coverage.json`：722 个 body 与 metadata ID 集合完全覆盖；冻结 glm_tokens 一致。
- `real_prompt_benchmark.json` / `real_benchmark.log`：原样 100214 / 256733 token 对话，各3次交替开关，两个路径预热；最大 loop lag 中位数 68.20→2.48ms、194.57→8.59ms。
- `real_results.json`：完整正确性汇总、派生 CPU 压力输入的时延、Rust encode_batch GIL 心跳实测；派生输入仅用于本地 CPU 计时。
- `patch_apply.log` / `reverse_check.log` / `receipt.json`：000→101→110→111→130 的零 fuzz 应用、语法编译、反向恢复逐字节一致、输入/脚本/补丁 SHA256。
- `unit_tests_initial.log`：首次14单测；`cpu_tests_dependency_mismatch.log` / `cpu_tests_metadata_attempt.log` 保留测试准备失败史（旧 tokenizer 依赖不支持、metadata 字段名假设），后续已全量重跑。
- `records_check.log`：最终记录一致性检查。

原实验使用锁定的 CPU Python 依赖和一次性测试脚本；脚本现已清理，旧命令不能直接复跑。需要重新验证 130 时，应基于当前补丁树重建测试，并把新运行与本目录历史证据分开。
未导入完整 GPU 服务、未运行 GPU/Trisol/bohr、未打镜像/提交；8卡验证仍待 Claude 安排（M3-06）。
