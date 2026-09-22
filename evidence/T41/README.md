# T41 / W15 CPU 交付证据

只使用本地 CPU；未运行 GPU、SSH、bohr/Trisol、镜像构建、队列或提交。

| 文件 | 含义 |
|---|---|
| baseline_apply.log | 在 base_exact/sglang 副本上 000→101→110→111，实际 -p3/fuzz=0 应用 |
| patch_sha256.json | 基线补丁链与 120 的 SHA256 |
| patch_apply.log / patch_reverse.log | 120 正向应用与反向回滚均成功 |
| validation.json / compile.log | 生成器重现一致、4684文件匹配、只改2文件、3617+3 py_compile、只读底包验证期间未变 |
| cpu_tests.log | 27/27 CPU 测试通过；实际 scheduler/PrefillAdder AST，mock batch/池/forward |
| interleave.json | 冷长+在跑decode+短命中，真实决策方法输出的四轮 trace |
| starvation_bound.json | 已准入冷请求在持续短到达下的4组完成轮次与条件上界 |
| off_parity.json | 24组 baseline/off 序列 hash 相同；22组40轮正常，2组同一基线断言 |
| off_decision_traces.json | 上述两臂的完整原始决策序列，含断言方法/语句/轮次 |
| cpu_tests_initial.log / cpu_tests_fixture_debug.log | 初次夹具缺stub/未执行chunk被错误视为已算完的调试失败，非最终生产结论 |
| records_check.log | 收尾账本检查结果 |
| delivery_sha256.json | 最终交付源码/说明/测试/方案的哈希 |

复现：

```bash
python3 scripts/make_120.py
python3 -B -m unittest discover -s tests -p test_sched_protect_chain.py -v
python3 scripts/verify_120.py
bash -n scripts/pod/jobs/dev_b120_template.sh
python3 scripts/check_records.py
```

结论边界与测试 ID P120-01…10 见 `patches/120-sched-protect-chain.md` / `tests/TEST_PLAN.md`。
CPU 通过不代表 GPU logits/overlap/NEXTN/池回收或 SLO 通过；LPM 未准入请求公平性保持原状。
