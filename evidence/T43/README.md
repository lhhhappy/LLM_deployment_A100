# T43 / W17 — 112交付证据

最终验收：**PASS**。入口 `patches/112-sm80-indexer-kernels.md`；自动摘要 `summary.json`。

| 文件 | 内容 / 复现 |
|---|---|
| `final_all.log` | 最终A100环境、SHA、256种fp8编码、88组数值、4种graph×3动态重放、8行性能；末行complete/PASS。`scripts/test_sm80_indexer_112.py --mode all` |
| `summary.json` / `summary_check.log` | 最终source/oracle/test/patch/compiler SHA绑定；`python3 scripts/summarize_112.py` |
| `performance_table.md` | 从同一最终日志导出的ms/加速表（单卡算子） |
| `generate_receipt.json` / `generate_apply.log` | 000→101→105→110→111生成基线、112产物SHA；`scripts/make_112.py` |
| `stack_receipt.json` / `full_stack_apply.log` / `verify.log` | 000→101→105→110→111→112→120→130 fuzz0、确定生成、3623编译、全栈回滚逐字节还原、只读底包未改；`scripts/verify_112.py` |
| `compiler/` / `compiler_run.log` | 最终代表形状PTX、sm80/bf16 MMA、寄存器/0 spills/共享内存与SHA；`scripts/inspect_sm80_indexer_112.py <输出目录>` |
| `gpu_environment.log` / `gpu_before_decode_tune.log` / `gpu_final_idle.log` | 已有CUDA13兼容库使torch2.13+cu130可用；开发机占用检查与收尾空闲 |
| `smoke_attempt1.log` / `numeric_attempt1.log` / `graph_attempt1.log` / `bench_attempt1.log` | 首版核验；与最终源码SHA不同，不能混为最终性能 |
| `tune_attempt1.log` | 首版48组BQ/BK/warp扫描；`scripts/tune_sm80_indexer_112.py prototypes/initial.py`（先将原型目录复制到运行目录） |
| `layouts_attempt2.log` / `layouts_wide.log` | 三种解码/矩阵布局与4/8/16warp；`scripts/compare_sm80_indexer_layouts_112.py prototypes [--wide]` |
| `decode_tune.log` | 两种解码×4/8/16warp×32k/190k，20调用CUDA graph；`scripts/tune_sm80_indexer_decode_112.py prototypes` |
| `prototypes/` | 初版、直接bf16位编码、q-major布局的完整源；离线调优可重现，不属于112补丁 |
| `verify_attempt1.log` | 全栈已还原，但GNU patch留下.orig使文件集检查失败；增加`--no-backup-if-mismatch`后通过 |
| `verify_attempt2.log` | 改kernel后未先生成补丁，verify正确拒绝旧patch；已再生成并完成最终verify |
| `layouts_attempt1.log` | 原型文件尚未传完时启动导致FileNotFound；完整传输后的attempt2成功 |
| `profile_decode_b112_n6_TP0.txt` | 用户给的原8卡profile；这里b112是服务源码版本名，其内容仍为110 torch基线，**不是本112融合补丁** |

源文件/补丁/测试均在仓库可复现；GPU文件位于 `/sjtu/linhang/arena/code/T43`，原始运行记录 `/sjtu/linhang/arena/runs/T43`。无需安装软件、模型、镜像或8卡。远端先source arena/env.sh，再使用现有CUDA兼容库；完整命令见补丁说明和 `scripts/run_sm80_indexer_112.sh`。

统计口径：逐行有限位置相对L∞和L2；topk比较有限定义域的集合，K=min(2048,有效key数)。真实模型形状、随机激活（标准/次正规数/大幅值），不是从真实模型采集的q/K/weights。表中的L是indexer key长度，kpool=4时不能直接当原始prompt长度。

授权范围仅开发机算子测试。未操作bohr/Trisol/pod、未打镜像、未启动8卡、未提交；完整服务及SLO待Claude。所有计时不含JIT编译和数据生成。
