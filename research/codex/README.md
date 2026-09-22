# Codex 调研

遵循仓库 [协作规则](../../rule.md)。报告按产出时的范围保留；后续获批的本地替身E1已完成，见[实验台账](../../notes/experiments.md)和[结果摘要](../../evidence/E1_stock/README.md)。本目录不是启动入口或提交目录；项目当前状态由Claude维护在根README。

主报告：[R1 — 方向调研与 Claude F3 / R2 交叉审阅](R1_directions_and_review.md)。标注了源码事实、推断、未来验证边界及请 Claude 复核的事项；尚不是双方共同确认的方案。

CPU 模拟工具：[R9 — 离线闭环模拟器](R9_offline_simulator.md)；[R10 — 公开正式包络粗校准](R10_sim_calibration.md)。R10 仅将其他队伍的 passing medians 当 prior；拟合与容量数字均为 MODEL OUTPUT，不能代替真实 stock/D1 A/B。

定向审阅：[R2 — HiCache、D1 角色边界快照、D6 A100 前置条件](R2_review_shared_directions.md)。回应 Claude 的交叉审阅请求，补充 D1 的调度和 slot 生命周期约束。

v2 分工交付（2026-09-22；全部仅调研）：

- [R3 — 底包访问路径](R3_base_images.md)：外部 registry 与平台入口、共享镜像权限限制、元数据/来源收据路线；未拉取或确认实际版本。
- [R4 — Prefill 调度与 D1 冲突](R4_prefill_scheduling.md)：#40024 / #39717 的依赖与移植面，单 chunk 不变量，F13 与 D1 设计的策略差异。
- [R5 — DP1/2/4 显存账](R5_dp_memory_accounting.md)：KV/KDA 几何、静态分池、部分权重复制、条件容量与请求槽限制；非实测。
- [R6 — 中文社区 / GitHub 先例与 rid 生命周期](R6_prior_art_cn_github.md)：llama.cpp 已合并角色边界切分、agent reminder 改写报告、六个仓库的 checkpoint/offload 证据与性能适用边界；§6 确认公开 dev 阶段复用 rid，并指出 flush 返回未校验。新增 F22–F23，全部仅调研。
- [R7 — KDA 单 forward 内部快照 / Plan B](R7_kda_internal_checkpoints.md)：追到 vLLM #52789 / #53614 合并实现；本地 SGLang 已有单点 FP32 累加器导出，但双点仍需 conv、索引、槽位及入树生命周期。区分 B0 替换与 B1 保守双点，纳入 F24；不改变 A 优先、不运行实验。

交叉审阅已追加至 [D0 接口设计 §5](../../patches/000-interface-compliance.md) 和 [D1 快照设计 §8](../../patches/001-role-boundary-mamba-ckpt.md)：D0 有条件同意，D1 需修改 branch 策略证据、最终准入检查和 cap=2 的保留假设。事实日志新增 F14–F18。

后续交付：

- [R15 — 实际底包源码探索](R15_base_source_exploration.md)：102单点vs双点、实际h路径与FP32参考树差异、103族/session/namespace语义、104负载观测依赖；407文件归档核验及静态数据统计。T36只读研究，不实施补丁。
- [R14 — 近期SGLang/vLLM PR与issue复核](R14_recent_pr_watchlist.md)：10个条目的状态/硬件边界，GLM内部快照、DP亲和路由、ReplaySSM接线与HiCache内部状态保留；容量统计增益误读，T35仅调研。
- [R13 — AgentX / AMD MLPerf双文精读](R13_agentx_mlperf_reading.md)：快照保留三个PR的范围与合并状态、DP/DCP和调度节拍区分、小batch调优映射及不执行的验证清单；T34仅调研。
- [R8 — AgentX优化与上游PR检查表](R8_agentx_checklist.md)，附[PR清单](R8_agentx_pr_inventory.json)：源码覆盖与适用边界。
- [R9 — 离线闭环模拟器](R9_offline_simulator.md)：参数化模型及校准限制，不能代替实测。
- [R12 — SLO感知调度](R12_slo_aware_scheduling.md)：chain-start统计余量、896次固定R10模型对照、003草稿及144项CPU验证；MODEL OUTPUT中EDF未胜SPF，live仍待验证。

早期脚本、审计快照、两个patch和config-only fixtures已在T8中原样移到[archive/](archive/README.md)，逐文件状态与SHA256均保留；仅删除6个可再生成的`__pycache__`字节码文件。它们不是E1依赖，不混入正式`scripts/`或根`patches/`；归档脚本的旧相对路径未修订，不应直接运行。

当前E1工具在共享`scripts/`：`e1_env.sh`、`setup_e1_env.sh`、`make_e1_kimi_standin.py`、`launch_e1_standin.sh`、`replay_chains.py`、`test_replay_chains.py`。E2状态和后续动作以[dispatch](../../notes/dispatch.md)与实验台账为准。
