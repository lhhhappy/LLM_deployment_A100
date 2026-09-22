# Codex 早期草案归档

2026-09-22，T8：以下11个历史文件原样迁入本目录，SHA256见[SHA256SUMS](SHA256SUMS)。**均非当前候选或E1依赖，本轮未执行、未部署。** 保留它们是为了追溯早期思路，不表示认可其当前兼容性或测试覆盖。

| 归档文件 | 一行状态 |
|---|---|
| [audit.py](audit.py) | ARCHIVED：早期config/公开负载静态审计脚本，不是GPU性能测量；E1不使用。 |
| [audit.json](audit.json) | HISTORICAL OUTPUT：上述审计的历史数值快照，保留原值，不作为最新事实或实测结果。 |
| [contract_probe.py](contract_probe.py) | ARCHIVED / NOT DEPLOYED：会向指定开发服务发请求并flush的旧接口探针，未提供成功的GPU合规结论。 |
| [remote_smoke.py](remote_smoke.py) | ARCHIVED / NOT E1：旧Qwen启动检查草案，不能验证KDA/DSA；不再作为当前启动入口。 |
| [prepare_patches.py](prepare_patches.py) | ARCHIVED / NOT CANDIDATE：旧flush与可选亲和路由补丁生成器，不用于当前D0/D1。 |
| [make_reduced_checkpoint.py](make_reduced_checkpoint.py) | ARCHIVED / NOT E1：GLM裁层方案草案；E1已选择随机Kimi-Linear替身，并未用它构建权重。 |
| [test_research.py](test_research.py) | HISTORICAL CPU/MOCK：早期局部方法与假worker测试，不证明GPU/DP/HiCache正确性，本轮未重跑。 |
| [patches/0001-flush-json-all-dp-workers.patch](patches/0001-flush-json-all-dp-workers.patch) | ARCHIVED / NOT DEPLOYED：早期flush草案；当前D0候选以仓库根`patches/000-interface-compliance.*`为准。 |
| [patches/0002-optional-header-affinity.patch](patches/0002-optional-header-affinity.patch) | ARCHIVED / NOT APPROVED：静态哈希亲和路由探索，未纳入E1/D1或当前候选。 |
| [fixtures/glm53-first5-config/config.json](fixtures/glm53-first5-config/config.json) | CONFIG ONLY：五层裁剪配置，不含可加载权重，不是可运行模型。 |
| [fixtures/glm53-first5-config/fixture_manifest.json](fixtures/glm53-first5-config/fixture_manifest.json) | INCOMPLETE FIXTURE：原收据标记`config_only=true, complete=false`，仅保留选层/分片规划。 |

迁移仅改变路径；旧脚本里`parents[2]`等默认根路径也原样保留，因此**不要在归档目录直接运行**。未来若采用某部分，先审阅再移植到`scripts/`或正式`patches/`并重新验证，不因归档而自动晋升。旧README所述历史验证不等于当前代码仍能直接运行。

原`research/codex/__pycache__/`内6个CPython 3.10字节码文件已删除，不进入归档；对应源码均在此，可在需要时重新生成。未删除任何源码、patch、fixture或审计数据。

当前入口：

- [E1实验记录](../../../notes/experiments.md)及[小摘要](../../../evidence/E1_stock/README.md)：完成的随机KDA+MLA stock功能基线。
- [E1启动](../../../scripts/launch_e1_standin.sh)、[替身构建](../../../scripts/make_e1_kimi_standin.py)、[链回放](../../../scripts/replay_chains.py)：实际采用的工具，文件头已说明用途。
- [D0](../../../patches/000-interface-compliance.md)、[D1](../../../patches/001-role-boundary-mamba-ckpt.md)：当前设计与候选审阅；是否验证以实验台账为准。
