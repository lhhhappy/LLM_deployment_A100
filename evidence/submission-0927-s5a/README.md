# 0927 S5a（有 MTP）：候选，未上传

用户 2026-09-27："你现在提交不了，先不急"；"我想选 chain 好的来作为提交"。两臂 chain 相同（本地 N34 开场 11 条，46757 配置 16 条）。

- 镜像 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0926e`（id 166310，FROM 0926c 叠 32,403 B 差异，`bohr image build` success；tag `image-lh-img-0926e`）。
- 引擎 791453ca（分支 claude/134-combo）= c0fcd486（Codex 128p，含 741f3eda 的 K3 分级预算与类别冻结）+ f34c7ac4/6976639e（Codex 本地续算路径与审查修正）摘取。
- S5a = 46758 的配置 + `SGLANG_AX_PREFIX_PRODUCER=1`（Codex 128p 共享前缀生产者/兄弟准入）+ `SGLANG_AX_DEADLINE_FREEZE_CLASS=1`（124 类别首见冻结）。命令与 46758 相同（保留 NEXTN MTP）。
- 实测依据：130ezf（N34 30 分钟，对 130ez1 同 1588 条：chain 16→11、turn 2→1、overall 43→57、fast 47→55、TPOT>0.10 76→73）、130ezg（N38 开场：37 个链首超时 14）、130ez6zzzz（N34 开场探针 chain 16→11）。冒烟 12/12。
- 审计（Sonnet 子代理，2026-09-27）：命令/env 与实测任务逐项一致，仅缺 3 个不影响行为的 env（两个 128p 日志上限，值等于默认；DEADLINE_FAMILY=0 等于默认）；三个新开关默认关，S4→S5 差异仅镜像 tag + 两个 env（S5b 再减 5 个 speculative 参数）；Dockerfile 内嵌差异与 `git diff e464d8ab 791453ca -- engine/sglang` 逐字节一致；check_submission 0 错误。提醒：N38 稳态只有 S5b 的 130ezi 在跑，S5a 没有 N38 稳态窗口。
- 文件：submission.json、candidate.json、Dockerfile、build-receipt.json。
