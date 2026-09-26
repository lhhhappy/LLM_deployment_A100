# 0926 S1：attempt 46676（arm A）

用户 2026-09-26 15:30 UTC 决定直接提交（“不用验证了，直接交吧”）。S1 = A′：46364 配置 + 117 Humming + 124 截止时间分层 + 激进 125 积压模式 + `SGLANG_AX_DEADLINE_WARM_S=15`，引擎带审查修正。目标：线上先过 N26/N30，主攻 chain 与 turn。

## 共同事实（实测）
- 镜像 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0926b`，digest `sha256:ac3e5b5d…`，平台构建 id 166021，成功，构建 30 秒；`FROM lh-img:0925a`（源码 759a6ebb，即 46251/46364 的镜像）只叠加引擎 diff。
- 引擎源码 `f546934eff67f7b1d9f9cc5516c5a89377aaeb71`（tag `image-lh-img-0926b`）= 4f9d1f0b（stack2 + 126/128 后续）+ 84dcca0e（124 warm 饥饿上限，默认等于 120 s，行为中性）+ Codex 审查分支的四个修正：124 停车重做（bab21ddf）、110 显式关闭守卫（67a00da8）、117 shape 元数据缓存（7dbe80f9）、180 主机池销毁释放（07f8de24）。128/170 的修正未合入（机制未启用）。
- `scripts/build_image.sh` 本地自检：嵌入 payload 逐字节等于 diff，0925a 源码树 + diff 逐文件等于 f546934e 树；Dockerfile 40,549 B。镜像内没有做源码摘要断言（本次 build_image.sh 无 ENGINE_SOURCE_VERIFIED 步骤），这是本次未核实的一项。
- 提交包：`outputs/`、`results/`、`execution/results/` 三份 submission.json 字节一致且等于本目录的 `submission.json`；`scripts/check_submission.py --final --trace` 0 错误 0 警告；命令与 46364 逐字相同。
- 8 卡验证：130b（A′，60 分钟 N30，引擎 4f9d1f0b）对 112 同 3063 条 chain 26→30、turn 24→16、overall 448→512、fast 506→554，能力冒烟 12/12；130ea（A′ + 124 停车修正，开场探针）对 130a 同 510 条 chain 14→14、turn 0→2、overall 47→34、fast 51→40、TPOT>0.10 34→39，冒烟 12/12；130eb（A′ + 四个修正，即本镜像引擎）机制行正确、冒烟 12/12（开场对照在 notes/kanban.md）。
- 文件：`submission.json`（上传的配置）、`candidate.json`（带注释的草案）、`bundle-audit.json`、`config-audit.json`（对 46364 的 env 差异）、`build-receipt.json`、`Dockerfile`、`upload.log`、`submission-state.txt`（排队收据）。

## 与 46364 的差异
见 `config-audit.json`：命令相同；env 新增 `SGLANG_AX_SM80_FP8_MOE_HUMMING=1`、`SGLANG_AX_DEADLINE_TIERS=1`、`SGLANG_AX_BACKLOG_RELIEF=1`、`SGLANG_AX_BACKLOG_COLD_CAP=8192`、`SGLANG_AX_BACKLOG_INTERVAL=0`、`SGLANG_AX_BACKLOG_HIGH_S=15`、`SGLANG_AX_BACKLOG_LOW_S=5`、`SGLANG_AX_BACKLOG_MAX_SLOW=80`、`SGLANG_AX_BACKLOG_GATE=0.10`、`SGLANG_AX_DEADLINE_WARM_S=15`；无删除。
