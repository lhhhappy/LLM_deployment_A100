# 0926 S2：attempt 46677（arm B）

用户 2026-09-26 15:30 UTC 决定直接提交。S2 = S1 + `SGLANG_AX_BACKLOG_MAX_SLOW=250`（其余与 46676 逐字节相同，同一镜像）。激进臂：允许 fast/overall 变差、TPOT>0.10 条数升高，换稳态 chain。

## 共同事实（实测）
- 镜像 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0926b`，digest `sha256:ac3e5b5d…`，平台构建 id 166021，成功，构建 30 秒；`FROM lh-img:0925a`（源码 759a6ebb，即 46251/46364 的镜像）只叠加引擎 diff。
- 引擎源码 `f546934eff67f7b1d9f9cc5516c5a89377aaeb71`（tag `image-lh-img-0926b`）= 4f9d1f0b（stack2 + 126/128 后续）+ 84dcca0e（124 warm 饥饿上限，默认等于 120 s，行为中性）+ Codex 审查分支的四个修正：124 停车重做（bab21ddf）、110 显式关闭守卫（67a00da8）、117 shape 元数据缓存（7dbe80f9）、180 主机池销毁释放（07f8de24）。128/170 的修正未合入（机制未启用）。
- `scripts/build_image.sh` 本地自检：嵌入 payload 逐字节等于 diff，0925a 源码树 + diff 逐文件等于 f546934e 树；Dockerfile 40,549 B。镜像内没有做源码摘要断言（本次 build_image.sh 无 ENGINE_SOURCE_VERIFIED 步骤），这是本次未核实的一项。
- 提交包：`outputs/`、`results/`、`execution/results/` 三份 submission.json 字节一致且等于本目录的 `submission.json`；`scripts/check_submission.py --final --trace` 0 错误 0 警告；命令与 46364 逐字相同。
- 8 卡验证：130b（A′，60 分钟 N30，引擎 4f9d1f0b）对 112 同 3063 条 chain 26→30、turn 24→16、overall 448→512、fast 506→554，能力冒烟 12/12；130ea（A′ + 124 停车修正，开场探针）对 130a 同 510 条 chain 14→14、turn 0→2、overall 47→34、fast 51→40、TPOT>0.10 34→39，冒烟 12/12；130eb（A′ + 四个修正，即本镜像引擎）机制行正确、冒烟 12/12（开场对照在 notes/kanban.md）。
- 文件：`submission.json`（上传的配置）、`candidate.json`（带注释的草案）、`bundle-audit.json`、`config-audit.json`（对 46364 的 env 差异）、`build-receipt.json`、`Dockerfile`、`upload.log`、`submission-state.txt`（排队收据）。

## 这一个变量的依据（实测，服务日志 `[ax-125] relief` 行）
- 125 的护栏累计“曾经慢”（运行中 TPOT > 0.10）的请求数，到 `MAX_SLOW` 就锁存（`decide()` 的 tripped 不复位）。112 与 130b 都在测量开始约 14 分钟时过线（slow 82/293、82/301），之后 46 分钟积压模式再没开过（relief_rounds 停在 307/300）。
- 14 分钟之后到达的约 170 个链首里仍有 13（112）/17（130b）条超 30 s，多数 TTFT 32–64 s，是 8192 冷块能救的区间。
- 250 按正式 tpot_p95 门反推：p95 ≤ 0.10 允许 5%（257/5150）最终慢；护栏数的是“曾经慢”，比最终慢多两三倍（130b：14 分钟内曾慢 82，整小时最终 >0.10 只有 37）。
- 未做 60 分钟本地确认（护栏在前 14 分钟不起作用，十分钟探针测不出）；确认运行 `130ec-v3_n30_A_warm15_fixes_slow250_60m` 已排队，对 130b。
