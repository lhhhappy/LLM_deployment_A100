# 0926 S4：attempt 46758（arm D）——S1 + DCP2 + Codex 本地续算路径（实验臂）

用户 2026-09-27 00:28 UTC："你把这个加上 说不定有效！"
- 镜像 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0926d`（id 166140，FROM 0925a，Dockerfile 56,693 B，本地自检通过）。
- 引擎 6976639e（tag `image-lh-img-0926d`）= e464d8ab（46757 引擎）+ Codex 的 DCP 本地 KV 预填充路径（f34c7ac4）+ 审查修正（B1 运行时 JIT 参数、B2 机制 token 与路由计数、N1 几何守卫）。
- 相对 46757：只多一个 env `SGLANG_AX_DCP_LOCAL_EXTEND=1`（LARGE_MAX/MAX_TOKENS 未设：只有缓存前缀 ≥4096、新算 ≤512 的暖续算走新路径；开场冷链首不走）。命令与其余 env 逐字节相同。
- 验证状态：两轮独立审查无阻塞项（notes/reports/review-dcp-local-extend-0926.md）；CPU 17/17；开发机 TP2 缩小模型数值与 MTP/HiCache trace 通过；**上传前未在 TP8 跑过**，三臂探针 130ez4/5/6 排在 Pod 队列里，结果作事后对照。
- 文件：submission.json、candidate.json、bundle-audit.json、build-receipt.json、Dockerfile。
