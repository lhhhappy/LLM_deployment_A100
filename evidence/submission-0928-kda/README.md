# 正式提交 47266：0928c KDA 全候选

2026-09-28 15:09 UTC，Codex 按用户明确授权完成提交。平台返回 attempt **47266**、worker job **25359**；初次独立查询状态为 **queued**，能力与压测成绩尚未返回。提交成功不等于评测通过。

提交物为镜像 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0928c`，引擎 `bf6b66faf3ffbe10e593e858d3bbedd1c48cefb2`，即已验证的钉池 400 + 118 + attention-TP 输入分片 + MoE 调参 + KDA prefill。未加入后续被否决的 133 调参。

- [submission.json](submission.json)：上传的服务配置，SHA256 `1178f8679349422b113b87af4b1998a3d51083cab880cdda20d03d728db6355d`，与冻结分支 `92aa33fd` 的候选逐字节一致。
- [submission.zip](submission.zip)、[bundle-audit.json](bundle-audit.json)：最终包 SHA256 `c64c462b7c94897d4cafde69928b8a6de92f83495a64b26ab7f671444ceb8415`，与平台回执一致。三个实际服务配置副本相同；ZIP 根目录的同名 `submission.json` 是 Playground CLI 的包元数据，不应与服务配置混淆。
- [image-build.json](image-build.json)、[image-current.json](image-current.json)、[Dockerfile](Dockerfile)：镜像构建成功，digest `sha256:0c2aea4df2c3e93e7b3afc60ed23eda2ea323badc4f7d899771bce360d264cc6`。提交前再次核了镜像 id、URL、状态和构建源码；目录查询以 base64 返回 Dockerfile，并去掉末尾换行，解码后构建命令一致。内嵌补丁等于 `git diff 20a58da9 bf6b66fa -- engine/sglang`。
- [preflight.json](preflight.json)、[dry-run.json](dry-run.json)、[arm_manifest.json](arm_manifest.json)、[stub-trace.jsonl](stub-trace.jsonl)：离线 schema/28 个启动参数/48 项环境变量检查通过；原 CLI dry-run 通过，轨迹按 task.md 规定使用占位轨迹。
- [upload-receipt.json](upload-receipt.json)、[upload.log](upload.log)、[upload-exit.json](upload-exit.json)：仅创建一次 attempt，上传退出 0。沿用已验证的 `DAY=0928 bash scripts/submit_official.sh KDA`，处理 CLI 0.1.39 创建 attempt 时空轨迹的问题。
- [attempt-47266-initial.json](attempt-47266-initial.json)、[attempt-47266-latest.json](attempt-47266-latest.json)、[submission-state.json](submission-state.json)：上传后及本轮末次独立查询均 queued，能力与压测尚未返回；另保留精简收据。只读查询命令为 `scripts/official_status.sh 47266`。

用户同时要求查看线上新进展。[47043 原始回报](attempt-47043-current.json) 已确认上一版 `0927a` 正式最高通过 N30：chain p95 25.608 s、fast 3.445 s、overall 4.044 s、turn 4.934 s、TPOT mean/p95 37.683/59.582 ms，能力 AIME 43/44、GPQA 151/156。这是 **47043** 的成绩，不是 47266 的成绩。平台只给最高 PASS 档，未返回更高失败档的具体失败门。

未停止、修改或删除现有八卡服务与本地队列。提交前尝试联系 Fable 避免并行重复上传，但其登记进程已退出；正式 attempt 和收据已写回共享提交记录。
