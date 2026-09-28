# Chain-max 16k / Mamba400：构建完成，未上传

2026-09-28 UTC，Codex。引擎审查已获 fable ACK；最终镜像/配置复核、修后 TP8 N30 结果和用户上传确认仍待完成。

- 引擎 **20a58da9**，分支 `codex/fix-chainmax-0927`；131 修复 `b3f8c3c0`，132 收据/依赖守卫 `20a58da9`。
- 镜像 **registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0927a**，构建 **166567**，平台 status=2。FROM 0925a，无新增安装依赖；包内源文件验证由构建中的 Python 执行。
- 候选：去 MTP、DCP=1、running/cuda graph=32、Mamba400、128p、131 interval1/auto、132、chunk/cold/backlog=16384、SHORT_TOKENS=2048、LOAD=1.05、MAX_SLOW=80，**118/126 均关**。2048 是准入阈值，不能称作真实预留。
- [逐项参数对照 46676](config-audit.md) / [机器收据](config-audit.json)：性能参数与 fable 已复核的 [N30 任务快照](n30-job-reviewed.sh) 一致。正式只额外显式固定 DCP=1/本地续算关闭，并关闭前缀逐决策日志和 HTTP access INFO；启动机制行、30 秒摘要、错误保留。
- [submission.json](submission.json) 是唯一候选配置；[expected-mechanisms.txt](expected-mechanisms.txt) 是期望，**不是已经观测到的修后 TP8 机制行**。fable 负责 `130ezna` N30/rot150 同 ID 对照，状态看主 checkout 的 notes/queue.md。
- [镜像准备日志](prepare-image.log)、[构建收据](build-receipt.json)、[构建原始日志](build-log-retry.json)：实际输出 `ENGINE_SOURCE_VERIFIED 31 5f0191240ba1c0cc69382c782183a89ae6dea414315c73c9b141d7e704235334`。平台存储的 Dockerfile 仅去掉最后一个 LF，其余与本地相同。第一次获取 build log 遇 TLS 超时，第二次成功；没有重复构建。
- [配置检查](check-submission.txt)：0 错误、0 警告。[本地打包收据](bundle-audit.json)：三个输出目录的 submission.json 字节一致；ZIP 根部同名文件是 CLI 元数据。只执行了 `--dry-run`，没有创建/上传 attempt。

本地待提交包：本 worktree 的 `build/submit_0927_CHAINMAX400.zip`；配置目录 `build/submit_0927_CHAINMAX400/`，禁止拿旧 A/B 目录代替。镜像已注册不等于正式评测提交。TP8 修后运行尚未闭合，不宣称 N30 已过。
