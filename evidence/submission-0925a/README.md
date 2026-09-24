# 0925a：069 配置正式提交

用户授权只提交一份，并要求与本地069一致。2026-09-24 23:23:11 UTC上传成功：**attempt 46251，job 24499，平台返回 queued**。该收据尚无成绩；不能将排队时的外层score=0解释成评测失败。

- 镜像：`registry.dp.tech/dptech/dp/native/prod-4727808/4650601/lh-img:0925a`，镜像ID165027。
- 源码：`759a6ebb8e31723519ad5daf438e26e24b32501a`；host64GB/rank、122关闭（pace=0）、mem0.87、新版180、MTP均与069一致。
- 镜像从0924d继承，只增加只读源码断言。构建成功，4692项（含1个符号链接）校验通过，未修改引擎或依赖；摘要见[源码校验](image-source-verification.json)。独立复核通过。
- 实际069启动参数与候选逐项核对，见[配置审计](config-audit.json)、[提交配置](submission.json)。部署端口30000→8000，代码路径从本地源码导出改为镜像安装包；平台控制正式数据、预热、清缓存与爬坡，不提交本地N30/rep16回放脚本。
- 三份打包的引擎配置副本字节一致；配置SHA256=`ea41c952c0228a7c2b0c655608c11581b21a6700195ae018a81aa71d12a2d8c9`，bundle SHA256=`4de54e297f2534830be9a7e7a97bfbaefb40c72a0ad165d41b9ce509b978aa61`，见[bundle审计](bundle-audit.json)。
- [上传日志](upload.log)、[状态收据](submission-state.json)、[构建收据](build-receipt.json)。第一次直接执行脚本因本地解释器权限失败，未创建attempt；改用bash后只创建并上传46251一次。

## 写盘审计与后续要求

本次为复现069，未临时改变日志参数。启动命令仅启动引擎，不启动本地harness、GPU采样器或CSV/报告生成脚本。提交配置未开启请求正文落盘、crash/tensor dump或性能trace；普通引擎/HTTP日志仍按默认INFO输出到标准流。源码默认值见`engine/sglang/srt/server_args.py`，基础日志配置见`srt/utils/common.py`。

不能据此承诺零写盘：JIT/Triton/FlashInfer等编译缓存会写文件，开启metrics后Prometheus多进程也创建临时目录；标准流日志由平台保存，其保留策略需由平台配置确认。069导出的`N30/server.log`为3,393,513字节（约3.4MB），只是这一份日志的大小，不是Pod总占用。旧Pod确因20Gi临时存储上限驱逐，但没有目录占用快照，不能把根因定为日志。

用户要求后续正式提交收敛写盘：保持请求/张量/trace调试落盘关闭，降低高频INFO和HTTP访问日志，保留启动配置收据与错误/警告；核验编译缓存、临时目录和磁盘预算。日志参数变化单独校验，不追改本次已上传配置。HiCache host64是CPU内存缓存，不是64GB磁盘缓存。
