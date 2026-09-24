# HC180 开发机 GPU 契约复核（2026-09-24）

## 结论

在开发机空闲的 A100 GPU0 上，冻结补丁 `180-hicache-glm-dsa.patch`（SHA256 `3b63d9c88cb28520102455bc9081e5e8a7d9f9b0654bd1f1dde1f401747c8848`）的三项关键真实 CUDA 测试通过：逐字节目标层和 MTP 草稿层 MLA/indexer/KDA 往返、128 处分叉后设备及主机恢复、未确认写穿期间 flush 后的槽复用。CUDA 模式下 `_cpu_harness.cpu_kernels()` 不替换拷贝函数，测试经过真实 SGLang 池、控制器、树和 `sgl_kernel`/JIT 搬运核。测试开始两张卡各占 4 MiB、利用率 0%，无计算进程；只设置 `CUDA_VISIBLE_DEVICES=0`。独立短跑 3/3，通过耗时 2.791 秒，SSH 命令总耗时约 28 秒。[命令](run-targeted.sh)、[原始 trace](targeted.log)；命令 SHA256 `61b281201764e94fa20baff1a1785e4b65e5327ddd0f9714ff7de895e7860db1`。

测试文件与开发机副本 SHA256 相同：`test_180_glm_host_pools.py` 为 `25243bdbcc4dfe053962d0c65cfa0a35786b5950167106fa3f48084e0949d25b`；`test_180_tree_hicache_e2e.py` 为 `4db1b1d4ea5b466d8919b377572df7ffb1b4f2f643b09133b5c0a31512007005`。Torch `2.13.0+cu130`，GPU 为 NVIDIA A100-SXM4-80GB。独立负对照用同一测试文件和同一 GPU 跑无 180 的树，退出码 1：目标 indexer 在有、无 MTP 两个子例都未恢复。[负对照 trace](negative-base-roundtrip.log)。

Claude 此前在同一开发机的最终全套 CUDA 日志为 52/52 通过，[原始日志](claude-gpu3-t180.log)。这 52 项中上游声明/检查点测试含桩与 CPU 逻辑，不能统称 52 项都完成了设备字节搬运。此前隔离运行的中间结果 47/52 失败 5 项，发生在测试夹具修正之前，[中间汇总](claude-isolated-summary.log)；不能将它当作最终补丁失败。

`run_tests.sh` 原用 `python -m unittest ... | grep` 且未设 `pipefail`，会吞掉 unittest 的非零退出码。仅将 `set -u` 改为 `set -uo pipefail`；本地最小正探针 `-h` 退出 0、缺失测试 `__missing_test_hc180__` 退出 1。未修改引擎补丁。

## 覆盖边界

上述逐字节检查验证同一有效 payload 的位置、毒化后换页/换状态槽、分叉及 flush 生命周期，不是自由生成 token 的逐字一致性测试。flush 测试保证写回已提交但确认尚未处理；测试没有记录 reset 瞬间 DMA event 的未完成状态，故不能声称必然覆盖物理 DMA 与 reset 同时进行。单卡池/控制器测试不验证 TP8 切分、rank 间确认、真实模型状态恢复后的输出或整档性能。

Claude 的缩小版真实模型四次尝试均在启动 CUDA graph 捕获时因开发环境缺少 `tilelang` 退出，未产生可用模型数值结果：[尝试汇总](claude-model-attempt-summary.log)、[一份服务原始日志](model-t180-off-server.log)。其余 GPU 数值与 TP8 边界仍待主代理处理。
