# 071 SGLang 冷启动：符号链接导致 JIT 重复编译

当前只处理 SGLang 071，Pod 未安装或启动 vLLM。071引擎/参数仍为759a6eb、host64、122on、mem.87、MTP；未更换模型或修改算法。

## 已确认原因

保护20Gi临时盘时，`/tmp/ax`链接到`/dev/shm/arena-runtime/ax`，但首版storage_env仍把`SGLANG_JIT_CACHE_DIR`设成`/tmp/ax/cache/sglang/jit`。
底包JIT `_to_entries()`将每个依赖解析为真实路径，却没有同样解析`build_dir`；生成的`cuda.cu`因此没被排除，被记录为临时staging目录下的绝对依赖。
编译发布时staging改名为deps目录，记录的旧路径消失。后一个TP rank拿锁后复核依赖失败，于是重新编译同一个内核。

Pod实测同一个`sgl_kernel_jit_gptq_marlin_bf16_t/build-d65f5727a061211c`已发布6份缓存，每份均记录失效的staging/cuda.cu。
6次生成耗时232.2、238.9、234.2、238.7、239.7、233.0秒；07:27:36第7份仍在编译。
这是**缓存路径迁移触发的底包边界缺陷**，不是20Gi已满、不是vLLM问题，也不能解释为不可避免的单次冷编译。

证据：[编译命令](../../evidence/L071-official_b_host64_full_n30_shortwarm/compiler-progress.json)、[重复缓存与依赖](../../evidence/L071-official_b_host64_full_n30_shortwarm/compiler-cache.json)。
当前源码与冻结759a6eb的JIT loader/cache无差异；源码见`engine/sglang/kernels/jit/utils/compile/cache.py::_to_entries`。

## 当前恢复与后续修复

原始启动器`lib.sh`最多轮询540次、每次间隔5秒。`ENGINE_TIMEOUT`是我们的45分钟等待预算耗尽，不是平台停止模型。
原模型PID20041仍在，编译器从cicc推进到ptxas并开始下一项；原等待进程已退出。

已部署一次性`adopt-wait.sh`：核对原PID/start ticks、完整argv、机制环境、源码COMMIT、数据校验收据；最多额外等待30分钟，每10秒检查就绪、每5分钟输出一行状态。
不调用原start_engine/stop_engine、不重载权重。只有原进程真正提供/v1/models、复用签名写好、pending任务哈希与worker身份一致，才恢复已授权071队列。
验证收据位于Pod该run的`adopt-verification.json`；脚本副本见[adopt-wait](../../evidence/L071-official_b_host64_full_n30_shortwarm/adopt-wait.sh)。

本地storage_env已改为先readlink -f，再导出所有缓存/临时路径为真实绝对路径；不改变/tmp/ax公共入口。
此修复**尚未覆盖正在运行的071环境**，避免在冻结运行中变更或重启。CPU3项回归用底包真实_to_entries复现失效，并验证真实路径会排除生成文件、发布后依赖仍存在；另有工作目录8项、评测工具18项通过。

## 赛题核对

- task.md 304节提供已按A100打好的底包；当前正使用列出的SGLang底包，不必重装第二套引擎。
- 424行要求SGLANG_OPT_USE_TOPK_V2=0；实际原模型环境已在adopt核验中检查。
- 226行明确/flush_cache只需清前缀KV，代码缓存和CUDA graph不用清。清空JIT目录会重做编译，不能解除这个路径失效问题。
- 123行3600秒是平台示例hang服务的启动超时；不同于我们容器内模型等待45分钟。462行21分钟是vLLM backport示例，不是SGLang耗时保证。

当前先保留在用的编译产物和模型，让071跑完。重复产物可在运行结束、原始证据归档并确认无编译者后清理；不通过删活动缓存来“加速”。
