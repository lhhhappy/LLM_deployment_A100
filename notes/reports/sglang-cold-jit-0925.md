# 071 SGLang 冷启动：符号链接导致 JIT 重复编译

当前只处理 SGLang 071，Pod 未安装或启动 vLLM。071引擎/参数仍为759a6eb、host64、122on、mem.87、MTP；未更换模型或修改算法。

## 已确认原因

保护20Gi临时盘时，`/tmp/ax`链接到`/dev/shm/arena-runtime/ax`，但首版storage_env仍把`SGLANG_JIT_CACHE_DIR`设成`/tmp/ax/cache/sglang/jit`。
底包JIT `_to_entries()`将每个依赖解析为真实路径，却没有同样解析`build_dir`；生成的`cuda.cu`因此没被排除，被记录为临时staging目录下的绝对依赖。
编译发布时staging改名为deps目录，记录的旧路径消失。后一个TP rank拿锁后复核依赖失败，于是重新编译同一个内核。

Pod实测同一个`sgl_kernel_jit_gptq_marlin_bf16_t/build-d65f5727a061211c`已发布6份缓存，每份均记录失效的staging/cuda.cu。
6次生成耗时232.2、238.9、234.2、238.7、239.7、233.0秒；07:27:36第7份仍在编译。
07:31:38已发布7份，第8个rank仍在编译；这是当时观察，不是整项启动完成百分比。
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
此环境路径修复留给后续部署，未更改071进程环境或重启。CPU3项回归用底包真实_to_entries复现失效，并验证真实路径会排除生成文件、发布后依赖仍存在；另有工作目录8项、评测工具18项通过。

07:43左右为当前进程部署了有限缓存恢复脚本`repair_jit_cache.py`：仅处理已发布的缓存，不碰活动staging；核对原manifest哈希、每一个真实依赖、已发布cuda.cu哈希。按底包本来应有的规则移除错误记录的生成包装文件路径，生成新manifest，以硬链接复用**原.so二进制**，原缓存不改不删。最终修复14类内核；编译器/ABI仍由原build key区分。CPU6项测试覆盖真实底包lookup接受、头文件变化拒绝、包装文件变化拒绝、其他缺失依赖拒绝、manifest篡改拒绝和幂等。

恢复进程nice=19、15秒检查一次、最多20分钟、输出总额128KiB；引擎就绪/任务开始即退出，不在测量期维护缓存。收据见[jit-repair-applied](../../evidence/L071-official_b_host64_full_n30_shortwarm/jit-repair-applied.jsonl)。07:46:49已有全部rank进入后续通信初始化，07:47:04 target/draft捕图均已推进完并开始HiCache host分配。此时仍未把启动等同于测量开始。

新进程必须使用canonical路径；当前缓存修复不是持续热补丁，也不保证未来从未见过的lazy JIT形状在071测量期绝不会编译。最终仍须保留冷缓存这一比较限制，不能把额外编译停顿归给122。

## TF32警告和正式配置对齐

对比实际PID收据与已上传0925a/46251：命令只差30000/8000服务端口，机制环境只差本轮明确要测的`SGLANG_AX_PACE_TPOT=0→0.085`。其余MTP、host64、mem.87、backend、模型与输出合同保持冻结配置。继承镜像的NCCL_VERSION单独记录，不能误当额外调参。可重跑[验证脚本](../../evidence/L071-official_b_host64_full_n30_shortwarm/verify-formal-alignment.py)，[收据](../../evidence/L071-official_b_host64_full_n30_shortwarm/formal-alignment.json)。这不代表本地数据/预热等同正式负载。

TF32不是本次迁移意外关闭：069完整server.log的server_args已有`enable_tf32_matmul=False`，本轮和0925a启动命令都没有打开它，冻结源码默认也为False。实际torch warning函数只检查CUDA可用、架构≥8和TF32未开，本身没有性能计时；它按进程缓存，多rank出现多行不能据此判重复编译。

只读查看已有Inductor产物，8个rank各有`[M,4096] FP32 × [4096,32] FP32`投影，与`dsa_indexer_kpool.py::_get_logits_head_gate`的x.float()→weights_proj→scale路径吻合。底包注释明确该投影保留FP32。此观察不能量化耗时占比，也不能证明开启TF32可使整档受益。快照扫描24个现有.py约0.003秒，无torch导入、无GPU调用、无profiler，见[precision-snapshot](../../evidence/L071-official_b_host64_full_n30_shortwarm/precision-snapshot.json)。

原生`--enable-tf32-matmul`在model_runner中调用`torch.set_float32_matmul_precision("high")`。它改变FP32矩阵乘法内部精度，不改变输出dtype；不是消除日志的无语义修复。[PyTorch官方说明](https://docs.pytorch.org/docs/main/generated/torch.set_float32_matmul_precision.html)。本轮保持关闭、不屏蔽警告；若另测，应先测此真实形状的时间/数值、DSA选择与长链影响，再按相同能力/延迟合同验收，不要求输出逐字一致。

07:45容量：RAM工作目录约1.07GiB，根盘/tmp仍1.32MiB，cgroup332.02/1509GiB、failcnt0；这不是完整Pod临时存储总占用，20Gi限额仍有效。缓存恢复产物以硬链接复用，没有另存一份大型二进制。

## 赛题核对

- task.md 304行附近提供已按A100打好的底包；当前正使用列出的SGLang底包，不必重装第二套引擎。
- 424行要求SGLANG_OPT_USE_TOPK_V2=0；实际原模型环境已在adopt核验中检查。
- 226行明确/flush_cache只需清前缀KV，代码缓存和CUDA graph不用清。清空JIT目录会重做编译，不能解除这个路径失效问题。
- 123行3600秒是平台示例hang服务的启动超时；不同于我们容器内模型等待45分钟。462行21分钟是vLLM backport示例，不是SGLang耗时保证。

当前先保留在用的编译产物和模型，让071跑完。重复产物可在运行结束、原始证据归档并确认无编译者后清理；不通过删活动缓存来“加速”。

## 已进入测量

071于07:48:06 UTC就绪，07:52:06—07:53:52完成rep16（106.287秒，16/16），07:53:53真flush成功，07:54:11首批测量请求派发。069同一预热计划/正文/输出预算为112.832秒；“快速预热”没有变慢，之前长等待主要在启动编译及preflight。冷编译在preflight内已有日志，测量后是否新增仍需在15分钟证据中核查。watcher按首批派发+900秒，之后每1800秒报告；归档进程与本地通知桥已核实在运行。
