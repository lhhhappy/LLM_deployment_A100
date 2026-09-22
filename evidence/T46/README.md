# T46 / W20 — 150 证据索引

补丁/方案：`patches/150-startup-warmup.patch`、同名说明；生成器`make_150.py`，运行源模板`scripts/p150/ax_shapes.py`。
完整基线000→101→105→110→111→112→113→140→120→130；仅副本变更，base_exact只读。

- **P150-01…03**：`cpu_tests.log`，21项CPU测试。生产预热module、flush wrapper，以及AST抽取的真实scheduler flush/tokenizer mixin/exporter方法，mock请求/池/IPC。请求计划48组564请求+最终零命中探针；`request_plan.json`给长度、输出上限、输入SHA。`cpu_tests_initial.log`保留最初测试的Python3.10 TimeoutError夹具错误；修正为asyncio.TimeoutError后通过。
- **P150-02 / Gloo**：`gloo.log`，两个真实CPU/Gloo进程调用生产verify_empty；全健康、仅rank1池泄漏、仅rank1拒绝flush三阶段，每个rank都得到预期结果。仅模拟池，无模型/TP8/GPU通信。
- **P150-04**：`stack_receipt.json`、`full_stack_apply.log`、`verify.log`。11补丁fuzz=0、确定性再生成、全部3623源码+8工具py_compile、应用树与生成树一致、反向整栈逐文件SHA恢复、base未改。`generate_receipt.json`绑定最终补丁；`verify_initial.log`为增加TP归并前的早期打包结果，不是最终SHA。
- **P150-05**：`jit_inventory.json`枚举667个显式Triton JIT定义、源SHA、constexpr/key/heuristics；`kernel_keys.md`逐行列63个FLA/conv/indexer函数，其中12个autotune。`inventory_summary.json`计数；实际固定模型预期域/未覆盖项详见补丁说明。库存含未被当前GLM选中的函数，不能视为launch trace。
- **P150-06**：A100单卡算子验证；`operators_cold*.log`保留所有尝试。前三次在独立包stub或Triton3.7 hook API初始化处失败、未执行算子；cold4完成12组首次/重复和新长度断言后，收据写入遇tuple-key JSON序列化错误，保留全日志；修正序列化后cold5从新空cache完整复验。`operators_persistent.log`是新进程复用磁盘cache。新cache运行开始时已有1个初始化生成的`cuda_utils.cpython-312-x86_64-linux-gnu.so`，不是预置kernel；12组首次的113个编译均disk_hits=0。每行记录JIT miss、真正编译/磁盘命中listener、autotuner._bench次数、新增/变化文件数；不能仅凭时间推断cache hit。
- **P150-07**：未执行。服务启动、真实模型、实际scheduler合批/graph fallback、140 on/off、真实pool/metrics与原dev全SLO/能力门交Claude。

GPU测试仅`/sjtu/linhang/arena/code/T46`、`runs/T46`、`cache/T46`；复用已有环境，未安装依赖。独立包初始化绕过无关SGLang服务依赖，真实FLA与113函数原样加载；没有模型权重、服务、8卡、bohr/Trisol/pod、镜像构建或提交。

复现开发机算子（先nvidia-smi核对空闲；只在arena目录）：

```sh
source /sjtu/linhang/arena/env.sh
cd /sjtu/linhang/arena/code/T46
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/sjtu/linhang/arena/cache/T46/new-empty-directory
export FLA_CACHE_RESULTS=1
/sjtu/linhang/arena/env/m0/bin/python -u test_warmup_cache_150.py --source sglang
/sjtu/linhang/arena/env/m0/bin/python -u test_warmup_cache_150.py --source sglang --persistent
```

传输源为候选树FLA目录、候选sm80_indexer_kernels.py与测试脚本；不修改只读底包。首次与repeat用同一组tensor/shape；不测试数值精度（既有112/113/140证据另见T43/T44/T45）。`seconds`包含编译/autotune及文件哈希取证，不是性能微基准、TTFT或TPOT。
`observed_keys_*.json`保存GPU hook实际specialization；只是这12个算子case，尤其8257为独立operator长度，不能声称chunk8192服务会launch单条8257。

最终结果：`summary.json`由`scripts/summarize_150.py`生成并绑定交付/证据SHA。
新cache首次12组113实际编译/102benchmark/0磁盘命中；重复12组全0新增。
新进程复用cache首次12组57个JIT内存miss、57磁盘编译命中、0实际编译，仍有42次autotune benchmark（未开启cache_results的装饰器）。重复12组又全部0新增。
新shape601在冷进程新增2编译；新进程命中它们的磁盘产物。不能把磁盘缓存说成“完全无需运行期autotune/加载”。
所有算子/Gloo进程已结束；GPU状态见`gpu_final_idle.log`。
