# T45 / W19 — 140 交付证据

补丁与设计：`patches/140-kda-dual-snapshot.patch` / `.md`。最终可机读收据：`summary.json`；生成器 `scripts/make_140.py`，汇总 `scripts/summarize_140.py`。

## VERIFIED

- GPU开发机A100-SXM4-80GB（仅GPU1），torch2.13.0+cu130、Triton3.7.1；随机权重算子。`numeric_final_v3.log` 为最终8组：7组有效角色边界 + 1组禁用角色槽；64×128和TP8分片8×128，长度65至8192，含变长、非连续slot stride、初始状态和三种NT_BUCKET。经实际forward_extend/dispatcher/TritonKDAKernel生产方法。
- 边界SSM、卷积历史、对齐末尾、恢复后suffix输出与最终状态，全部最大逐元素误差0；关闭与原始recurrence同样逐元素相同。`numeric_table.md`。两臂共享源码相同的gate/output辅助函数和autotune选择，详见补丁说明；未运行完整服务。
- `cpu_final_v2.log`：21/21 CPU，真实UnifiedRadixCache/tree/components/tracking/pool/flush方法；checked allocator、session stub。每测试末尾调用原tree sanity_check。CPU由开发机执行，`CUDA_VISIBLE_DEVICES=`，无GPU计算。
- `cache_off_{baseline,candidate}.json`：三种branch情形各三次请求+最终evict，两个JSON字节相同（包含树、命中、锁与分配器账）；`cpu_final_v2_command.log`保留命令结果。
- `scheduler_{off,on}_traces.json` / `scheduler_summary.json`：32组关闭调度轨迹×30轮=960轮与完整原栈JSON字节一致；8组开启无双partial。定向role请求off两次extend，on一次。
- `full_stack_apply.log` / `stack_receipt.json`：000→101→105→110→111→112→140→120→130实贴fuzz0、确定生成、所有Python编译、全栈逆向恢复底包文件字节、原kernel源字节不变。
- `replay/`：真实722请求、311链，原Renderer/GLM tokenizer与冻结glm_tokens全部一致。缓存无限、每链独立串行、只包含prompt，不含decode/eviction/retraction/真实准入压力。8192：off/on命中16,886,400/16,903,104，extend3284/2616；2048：16,806,912/16,905,152，9430/8949；0请求命中退步。是离线估算，不是服务cached_tokens或SLO测量。
- `gpu_preflight.log` / `gpu_final_idle.log`：占用检查与算子结束后释放。工作目录仅 `/sjtu/linhang/arena/code/T45`、`runs/T45`；无服务、8卡、bohr/Trisol/pod、镜像或提交操作。

## 失败史与证据边界

- `numeric_01.log`：初版跨底包small_grid融合阈值出现8.535385e-5误差。开启时固定非融合intra后`numeric_02.log`6组全部逐bit通过；最新v3在此基础上增加原kernel off对照、实际backend接线、长extend。
- `numeric_final.log`：独立导入两份原本相同helper的冷off比较首次失败，未记录差值；未改源码的`off_debug.log`暖重跑通过。最终对照共享unchanged helper的autotune决定；不把该单次冷差异归因于140，完整服务冷启动仍需L2观察。
- `cpu_01.log` / `cpu_02.log`：夹具缺enum/session无操作接口，补齐后`cpu_03.log`11项通过；`cpu_04.log`20项通过。`cpu_final.log`配置守卫测试读取旧远端helper失败（同步和本地再生成重叠导致一次scp丢文件）；重新顺序同步后的`cpu_final_v2.log`21项通过。最终`source_hashes.json`核对远端实际文件与交付文件。
- 不证明模型能力、TP8/overlap长跑或SLO；固定非融合intra可能增加短extend启动成本；额外slot可能因压力跳过；NEXTN、HiCache、lazy等显式不支持。8卡A/B方案见补丁说明，交Claude审阅安排。

参考：`refs/vllm-pr56960/vllm/model_executor/layers/mamba/kda_checkpoint.py` 与 `tests/models/glm5next/test_kda_recurrent.py` 的exporter/conv恢复检验；底包为本仓库base_exact，不是v0.5.20。
