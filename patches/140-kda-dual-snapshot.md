# 140 — KDA 双点 fp32 快照（T45 / W19）

> T57（09-24）：105 已并入 101，112/113 已并入 110，116 已并入 115；文中的旧编号指这些现已合并的部分。补丁按数字顺序叠加，单独叠在 S0 上可打（`scripts/patch_stack.py`）。

140 属于 S0 基线，按数字顺序叠加（000→101→106→110→111→120→140）。
启动时 `SGLANG_AX_KDA_DUAL_SNAPSHOT=1` 开启；默认/`0` 保持原栈。未加入 RELEASE、构建脚本或任何队列。

## 设计与范围

一次 extend 同时导出最后角色边界和对齐末尾；实际未对齐末尾的 fp32 状态仍在 active slot，供当前请求继续 decode。正常 token、thinking、tools、历史、计数和 flush 路径均保留。

- **边界**：在当前请求的 `full_untruncated_fill_ids` 中反向找最后一个 `<|user|>=154827` / `<|observation|>=154829`。设其 token 下标为 `r`（不包含这个 marker），`G=lcm(64, mamba_checkpoint_grid(page_size))`，角色深度 `R=floor(r/G)*G`。只读当前请求，不查看下一轮输入。
- **末尾**：prefix 为 `P`，本次 extend 长 `L`，可入树末尾 `E=P+floor(L/G)*G`。`P` 必须对齐 `G`。kernel 接收相对 offset `E-P` / `R-P` 和物理槽号。每条序列恰好两个描述项，禁用项 `-1`。
- **何时多存一份**：`P<R<E`，而且额外槽可申请时才导出角色节点；`R==E` 时同一个节点标为角色节点。角色不在当前 chunk、对齐后落入已有 prefix、extend 不足一 grid、无 marker，或者槽不足时不申请额外槽。一般角色不在 grid 上时向下取整，最多重算 `G-1` 个 marker 之前的 token。
- **非对齐 prefix / streaming session**：整个该 batch 回到原 tracking；140 不额外切调度。普通 GLM TP 的缓存 prefix 本来对齐，属于测试覆盖外的保守路径。
- **不会每遇到一个角色就快照**：仅最后一个全请求角色边界；它即使位于中间调度 chunk，也在那次 extend 导出，不必等最终 chunk。

开启要求 GLM target、FULL+MAMBA Python UnifiedRadixCache、fp32 temporal、普通 `extra_buffer`、TP（允许默认 overlap）。启动检查拒绝 lazy、int8、HiCache、unified memory、SWA、session radix、DP/CP/PP、PD、mixed、TBO、NEXTN/其他 speculative 与 ReplaySSM。请求级 streaming session 用上述回退。decode CUDA graph 保持原样；140 只导出 eager extend 的状态。NEXTN 应另做接线和验证，不能直接打开。

## Kernel 与数值

`chunk_delta_h_snapshot.py` 由生成器从底包原 kernel 机械派生。在完成指定 64-token chunk 的 recurrence 后，把 `b_h1…b_h4` fp32 累加器直接写进该层状态池；不经过 bf16 `h`。该状态池的 slot stride 由 tensor 提供，使用 int64 地址乘法。active slot 的原 epilogue 继续写真正末尾状态。

卷积 exporter 在原位 causal conv 之前，从 raw QKV 拷贝 offset 前 `kernel_size-1` 行到 `[slot, history, channel]` 状态。offset 至少 64，当前模型 history=3，因此不需要拼接上轮历史；raw QKV、conv slot stride 都显式传递。与 vLLM #56960 的 exporter/history 恢复方法同原理，但这里 chunk 对齐为64，不是 FlashKDA的16；测试也比较截断 prefill 与从快照续算。

**开启时固定 intra 算法是数值前提**：底包 `_small_grid=B*NT*H<=256` 会改变 `fuse_diagonal/fuse_recompute`。首版在跨该阈值的两条序列上出现 `8.535385e-5` 最大状态差，超过目标。140 开启时统一 `False`，长 extend 与短前缀走同一非融合计算路径。短 extend 因此可能增加 kernel 启动成本，必须纳入8卡A/B；不能把省一轮调度直接当成性能提升。

关闭时仍调用**原始 Triton kernel**（源函数逐字节保留）；新 exporter 为独立生成文件。KDA原 intra选择、原 bf16 tracking与101/105都恢复。数值比较的两臂共享源码完全相同的 gate/output helpers 及其 autotune配置，以避免独立导入两份 unchanged helpers 带来的冷调优配置差异。最初独立副本冷对照曾失败，未记录差值；相同源码暖重跑通过；不能将其单独归因为140。早期失败日志保留。

## 槽位与缓存所有权

1. 正常准入先保留 active + 原 ping-pong 槽。140只为角色申请一个额外槽，挂在 `ReqKvInfo.ax_kda_snapshot_slot`；在申请前留下至少 batch_size 个 donation replacement 空位。申请失败只省略额外节点，不主动淘汰已有缓存、不加一轮forward。
2. 正常末尾仍写当前 track ping-pong 槽，由原有 `prepare_for_caching_req` donation/finished逻辑入树。140开启时跳过旧 branch 优先覆盖末尾的规则。
3. 末尾 insert 完成后，暂时锁住末尾，重新匹配得到**树已拥有的KV**，再插入角色节点。角色 insert 的 `prev_prefix_len=R`，不会把已有KV当成重复的新分配而释放。不能读取前一次 insert 可能已释放重复值的旧 req_to_token 切片。
4. 新角色节点接管额外槽；已有同深度状态则释放额外重复槽。slot移交后清空请求字段。新parent的Mamba引用计数从0开始；Full继承正常split锁；临时末尾锁成对释放，再由原cache方法做自己的锁转移。下一轮正常 `match_prefix → cow_mamba → deferred copy` 同时恢复conv+SSM。
5. 不插入的 finish/abort/retraction经原 `free_mamba_cache` 释放未移交额外槽。flush不增加独立持久缓冲：仍由真实scheduler idle检查后 reset树、clear请求池和token池；busy返回false。

**淘汰顺序**：角色标签属于精确深度；split parent不继承，旧child保留。reminder之后的末尾标为tail，若同深度曾是role则role优先。Mamba压力先选可淘汰tail，再沿原LRU（不会越过lock）；Full KV压力的leaf heap先选tail。path cap在原eligibility内先处理tail/普通状态，再处理role，仍是soft cap，locked/fork/当前tail保护不变。正在运行请求锁住的tail不能提前丢，不能保证所有情况下都先释放tail。

## 为什么叠加101/105

140的 `_role_boundary_token_ids()` 在开启时返回空，因而101的admit/tail split和105为该split增加的准入限制均不触发；角色位置由140自己的helper识别。这样只需一个启动开关即可回到完整已知基线，120的独立单partial保护不受影响。关闭时不需要换补丁栈。不得运行时热切开关，需重启服务。

## 验证与证据

结果索引：`evidence/T45/README.md`、`summary.json`；完整数字以该收据为准。

- **历史栈校验**：当时的完整000→101→110→111→140→120旧栈应用、编译、再生成和反向恢复见[收据](../evidence/T45/verify.log)；旧 runner 已清理。现行合并补丁按[patches/README.md](README.md)使用。
- **历史调度/准入测试**：关闭32组×30轮JSON轨迹字节相同，开启8组；role prompt从2次extend变为1次。[测试日志](../evidence/T45/scheduler_tests.log)保留，原一次性 runner 已清理。
- **开发机 CPU/GPU 与离线回放**：真实 controller/tree/components 的缓存测试、64×128 KDA 随机权重数值、原 Renderer+glm_tok 的722请求回放分别见[证据索引](../evidence/T45/README.md)。这些一次性 runner 已清理；离线回放不模拟真实并发、淘汰或解码，不能当作缓存命中或 SLO 结果。

| 调度chunk | off命中token（101+105） | on命中token | 增加 | off→on extend次数 |
|---:|---:|---:|---:|---:|
| 8192 | 16,886,400 | 16,903,104 | 16,704 | 3284→2616 |
| 2048 | 16,806,912 | 16,905,152 | 98,240 | 9430→8949 |

两个chunk分别6/73请求改善，均0退步。8192删除666次101额外split，其余2次extend差来自多命中；2048删除428次split。原始summary的fast_intra p95沿用数据集phase标签，包含在此回放中新建链的首条请求；不能当真实intra SLO，交付只引用上表总量。

## 开放问题与8卡A/B（未执行）

VERIFIED限于源码、CPU方法与开发机算子；完整服务启动、权重、34个KDA层/TP8 collective、overlap压力、长跑能力与SLO尚未验证。额外状态每rank约17.6MiB，在有限池压力下可能跳过快照；固定非融合intra的短请求成本、命中后减少一轮准入的收益、tail优先淘汰对跨请求共享的影响需实测。此次未证明吞吐提升或N@SLO提升。

由Claude交叉审阅后安排，元数据遵守中性命名，开关写镜像内部profile；本任务不建队列/镜像/服务：

1. 同一候选栈、同参数，普通TP8无NEXTN，A=140开关0、B=1，101角色IDs两臂均保留；120/130状态一致。先验证运行后完整源码hash与配置守卫。
2. 一条真实reminder链（含branch冲突、非整页尾），记录raw prompt token、cache命中长度、每次extend范围/次数；单独比较全量重算/前缀续算输出、logits合理误差和原计数。覆盖立即finish、abort、retraction、并发duplicate与槽耗尽。
3. default overlap、多短命中+长冷请求、Mamba/KV分别制造可控压力，检查tree/allocator/lock不变量和无双partial；idle flush恢复全部池，busy flush拒绝，之后首请求cached_tokens为0。
4. 原dev真flush，交替A→B→A，从N6/10再14/18/22；比较全部TTFT桶、TPOT、错误门及显存、slot_skip频率。先固定chunk8192，再单独消融120的2048；不要与113或其它新方向同时变化。若short-extend intra成本抵消收益，先做算子profile再调整，不放松数值门。
5. NEXTN、HiCache/lazy/TBO等目前显式不支持；另行实现/验证后才可纳入A/B。

回滚：重启时置0；或在副本按反向顺序撤掉下游补丁再反向140。未修改base_exact、src/sglang、s1-dev或赛题原件。
