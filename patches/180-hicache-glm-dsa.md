# 180 — GLM-5.3-Flash 的主机内存二级缓存（HiCache）做对：DSA indexer + 256 组所有权

状态（2026-09-24）：**第 1 阶段（移植上游）完成**。补丁可打在部署栈、S0 和裸底包上；CPU 测试 52 项全过，并且能区分有无本补丁。**没有在 GPU 上跑过，没有做模型数值，没有上 8 卡。** 开关不开时，行为与现栈相同（依据见 §6）。

术语只解释一次：
- **HiCache / 主机层**：GPU 放不下的前缀缓存挪到 CPU 内存，需要时再搬回 GPU。
- **indexer 行**：DSA 稀疏注意力的“挑选器”给每个历史 token 存的打分键。GLM 用 kpool=4 压缩，4 个 token 合成 1 行。
- **组**：连续 256 个 token。每组合成 64 行 indexer，全部存在**本组第一页**（64 token 一页）里（`kpool_fp8_index.compute_pooled_write_locs`），同组其余 3 页的 indexer 位置空着。
- **KDA 状态**：线性注意力层把全部历史压成的固定大小矩阵；前缀缓存只能从存了状态的位置接着算。
- **节点**：radix 树里的一段前缀；**树页**：树切分节点的最小单位。

## 1. 问题

底包打开 `--enable-hierarchical-cache` 后，GLM 从主机恢复前缀是错的，有两处：

1. **indexer 根本没进主机层**。GLM 走 `_MambaStrategy → build_hybrid_mamba_stack`，只挂了 MLA 潜向量和 KDA 状态两个主机池；indexer 的主机池只在纯 DSA 模型的 `_DsaStrategy` 里有。恢复后 MLA 对了，indexer 是那几页原来残留的内容，稀疏注意力选错 token，不报错。上游 #40915 报告主机恢复 KL 0.06–0.71（修后 ~2e-3）【上游报告】。本地 CPU 测试复现：底包恢复一组 256 token 后，MLA 和 KDA 状态逐字节正确，indexer 行 **全部** 不对（`test_180_glm_host_pools`）。
2. **组内分界让“组头页”被别人改写**（只补 1 不够）：
   - (a) 陈旧主机副本：请求在 960 处存了检查点并写穿到主机（此时组 768–1023 的组头页只写到 959），随后 decode 把 960–1023 的行写进**同一个已归树所有、已备份的组头页**；主机那份永远缺这 16 行。恢复后这 16 行是错的。这正是本负载最常见的形状（提示词末尾 64 对齐、不是 256 对齐，然后 decode）。
   - (b) 分叉覆盖：两个分支在组内 64/128/192 处分开，后算的分支把自己的行写进共享组头页，改坏另一分支（设备上就坏，主机上也跟着坏）。
   本地 CPU 端到端测试（真实树 + 真实 HiCache 控制器）实测：**只移植上游 #40913–#40915（indexer 主机池）时，(a) 恰好错 pool 240–255，(b) 错 pool 32–63**；加上 256 组所有权后全对（`test_180_tree_hicache_e2e`，§8）。

## 2. 设计（为什么这样选）

**A. 主机池：原样移植上游声明式栈 #40913 → #40914 → #40915**（三者是一条栈，#40915 依赖前两个，不能单拿）。
- 设备池自己声明需要哪些主机池（`host_pool_decls()`）：DSA 池声明 KV + INDEXER；混合池 `HybridLinearKVPool` 转交给它的全注意力子池；Mamba 状态仍由原 Mamba 路径负责。
- `build_hybrid_mamba_stack` 按声明建 KV、INDEXER，再建 MAMBA；INDEXER 用 KV 的主机槽号（sidecar，`indices_from_pool=KV`），不自己分配。
- 已迁移的策略（DSA、Mamba）少了声明的池就**启动报错**（`_check_declared_pools_present`）。
- MTP 草稿层（160 的 NEXTN，1 层 DSA）按上游 PACKED 方式挂在目标主机池尾部：MLA 和 indexer 各多 1 层。草稿行宽、dtype、indexer 格式与目标不一致时**启动报错**，不会静默漏拷。

**B. 所有权：从上游 #38212 只取“压缩 indexer 按组所有”和“检查点对齐”两部分**，它的主机池那一半被 A 取代，不重复移植。
- 开 HiCache 且设备池是 kpool 压缩的 DSA 时，树页放宽为 `lcm(64, 64×4)=256`；分配器、主机槽、传输仍按 64 token 物理页（`init_hicache` 拿原始参数）。
- 于是节点的注册、匹配、分叉、切分、淘汰、恢复全部落在 256 边界上：每个节点拥有完整的组；**还没写满的尾组一直归请求私有，不进树**；进树的组此后不再有人写，写穿的主机副本就是最终内容。
- KDA 检查点跟着对齐：extend 的检查点取**绝对**位置 `floor(extend_end/256)×256`，且必须落在 kernel 的 64 快照网格上；分支点也必须在 256 网格上；缓存时状态位置与键长不一致（未对齐、超长、None）就**不发布**，状态留给请求在结束时释放（上游称“state cannot be shortened”）。
- 这样“每个 256 边界都真有检查点”是构造出来的：chunk 截断（120 的 `// grid * grid`，grid 由树页推出）、101 的角色切分、decode 记录（间隔 256）、MTP 记录都用同一个树页推出的网格。

**为什么不选上游 #40134（每页 indexer 单独存）**：它改 indexer 的写地址和读页表（tilelang kernel、`dsa_indexer_kpool`、`kpool_fp8_index`、`kpool_plan`），而这些正是我们 110/114/170 改过的 sm80 kernel 路径；只移植池那一侧会读错行。要走这条路必须连同 110/114 的地址约定一起改、重做数值验证，属于执行层的另一项工作。B 方案不碰任何 kernel，代价只是命中长度按 256 取整（见 §5）。

**为什么放宽只在开 HiCache 时生效（相对上游的唯一行为差异）**：上游 #38212 对压缩 DSA 无条件放宽。设备上的分叉覆盖 (b) 在现栈里本来就存在，但它是另一个问题；为了让“打了 180 不开开关 = 现栈”严格成立，这里只在 HiCache 开时放宽。要单独修设备侧 (b)，可以以后把这个条件拿掉做单变量实验。

**位置语义（“同一位置”指什么）**：树里键长为 L 的节点拥有位置 `[0, L)` 的 MLA 潜向量行、完全落在 `[0, L)` 里的每个 256 组的 indexer 行、草稿层的同样两类行，以及**消费完 token 0…L−1 之后**的 KDA/卷积状态。MTP 开时键是二元组（bigram），键长 L 覆盖 L+1 个 token，但 KV 位置仍是 `[0, L)`；草稿 KV 在位置 L−1 依赖 token L，这由 bigram 键保证。上述 L 在开 HiCache 时总是 256 的倍数。

## 3. 改了哪些文件（相对部署栈 000…170；9 个文件）

| 文件 | 来源 | 内容 |
|---|---|---|
| `mem_cache/pool_host/host_pool_decl.py`（新） | #40913、#40914 原样 | 声明类型 `HostPoolDecl` / `HostPoolStorageInfo`；草稿改名 `make_draft_sidecar_decls` |
| `mem_cache/hybrid_cache/host_pool_config.py`（新） | #40913、#40914 原样 | 声明校验、层绑定、打包草稿校验 |
| `mem_cache/memory_pool.py` | #40913 原样 | `KVCache` / `HybridLinearKVPool` / `DSATokenToKVPool.host_pool_decls()` |
| `mem_cache/pool_host/dsa.py` | #40913 + #38212 的 `destroy` | indexer 主机池由声明构造；只镜像真有缓冲的层；草稿层号换算；销毁时注销锁页内存 |
| `mem_cache/hybrid_cache/hybrid_pool_assembler.py` | #40913/#40914/#40915 + 本地 3 处 | 声明式组装；Mamba 栈挂 INDEXER；见下 |
| `mem_cache/unified_radix_cache.py` | #38212 | 树页放宽（加 HiCache 条件）；拒绝外部 linker 与 L3；缓存时先定其它上限、最后定状态 |
| `mem_cache/unified_cache/components/mamba_component.py` | #38212 | 状态位置不合法就不发布 |
| `managers/schedule_batch.py` | #38212 | extend 检查点取绝对 256 网格；分支点须在网格上 |
| `managers/scheduler.py` | 本地（2026-09-24 夜加入） | 120 的总开关 `_ax_sched_protect_enabled` 原先见 HiCache 即关闭，改为只在 L3 存储（`enable_hicache_storage`）时关闭；见 §6a |

**对上游的全部改动（逐条）**，脚本 `scripts/tests/hicache180/port/` 可从 `refs/pr*.diff` 原样重建本补丁（已核对重建树与补丁树逐字节相同）：
1. `dsa.py` 构造函数（#40913 的 hunk 4/5 打不上）：去掉上游后来才有的 `is_dummy` 空跑模式，其余照抄。
2. 我们底包的关键字叫 `transfer_layer_num`，上游后来改名 `transfer_layer_id_max`；新函数内部沿用上游名，传给底包 `PoolEntry` / 控制器时用 `transfer_layer_num`（值相同：GLM 的传输层号 0…44 连续）。
3. 底包 `_build_dsa_device_pool_group(kvcache, page_size)` 只有两个参数（上游多一个草稿参数）。
4. 底包还有一个旧的 HiRadixCache DSA 挂载函数调用已删除的 `build_anchor_sidecar_stack`，改走同一个声明式组装函数（我们不用这条路径，只为能导入、行为等价）。
5. #40914 的 `build_full_draft_pools`、#40915 的 `build_hybrid_mamba_stack` / `_MambaStrategy.build` 因上下文漂移（上游多了 `_stage_local_layer_mapping`、Mamba 布局覆盖）整段打不上，按上游新代码的结构在我们的底包上重写，逻辑逐行对应。
6. #38212：放宽加了 `enable_hierarchical_cache` 条件（§2）。
7. 本地新增（上游没有）：
   - `_carve_declared_sidecars`：`--hicache-size` 是每卡总量，indexer 镜像从 KV 份额里扣，而不是另加（上游在 #38212 里自己承认 sidecar 会超预算）。
   - 打包草稿的设备池比目标小就启动报错（主机恢复用目标的槽号寻址草稿缓冲）。
   - `DSAIndexerPoolHost.destroy` 取自 #38212。

## 4. 开关

全部是已有参数，没有新环境变量。
- `--enable-hierarchical-cache`：开 = 本补丁生效；不开 = 与现栈相同。
- `--hicache-size N`：**每卡**主机内存 GB（总量，含 MLA、indexer、KDA 状态）。建议从 **32** 开始（§5）。不给时按 `--hicache-ratio`（默认 2 倍设备池）。
- `--hicache-io-backend kernel --hicache-mem-layout page_first`（都是默认）：每层一次核函数按索引表搬运，不是逐页拷贝；JIT/`sgl_kernel` 的搬运核只用普通全局读写（`ld/st.global.L1::no_allocate`，sm70 起可用），sm80 可用。KDA 主机池只支持 page_first 系布局，`layer_first` 会在启动断言。`direct`（逐页 memcpy）不建议，也不在测试里。
- `--hicache-write-policy write_through`（默认）：每个新节点插入时就异步写穿；**不要用 `write_through_selective`**（要命中 2 次才备份，第一次复用前可能已被挤掉）。
- NUMA：`--numa-node 0 0 0 0 1 1 1 1`（按 pod 上 GPU0–3 在节点 0、GPU4–7 在节点 1；上 pod 前用 `nvidia-smi topo -m` 核对）。调度进程在建主机池之前绑到对应节点（`numa_bind_to_node` 设 CPU 亲和并 `numa_set_preferred`），主机池在启动时分配并 `cudaHostRegister` 锁页，热路径不再锁页。
- 不支持、会**启动报错**的组合：L3 存储后端（`--hicache-storage-backend`，树页≠传输页）、外部 cache linker、140 双快照（140 自己的守卫拒绝 HiCache）、int8 Mamba 检查点（底包拒绝）、草稿行宽/格式不一致或草稿池比目标小。
- 未验证、没有拦截的组合：170 的 breakable 预填充图（静态审查：DSA 注意力、kpool indexer、KDA 都在图外的 eager 段里读缓存，逐层等待可以生效；需实测）；115 DCP。

## 5. 内存与时间账（标注：【按config推算】/【推算】）

每卡每 token 的主机字节【按config推算，与代码公式一致】：
| 组成 | 无 MTP（11 层 DSA） | 有 MTP（+1 草稿层） |
|---|---|---|
| MLA 潜向量 bf16，512×2 B/层 | 11,264 | 12,288 |
| indexer 镜像，132 B/层（按页镜像，3/4 是空位） | 1,452 | 1,584 |
| 合计 | **12,716** | **13,872** |
| KDA 状态（每个检查点，34 层，fp32 SSM + bf16 conv） | 18,452,480 B = 17.6 MiB | 同左 |

MLA 在 TP8 下每卡一份完整副本，所以 8 卡各存一份相同内容，主机总用量是单卡的 8 倍。

`--hicache-size` 的分配【推算】：先按设备上 KV 与 KDA 池的字节比例分，再把 indexer 从 KV 份额里扣出。例：设备 KV 104 万 token（13.2 GB）、KDA 200 槽（3.7 GB）、有 MTP：
- 32 GB/卡 → KV 份额 25.0 GB → MLA 22.1 GB + indexer 2.9 GB ≈ **180 万主机 token**；KDA 7.0 GB ≈ **380 个状态**。8 卡共 256 GB。
- 128 GB/卡 → 约 720 万 token + 约 1,500 个状态；8 卡共 1 TB。
- 代码允许的上限约为 (空闲内存 − 10 GB) / 8 ≈ 236 GB/卡（按 1.9 TB 空闲）。
- 扩容看两件事：主机层淘汰次数，以及命中是否因为“状态没了”而截短。KDA 份额取决于设备上 KDA 池的字节数；MTP 的 verify 中间缓冲是否算进 `mamba_cache.mem_usage_bytes()` 没核实，可能让 KDA 份额偏大。

恢复时间【推算，未在 pod 上测】：6 万 token × 13,872 B = 0.83 GB/卡。
- PCIe 4.0 x16 有效约 20–25 GB/s，单卡独占时 33–42 ms。
- TP8 下 8 卡同时搬同一前缀；两卡共用一个 PCIe 交换机，若上行也是 x16，每卡约 12.5 GB/s，约 66 ms。
- KDA 状态 17.6 MiB，约 1 ms。按层流水，第 N 层到了第 N 层就能算。
- 对比：同样 6 万 token 冷算约 4.9 s（按 19 万 token 15.44 s 的实测速率换算）。
- 写穿是 D2H 方向，恢复是 H2D 方向，PCIe 双向各自独立；但两者都占拷贝引擎和少量 SM（JIT 搬运核带 `block_quota` 限制，默认 2，用来减少对计算的干扰；实际占用未测）。

命中长度的代价【推算】：开 HiCache 后命中按 256 取整，比现在的 64 最多多重算 192 token（约 15 ms）。角色边界快照也落在 256 上。

## 6a. 让 120/121/122 的调度保护在 HiCache 下生效（2026-09-24 夜并入）

059（正式 A + 旧版 180，lite N14）实测发现：120 的总开关里有 `not self.enable_hierarchical_cache`，开 HiCache 后 120 整体关闭，挂在同一开关上的 122 也失效。结果是冷请求不再压块、短命中不能插队。有请求排队时，8192 token 大块的批次在 058 是 10/1061，在 059 是 648/866。059 新增的 25 条 fast 超时里，多出的时间全在准入到开始执行之间（中位 +5.1 s），执行本身只多 0.34 s。所以 059、060 实际是"A 去掉 120 再加 180"，不是单变量。

改法：只在 L3 存储预取时绕开保护，只开 L1/L2 主机层时保留。依据：
- 冷块上限和 decode 让轮与缓存层级无关；
- 短命中插队本来就用 `needs_host_load_back()` 排除了待主机搬回的请求；
- 搬回由底包在准入之后照原流程做。

L3 预取会改变准入流程，180 也拒绝 L3，所以继续绕开。

测试 `tests/test_sched_protect_chain.py::HiCacheTierTests`（在正式 A + 180 树上运行，树由 `patch_stack.py` 构建到 `build/p180/candidate`）覆盖三点：
- 只开 HiCache 时保护生效，开 L3 时关闭；
- 纯设备命中负载下，开与不开 HiCache 的调度决定逐步相同，冷块仍被压、短命中照样插队；
- 待主机搬回的请求不插入进行中的块，也不触发搬回。

3/3 在新 180 上通过，在旧 180 上全部失败。原有 120 的 27 项与 122 的 12 项测试仍全部通过。

未验证：GPU 上这一改动的端到端效果，要等 A+新 180 的 8 卡运行。命中在主机上的短请求仍要等进行中的块做完才能搬回并开始；是否值得让它们带着搬回成本插队，要看搬回耗时的实测。

## 6. 不开开关时为什么与现栈相同

- 树页放宽只在 HiCache 开时发生。
- `schedule_batch` 的绝对检查点：前缀长度是 64 的倍数时，与原公式结果相同。现栈的 chunk 截断（按页或按 grid）、101、120 都保证这一点。
- `mamba_component` 的“不发布”：只在状态位置未对齐、超过键长或为空时触发，这些情况原代码会把状态挂到错误的键上。
- 缓存时先定其它组件上限、最后定状态：GLM 只有 FULL+MAMBA，FULL 不设上限，顺序与原来相同。
- 主机池相关代码只在 `init_hicache` 里跑。
- `scheduler.py` 的改动只替换总开关里 HiCache 那一项；不开 HiCache 时 `enable_hicache_storage` 也为假，结果与原条件相同。
- 实测：底包自带的 `test_unified_radix_cache_unittest`（2436 项，在本 CPU 机上能跑的 471 项通过，其余需要 CUDA 或更多内存）在打与不打 180 时，失败集合**完全相同**。

## 7. 与外部专家要求的对照

| 要求 | 现状 |
|---|---|
| 完整检查点（MLA+indexer+KDA+草稿同一位置） | 180 做到，位置语义见 §2；CPU 测试逐字节验证 |
| 256 组：注册/匹配/分叉/尾组/淘汰/恢复一起约束，确认对齐处真有检查点 | 180 用放宽树页一次约束全部；检查点由同一网格产生（§2B）；CPU 测试覆盖 320 步长 chunk、短 chunk 跨界、分支点、完成请求 |
| 异步、增量、提早备份；源页等 D2H 完成事件才可释放；已备份不重传 | 底包 write_through 已如此：插入即异步写穿，只写没有主机副本的节点；写穿期间节点加锁，`writing_check` 在完成事件就绪且各卡一致后才解锁；已备份节点淘汰时只丢设备副本，不需同步 D2H。180 未改 |
| 不要只按命中≥2 备份；会话结束不要立刻赶出 GPU | 用 `write_through`；底包按显存压力做 LRU，不在请求结束时淘汰。180 未改 |
| 恢复优先但不饿死写回；限制在途字节 | 底包 H2D、D2H 各用一条流，无优先级、无在途上限 → **181 候选** |
| 拷贝粒度：批量、按层流水 | kernel 后端：加载每层 1 次核、备份所有层 1 次核，索引表驱动，按层事件放行计算。180 未改 |
| 锁页、NUMA、启动时分配 | 底包启动时分配并 `cudaHostRegister`；`--numa-node` 绑节点（§4）。需在 pod 上核对拓扑 → 实测项 |
| 测试：逐字节往返、分叉污染、强制真主机命中（毒化）、在途 flush | 全部已写，CPU 通过，GPU 待跑（§8） |
| 指标：restore 排队/开始/就绪时间、回退原因、重算 token | 未做 → **181 候选** |
| 只做 GPU↔主机两级，不做 L3 | 180 在树页放宽时拒绝 L3 |

## 8. 测试（`scripts/tests/hicache180/`）

CPU 上跑真实代码：真实设备池、主机池、组装策略、`HybridCacheController` 写读路径、L2 引擎逐层循环、radix 树。只把 CUDA 字节拷贝核换成等价的 torch 拷贝，并校验调用方声称的行宽。所以“搬什么、搬哪层、搬到哪、何时提交”全是被测代码自己的决定。

`run_tests.sh TREE PYTHON` 在给定树上跑全部测试。实测结果：

| 树 | 结果 |
|---|---|
| 部署栈 000…170 + 180 | 52/52 通过 |
| S0（000 101 106 110 111 120 140）+ 180 | 52/52 通过 |
| 裸底包 + 180 | 52/52 通过 |
| 只有上游 #40913–#40915（无 256 所有权） | 23 项失败：分叉与陈旧组头页、#38212 检查点测试 |
| 部署栈、无 180 | indexer 往返、结构、flush、分叉、陈旧组头页全部失败 |

测试文件：
- `test_180_glm_host_pools.py`（5 项）：GLM 几何（有/无 MTP 草稿），经真实控制器写穿一组 256 token 和一个 KDA 槽；毒化设备上的源页、目标页和源状态槽；恢复到**不同的**页和槽，逐字节比较（纯拷贝，容差 0）。另测 flush 清空每个主机池、`--hicache-size` 是含 indexer 的总量、草稿池过小报错。
- `test_180_tree_hicache_e2e.py`（6 项）：真实树 + `init_hicache`。玩具“模型”按真实 `compute_pooled_write_locs` 写 indexer 行。测 (a) 陈旧组头页、(b) 组内 128 处分叉（设备上与主机往返后都查）、真 flush、写穿未确认时 flush（旧确认不得在 flush 后提交、槽可复用、新内容往返正确）、不开 HiCache 时树页仍为 64。每次恢复前整池毒化成 0xAB，读到旧显存就会失败。
- `test_180_upstream_38212_checkpoint.py`（7 项）：上游 #38212 检查点测试的适配版（改动写在文件头）；去掉了依赖上游较新 SWA 字段的 1 项，GLM 不用 SWA。
- `test_180_upstream_decl_stack.py`（34 项）：#40913–#40915 新增或改动的上游单测。上游用 `is_dummy` 不分配内存，这里给桩对象挂真实小张量，真的分配，覆盖的代码更多。
- 底包旧的 `test_hybrid_pool_assembler` 有 2 项因 API 改变而失败，属预期：上游 #40914 同步改了这 2 项，改后版本在上面的 34 项里通过。

**GPU（开发机，已写未跑）**：
1. `HC180_DEVICE=cuda run_tests.sh TREE python`：同一批测试换成真实 `sgl_kernel`/JIT 拷贝核、锁页内存和 CUDA 流。写穿未确认时 flush 的那项在 GPU 上是真正的 DMA 在途。
2. `gpu/hc180_numeric.py`（配 `gpu/sitecustomize.py`）：缩小版 GLM（8 层，KDA 0–2/4–6、DSA 3/7，另加 NEXTN 层；dummy 权重按 dcp_check 的办法重新初始化，让注意力起作用），TP2 真服务。对分支偏移 64/128/192/256 各跑一轮：
   - 冷算 → 设备命中 ×2（得到重跑噪声底，预期 0）→ 分叉 → 用无关请求挤掉 → 主机恢复；
   - 判据：生成 token 完全相同；主机恢复与设备命中的 logprob 差不超过噪声底；`--dump` 时逐层比较 DSA o_proj 输入；另外报告与冷算的 k3-KL（仅参考）。
   MTP 关、开各跑一次；同样的脚本在“无 180”的栈上应当失败，作对照。

## 9. 风险与未验证

- **全部是 CPU 结论。** 真实拷贝核、锁页、流并发、8 卡一致性、模型数值都没跑过。
- **KDA 主机恢复本身**：上游 #39830 报告 GDN 模型在上游 main 上主机命中 20/20 答错，原因未明，不是 indexer 的问题。本地 CPU 测试证明 KDA 状态字节逐位还原、位置对齐，但没有排除那类问题。GPU 数值检查必须单独看 KDA（无 DSA 差异时输出仍须一致）。
- 放宽后命中取整到 256，多重算最多 192 token（§5）；101 的角色切分也按 256。
- indexer 镜像 3/4 是空位（每 token 多存、多搬约 1.2 KB）。可以只搬组头页来省掉，另做。
- MLA 在每卡各存一份、各搬一份，这是底包 TP 设计，180 未改。
- MTP 下 verify 的中间状态缓冲不进主机层（不需要）。KDA 份额是否被 spec 缓冲放大，未核实（§5）。
- 170 breakable 预填充图、115 DCP 与 HiCache 的组合未验证（§4）。

## 10. 复现

```bash
python3 scripts/patch_stack.py apply /tmp/t180 000-interface-compliance.patch 101-role-boundary-split.patch \
  106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 114-indexer-row-shard.patch \
  120-sched-protect-chain.patch 121-sched-cap-while-decoding.patch 130-async-tokenize.patch 140-kda-dual-snapshot.patch \
  150-startup-warmup.patch 160-nextn-sm80.patch 170-glm-bcg-prefill.patch 180-hicache-glm-dsa.patch
scripts/tests/hicache180/run_tests.sh /tmp/t180 <python with torch>
```
上游原始 diff：`refs/pr40913.diff`、`pr40914.diff`、`pr40915.diff`、`pr38212.diff`（本补丁使用）；`pr40134.diff`（未采用，§2）；`pr38474.diff`（KL 测试设计，参考）。
