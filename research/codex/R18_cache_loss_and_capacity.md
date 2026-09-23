# R18 — b113 N6 缓存丢失与容量账（T49）

作者：Codex 主会话；2026-09-23。范围：只读底包、补丁与本地证据；未修改引擎/补丁，未访问开发机、pod、bohr、Trisol，未构建或提交。

**标记**：VERIFIED = 本轮读源码、执行原方法的 CPU 夹具或本地数据计算；INFERRED = 机制对真实异常的解释、容量情景或待验证方案。引用 `S/...` = `build/base_exact/sglang/srt/...`，`K/...` = `build/base_exact/sglang/kernels/...`；补丁引用为 `.patch` 文件本身的行号，不冒充应用后行号。实测性能引用 F59/F73–F76 与 `evidence/N6_b113/analysis.txt`，不把源码推断写成 8 卡实测。

## 1. 可以立即纠正的结论

1. **VERIFIED：KV 50%、KDA 4% 是扣除可淘汰缓存后的占用，不是物理驻留率。** 两个池都可能已满。这解除了“只用一半显存却淘汰”的表面矛盾，也推翻了由 4% 推出状态槽不紧张的依据。`S/managers/scheduler_components/pool_stats_observer.py:249`、`:271`。
2. **VERIFIED：252115-token / cached=64 的请求是本次 cohort 的链首，且该链只发送这一条。** 冻结未命中 1281 不等于本轮应当仅计算 1281；本轮没有发送它的同链前驱。摘要中损失最大的五条均是 cohort 链首。不能用它们证明运行中丢了 25 万缓存。见 §4。
3. **VERIFIED：原“19个intra”实际是18个intra+1个turn_start，且其中一条冻结LCP并非本次相邻prompt的LCP。** 5条短回退精确落在前请求8192网格的较早chunk末尾，101只在最后chunk拆角色点，能解释这个现象；另有branch覆盖end的源码路径。逐条表见§4.3，具体树事件仍须区分INFERRED。
4. **VERIFIED：140 已同时涉及快照和淘汰。** 它保存当前角色与末尾、关闭101拆分、修改 FULL/MAMBA/path-cap 的尾部优先淘汰；不是只改 kernel。但额外角色槽只用 allocator 真空位，压力下直接跳过，不能保证长期一直获得双快照。见 §5。
5. **VERIFIED：每卡当前两池约 21.23 GiB，其中 KV 11.17、KDA 10.05（不是 9.7）。** 584 是缓存与活动请求共同使用的总槽数；“5槽/请求”是并发上限的预算比例，活动态本身通常是 active+两个 ping-pong。见 §7。
6. **VERIFIED：012真实graph捕获增量是1.31GiB、实际bs≤116，不是简报的6.1GiB。** graph配置512→64，连同本模型VLM预算折减，直接约+3.6万KV token（3.8%），不能承诺+40%或腾6GB。见§7/§8。
7. **VERIFIED：完整measurement的KV不可淘汰峰值达到863296 token / 0.92，50%只是较早实时观察。** **INFERRED：N26中心情景124–130万；按完整N6峰值线性放大是374万压力情景**，不能保证N26可容纳。优先测状态/淘汰和静态预算；DCP有容量潜力，但115以外仍须核验DSA地址协议。

本轮收据：`evidence/T49/local_audit.json`（最初源码夹具/算术，含尚未套VLM折减的预算情景）；`remote_analysis.json`（Claude补回raw/日志后，真实prompt重渲染、19对LCP、日志行号与SHA）；`previous_prefill_slices.txt`。原始数据在`evidence/T49/remote/`，本会话只读本地回传文件。日志有批量统计，无逐节点eviction/track事件；报告不伪造这些事件。

## 2. 缓存生命周期：实际使用的是哪一个池、哪一个状态

### 2.1 匹配并不等于 FULL KV 最长公共前缀

**VERIFIED**：默认注册构建 `UnifiedRadixCache(FULL,MAMBA)`，不是旧 `MambaRadixCache`。`S/mem_cache/registry.py:143`、`:165`。

- FULL walk 得到 `full_kv_hit_length`，但 MAMBA validator 要求对应节点确有状态；返回给请求的 `device_indices` 截到满足所有组件的最深节点。树内分裂只分 FULL，新的中间父节点没有凭空可得的 KDA 状态。`S/mem_cache/unified_cache/components/mamba_component.py:142`、`:310`；`S/mem_cache/unified_cache/unified_tree_core.py:843`。
- 如果 FULL 的对齐命中深度超过 KDA 深度，`mamba_branching_seqlen=floor_grid(full_kv_hit_length)`。这是**当前已有树的匹配分叉点**，并不知道下一轮角色边界。它也可能来自别的请求/会话，不能一律解释为上一轮本链的角色位置。`mamba_component.py:155`。
- COW 给请求分配 active 状态槽；没空位就暂锁源节点、淘汰一个 MAMBA 槽、再分配。状态复制延迟到 forward stream，保护的是源槽生命周期。`mamba_component.py:187`；`S/managers/schedule_batch.py:2862`。
- 即使 FULL 在树里仍有很多 token，缺少相应状态也必须从较浅节点重算。重算插入时释放重复 KV，保留树中原份；不能只看 GPU 已计算过多少 token。`unified_tree_core.py:1071`。

### 2.2 extra_buffer、提交与 decode

**VERIFIED**：默认 overlap 下每活动请求有 active+两个 ping-pong，来自**同一个** `MambaSlotAllocator`。它们不是额外独立的584槽之外的池。`S/mem_cache/memory_pool.py:1226`、`:1276`、`:1465`。

| 时点 | 保存/提交规则 | 源码 |
|---|---|---|
| extend准备 | 只有一个常规 tracking 点；默认末尾向下对齐，chunk内合法branch会覆盖它；强制中间h读取用深度+1标记 | `S/managers/schedule_batch.py:2795`、`:2838` |
| extend结束/续chunk暂存 | 将最新tracking槽捐给树，先申请替换槽；只提交到tracking深度。重匹配后把请求锁移到新节点，尾部KV仍由请求拥有 | `mamba_component.py:529`、`:589`；`S/mem_cache/unified_radix_cache.py:949`、`:1016` |
| 提交后 | `mamba_last_track_seqlen=None`，不能把已捐出的槽当成本轮新的decode快照再次提交 | `mamba_component.py:645` |
| decode | 序列长度落在 `lcm(checkpoint_grid,256)` 时复制状态到tracking槽；结果处理更新深度与ping-pong索引 | `S/managers/schedule_batch.py:3348`；`S/managers/scheduler_components/batch_result_processor.py:1213` |
| 请求结束 | 只把**最新**decode tracking提交到树，不把每256处都保留下来；没有新tracking则有效新cache_len=0。释放 `[max(cache_len,cache_protected_len),kv_len)`，释放active和未捐的ping-pong，解请求锁 | `mamba_component.py:554`、`:608`；`unified_radix_cache.py:838`、`:882`、`:912` |

因此“输出少于256”不是精确判据：应看实际forward的绝对序列位置有没有跨网格；输出可能很短却刚好跨界。反过来，结束时已生成的最后token可能尚无KV，不能用 `prompt_tokens+completion_tokens` 直接冒充 `kv_committed_len`。

普通TP8网格64、decode网格256；DCP8的树页变512时两个网格都变512。`S/runtime_context.py:1920`、`:1928`。原版中间h为bf16，再写回fp32池不会恢复丢失精度；140导出fp32解决的是另一个正确性前提，不能仅靠多存一个bf16 h替代。`K/ops/attention/fla/chunk_delta_h.py:349`；`patches/140-kda-dual-snapshot.md:1`。

### 2.3 锁、LRU、淘汰的联动

**VERIFIED**：FULL锁沿路径保护KV，MAMBA锁只保护选中的状态节点；active是COW副本，并不意味着历史路径上的所有状态都被锁。请求完成后正常解锁，harness两次请求之间的sleep不会保留这些锁。`S/mem_cache/unified_cache/components/full_component.py:263`；`mamba_component.py:433`；`unified_radix_cache.py:912`。

- KV分配时只比较 `allocator.available_size()` 与实际申请量，不比较日志usage；空位不足才按短缺淘汰。`S/mem_cache/common.py:149`。
- MAMBA在COW、`alloc_req_slots`预留、捐赠替换槽等位置各自可能触发淘汰。`mamba_component.py:201`、`:493`；`S/mem_cache/allocation.py:241`。
- FULL按可淘汰叶子的LRU堆取节点，删除叶子会一并释放它的MAMBA；之后父节点可能成为下一可淘汰叶子。`full_component.py:194`；`unified_tree_core.py:1429`。
- MAMBA按状态LRU取未锁节点；**内部节点**只丢状态，FULL可留下；**可删除叶子**会连KV删除。一次状态槽短缺可能导致大量KV被连带释放，尤其长路径已被删去中间状态时。`mamba_component.py:360`、`:402`。
- MAMBA匹配只刷新实际使用的最深状态，不刷新祖先；FULL只刷新选中匹配节点向根的路径。LRU时钟是事件计数，不是秒数。`mamba_component.py:118`；`unified_tree_core.py:823`；`S/mem_cache/unified_cache/components/tree_component.py:110`。
- `mamba_max_states_per_path` 默认 **-1无限**。显式启用才删浅层、单子、未锁内部状态，保留tail/fork/leaf，是软上限。012真实args也为-1、extra_buffer、interval256、无统一内存/HiCache/session。`S/server_args.py:2595`；`mamba_component.py:252`；`evidence/T49/remote/012_server.log:14`。

## 3. (a) 244–300秒空闲后几乎全丢：先纠正观测，再定位触发池

**VERIFIED：实际日志公式**（`S/managers/scheduler_components/pool_stats_observer.py:249`）：

```text
KV_usage    = (KV_capacity - KV_free - KV_evictable) / KV_capacity
Mamba_usage = (state_capacity - state_free - state_evictable) / state_capacity
physical_resident_fraction = 1 - free / capacity
```

这里的evictable是**已占槽但可以释放**，不是已经free。一个真实方法CPU夹具输入KV capacity=943360/free=0/evictable=471680、MAMBA capacity=584/free=0/evictable=560，输出正是 **0.50 / 0.0411**。两池都满，但日志看起来50%/4%；见 `evidence/T49/local_audit.json` 的 `actual_observer_full_pools`（T49-01）。这是计数公式验证，不是声称N6当时正好是这组free值。

**INFERRED，当前最有依据的解释**：本链空闲时其他agent持续插入，旧路径解锁且逐渐变LRU，触发FULL或MAMBA分配淘汰；再次出现时只余共同开头。不能从244/300秒反推“5分钟TTL”。Unified路径没有读 `cache_ttl_seconds` 的淘汰分支；初始化参数虽保留此字段，不等于当前cache实现有TTL。

**两池谁先满尚未VERIFIED**。即使总KV尚有真空位，584状态槽也可能被历史chunk/角色/分支状态填满；反之FULL压力会同时删状态。普通chunk8192、无path cap会积累中间状态；许多短分支每新增不到1451个KV token就占一个状态，状态池相对更易成为历史缓存约束。长单链每8192 token一个状态则偏向KV先满。这个密度判断比活动4%更有用，但不能代替事件日志。

**补回完整日志后的VERIFIED**：measurement为04:55:54.865–05:25:54.477 UTC，峰值863296 token/0.92（`012_server.log:5644`），Mamba非evictable最大24/584。50%并非完整N6峰值。raw`:476`隔244.00s后cached77952→33600，实际两prompt LCP78561；raw`:626`再隔300.81s后cached33600→1216，LCP78972。两个当前LCP都超过上次已使用的命中深度，证明**原可复用路径退化**，不能用“下一prompt不同”或“前请求没存end”解释全部损失；谁驱逐了它仍是INFERRED。raw`:695`也有短idle5.28s+排队14.77s后17664→576，LCP17938，容量淘汰不只发生在长idle。

**flush/namespace的证据边界**：

- 本地 `run_dev.py:231` 在warmup之后、measurement之前调用flush；`S/managers/scheduler.py:4763` 只在全闲时真清。012日志最后一次测前flush在04:55:48（`:1538`），下一次在05:25:55（`:5754`）；**measurement中没有flush成功、重启/重新载权或retraction日志**，也没有TTL证据。无法从未打印的eviction行推导“没淘汰”。
- harness `X-S1-Cache-Namespace` 只在header；b113的000/101/105/110–113没有把它接入 `cache_salt`，原header表也没有X-S1项。`s1-dev/harness/s1_loadgen.py:100`、`:110`；`S/entrypoints/request_headers.py:10`。故不能用这个header解释本轮跨请求隔离；但必须记录有效 `extra_key/cache_salt` 才能排除实际部署额外配置。
- 同理X-S1-Session-ID不自动建立有锁的引擎streaming session；只知道同一个session_id不能证明历史缓存被保护。

## 4. (b)/(c) 短回退与25万极端例是两类证据

### 4.1 短间隔5–8k回退的精确机制

**VERIFIED（源码机制，T49-02）**：101 `_role_split_len` 在 `prefix < branch <= full_len` 时返回None，统计 `skip_branch_conflict`；角色扫描仅用于可完成的尾块/准入完整请求，且受已有partial、32768扫描窗口、grid等条件限制。105只补“继续中的partial存在时不要再截第二个partial”的保护，没有新增状态，也没有保活TTL。`patches/101-d1v12-on-base.patch:47`、`:59`、`:92`、`:110`；`patches/105-role-split-single-partial.patch:3`。

一个可手工检查的机制例子（**合成，不是那19条的伪造trace**）：

1. 当前请求prefill起点36864、结束45000；树的FULL分叉点38912，当前希望保留的角色位置44040→floor64=44032。
2. 本次extend长8136，能装进8192。101看到38912分叉，跳过角色split；原tracking选择38912而不是44992。
3. 真实原方法CPU执行给出 `track_seqlen=38913`（强制h标记）、`mamba_last_track_seqlen=38912`；无branch对照则为44992。
4. prefill提交38912，reset last_track。假设只输出32且无有效decode网格提交，结束释放38912之后请求拥有的KV。
5. 下一prompt的LCP在当前角色44040，树却只留38912有效状态，额外重算约 **44032−38912=5120**。时间间隔可以很短，不需任何LRU淘汰。

还有两条必须分开的机制：

- **有更深decode快照也可回退**：下一prompt在reminder处改变，末尾/decode状态处于已分叉的旧后缀，不能拿来恢复。只能找该LCP以内最深状态。降低decode间隔不能修复这个位置错误。
- **101不在所有chunk补角色点**：冷长请求只在最后可完成chunk尝试role split；若最后角色位于较早chunk中间，原chunk-end网格未必正好有该点。即使角色被保存，LRU也可删它。不能把任何5–8k损失直接断言为一个固定branch-conflict。

F76“短间隔停在上一轮角色边界”应收窄：下面19条没有一条cached恰等于前请求**最后**user/observation标记floor64；有的落在更早chunk/分叉点。仍需branch/chosen_track/finish事件将每条分类。只用冻结 `glm_lcp_with_prev` 不够：harness重放冻结prompt，本次生成输出不会直接拼入下一prompt（`s1_loadgen.py:359`）。

### 4.2 252115→cached64：本地已有更直接的解释

**VERIFIED（T49-03）**：测量使用的 `s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json` 中：

| prompt | 摘要cached | 冻结未命中 | 实际cohort链内序号 / 该链发送条数 | requests.jsonl行号 |
|---:|---:|---:|---:|---:|
| 252115 | 64 | 1281 | 0 / 1 | 696 |
| 251916 | 384 | 1196 | 0 / 2 | 697 |
| 222738 | 7296 | 1298 | 0 / 5 | 699 |
| 229576 | 18240 | 504 | 0 / 1 | 17 |
| 203233 | 8704 | 1295 | 0 / 1 | 704 |

第一条rid=`scimaster:canon:QSdTYVbowNG_k_R8lOG7T:llm:1`，cohort只含这一条，没有该链的`:llm:0`。完整chain元数据中的 `n_requests=4` 是原链规模，不能替代measurement cohort。五条的cohort位置/ID详见本轮JSON；数据集一共有115个cohort链首仍标 `phase=intra`。

回传raw再次确认：252115这条在`012_dev_raw__.jsonl:688`（若后续重排行号，以rid定位），`idx_in_chain=0`；其余四条也均为0。

scorer按 `idx_in_chain==0` 优先归为chain_start，见 `s1-dev/harness/s1_common.py:107`、`:131`；`analyze_run.py:39` 的缓存损失榜仅用 frozen expected减actual，并未排除这些链首。因此1281不是本轮可实现命中的保证，27.47秒也不是fast_intra的27.47秒超标。**已证实的是缺少同链前驱，不是证明从未存在任何跨链可复用内容。** 缓存64到底来自哪个其他请求、是否曾有更长跨链路径被淘汰，仍需树日志。

对真正有已完成本次前驱、LCP约25万却只命中64的情形，才依次查：FULL被删；FULL仍在但MAMBA内部状态被删；namespace/flush/restart；实际输入早期差异；retract/abort不提交。普通末页对齐最多损失63、decode末尾网格最多约255，**都不能单独解释25万回退**。

### 4.3 回传raw的19条逐项核对（T49-08）

**VERIFIED**：筛选`idx_in_chain>0 && prompt−uncached_expected−cached>4096`，得18 intra+1 turn_start。使用原harness Renderer、原tokenizer本地重渲染这19对涉及的36个prompt，长度逐一等于raw；计算真正的prompt-prompt LCP。raw不含本次生成token文本，若下一prompt完整包含前prompt，其后的生成输出匹配仍未知，因此下表LCP是已能直接验证的基准。

表中行号均指`evidence/T49/remote/012_dev_raw__.jsonl`；Δ为**实际prompt LCP−cached**；gap为前请求完成到当前dispatch，未包含queue。C=chunk角色点漏存证据强；P=已有命中深度退化，压力淘汰解释强；B=branch/准入保护/其他树状态原因待事件；D=冻结前驱口径不适用。分类均为**INFERRED**，数值为VERIFIED。

| raw行（前驱行） | prompt | cached | 实际LCP | Δ | gap秒 | 分类 |
|---|---:|---:|---:|---:|---:|---|
| 165（158） | 45229 | 35648 | 40604 | 4956 | 2.34 | B |
| 169（165） | 48633 | 40576 | 45229 | 4653 | 2.82 | B |
| 210（206） | 41775 | 36672 | 40994 | 4322 | 3.81 | B |
| 257（255） | 40098 | 31232 | 38781 | 7549 | 0.64 | C |
| 265（264） | 91386 | 82816 | 90193 | 7377 | 0.99 | C |
| 270（267） | 29533 | 21312 | 27195 | 5883 | 1.03 | B |
| 299（292） | 35212 | 25792 | 33673 | 7881 | 6.12 | C |
| 347（346） | 79629 | 70656 | 77651 | 6995 | 1.48 | C |
| 358（344） | 44486 | 37312 | 42304 | 4992 | 17.63 | B |
| 374（371） | 95880 | 49216 | **52400** | **3184** | 1.50 | D；原冻结“丢43797”不能成立 |
| 451（447） | 83794 | 74560 | 81124 | 6564 | 3.38 | C |
| 476（366） | 80607 | 33600 | 78561 | 44961 | 244.00 | P |
| 608（606） | 16553 | 9280 | 14960 | 5680 | 0.27 | B |
| 623（609） | 29162 | 8896 | 14117 | 5221 | 5.81 | B |
| 626（476） | 80803 | 1216 | 78972 | 77756 | 300.81 | P |
| 660（658） | 87765 | 16576 | 21884 | 5308 | 0.51 | B |
| 661（655） | 9618 | 4096 | 9115 | 5019 | 8.97 | B |
| 695（690） | 18472 | 576 | 17938 | 17362 | 5.28 | P；另排队14.77秒 |
| 721（714） | 16982 | 9600 | **14653** | **5053** | 9.83 | B；phase是turn_start |

**C类的具体机制**：以raw255→257为例，前请求从cached23040开始，两个8192块分别到31232、39424，最后剩50；最后角色在38720对齐点，落在**第二个仍被truncated的chunk**。101的`:92`只在not truncated时尝试role split，而最后50-token块扫描起点39424已越过角色。因此树可能有31232和39424，却没有38720状态；下一prompt在38781分叉，39424及更深decode状态都在分叉后，实际回落31232、额外7549。真实批日志`012_server.log:3004`、`:3005`给出8192/8192及pending8242/50，与此一致。即使decode保存到39680，也不能用于38781以内的前缀。

其余4个C类同样满足`cached=前请求cached+k×8192`，k分别6、3、5、5；下一个chunk末尾已越过前请求最后角色，最终剩余分别137、99、438、560。raw292日志`:3268`/`:3269`还能直接看见最终pending99、尾128对齐。**这些已有真实token与批日志支持，比笼统“上一轮角色滞后”更精确**；但缺少节点事件仍不排除同时有LRU/branch作用。140在角色所在的任意extend导出快照，直接覆盖这类最后chunk限制，前提是角色槽成功分配。

D类证明冻结LCP本身需审计：raw374冻结93013而实际与raw371只有52400，差40613；不能要求本次未处理过的“源输出”前缀被缓存。raw721冻结LCP16136而实际14653也有1483差。剩余B类保留明确的rid/前驱/track取证位置，不用单点机制图替代运行时证据。

## 5. 修复建议及140的边界

**VERIFIED，140已有实现**：

- 开启后101 helper返回空，避免额外拆分与105相应限制；tracking不再让branch覆盖end；每个extend可导出end及当前最后user/observation角色点，含fp32 SSM和conv历史。`patches/140-kda-dual-snapshot.patch:638`、`:669`、`:748`。
- 先提交常规end，再临时锁住end，用**树拥有的KV**插入角色状态，处理duplicate，不重用已free的请求旧KV。`同文件:784`。它不是每多一个状态就复制一整条KV。
- 同时改变FULL叶子优先级、MAMBA扫描和path-cap顺序，优先牺牲标记为tail且非role的状态。`同文件:845`、`:878`、`:892`、`:910`。R8把“tail优先淘汰”全部留给141已过时。
- 额外role槽条件是 `available_size() > len(batch.reqs)`，**不把evictable算成free，也不先淘汰**。因此状态池满时140可能只存end。`同文件:762`。

**INFERRED，建议优先级**：

1. **先做池/树归因日志，再测140完整服务**（承接P140-08）。短请求额外split减少与当前角色状态才是直接收益；T45离线chunk8192只有+16704命中token、extend −20%，不能承诺修复所有19条。
2. 若日志确认role `slot_skip` 出现在满状态池，新增**准入阶段的可选角色槽预留**：在源节点/旧tracking受正确锁保护时，为本批优先回收无用tail并预留至多一槽/请求；不得在已launch的forward之间抢占overlap仍在用的slot。失败时沿用当前安全跳过，不改变用户token。必须对duplicate、并发COW、abort/retract、flush与池守恒重验。
3. 若FULL先满，压缩/回收**没有状态且没有任何后代需要它的无锁叶子**、改善tail优先和保留共同角色/工具边界。不要把“某请求KDA深度之后的KV”一刀切：那些KV可能是其它有状态后代的公共祖先，或被活请求引用。
4. 若MAMBA先满，减少低价值中间/旧tail状态，保护当前可复用角色、共享system-tools边界和近期prompt end；之后才考虑减池。过小的 `mamba_max_states_per_path` 会先删浅层共享点，role也不是硬锁保护，不能只设2就当“role+end保证”。141的首角色/网格方案要同时付出额外槽成本。
5. `--mamba-track-interval 64` 是append类补充消融，可减结束尾部损失；它不替代role快照，也不修容量淘汰。每次checkpoint拷贝约17.6MiB/rank、频率增加，应实测decode成本。

## 6. 交给Claude的8卡取证与最小验证

Claude已回传任务012 raw/run/job/server四文件，`012_job.log:1`绑定b113源码签名、`:3`记录role IDs。已有证据能确认实际参数、flush边界、批次与真实LCP，**仍不能区分每次FULL/MAMBA淘汰**。下一次只需增加下列事件；不要重复拉全环境或凭据。

建议TP0事件日志（其余rank只报不一致校验），用单调时间+forward迭代+rid+节点ID，避免每步打印大张量或同步GPU：

| 位置 | 最小字段 / 判定 |
|---|---|
| 每次KV/MAMBA申请及evict前后 | 各capacity/free/evictable/protected，申请量、触发调用点、被删组件数、连带KV数；分别算resident与不可淘汰率 |
| match结束 | FULL LCP、最终KDA深度、branch、选中节点/slot、extra_key/cache_salt摘要、cache generation；FULL长/KDA短定位状态缺失 |
| 每次extend/finish | prefix/end、role、branch、101 skip原因、实际track深度、140 role allocated/skipped、insert/duplicate、`cache_protected_len`/`kv_committed_len`、tail free起止 |
| eviction | trigger=KV alloc/COW/donation/req reserve/path cap/retract、node depth与key长度、FULL/MAMBA锁计数、LRU序号、role/tail、deleted或tombstoned |
| flush/restart | cache generation递增、请求入口、busy/success、清后allocator counts；重启PID/源码签名 |

**不把 `/metrics cache_hit_rate=0` 当零命中证据**：`S/managers/scheduler_components/metrics_reporter.py:876`、`:935` 的decode上报直接写0；prefill另在`:687`计算。采样时机即可出现0，无需先假定Unified未记账。cached_tokens仍保留原如实计数。

最小实验（均交Claude安排，**本轮未执行8卡**）：

- **T49-04 / idle vs pressure**：固定合成prompt与续写prefix，真flush后建缓存；无其它请求等310秒后复用；另一臂相同等待期插入无共享长prompt（偏KV压力）；第三臂插入大量短分支（偏状态压力）。用free和eviction trigger确认根因，不能只看usage。等时但无流量不丢、加流量丢，支持LRU而非TTL。
- **T49-05 / stale boundary**：复现§4.1构型，原栈与140 off/on配对；分别令输出跨/不跨256边界，并让下一prompt在role前后分叉；记录每次extend/track/finish和matched depth。不得仅看decode结束时有一个更深状态便判PASS。
- **T49-06 / memory A/B**：原启动、graph64、graph64+经峰值证明安全的mem-fraction；再单变量ratio 0.5/0.3。记录load前后free、池实际bytes、capture bs列表及before/after free、prefill峰值、两池驱逐与全部SLO。承接P140-08的槽压力/flush门。
- **T49-07 / DCP+115**：DCP1/2/8，同prompt cold与命中续算logits比较；跨64/256/512边界、分叉、retract/flush、graph/eager；故意令allocator使用超过旧物理容量的高虚拟slot并核对indexer读写界限。短启动/数学冒烟不替代此项。

## 7. 每卡显存账

### 7.1 固定分池与精确字节

**VERIFIED（源码+F59数值）**：基线普通TP8、非MTP、bf16 MLA、34 KDA+11 DSA、无unified memory。`S/mem_cache/kv_cache_configurator.py:2045`：

```text
R = post_weight_free - pre_load_free × (1 - mem_fraction_static) - mm_reservation
Mamba_budget = R × r/(1+r), r默认为0.9
state_slots = floor((Mamba_budget - one_state) / one_state)
KV_budget = R - (state_slots+1) × one_state
```

`r=0.9`表示MAMBA约占缓存预算47.37%；**不是显存47%**。余量在进程初始化时一次定池，graph缩小后不动态让KV长大；本模型MLA被post-capture sizing排除（`S/arg_groups/overrides.py:1753`）。`max_total_tokens`只能下限约束/截小已算容量，不能强行增加实际池。`kv_cache_configurator.py:2131`。

| 分项（每rank） | 计算/数值 | 证据、边界 |
|---|---:|---|
| MLA KV | 11×512×2 = **11264 B/token** | 无RoPE维；`s1-dev/glm_tok/config.json:248`；`S/model_executor/pool_configurator.py:332` |
| indexer K+scale | 11×(128+4) = **1452 B/token** | uint8存FP8+fp32 scale，物理按原token容量分配；kpool4并未令分配/预算除4。`S/mem_cache/index_key_cache.py:15`、`:32`；`pool_configurator.py:426` |
| KV合计 | **12716 B/token**；943360→**11.17193 GiB**，另哨兵页不到1MiB | TP8每rank复制这一完整逻辑KV；不能乘8当可服务token容量 |
| KDA一槽 | 34×[8×128×128×4 + 3×3072×2] = **18452480 B = 17.59766 MiB** | fp32 SSM+bf16 conv；`S/configs/mamba_utils.py:116`、`:294`；`S/configs/glm5_next.py:264` |
| KDA池 | (584+1)槽：SSM **9.71191** + conv **0.34143** = **10.05335 GiB** | 0号padding也分配；`S/mem_cache/memory_pool.py:583`、`:608`；F59 `notes/findings.md:325` |
| 两池总量 | **约21.2253 GiB** | 不含小元数据、KV哨兵页与以下运行内存 |
| 请求映射/小池 | 116×约context_len×4字节；约0.11GiB量级，另kpool K/score tails约数MiB、分配器表 | `S/mem_cache/memory_pool.py:254`、`:4807`；真实shape为准 |
| decode graph | **简报/R8报告约6.1GiB** | 是capture前后free差，不是单独查出的纯graph张量；可能含capture/warmup相关持久分配。`S/model_executor/model_runner_components/cuda_graph_setup.py:530`、`:570` |
| 权重与加载后常驻开销 | **本地缺启动原日志，不能报精确实测值** | `S/model_executor/model_runner.py:1128`、`:1225` 有load前后记录；磁盘328GB/8不是运行权重显存 |
| 激活/通信/graph预留 | 默认启发式见下，**预算而非实测占用** | 不能把它与6.1GiB再全部相加计费 |

按默认8192 chunk、TP8、配置graph512、KDA prefill graph disabled，`S/arg_groups/memory_hook.py:245`、`:292` 的预留为：512MiB元数据 + **12288MiB激活启发式** + 1024MiB并行余量 + 1024MiB graph启发式 = **14848MiB=14.5GiB**，再按设备容量算并舍入mem-fraction。精确预留用 `pre_load_free×(1−实际f)`，不会与此完全相等。

**INFERRED条件反算**：如果pre-load free约79–80GiB、实际预留约14.5GiB且两池如上，则load及此前常驻开销约 **43.3–44.3GiB**；不是已读出的权重测量。graph6.1GiB属于14.5GiB运行余量内的后续消耗，理论余下约8.4GiB供其它运行峰值，不能声称“另有14.5GiB空闲”。精确账需启动阶段free/allocated/reserved及prefill峰值闭合。

### 7.2 N26需要多少KV

**VERIFIED（数据算术）**：722个prompt共34416777 token，按请求平均47668.7；输出预算平均298.6。**INFERRED（驻留估算）**：

| 情景 | 逻辑KV token | bf16 KV每卡 |
|---|---:|---:|
| 26×请求均值 | 1239385 | 14.68GiB |
| 26×50000 | 1300000 | 15.40GiB |
| N6不可淘汰峰值0.50线性放大26/6 | 2043947 | 24.21GiB |
| 26×257000长上下文压力情景 | 6682000 | 79.13GiB |

前两项给设计中心，第三项给压力参考，第四项说明尾部不能忽略；全都不是N26容量实测。N是agent槽，sleep和等待不一定拥有active KV；长请求服务时间偏置会抬高同时驻留均值，共享前缀则降低唯一token数。真正需要的是：**同时运行的唯一受保护KV + decode增长/页碎片/重算重复的瞬时空间 + 希望留给空闲链的历史缓存**。历史cache目标还取决于reuse distance，而非只看N。不能把50%按N外推成“N14开始首次淘汰”：N6可能早已淘汰历史缓存。

KDA活动部分通常3N=78槽≈1.34GiB，另树中受保护状态和捐赠瞬时峰值；并发预算5N=130槽≈2.23GiB；140额外角色的同时在途上界可再加N槽≈0.447GiB。这些是当前活动部分，**不是历史状态总需求**，不能据此把584直接缩成130。

## 8. 杠杆：按当前收益证据/风险排序

### 8.1 先收graph，再根据真实峰值调静态预算（较低风险、收益须分两步）

**VERIFIED**：配置默认max_bs512（`memory_hook.py:102`），但实际capture列表还按请求池大小裁剪（`S/model_executor/runner/base_cuda_graph_runner.py:64`）。F59并发cap116，所以不能用“已经capture512”计算节省。需读capture日志 `bs=[...]`；考虑padding时也不能机械断言最大恰为116。

建议普通N26先试 `--cuda-graph-max-bs-decode 64`（旧 `--cuda-graph-max-bs` 仍是alias，`S/server_args.py:3929`）。实际batch超过capture上界会走eager，正式能力门并发、MTP verify宽度也要另看，不能只按dev N26保证所有请求都进graph。

- 如果mem-fraction仍自动：512→64只少预留 **896MiB**；r不变时约52.63%变KV，约47.37%变状态，预计 **KV +38887 token，约4.1%**。f舍入、其它分配有小误差。
- 如果显式固定mem-fraction：减少graph本身通常只增加运行空余，**KV初始大小基本不变**。
- 如果实测真省5GiB且据此安全提高静态预算：r=0.9时也只约2.63GiB给KV，即+22.2万token、约23.6%；只有固定/缩小MAMBA池并把全部5GiB转给KV，才接近+42万token/+44.8%。5GiB本身尚未实测。

**INFERRED建议**：保持prefill工作量/120/140一致，先收graph测峰值，再小步提高f。每增加0.01f，若pre-load free≈79–80GiB，池预算约+0.79–0.80GiB，固定r时KV约+3.5万。激活随chunk、长context indexer、ragged batch变化；不能照其它模型的0.9/0.95直接套。

### 8.2 `mamba_full_memory_ratio`（中等风险，必须知道触发池）

**VERIFIED条件算术**：固定本轮两池总预算21.2253GiB、每槽/每token字节不变、非spec：

| r | 状态槽（不含padding） | 并发上限 floor(slots/5) | KV token | 相对基线 |
|---:|---:|---:|---:|---:|
| 0.9 | 584 | 116 | 943360 | 1.00× |
| 0.5 | 410 | 82 | 1195840 | 1.27× |
| 0.3 | 284 | 56 | 1378688 | 1.46× |
| 0.2 | 204 | 40 | 1494784 | 1.58× |

输入R由已分配两池反算，原始budget的取整剩余未取回，槽数可能相差1；重启后graph/weight/workspace变化也会改变R。0.9→0.5不是旧文所说1.5×。0.3可覆盖130万中心情景但缺历史KV余量，状态槽减半也可能增加重算；若当前MAMBA先触发，盲减r可能变慢。

`extra_buffer_lazy`节省的是每活动请求约一个槽，N26约0.447GiB可用于更多历史状态，**固定池大小下不直接增KV**；并发预算比例5→4。当前140明确拒绝lazy（`patches/140-kda-dual-snapshot.patch:708`），不能直接组合。`--max-mamba-cache-size`可在扩大f时固定状态池，使增量更多给KV，但须用历史状态/slot_skip证据确定大小。

### 8.3 DCP `--dcp-size` +115（理论容量收益最大，但正确性风险高于上述调参）

**VERIFIED源码意图**：普通TP8的MLA/KV复制；DCP把虚拟token按rank取模存、再除DCP宽度，allocator大小与逻辑页分别乘D，物理page仍64。`S/mem_cache/kv_cache_configurator.py:1994`；`K/ops/kvcache/mla_buffer.py:42`。若完整KV/indexer链都正确分片、物理池预算近似不变，逻辑token上限理论变D倍：D=2/4/8约188.7/377.3/754.7万。

**DCP不是DP8，也不是prefill CP**：只设dcp-size不会把 `attn_tp_size` 从8改成1；KDA仍按8头/rank划分，不产生DP8的8倍状态槽开销。`S/runtime_context.py:135`、`:172`；`S/configs/glm5_next.py:267`。decode会all-gather Q到64头并合并各rank局部attention/LSE，prefill会gather前缀KV，未必直接加速冷prefill。`S/models/deepseek_common/attention_forward_methods/forward_mla.py:633`、`:767`。115修的是64头tilelang共享内存，不修这些协议。

**INFERRED，必须先排除的源码接线风险**：

1. DSA backend从 `req_to_token` 取逻辑loc，`real_page_table=loc//64`（`S/layers/attention/dsa_backend.py:985`、`:1275`）；kpool以这些页号构造压缩写入地址（`S/layers/attention/dsa/kpool_fp8_index.py:344`）。Hybrid DSA池却仍以物理 `size` 构建默认 `index_buf_size=size`（`S/mem_cache/memory_pool.py:3762`、`:4737`）。这些函数没有DCP owner/filter或除D变换。高虚拟loc是否越过index buffer，是必须验证的具体疑点，不能用声明的8倍allocator容量证明可用。
2. sparse decode取得的是rank物理 `get_key_buffer`，topk到页表变换及tilelang入口没有在这里显式mask owner/除D（`dsa_backend.py:3312`、`:3345`、`:3393`）。extend侧通用MLA已准备gather buffer，但DSA路径在`:3001`仍取物理pool。需要查实际运行page_table/topk地址与输入buffer是否在其它路径已正确转换；**本轮未运行完整DCP，故不把此疑点写成已实测错误**。
3. DCP8树页512使角色对齐损失从最多63变最多511，decode tracking也变512；小append更易没有结束快照。140守卫检查CP/DP等，但没有显式拒绝dcp-size；“能通过守卫”不等于已经验证140+DCP。先做T49-07，再组合。

若选择保留indexer全局复制、只分MLA，每逻辑token每DSA层成本应为1024/D+132字节；D8理论只有 **1156/(128+132)=4.45×**，不是8×，且需把index容量/预算/地址一起修正。若要index也分片，则需全局topk一致性与kpool跨rank归属协议。当前115没有解决这部分。

### 8.4 FP8 KV（较高实现与能力风险，容量也不是精确2×）

**VERIFIED**：当前indexer已用FP8 storage+scale；需要改的是MLA KV。按现有scaled FP8布局，512值+4组fp32 scale=528B，加原132B index，11层为 **7260 B/token**，固定KV预算容量 **1.7515×**，943360→约165.2万。`S/mem_cache/kv_cache_configurator.py:2472`。除非再改变index分配/scale布局，否则不能承诺2×。

sm80的当前tilelang CUDA v1输入为bf16（`K/ops/attention/dsa/tilelang_kernel.py:303`、`:325`、`:1403`），不是只改dtype flag便可用FP8 MMA。需要在KV store与稀疏gather处做正确量化/核内解码，并覆盖有前缀/无前缀、decode、graph、DCP、kpool tail。若整池解码成持久bf16副本会抵消容量收益；全context反量化也可能吃掉带宽收益。112/113负责indexer本来就读其FP8编码，不应笼统重写成“它们尚不支持FP8”。长上下文输出/logits与正式能力门必须验证。

int8 KDA checkpoint/降低SSM dtype排在更后：当前源实现另建int8池（`S/mem_cache/memory_pool.py:1282`），不能假定直接把fp32原池一半字节返给KV；140也明确拒绝int8。递归状态精度风险与额外池预算需要独立解决。

## 9. 请Claude更新的过时/被纠正结论清单

以下仅列改写建议，**未覆盖Claude的research文件，也未改旧findings原文**。区分“真实数据更新”与“源码/口径纠正”，保留历史版本条件。

| 文件:位置 | 原结论/问题 | 应改成什么 |
|---|---|---|
| `notes/findings.md:441`（F75） | KV峰值50%、状态4%→N14起才满/驱逐 | **源码纠正**：是不可淘汰占用；物理free未知，N6可能已历史淘汰，不能确定哪个池先满 |
| `notes/findings.md:449`（F76缓存段） | “当时KV峰值仅50%”被当作低容量压力 | 保留观察，去掉“因此不应淘汰”的含义；短回退机制逐条归因仍待日志 |
| `research/claude/R8_next_directions.md:57` | KDA584槽9.7GB | 9.71是SSM，conv另0.34，总10.05GiB；日志GB标签实际以2^30计 |
| `research/claude/R8_next_directions.md:58` | 130万>94万→容量将先于算力约束 | 是中心容量情景；**实测N6先卡intra排队**。容量/状态/调度共同测，不能先确定约束顺序 |
| `research/claude/R8_next_directions.md:61` | graph64预计腾5GB→KV+40% | 未实测5GB；默认预算直接+约4.1%KV；实际capture受116请求池约束；更大转KV收益需调f/池 |
| `research/claude/R8_next_directions.md:62` | lazy/int8可作为状态池参数与140并用 | 当前140拒绝lazy/int8；lazy固定池不增加KV；int8是额外池，先闭合预算 |
| `research/claude/R8_next_directions.md:63` | tail优先淘汰列为141待实现 | 140已改FULL/MAMBA/path-cap优先级；141新增共享首边界/网格，另测池压力 |
| `research/claude/R8_next_directions.md:64` | FP8 KV容量×2 | 当前布局固定预算约1.75×，需sm80 MLA KV解码与能力验证 |
| `research/claude/R8_next_directions.md:20` | system-tools-changed“结构性零命中” | 改为可能在变更点前复用共同前缀；R8自身§6已给非零潜力，F76总命中也显示跨链复用，不能字面全零 |
| `research/claude/R8_next_directions.md:48` | 以chain_start为当前约束门安排P1 | 前排画像不能替代本栈：F76 fast/overall FAIL、chain有余量；预填充仍通过队头阻塞影响intra |
| `research/claude/base/00-summary-mainline.md:10`、`:48`（M0行） | A100长prefill风险运行时未证，候选fa3 | F57/F59/F73已证tilelang+110/111可启动；fa3不适合A100；原A/B已失败是历史状态 |
| `research/claude/base/00-summary-mainline.md:29`；`03-hybrid-cache.md:99`附近 | “先耗尽KV不是状态槽” | 只对特定状态密度成立；F75指标不支持该断言，必须看free/驱逐原因 |
| `research/claude/base/03-hybrid-cache.md:221` | ratio0.9→0.5使KV约1.5× | 固定总预算为1.9/1.5≈1.267×；本轮943k→约1196k |
| `research/claude/base/03-hybrid-cache.md:139`、`:240` | 深状态之后KV“never served”、回收低风险 | 当前匹配不能用不等于全树永远无用；有状态后代/活锁依赖祖先KV，回收应限无依赖无锁叶子 |
| `research/claude/base/03-hybrid-cache.md:194`附近 | decode interval64 “No new risk” | 功能有现成开关，但复制频率/overlap压力需实测；不修角色分叉，DCP8有效grid仍512 |
| `research/claude/base/04-model-kernels.md:12`、`:78`、`:79` | 当前FP8 MoE走Triton，Marlin不可用/待实现 | b113用111 Marlin W8A16，F58/F73/F74已验证；调优目标不再是该不可跑FP8 W8A8默认路径 |
| `research/claude/base/04-model-kernels.md:68` | 冷36k约1–2s；40–60% MFU猜测 | F76约1万tok/s，36k粗推约3.6s，不能当实测36k；profile明确MoE/DSA/mHC/allreduce占比 |
| `notes/findings.md:351`（F63） | indexer占60%主瓶颈 | 明确仅110 torch版本；b113真机F76 indexer4.3%，主要预算已迁到MoE/DSA/GEMM/mHC/通信 |
| `notes/findings.md:415`（F71）、R8结构方向 | 单卡成本外推19.5万约15s | 保留“无allreduce估算”，生产规划用F76 19万19.3s；稀疏attention+ mHC实测28.8%，非整服务必减1/3 |
| `research/claude/base/04-model-kernels.md:33`、`:43`、`:82` | MTP可用，预计1.5–2×/笼统无A100验证 | 原版不能直接推可用；160已有算子证据、普通TP8服务与接受率未测；r/A情景不是TPOT承诺，140组合仍拒绝 |
| `notes/findings.md:443`（F75） | cache_hit_rate=0疑似Unified不更新 | decode reporter会显式写0；先区分prefill/decode采样，再判统计缺陷 |
| `evidence/N6_b113/analysis.txt:44`（不改原证据）及据此写的结论 | 最大“lost”视为同链缓存丢失 | 给衍生分析增加idx_in_chain/前驱是否实发；本轮最大五条均链首，不能混入19条真实前驱损失 |

收尾：本轮结果止于源码/CPU数据审计与可审阅验证方案；没有宣称19条已逐条定责、N26已可容纳、DCP/FP8/140完整服务已通过。
