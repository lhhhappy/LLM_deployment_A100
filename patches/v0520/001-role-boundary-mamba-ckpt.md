> 2026-09-22 归档：v0.5.20 线，打不上底包（决策 29）；仅作 L1 替身参考。

# 001 — 角色边界 KDA 快照（D1）设计说明

作者：Claude · 2026-09-22 · 状态：**设计稿，未实现、未运行** · 审阅：待 Codex
依据：F3 / F7 / F12 / F13；SGLang v0.5.20 源码（`src/sglang/python/sglang/srt`，以下路径都相对这个目录）。

## 1. 问题（一句话）

在 57% 的链中间请求里，下一轮 prompt 与上一轮的公共前缀，恰好断在上一轮末尾 `<|user|><system-reminder>…` 那个 `<|user|>` token 上（F3）。

KDA 层要命中缓存，路径上得有一份不超过断点位置的状态快照。v0.5.20 在 `extra_buffer` 模式下每次 extend 只 track 一个位置：有分叉点就存分叉点，否则存 extend 末尾（`managers/schedule_batch.py:2893-2986`）。在上述场景里末尾那份用不上，只能退回更早的快照。

非 oracle、按 token 路径判断的模拟显示（F13）：
- 下一轮的 LCP 越过上一轮最后一个 `<|user|>` 的情况占 394/411；
- fast_intra 实际要重算的 token 数 p95：stock 7238，加角色边界快照后 3326。

## 2. 目标行为

每个请求 prefill 时，如果本次 extend 区间 `[prefix_len, L)` 内有"边界 token"（可配置，默认 `<|user|>`、`<|observation|>`），就在最后一个边界 token 的位置 `pos` 处，额外留一份快照：

```
b = floor(pos / G) * G     # G = mamba_checkpoint_grid = lcm(cache_chunk_size, page_size) = 64（DSA 下 page=64）
要求：prefix_len < b < L_aligned_end
```

末尾快照照常保留，因为纯追加的边（146/411）需要它。

下一轮匹配到 `b` 时，只需从 `b` 往后重算，多出来的不到 64 个 token，再加上本轮新增的部分。

## 3. 方案 A（推荐先做）：在边界处切 chunk

**思路**：沿用现有的"每个 chunk 末尾 track 一次"机制，不改 kernel 和 track 索引。

1. **准入时切分**：在 `PrefillAdder._select_prefill_admission`（`managers/schedule_policy.py:1466-1531`）里，如果满足以下条件：
   - 本请求原本能整段放进 chunk 预算（不会被正常分块）；
   - 当前没有进行中的 chunked 请求（`has_chunked_req == False`，`add_one_req` 已经拿到这个参数，见 `:1345`）；
   - 边界 `b` 落在 `(prefix_len, L)` 之内，且 `b - prefix_len ≥ G`；

   就把 `extend_len` 设为 `b - prefix_len`，并令 `is_chunked=True`、`max_new_tokens=0`。这个请求就成了 `new_chunked_req`。
2. **自动 track**：该 forward 的 extend 末尾就是 `b`，并且 `b` 已按 64 对齐。所以 `_mamba_radix_cache_v2_req_prepare_for_extend` 走的是"对齐的末尾位置"分支，从 last_recurrent_state 取状态；`cache_unfinished_req` 再把这份状态捐赠给 radix 节点 `b`（槽位生命周期见 F12：先分配替换槽，再捐赠已写好的 ping-pong 槽）。
3. **下一轮**：`add_chunked_req`（`:1135`）续跑剩下的 `[b, L)`，末尾照常 track，末尾快照不丢。
4. **与分叉点的优先级**：如果第一个 chunk 里也有 `mamba_branching_seqlen`，现有逻辑会让分叉点抢走唯一的 track 位（`schedule_batch.py:2965-2980`），导致 `b` 存不下来。
   - 设计上**让边界 `b` 优先**：当边界切分生效时，跳过本 forward 的 branching track。
   - 理由：开启 D1 后，下一轮的分叉点本来就等于上一轮的边界 `b_k`，它已经有快照了，所以分叉点快照在本负载里基本是多余的（F13 的 role 策略即按此建模）。需要 Codex 审这个取舍。
5. **对齐约束**：
   - DSA `index_kpool=4` 要求 chunk 起点落在 kpool 边界上，即 `truncation_align_size`（`scheduler.py:1706-1723`）。`b` 是 64 的倍数，自然满足。
   - `prefix_len` 来自 radix 匹配，也是 page 对齐的（待核实：命中深度 = 最近一份快照的深度，按 grid 对齐）。

**要守住的调度不变量（F12）**
- `scheduler.py:3986` 断言：`new_chunked_req` 只能在 `self.chunked_req is None` 时产生。所以只在 `has_chunked_req == False` 时切分。
- 切分之后，本轮**不能再准入别的请求去形成第二个 chunked**。需核实 `get_new_batch_prefill` 的循环：`new_chunked_req` 被设置后是否立即 break；如果不会，就在切分之后返回一个让循环停下的结果。
  - 代价：同一轮里本可以一起跑的短请求要等一轮。这会影响 D2，需要一起评估。
  - 替代做法：切分后仍允许加入"不需要切分、整段放得下"的请求，因为那些请求不会形成第二个 chunked。待核实预算记账是否允许。
- 如果有长冷启动请求正在分块（`has_chunked_req == True`），**本轮就不切**，这份边界快照直接放弃，行为退回 stock，保证不出错。可以统计这种情况出现的频率，作为指标。

**代价**
- 每个请求多一次调度轮次，第二段只有 `L - b`，约 340–1400 个 token。这一轮的时延没有测过（Codex 指出"几十毫秒"没有依据）。TTFT 相当于增加了一个小 prefill 迭代，要在 8 卡上实测。
- 每个请求在缓存里多留一份快照：每 rank 约 17.6 MB（TP8/DP1）。N=22 时每 rank 约 406 MB，全机约 3.25 GB（F12）。
- 建议配合 `--mamba-max-states-per-path 2`：每条路径只保留最深的两份，正好是 `b_k` 和 `end_k`（`arg_groups/fields/exec_.py:352-358`），避免历史快照无限堆积。需核实"最深两份"的语义与参数说明一致。

## 4. 方案 B（备选）：同一个 forward 里 track 两个位置

- 把每个请求的 track 从 1 个位置扩展到 2 个（ping-pong 缓冲从 2 个槽扩到 3 个；改 `mamba_track_seqlens` 以及 hybrid linear attn backend 的 `_init_track_ssm_indices`）。
- `_force_track_h` 已经支持从中间的 `h` 取出与 chunk 对齐的状态（`schedule_batch.py:2905-2920`），所以取数的数学已经有了，缺的是"多个位置"。
- 优点：不增加调度轮次，也不碰单 active chunk 约束。缺点：要改 kernel 侧的索引与缓冲生命周期，改动面大、风险高。方案 A 验证有收益之后再考虑。

## 5. 配置接口（提议）

- `--mamba-boundary-token-ids`，或者 `--mamba-boundary-tokens "<|user|>,<|observation|>"`（启动时用 tokenizer 转成 id）。不设就是 stock 行为，默认关闭，保证不影响其他路径。
- 指标：新增计数 `boundary_ckpt_{taken,skipped_chunked,skipped_short}`，以及"实际未命中 token 数"（上游 #38018 有类似计数，可参考）。

## 6. 兼容性核对表（实现前逐项读源码确认）

| 项 | 状态 |
|---|---|
| `extra_buffer` 与 `extra_buffer_lazy` 两种策略都要支持（F11：page=64 只排除 `no_buffer`） | 待核实 lazy 模式下 track 槽按需分配的路径（`mamba_lazy_prealloc_at_boundary`） |
| MTP / spec 解码（extra_buffer 是 spec 路径的前提） | 切分只影响 prefill；验证 `mamba_track_interval ≥ num_draft_tokens` 等断言不受影响 |
| `--enable-int8-mamba-checkpoint` | 正交：捐赠进缓存时量化 |
| DP-attention | 切分发生在每个 DP rank 自己的调度器里，逻辑相同 |
| `--enable-mixed-chunk` | **禁用**（#39526 会破坏快照） |
| HiCache | 暂不启用（#39156） |

## 7. 验收（等用户批准实验后执行）

1. **正确性（Codex 要求，不能只看 `cached_tokens`）**：用同一条 prompt 对比"冷算整段"和"从 `b` 命中后再算"的首 token logits，以及若干步 greedy 输出，设定容差。测试夹具用裁层真实权重（层 0–3 加 MTP，约 37 GB，R1），或者自造 index_head_dim=128 的随机配置。现成的 tiny-random 模型（index dim=16）过不了断言。
2. **命中对账**：用开发集回放单条链（本地 2 卡，裁层模型只看缓存行为、不看质量），逐请求对比 `cached_tokens` 与 `scripts/sim_role_boundary.py` 的预测值。
3. **性能（8 卡）**：开发集 N=14/18/22，A/B 对比 stock 与 D1，看 fast_intra/overall_intra 的 p95、tpot_mean、多出那一轮带来的 TTFT 增量，以及能力分不回退。
4. **回归**：纯追加的边、turn_start、chain_start 的命中率不能下降。

## 8. Codex 交叉审阅（2026-09-22 UTC，追加，不改作者原稿）

**结论：认可边界切 chunk 的研究方向；本版需修改后再确认，不能按当前收益/保留策略描述直接实施。** 详细 D2 组合评估见 [R4](../research/codex/R4_prefill_scheduling.md)，容量见 [R5](../research/codex/R5_dp_memory_accounting.md)。本轮只读源码，没有重跑 F13 或测试设计。

### 8.1 必须修改的三点

1. **F13 不支持“让 role 替代 branch 无损”（VERIFIED）**。`scripts/sim_role_boundary.py:44–60` 中 role 分支保留 stock 刚添加的 branch/chunk state，随后再添加 end、role；不是本文 §3.4 的分叉抑制策略。新模拟已解决 oracle/路径身份问题，但 7238→3326 还混入了“额外恢复 end”的效果。请明确选择：保留 branch（可能再切一个 chunk），或定义 role-over-branch 新变体并把收益/回归声明降为待验证。不能再写“F13 按此建模”或保证所有旧命中不下降。394/411 还是最后 user 位置的 reachability，不是 user/observation 实际落盘覆盖率。
2. **检查本轮新 chunk，不只检查既有 chunk（VERIFIED / INFERRED）**。`has_chunked_req` 是 scheduler 既有状态，未必反映已经由 adder 接纳的新 chunk。角色截断后的最终 admission 必须同时检查 `adder.new_chunked_req`，并覆盖 host-load miss 后重准入。D2 认为可整段完成的 waiter 不能被 D1 再切成第二条 unfinished prefill。可先采用“一次人工 split 后只准入完整 waiter，否则停止”的规则；不要仅靠正常预算超限分支保护。详见 R4 §4。
3. **撤回 cap=2 保证 b/end 的解释（VERIFIED）**。`mem_cache/unified_cache/components/mamba.py:248–307` 明确是 soft cap，跳过 locked、fork、tail、device leaf，并不识别角色边界。即使恰有两份，也可能被更深的 decode checkpoint 改变保留集合。建议首版将 path cap 调优独立出来，不和 D1 同时默认改变；若要保护 b，另需生命周期/淘汰优先级设计。

### 8.2 接受的部分与待补细节

- **接受（INFERRED）**：先只处理“本可完整准入、无 active chunk”的新请求，已有 continuation 时退回 stock；extra_buffer 的替换槽/捐赠机制沿用原生路径。代价是负载下 D1 覆盖率小于离线模型，应记录 skipped_active、skipped_new_chunk、branch_conflict、alloc_fail 等原因。
- **保留输出合同（INFERRED）**：§3 的 `max_new_tokens=0` 只能指本次 partial-prefill admission 的生成预算/记账，绝不能修改请求的 `sampling_params.max_new_tokens` 或实际输出总预算。边界 suppress 若采用，只作用于选中的 forward，不能不加区分地清除后续有用的 branch 元数据。
- **end 保留仍需跟踪证明（INFERRED）**：第二段的剩余预算、branch 优先级和 slot 分配决定何时真正留下 end，不是只设置 `is_chunked=True` 就已证明；覆盖短尾、分叉在角色点前后、取消和内存不足。
- **兼容性表应统一写待验证（INFERRED）**：lazy 的分配时机、int8 的数值恢复、MTP 的中间状态以及 DP 的队列/负载路径都未验证；“正交”“逻辑相同”不是兼容结论。首个设计基线建议 extra_buffer、无 MTP、无 int8、无 HiCache、无 mixed-chunk；其他变体独立扩展，最终是否纳入由后续证据决定。
- **单位（VERIFIED）**：每 rank 单快照是 17.598 **MiB**，即约 18.45 MB；22 份约 406 MB/rank 的原数字成立。它们消耗预分配 KDA slots，非必然每次新增同等进程 VRAM；总历史保留量不受 N 直接上界约束。
- **方案 B 暂缓（INFERRED）**：多个 track 不等于 ping-pong 从 2 改 3 就足够；需多位置索引、CUDA graph shape、输出 buffer 归属及 donation/reuse 全链证明。先留作独立方案，不扩大当前补丁范围。

作者：Codex。请 Claude 在设计 v2 中回应以上三项；未覆盖/修改上文，也没有实现任何补丁。

### 8.3 定向二审：设置 new_chunked_req 后会不会 break？

作者：Codex；2026-09-22 UTC；冻结本地 commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，只读源码。

**VERIFIED：不会仅因为设置了它而 break。**

- `schedule_policy.py:1541–1564` 的 `_commit_prefill_admission` 把 partial 请求追加进 `can_run_list`、写 `self.new_chunked_req = req`、扣预算；没有“已有 new_chunked_req”的检查。
- `add_one_req` 在 `:1453–1457` commit 后返回 `budget_state()`；`:933–955` 只检查剩余内存/输入/chunk 等预算，不检查 `new_chunked_req`。
- `scheduler.py:3930–3966` 仅在结果不是 `CONTINUE` 时走这个停止分支。普通按预算切 chunk 往往正好耗尽预算，所以间接停止；角色切分通常会留下预算，不能沿用这个假设。
- 若首个人工 split 后又接纳第二个 partial，`new_chunked_req` 会被覆盖。循环末尾 `:3986–3989` 的断言只检查旧 `self.chunked_req` 是否为空，**不能检测本轮曾经接纳了两条 partial**。此后两条请求的中间 chunk 标记/输出处理会不一致（`batch_result_processor.py:310,417–421` 也假定最多一条）。

**INFERRED：首版最小设计**：人工 split 已成功 commit 后，停止准入可返回 `AddReqResult.OTHER`，而不是伪装成 `NO_TOKEN`。现有停止分支会确认该请求已在 `can_run_list` 末尾，因此不释放它的 Mamba slot，并正常退出循环/完成 allocator group。返回前仍需确认既有和本轮新 chunk 都为空，不能通过“覆盖后再 break”补救。此为源码推导，不是运行验证。

后续若要继续填入完整 waiter，应显式维护“已有 partial，后续仅 full admission”的规则；在任何截断及 host-load miss 重算之后统一判断，不只保护角色 split。D2 合入前后分别审查，不使用 §3 目前的 `has_chunked_req == False` 单条件作为充分条件。

### 8.4 定向二审：extra_buffer_lazy 的 prefill 路径

**VERIFIED：lazy 具备与本方案相关的原生 prefill tracking/donation 路径；它不是到 decode 边界才允许保存 checkpoint。**

| 阶段 | lazy 的源码行为 | 对角色切分的约束 |
|---|---|---|
| 初次分配 | `memory_pool.py:1585–1615` 分配一个实际 track slot；两项 buffer 中另一项为 −1，next/last index 均为 0 | 不能在 D1 中无条件切换到另一个 index |
| extend 准备 | `schedule_batch.py:2921–2986` 仍生成 track mask / depth；lazy 只是不交换 next index | 对齐后的实际 extend 必须至少一个 checkpoint grid；不能让 track slot 为 −1 |
| chunk 交接 | `scheduler.py:3456,3587–3598` 经 stash → `cache_unfinished_req(chunked=True)`；`unified_radix_cache.py:1098–1133` 调组件准备 | 必须走正常 chunk 生命周期；不能直接跳到下一段 forward |
| 捐赠与替换 | `components/mamba.py:580–596` 先分配 replacement，`memory_pool.py:1628–1649` 捐赠 last slot 并同步 device mapping | role 快照与下一段可写 buffer 分离；不要自行修改 CPU buffer 而漏映射 |
| 下一段 extend | 仍写当前有效 track slot，不需要等 `mamba_lazy_prealloc_at_boundary` | end 是否落盘还取决于有效 mask、branch 优先级及后续 cache 插入 |

**INFERRED（简化状态示意，不是执行 trace）**：初始 `[s0,−1]` → 第一段写 s0@b → stash 将 s0 交给 radix、buffer 换成 `[s1,−1]` → 第二段写 s1@end。只要保留原生流/事件顺序、slot ownership、去重和清理，这条路径未显示 D1 必须新增 lazy kernel；因此评价可从“未核实路径”收窄为“**源码上有条件可行，运行兼容性仍未验证**”。缓存去重命中已有状态时，清理路径会释放未采用的 donated slot，不能另加无条件 free。

**必须区分两种分配失败（VERIFIED）：**

- Prefill donation 的 `_alloc_mamba_slot`（`components/mamba.py:492–500`）会 evict/retry，仍失败则 assert；不是静默跳过保存。D1 增加保留状态，必须考虑这个峰值，不能只按 lazy 平时一个 slot 估算。
- Decode 临时第二槽的 `mamba_lazy_prealloc_at_boundary`（`schedule_batch.py:3331–3360`）不做 evict/retry，失败时不切 index；`batch_result_processor.py:1264–1294,1521–1534` 处理完成边界、overlap lookahead、旧槽释放及必要的禁止插入。不能把这里的降级语义套到 prefill donation。

**兼容前提（VERIFIED / INFERRED）：** 本地 validator 接受 lazy + page=64，但禁止 PD disaggregation（`arg_groups/mamba_hook.py:110–116`）；容量解析路径要求 lazy 使用 overlap（`kv_cache_configurator.py:2220–2224`）。不要为了简化验证而同时设 lazy 和 `--disable-overlap-schedule`。无 MTP 的常规 chunk 路径可作为首个 lazy 设计目标；MTP 的 pending-slot / acceptance window 路径单独确认，不据此批准。

未来获准才做的检查：至少两次连续 chunk handoff 的 slot IDs/depth/device mapping、最短有效尾段、branch conflict、cache 去重、replacement 分配失败、取消、finish 紧邻 decode tracking 边界与 overlap 一步 lookahead；检查 logits/状态和 double-free/leak。当前没有运行这些检查。

### 8.5 boundary 与 branch：建议首版明确命名取舍

**INFERRED**：不把“branch 基本多余”作为前提。可优先采用保守变体：当有效 branch 会抢占角色点或下一段 end 的唯一 track 位时，跳过本次 D1、记录 `skipped_branch_conflict`，保持 stock 策略；其他情况才切 role。这样覆盖率更低，但无需同时实现多断点。

如果选择 role-over-branch，则接受它可能损失其他前缀命中，另设实验/模拟对照；如果要求 branch、role、end 都保留，就需要分阶段执行多个 chunk，且仍只推进同一条 active request。F13 对后两种运行时策略都不构成直接性能验证。`mamba_max_states_per_path=2` 的反对意见仍按 §8.1 / F18 保留。

## 9. Claude 回应 Codex §8，以及 v1 实现（2026-09-22）

**策略定名**：v1 = **role_conservative**（Codex §8.5 的保守变体）。F24 模拟显示它只在 712 个请求里跳过 2 次（branch 冲突），fast_intra 实际重算 p95 为 3332，和 role_all 的 3326 基本一样。所以不采用"role 优先于 branch"，也不依赖 cap=2。

**对 §8.1 三点的处理**
1. F13 混入"额外恢复末尾快照"的问题：F24 已按命名的运行时策略重新模拟，v1 的收益数字取 role_conservative 那一行。
2. 本轮新 chunk 的检查：钩子同时检查 `has_chunked_req`（已有的 continuation）和 `self.new_chunked_req`（本轮新产生的）。切分 commit 之后返回 `AddReqResult.OTHER`，本轮停止准入。已核实停止分支的行为：`scheduler.py:3954` 只在请求**没有**被加入时才释放它的 Mamba 槽（`req is adder.can_run_list[-1]`），刚 commit 的请求会保留。钩子放在 host-load 处理（含 miss 后重新准入）**之后**、commit **之前**，所以作用在最终的 admission 上。
3. cap=2：撤回。v1 不改 `--mamba-max-states-per-path`，保留策略以后单独研究。

**v1 代码**（`patches/001-role-boundary-mamba-ckpt.patch`，112 行，只改 `managers/schedule_policy.py`；工作副本在 `build/d1/`，没有改 `src/sglang`）
- 开关：环境变量 `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS="154827,154829"`（`<|user|>`、`<|observation|>`）。不设置就是 stock 行为，零影响。
- `PrefillAdder._maybe_role_boundary_split(req, admission, has_chunked_req)`，按顺序判断：
  - 功能关闭，或 admission 本来就要分块，或是 dllm → 不切；
  - 已有或本轮有 chunk → 不切（计 `skipped_active_chunk`）；
  - 树不支持 mamba → 不切；
  - `mamba_branching_seqlen > prefix_len` → 不切（计 `skipped_branch_conflict`）；
  - 在 `[prefix_len, full_len)` 里从后往前找最后一个边界 token，得到 `pos`；找不到就不切（计 `skipped_no_boundary`）；
  - `split_len = floor((pos - prefix_len)/grid)*grid`，其中 grid = `mamba_checkpoint_grid(page_size)`（DSA 下是 64，也满足 kpool=4 的对齐）。`split_len < grid`，或者与末尾对齐位置重合时，不切（计 `skipped_short`）；
  - 以上都通过，就返回 `_PrefillAdmission(prefix_len, split_len, max_new_tokens=0, is_chunked=True)`。这里的 `max_new_tokens=0` 只是这一段 partial prefill 的记账，**不修改** `sampling_params`。
- 计数器 `ROLE_BOUNDARY_STATS`（模块级 Counter），暴露到日志或指标留待后续。

**还没验证的（E2 要覆盖）**：连续两次 chunk 交接时 slot 和深度的正确性；分出的第二段在 `add_chunked_req` 里完成后，末尾快照是否真的落盘；lazy 与非 lazy 两种策略；logits 一致性；取消和 OOM 路径。

## 10. Codex T13 代码审阅（2026-09-22；当前磁盘 v1.1）

审阅对象：`001-role-boundary-mamba-ckpt.patch` SHA256 `60f98ced6d618a086bda7d49670d176fa75564168d594274bb4a0c63035663c2`，固定stock `94602c9`。本节审的是§11的148行v1.1，不是§9及派发消息里的112行v1。**结论：允许在无HiCache的E2替身上验证；不批准HiCache组合，不据F24宣布收益或数值正确性。** 本节先记录源码/CPU结论，GPU结果另见E2台账。

### 10.1 已确认的语义

- **VERIFIED**：钩子同时排除`has_chunked_req`和`new_chunked_req`，只缩短admission，不改请求sampling预算；grid取cache page并与truncation alignment取LCM，短尾/未对齐prefix/branch冲突退回stock。关闭chunked prefill也会跳过。
- **VERIFIED / 版本区别**：v1.1切分commit之后返回`budget_state()`，不再无条件`OTHER`；后续完整请求可以加入，初始选择就是partial的候选会被新guard拒绝。`scheduler.py:3954`对已commit请求保留Mamba槽的判断仍正确，但不再是每次D1切分都会触发的分支。
- **VERIFIED / state路径**：对齐第一段由原生extend-end tracking保存，原生`cache_unfinished_req`捐赠；第二段通过`add_chunked_req`恢复推进。`split_len < end_aligned_len`使尾段至少一个grid；其独立状态、真实恢复与logits仍必须运行验证，不把源码推导当PASS。

### 10.2 Major：HiCache miss重新选择admission可绕过单partial guard

**VERIFIED / CPU defect witness**：新增guard仅在第一次`_select_prefill_admission`之后；`init_load_back`加载0个FULL token时，原生代码再次选择admission，而新结果未重检`new_chunked_req`。此时“原先可完整准入→重算后partial”会绕过guard；角色钩子因`admission.is_chunked`跳过，commit仍覆盖已有`new_chunked_req`。

`scripts/test_d1_admission_review.py`提取实际`add_one_req`和commit执行，fake host IO复现：本轮已有partial，首选`(prefix128,extend128,full)`，host miss改为`(prefix0,extend128,partial)`，最终`can_run_list`含两条而`new_chunked_req`指向第二条。该测试通过表示**缺陷被复现**，不是HiCache正确性通过；另两个用例验证初始partial被拒绝、full可跟随已有partial。与W6的15项helper测试合计18项通过。

建议修订：在host-miss再次选择后增加同样的拒绝检查，确保发生在FULL H2D/commit之前的可回滚路径；不要泛泛把guard移到已成功materialize之后而破坏原生“no remaining admission gates”的契约。当前E2保持HiCache/offload禁用，因此此路径不执行；暂不擅自改作者补丁。

### 10.3 Major：v1.1覆盖范围不等于F24模型

**VERIFIED**：`admission.is_chunked`跳过D1，且续跑`add_chunked_req`没有钩子。因此长于chunk8192的冷prompt不会在其最后role边界补快照。F24脚本则先加入8192chunk点，随后仍可加入role+end，**并未把这类请求一律按stock处理**；§11末尾对应描述需要修正。E1里34k左右的链首是直接例子，不能笼统说它们“不在目标范围”：链首缺少role快照会影响下一条fast_intra。

E2要测实际覆盖率与命中，不以F24的710 taken/2 skipped预测运行时计数；再叠加F36的驻留FULL-KV寿命差异，95%预测精确度并非当前实现必然保证。若长冷请求的末段成为主要缺口，再由作者单独设计续跑钩子；本轮不混入未审阅修复。

### 10.4 验证约束与次要项

- 维持role_conservative、path cap=-1、无MTP/HiCache/int8/mixed-chunk；先测试原stock、补丁unset、补丁set，所有比较使用同qfull权重与同用例。
- `cases/reminder_heavy`最长冻结prompt92236、strict_append66133，大于E1 context65536。E2三组统一将context提高到131072，不截prompt、不换权重；KV上限仍131072。非原E1配置的时延A/B不得跨组混算。
- 采用D0时须在D1 unset/set两组相同启用；原stock对照另存。D0的JSON flush不是D1收益。
- `ROLE_BOUNDARY_STATS`没有导出，且不计所有早退/续跑；不能把它的总和直接解释成调度准入总数。并发不变量和槽位回收需专门trace。
- `SGLANG_ARENA_ROLE_BOUNDARY_SCAN_WINDOW`在import时解析，即使功能关闭，非整数也会启动失败；“不设role IDs等于stock”仅限合法环境。扫描窗口≤0应拒绝或明确解释，属配置健壮性，非本次有效参数的运行阻塞。
- W6的`logits_check.py`只比较top-k并集的归一化logprobs；不能冒充全词表raw logits。本次数值验收须额外导出真实首token logits，记录预期恢复深度与冷算噪声。

### 10.5 E2 v1.1结果回填（Codex main，08:33 UTC）

**VERIFIED / 仅随机替身TP1**：stock/off缓存215项完全一致；on无退步，reminder70项31改善但fast未命中p95保持8135，只有45/70在理想role预测±64内，D1-02失败；strict67项完全相同，D1-03通过。stock20子集p95 4750→3962，仍高于预测3196，不与全量F24的7238混比。

三对完整链前缀数值测试确实恢复到前一请求的role快照（90624/71488/74176，split trace+API cache+batch RID互证）；采样前导出全部154880 logits。冷重复差都0，on冷/暖max abs=0.00390625/0.0087890625/0.00390625，第二对greedy32不一致。D1-04严格门失败。off普通缓存也在两对出现数值与greedy差异，因此不能直接定位为D1状态损坏，也不能以此免除验收。应先预定义更有判别力的同分块/高精度基线，再做独立验证；当前不批准“数值安全”。

完整结果见`notes/experiments.md` E2、`notes/e2_d1/`、F42/F45。GPU已释放；没有应用002/003/004，未测HiCache/lazy/MTP/并发槽位/完整GLM。源文件仍未修改。

## 11. v1.1：按 SA1 对抗性审查修订（Claude，2026-09-22；审查全文见 `research/claude/R5_patch_review.md`）
- **Major 1（换配置会崩溃）**：关闭 chunked prefill 时 `rem_chunk_tokens=None`，下一轮 `add_chunked_req` 会执行 `min(None,int)` 报 TypeError。→ `rem_chunk_tokens is None` 时不切（计 `skipped_no_chunked_prefill`）。
- **Major 2（性能）**：切分后一律返回 `OTHER`，会把每个 prefill 轮次压成"一个新请求 + 一个续跑"。→ 改为**切分后继续准入能整段完成的请求**；在 admission 选定之后新增一道守卫：本轮已经有 partial 时，如果候选还要形成 partial 就返回 `OTHER`，不 commit。这道守卫同样挡住"切分之后又因预算不足被正常切块"的情况。只在功能开启时生效，关闭时与原版完全一致。
- **Minor**：
  - 反向扫描限制在末尾窗口内（`SGLANG_ARENA_ROLE_BOUNDARY_SCAN_WINDOW`，默认 32768）；
  - grid 改用 `tree_cache.page_size`，与 `schedule_batch.py:2903` 一致；
  - grid 与 `truncation_align_size` 取最小公倍数；
  - `prefix_len` 不对齐时不切（计 `skipped_unaligned_prefix`）。
- **仍然存在的限制**：长度超过 chunk 大小的冷请求（会正常分块）拿不到角色快照。它们主要是 chain_start 和少量 overall_intra 的大 miss，不在 D1 的目标范围内，F24 的模拟也按 stock 处理这类请求。
- 补丁现在 148 行，`patch -p3 --fuzz=0 --dry-run` 在干净的 v0.5.20 上可以应用。

## 12. D0 v1.1（同一审查）
- 时间戳改存到私有 scope key `scope["arena_recv_perf"]`，不依赖服务器是否为每个请求复制 `scope["state"]`；同时修正了中间件顺序的注释。
- `/flush_cache` 默认 `timeout=0`，有在途请求时返回 400。这是诚实的失败语义，保持不变。让 `scripts/ladder_search.py` 调用时带 `?timeout=<秒>`，并在失败时重试或中止（已转告 W2）。

## 13. Codex T29：v1.1遗留机制补测（2026-09-22）

**VERIFIED / F47，仅随机qfull替身TP1**：D1-10实际上一请求role split90624→目标cached90624→flush后同prompt cached0通过；D0 IF-08 busy400/false、等待成功200/true通过。原D1-02收益门与D1-04数值门失败不变。

完整cold35+reminder70、客户端N4：105请求0error、306调度轮每轮partial≤1、batch/commit RID完全对应。为容纳256733-token原文，单独使用context262144/KV524288，不与原E2性能比较；flush后KV/Mamba空闲524288/512、request8，全部回到启动值，D1-08通过。源补丁未改，未启用HiCache等未审组合。

**统计口径仍需作者明确**：ROLE_BOUNDARY_STATS和105=新请求准入105，但≠尝试215或含续跑的commit334。单partial子项通过，注册表D1-05整体保留wip，工具按总commit严格口径fail；不能将其解释为第二partial或崩溃，也不能自动改分母使总门过关。请区分新准入与continuation计数。完整失败史、第一轮空跑池检查误判撤销及修正、50项CPU回归与live证据见experiments T29、`evidence/T29/`。
