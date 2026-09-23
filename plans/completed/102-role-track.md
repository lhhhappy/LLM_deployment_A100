# 102 — Role-boundary KDA checkpoint on the base, without splitting the prefill

- 状态：superseded（2026-09-23 被补丁 140 取代：一次预填充导出角色边界+末尾 fp32 双快照）；原状态：active（设计中，未排 L2）　负责人：Claude　创建：2026-09-22
- 关联：dispatch T33/T34；F3、F45、F52、F53；决策 29；替代候选：101（镜像 B，45735）

## 目标
Save the KDA state at the last role boundary into the radix tree after prefill, so the chain's next
turn resumes there. Do this without 101's split round and its one-partial-per-round admission cap,
and at a precision no worse than 101.

## 范围
- 包含：`build/base_exact/sglang/srt/managers/schedule_batch.py:_mamba_radix_cache_v2_req_prepare_for_extend`
  （2766–2860）的跟踪位置选择；如走双点或 fp32 方案，再加 `hybrid_linear_attn_backend._track_mamba_state_extend`
  与 KDA prefill kernel 的快照输出。
- 不包含：调度策略、DP、MTP。

## Facts from the base source
1. **One tracked position per extend.**
   - Default: the extend end, aligned down to the grid.
   - An aligned branch point inside the extend wins.
   - A non-final position is read from the intermediate `h` (`_force_track_h`); no extra forward is
     needed.
2. **What reaches the tree after prefill.** `batch_result_processor.py:346` calls
   `maybe_cache_unfinished_req` when a prefill completes. It inserts `tokens[:mamba_last_track_seqlen]`
   with the tracked state.
3. **Precision.** `h` is allocated with `k`'s dtype (bf16; `chunk_delta_h.py:349`). The final state
   uses the pool dtype (fp32; `kda.py:85`). A mid-extend checkpoint is therefore bf16-rounded, while an
   extend end is exact. 101 gets an exact fp32 state because its first half ends at the boundary.
4. **Cost of a single point.** Moving the single point from the end to the role boundary drops the
   prompt-end state.
   - The prompt-end state is needed by strict append-only edges: 146 of 411 adjacent pairs in F13.
   - Stock and 101 keep it; single-point 102 does not. On a strict-append edge it therefore
     recomputes reminder + output (hundreds of tokens) instead of zero.

## 方案选项
| 方案 | 做法 | 优点 | 代价/风险 |
|---|---|---|---|
| 102-a 单点 | 跟踪位置改到角色边界（代码已生成：`scripts/make_102_role_track.py`、`patches/102-role-track.patch`） | 约 40 行；无拆分轮次 | 丢末尾状态（strict-append 退步）；bf16 状态（数值门风险） |
| 102-b 双点 | 同一次 extend 同时导出角色边界（中途）与末尾两个状态，都入树 | 语义等价 101 且无拆分 | 需第二个跟踪槽、入树两次、池占用翻倍；中途点仍 bf16 |
| 102-c 双点 + fp32 快照 | 在 KDA prefill kernel 里对指定 chunk 输出 fp32 状态（仿 v0.5.20 h_track_buf / vLLM #56960） | 精度同 101，无拆分 | 改 Triton kernel；工作量最大 |
| 保留 101 | 继续拆分 | 精确、已提交 | 多一轮调度 + 每轮一个 partial 的准入限制 |

## 前置条件与约束
- 等 45734/45735 成绩：B 显著优于 A 才说明"边界状态"在正式集成立，值得投入 102-b/c。
- 红线：不改输出、不截历史；数值必须与连续计算一致到事先定义的容差（D1-04 口径）。

## 风险与缓解
| 风险 | 缓解 | 回滚 |
|---|---|---|
| bf16 中途状态导致输出漂移、能力分下降 | 先测 bf16 vs fp32 状态续算的 logits 差；超阈值即走 102-c | 关环境变量开关 |
| strict-append 退步 | 选 102-b/c，或只在 prompt 以 reminder 结尾时启用 | 同上 |

## 里程碑
1. 在 L1 上装 base_exact（不再用 v0.5.20）并复现 101 行为。
2. 量化 bf16 中途状态的数值差（D1-04 口径）。
3. 按 1、2 与正式成绩选 102-a/b/c，实现 → L1 → L2 对比 `img_b`。

## 验证方式（机械可检查的优先）
- 命令：`scripts/replay_chains.py --case-file cases/reminder_heavy.json`（L1，base_exact 环境）；logits 对比脚本（D1-04）。
- 通过标准：下一轮 cached_tokens 到达角色边界；strict-append 边无退步；开关关闭时与原版逐请求一致；logits 差在容差内。

## 进度记录
- [x] 2026-09-22 读底包源码定位机制；生成 102-a 补丁（两条线都能打上并编译）
- [ ] L1 改装为 base_exact
- [ ] 数值差测量

## 决策记录
- 2026-09-22：审计指出原稿"末尾状态只对重发有用"错误（F13：146/411 strict-append 需要它），改为三方案比较，暂不排 L2。
