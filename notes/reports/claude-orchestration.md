# Claude 编排报告

会话 `91cd0b57-498a-4428-b192-9ba6fe4f2404`。按 [coordination.md](../coordination.md) 汇报；主会话主动读取。

## 2026-09-24 交接确认

**已确认接手新分工：** 执行层主 Codex 是唯一的 8 卡队列管理者，负责复核与提交准备；Claude 负责编排、准入、块预算、缓存策略，提出单变量候选并提供原始证据。Claude 不再入队、不操作 pod、不做提交。

### 当前占用（04:20 UTC 前后查询）

| 资源 | 状态 |
|---|---|
| 8 卡 pod 队列 | running/pending 均空；046r 已进 done，按指示未干预 |
| 我入过的 8 卡 job | 044r、045r、046r，均已结束；此后没有新入队 |
| GPU 开发机（2×A100） | 两卡 0% / 4 MiB，没有我的任务。tmux 会话 arena、arena-daemons、batrl2、rescore3、rl480、seeds480 都是空闲 shell 或已结束的旧任务，没看到守护进程或会删服务的路径 |
| 本地后台进程 | 无 |

### 待同步项（均未同步到 pod 或开发机）

| 项 | 位置 | 说明 |
|---|---|---|
| 补丁 122 v2 | `patches/122-tpot-paced-prefill.patch`，sha256 `644cef8c4bc11942b7550e67e0476d1036562403b84f8c618de037d4e64f98ce` | pod 上的 122 仍是旧版 adaptive-decode-rounds（037c/d 使用）。本地旧版已删除，git 保留历史。新版文件名不同，不会覆盖旧文件 |
| 草案 job | `scripts/pod/jobs/drafts/offA_122_n14.sh` | 未入队。与 `cal_offA_n14.sh` 只差 3 行：G_NAME、插入 122、`SGLANG_AX_PACE_TPOT=0.085` |
| verify_kit | 无改动 | — |
| 本地未提交的共享文件改动 | `notes/knowledge.md`（"闭环负载"一节，已按复核边界收紧） | 我只改了这一节，未提交，请主会话决定是否并入 |
| `notes/queue.md` 的 122 行 | 仍描述旧 122（target=.13、11/6 轮） | 该文件现归主会话，我没改；新 122 与它无关 |

## 122 复核结果（本轮任务）

### 真实调度器边界
| 边界 | 结论 |
|---|---|
| TP rank 一致性 | 决策输入（运行批、等待队列、续算请求、配置）在各 rank 相同；时钟经 `tp_cpu_group` all_reduce(MAX) 后各 rank 用同一值；collective 只在各 rank 相同的条件下进入。未在真实 8 卡上验证 collective 开销 |
| MTP 计时与产出计数 | 按 `len(req.output_ids)` 计数，spec 结果处理时 extend 接受的 token（`batch_result_processor.py:956`）。计数滞后只会让余量偏小，是保守方向 |
| overlap：首 token 未处理 | **发现并修复。** 原版跳过无输出的新 decoder，可能在它首 token 后插进一整块而不计账。现在按"刚出首 token"计入 |
| 冷启动：新到请求 | **发现并修复。** 真实 Req 的 `full_untruncated_fill_ids` 在准入前为空（`schedule_batch.py:974`），原版把新到请求的待做量算成 0/1，预留量也算不到。改用 `seqlen`。原测试的假 Req 把该字段预先填满，掩盖了此问题；已加专门用例 |
| 时间预算不足 | **修改。** 连续 decode 满 `MAX_DECODE`（32）后放行一整块，原版只放一个 256 碎块 |
| 空批 / 仅 prefill 批 / 全部已完成 / 无待做 prefill | 走原路径，不进 collective |
| 短输出 | 欠账上限 (0.10−τ)·(n−1)；n=1 的请求出首 token 即完成，不参与 |

### 测试：真实调用 vs 模拟
- **真实代码：** 测试从 `build/p122/candidate`（正式 A 13 补丁 + 122）抽出生产方法执行，包括 `get_next_batch_to_run`、`_get_new_batch_prefill_raw`、`PrefillAdder` 与 122 的全部新方法。模型前向、KV/状态池和 ScheduleBatch 是假的，也不涉及 torch.distributed（rank 用例用假 collective）。
- **结果：** 122 共 11 个用例，120 原有 27 个用例，共 38 个通过。
- **仿真：** 在上述真实调度代码外加假时钟与假成本模型，只证明机制按设计工作。常数为推断值：C0=0.08 s、C1=60 µs/token、decode 一步 0.036 s、MTP 3.3 token/步。**不是性能预测。**

复现：
```
A="000-interface-compliance 101-role-boundary-split 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 114-indexer-row-shard 120-sched-protect-chain 121-sched-cap-while-decoding"
B="130-async-tokenize 140-kda-dual-snapshot 150-startup-warmup 160-nextn-sm80 170-glm-bcg-prefill"
python3 scripts/patch_stack.py apply build/p122/baseA $A $B
python3 scripts/patch_stack.py apply build/p122/candidate $A 122-tpot-paced-prefill $B
cd tests && python3 -m unittest test_tpot_paced_prefill test_sched_protect_chain   # Ran 38, OK
```
`test_srpt_admission` 需另建 p123 树，本轮未跑。

### 文档收紧
- `notes/claude-进展-2026-09-24.md`：
  - 正式负载满载改为待验证假设，注明逻辑 TPM 含缓存、两点配置不同；
  - 删去"榜首把等待转入 decode"的推断，以及 N 与 tpot 的粗算表；
  - KV 一节改为"需实测"，注明平均 prompt × 在飞数不是物理驻留。
- `patches/122-tpot-paced-prefill.md`：
  - 动机改为实测加假设，注明 0.446/0.739 s 是日志间隔（含 2 轮 decode，8192 仅 43 个样本）；
  - 列出测试范围与仿真边界、已知限制（包括 collective 开销未测）。
- `notes/knowledge.md`：同步收紧（未提交，见上表）。

## 草案 job：正式 A + 122（未入队）
- 文件 `scripts/pod/jobs/drafts/offA_122_n14.sh`。
- 相对 044r 唯一变量：补丁 122 与 `SGLANG_AX_PACE_TPOT=0.085`。启动参数不变；`--prefill-decode-interval 2` 仍在命令中，122 开启时被忽略。
- 判据：全部 11 门与 044r 逐门对照。另看 `[ax-pace] on:` 的生效值、30 s 统计行中 `mean_budget` 是否高于 4096、prefill 块长分布、KV 使用率。
- 推翻条件：首 token 各门都无改善，或 tpot_p95 > 0.10，或块并未变大。
- 建议：若差异接近 036/042 的重跑波动，重跑 A 量噪声后再下结论。

## 阻塞
无。需要主会话：审核 122、同步补丁、决定是否入队。
