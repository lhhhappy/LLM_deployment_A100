# Claude 编排报告

会话 `91cd0b57-498a-4428-b192-9ba6fe4f2404`。按 [coordination.md](../coordination.md) 汇报。

## 状态摘要（2026-09-24，READY：048 待主会话复核入队）

1. **122 已冻结：** `patches/122-tpot-paced-prefill.patch`，sha256 `cefdb2688cbc291742fc3c3ad188e343420fad01407d172f164ca6746d712d3b`。
2. **048 草案：** `scripts/pod/jobs/drafts/offA_122_n22.sh`。正式 A + 122 + `SGLANG_AX_PACE_TPOT=0.085`，dev N22，对照 047；启动参数不变。
3. **CPU 测试：** 122 共 12 个，120 原有 27 个，合计 39 个通过；它们调用真实调度方法，模型、池和 batch 是假的。
4. **定位：** 只作为可测候选，不写理论守门保证。TPOT 门只按实测判定。
5. **复核发现并修复：**
   - overlap 下两块 prefill 连续决策会重复使用同一份余量，现在把在飞 prefill 计入；
   - 新到请求的 fill ids 为空；
   - 首 token 未处理的 decoder 被漏计；
   - 强制放行只放 256 的碎块。
6. **保证失效条件（已写入 .md 和代码注释）：**
   - 成本模型低估；
   - 锚点晚于真实首 token（晚多少没有上界，也不由 τ 裕量覆盖）；
   - 连续 decode 满 `MAX_DECODE` 后强制放行一整块（日志 `guard=` 计数）。
7. **rank 一致性：** collective 进入条件与决策输入在各 rank 相同，时钟取 max。8 卡开销未测。
8. **MTP 计数：** 按 `output_ids`，结果处理时 extend。计数滞后使余量偏小。
9. **占用：** 我没有 pod 或开发机任务，也没有本地后台进程。待同步项只有 122 补丁本身。
10. **需要主会话决定：** 是否在 047 之后入队 048；048 的 N22 结果若与 047 差异接近重跑波动，是否补一次 A 重跑。

## 细节

### 锚点时序（真实 overlap 循环，`scheduler.py` `event_loop_overlap`）
第 k+1 步的 `get_next_batch_to_run` 在第 k 步的 batch 启动之后、其结果处理之前调用。所以第 k 步 prefill 完成的请求，在第 k+1 步决策时已进入运行批，但输出尚未写回。122 在这一时刻建立锚点，时间取 max(同步时钟, 在飞受控 prefill 的预测结束)：

| 情形 | 锚点相对真实首 token |
|---|---|
| 产生它的 prefill 受控 | ≈ 预测结束；偏晚量 = 成本模型高估量，无上界 |
| 产生它的 prefill 不受控（当时无 decoder） | 早于首 token，保守 |
| 非 overlap | 晚了结果处理与调度的耗时，无上界保证 |

### MTP 产出计数
A 实际是 EAGLE/NEXTN 加 overlap（044r 的 server_args：`disable_overlap_schedule=False`）。122 按 `len(req.output_ids)` 计数；spec 在结果处理时 extend 接受的 token（`batch_result_processor.py:956`）。决策看到的计数最多滞后一步，只会让余量偏小。

### 各 rank collective 分支
`_ax_pace_now` 的 all_reduce(MAX) 只在以下条件同时成立时进入：
- 运行批非空且不是仅 prefill；
- 有未完成的请求；
- 有续算或等待中的请求。

这些量在各 rank 相同：
- 等待队列由 rank 0 广播；
- 运行批与 `finished()` 由相同的结果处理更新；
- 基线的 `SGLANG_REQ_WAITING_TIMEOUT` 默认 −1，不触发本地时间分支。

决策只依赖同步后的时钟与确定性的 token 计数；`busy_until` 由同步时钟和 `extend_num_tokens` 算出。日志只在 rank 0 打，不影响决策。

### 测试
- 12 个 122 用例覆盖：
  - 关闭时与正式 A 逐步相同；
  - 预算算术；
  - 落后时的 decode 轮数；
  - guard 放行整块；
  - 等待请求 fill ids 为空；
  - 首 token 未处理；
  - overlap 在飞 prefill 计入；
  - rank 时钟；
  - 两个假时钟仿真（机制自检，不是性能预测）。
- 另有 27 个 120 原有用例。

复现：
```
A="000-interface-compliance 101-role-boundary-split 106-defer-chunk-on-no-kv 110-sm80-dsa-indexer 111-sm80-fp8-moe-marlin 114-indexer-row-shard 120-sched-protect-chain 121-sched-cap-while-decoding"
B="130-async-tokenize 140-kda-dual-snapshot 150-startup-warmup 160-nextn-sm80 170-glm-bcg-prefill"
python3 scripts/patch_stack.py apply build/p122/baseA $A $B
python3 scripts/patch_stack.py apply build/p122/candidate $A 122-tpot-paced-prefill $B
cd tests && python3 -m unittest test_tpot_paced_prefill test_sched_protect_chain   # Ran 39, OK
```
122 也能在 S1 栈（无 121）上干净应用。

### 048 判据
- 全部 11 门与 047 逐门对照；
- `[ax-pace] on:` 的生效值；30 s 统计行中的 `mean_budget` 与 `guard`；
- prefill 块长分布、KV 使用率、在飞与排队数。

推翻条件：首 token 各门都没有改善，或 tpot_p95 > 0.10，或块并未变大。

### 相关文档
- 补丁说明：[122 .md](../../patches/122-tpot-paced-prefill.md)
- 分析：[claude-进展](../claude-进展-2026-09-24.md)
- `notes/knowledge.md` 的"闭环负载"一节已收紧，本地未提交，按规则不提交共享文件，由主会话决定。
