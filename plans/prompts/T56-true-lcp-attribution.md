# T56 — 026 N18 按真实 token LCP 逐条归因（Claude 派发，Codex worker W25，astra/xhigh）

先读 `plans/prompts/_context-0924.md`、`research/codex/R18_cache_loss_and_capacity.md`（重新渲染 prompt、算真实 LCP 的做法）、`research/codex/R19_progress_and_cache_review.md`。

## 输入
- `evidence/T53/026_N18_raw.jsonl`（本地；SHA256 486f041027ab…）。
- pod 上 026 的完整日志 `/tmp/ax/runs/026-ladder_best140/server.log`，读取方式：
  - 用 `scripts/pod/pread grep|tail|head` 只读；
  - 或把你的分析脚本先放到 GPU 机 `/sjtu/linhang/arena/repo`，再用 `scripts/pod/ppush /tmp/ax/codex <文件>` 上传，并用 `scripts/pod/pexec_codex '<命令>'` 在 pod 上以纯 CPU 运行。本地调用写法：`scripts/gssh "cd /sjtu/linhang/arena/repo && scripts/pod/pexec_codex '<命令>'"`。
- dev 数据 `s1-dev/data/dev-combined-v1`、harness 渲染器、tokenizer `s1-dev/glm_tok`、cohort。

## 任务
1. **411 条后续请求**（本轮 `idx_in_chain>0`，且前驱在本 raw 中）。先重新渲染本轮实际发送的前驱 prompt 和当前 prompt，算真实 token LCP，再与 `cached_tokens` 比较，每条归入一类：
   - (a) 无损失：cached ≥ 真实 LCP − 64；
   - (b) 命中停在真实 LCP 之前的某个 KDA 状态点：要指出是哪一个点，前驱的 chunk 末尾、角色边界还是更早的位置；
   - (c) 命中远低于 LCP：前缀已被淘汰，命中≈0 或≈跨会话共享深度；
   - (d) 真实 LCP < 冻结 LCP：冻结代理值高估了，不算损失。

   分类报数量与 token 合计，并给出"真实可复用却丢失"的总量，即修复的上限。
2. **harness 分桶下 fast_intra 超过 3s 的 10 条**，逐条给出：
   - 排队与执行的拆分；
   - 按真实 LCP 算的损失；
   - 到达时正在进行的 prefill（server.log 该时间窗内的批次）；
   - 原因：缓存丢失、队头阻塞，或两者都有。
3. **短输出 TPOT**。对 `output_tokens<100` 且 tpot>0.10 的请求：
   - 列出其 decode 窗口内与之重叠的 prefill 批次（server.log 的时间戳、`#new-token`、`#cached-token`）和推算出的停顿时长；
   - 汇总每条请求被停顿几次、各种块大小的典型停顿时长。
4. **时间对齐**：server.log 用的是 pod 本地时间。用 raw 的 epoch 字段（t_recv_s、t_exec_start_s）与日志行推出偏移，并写清楚。

## 交付
- `research/codex/R20_true_lcp_attribution.md`，每条结论标 VERIFIED 或 INFERRED；
- `evidence/T56/`：可复现脚本 `attribute.py`（记录输入 SHA）和输出；
- 用 `python3 scripts/next_id.py F` 取号追加一条 finding，再跑 `python3 scripts/check_records.py`；
- 更新 `notes/dispatch.md` 里 T56 那一行的状态。

## 约束
- 只做分析：不改生产代码、补丁或工具；不操作 pod 队列、服务或引擎；不用 GPU。
- 在 pod 上只运行纯 CPU 分析，只写 `/tmp/ax/codex`。
