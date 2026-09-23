# T54b — 诊断脚本统一口径、数值比较器与探针断言、120 版本说明（Claude 派发，subagent Opus）

先读 `plans/prompts/_context-0924.md` 和 `research/codex/R19_progress_and_cache_review.md` §2 的 S2/S4/S5/S6/S8。

## 要读的代码
`scripts/pod/verify/{prefill_waste.py,cachecmp.py,ttft_decomp.py,logstat.py,numcheck.py,numcheck_cmp.py,lost_cache.py}`、`scripts/pod/jobs/dcp116_probe.sh`、`scripts/make_120.py`（Codex 的，只读）、`patches/120-sched-protect-chain.md`、`patches/drafts/120-*`、`s1-dev/harness/s1_common.py`。

## 交付
1. 新 `scripts/pod/verify/replay_roles.py`，公共口径：
   - 读 raw（和 cohort）；
   - 逐请求给出门归属（import harness 的 `in_ttft_gate`）、回放角色（按 `idx_in_chain` 分 head/followup）、`predecessor_sent`（本 raw 里同链 idx−1 是否存在）；
   - 原始 edge/phase 标签作为独立字段保留。

   所有诊断脚本都 import 它。
2. 在它之上重写 prefill_waste.py、cachecmp.py、ttft_decomp.py：
   - 按（门、回放角色、predecessor_sent、原始标签）分组报告；
   - 冻结差值明确标注"proxy（冻结值）"，净差和正差分开；
   - 不再按 phase 写死阈值。

   Golden：`evidence/T53/026_N18_raw.jsonl` 必须复现 F93，即链首 311 条实际 8,100,837、后续 411 条正差 913,499、append-only/head 53 条实际 4,605,181、fast_intra n=328 超时 10。
3. `logstat.py`：
   - 支持 `--since/--until`（epoch）。server.log 是 pod 本地时间，从日志与 raw 的对应事件推算时区偏移，或接受 `--tz-offset`，写清楚。
   - 修掉"new≥8192 就当 8192 块"的假设，按每个 batch 的实际 `#new-token` 计。
   - 给了窗口就绝不混入 warmup 或前一档。
4. `numcheck_cmp.py`：
   - 长度不一致判 WRONG；
   - 任意位置 token 分叉判 WRONG，除非有标定过的容差文件（eager 对 eager 重启得到）明确允许该用例和位置；
   - wrong>0 时退出码 1，输入畸形时退出码 2；逐用例打印原因。
   - T53 反例（参考 [1,2,3]，候选 [1] 和 [1,9,8]）必须判 WRONG 且 rc≠0。
5. `numcheck.py`：
   - 严格 flush：HTTP 2xx 且 JSON `success:true`，否则非零退出；
   - 工作线程异常要传播，非零退出；
   - 逐用例记录 cached_tokens，声明为"命中"的用例 cached==0 就判失败。
6. 新 `numcheck_text.py`：
   - 固定 20 条真实 dev prompt，用 harness 渲染器渲染；tokenizer 本地用 `s1-dev/glm_tok`，pod 上用 `/mnt/models`；
   - 截到固定 token 长度，要覆盖 16384 分块边界附近和一条约 100k 的长 prompt；
   - 走 `/generate` 贪心 64 token 并带 logprobs，写 `numcheck_text.json`，用 numcheck_cmp.py 比较。
   - eager 对 eager 的标定由 Claude 之后在 8 卡上跑；你提供 job 片段。
7. `dcp116_probe.sh`：
   - 结构做到在 `set -euo pipefail` 下安全；
   - 文件头每条验收标准都要真正断言，失败时打印 `PROBE_FAIL <标准>` 并以各自不同的非零码退出：引擎挂或日志有 illegal memory、numcmp wrong>0（用修好的比较器）、CAP_SMOKE <10/12、高负载命中请求 cached_tokens==0、分配器高水位没超过旧的每卡行数。
   - 现有日志看不出分配器高水位时，打印 `NOT_COVERED allocator_high_water`，不能算通过。
8. 120 版本说明：
   - 不改 `make_120.py`（Codex 的）。在 `patches/120-sched-protect-chain.md` 和 `patches/README.md` 写清：v2/v3 是手工修改，生成器只能复现 v1；`drafts/` 存 v1/v2；026 用的是 drafts v2（sha256 ef1744b3…）；当前文件是 v3（sha256 8b7366b7…）。
   - 若认为生成器必须改，把建议 diff 写到 `evidence/T54/make_120_proposal.diff` 交 Codex。
9. CPU 测试 `scripts/pod/verify/test_diagnostics.py`，覆盖 2、4、5、7（flush 用 mock HTTP）。日志存 `evidence/T54/`。

## 约束
- 同 T54a：不推 pod、不起引擎、不改 Codex 文件和 s1-dev。
- 不要改 T54a 负责的文件：run_level.py、dev_ladder_template.sh、lib.sh、analyze_run.py、mkjob.py、make_kit.sh。需要改时在报告里说明。
- 只更新 `notes/dispatch.md` 里 T54 那一行的状态。

## 最后回报（≤300 字）
改了哪些文件、测试计数、golden 数字是否复现、仍未覆盖的项。
