# rule.md §6 旧交接（Claude → Codex，2026-09-22 上午；已过时，基于"底包 = v0.5.20"假设）

**已完成**
- [Claude] R3 英文社区/厂商先例 — `research/claude/R3_prior_art_en.md`
- [Claude] D0/D1 设计稿（未实现）— `patches/000-*.md`, `patches/001-*.md`
- [Claude] 回应 Codex 审阅 + 非 oracle 复核 — `research/shared/directions.md` §F/§G，F13，`scripts/sim_role_boundary.py`
- 环境：playground 登录、赛题下载、开发集就位（本地 + GPU 机）、GPU 机连通与 env.sh。
- 竞品摸底：`data/all_att.json`（479 次提交成绩）；最好 N=18，N=18 时的绑定门是 fast_intra（2.2–3.0s）与 overall_intra（3.5–5.2s）；chain_start 45s 仍 PASS（样本少、统计余量大）。
- 模型结构（F1）：**混合架构**——34 层 KDA 线性注意力 + 11 层 MLA/DSA；MoE 288 专家 top-8；FP8 块权重 ≈328 GB；1 层 MTP。SGLang v0.5.20 原生支持 `glm5_next`。
- 负载（F2）：闭环回放（N 个线程各自串行跑链，gap 后才发下一条）；fast_intra 的归类用**冻结的** `uncached_expected ≤ 4096`，实际未命中多了直接砸 3s 门。
- **关键发现 F3**：57% 的链中间请求，前缀在上一轮末尾的 `<|user|><system-reminder>…` 起点处分叉（比上一轮末尾早 340–1400 token）。KDA 前缀命中需要该位置有状态快照；v0.5.20 只在 extend 末尾/分叉点存。模拟（`scripts/sim_checkpoints.py`）：fast_intra 实际重算 p95 ≈ 7.5k vs 门假设 3.1k；在最后一个 user 角色边界补存快照可回到 ≈3.3k。
- R2 调研（`research/claude/R2_serving_techniques.md`）：长 chunked prefill 独占 chunk 预算（上游 PR #40024 / #39717）；MLA KV 在 TP8 下每卡复制、N=18 可能容量吃紧 → HiCache（注意 issue #39830 正确性风险）、session radix cache；A100 上排除 PD 复用/DeepEP/单机 PD 分离。

**待办（当时的建议分工）**
- [Claude] R1 调研收尾 → `research/claude/R1_model_and_engines.md`。
- [Claude] 汇总方向清单 → `research/shared/directions.md`。
- [Codex] 独立审阅 F3 与 R2 结论 → `research/codex/`。
- [Codex] 调研组织方各底包版本与差异、`registry.dp.tech` 底包内容如何获取；vLLM 路线。
- [待定] 设计「角色边界快照」补丁方案 → `patches/001-role-boundary-mamba-ckpt.md`。

**当时的未决问题**
- 评测榜单网站显示的是能力分平均，与 task.md 的 N@SLO 排名口径不一致，需向主办方确认。
- 正式评测机的主机内存大小未知（影响 HiCache 容量）。
- 底包 `arena-sglang-glm53:260918` 是否就是 v0.5.20，未确认。→ 已解决：不是，底包 = fe236ea6c3 + mmfix1（F53/F54）。
