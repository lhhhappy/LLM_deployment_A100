你是 Codex worker W15，任务 T41（M1：调度保护链中间请求）。仓库 /workspace/Agentic_science_challenge。
先读：AGENTS.md、rule.md §4（红线）、research/claude/base/00-summary-mainline.md（第一节 2）、research/claude/base/02-scheduler.md、llm-challenge-arena-v1/task.md（SLO 门：fast/overall intra TTFT、chain_start、tpot 等）。
基线代码 = build/base_exact/ + patches 000、101、110、111（按此顺序 `patch -p3 --fuzz=0` 打到 base_exact/sglang 的副本上；不要照 src/sglang）。

问题（源码已证）：有预填充就不跑 decode（scheduler.py:3476-3484）；正在分块的冷启动请求每轮吃满整个 chunk 预算并使准入循环退出（scheduler.py:3657-3680，schedule_policy.py:810）。一个 10 万 token 冷启动≈13 轮连续预填充，期间在跑请求停止出字、链中间的短请求（前缀缓存命中、新增 token 少）全部排队 → 打爆 intra TTFT 与 tpot_p95。

目标：写补丁 `patches/120-sched-protect-chain.patch`（+ 同名 .md：假设、机制、开关、风险、回滚），默认开、可用环境变量 SGLANG_AX_SCHED_PROTECT=0 完全回到原行为。建议机制（可改进，须说明理由）：
1. 冷启动分块续算期间穿插 decode（例如每轮分块后至少跑一次 decode，或 mixed/交替），保证在跑请求的 tpot；
2. 缓存命中的短请求（新增 token 小于阈值）优先于冷启动分块续算入批，冷启动分块在剩余预算内继续；
3. 冷启动请求每轮分块预算上限（可配置），避免独占；
4. 不破坏 101 的角色边界分块与 lpm 排序；不能引入饿死（冷启动须在有限轮次内完成，给出上界论证）。
红线：不改 meta_info 时间戳与 token 计数真实性，不截断、不压输出、/flush_cache 语义不变。
验证：CPU 单元测试（mock ScheduleBatch/请求队列，至少覆盖：冷启动+短命中请求交错、仅冷启动、仅 decode、开关关=原行为逐字节同决策序列、饿死上界），放 tests/ 或 build/p120/；在 base_exact 副本上 py_compile 全过。不要起 GPU、不要碰 Trisol/bohr、不要打镜像或提交。
落盘：notes/dispatch.md 里 T41 行状态 accepted→in-progress→done/blocked（落盘即通知）；证据放 evidence/T41/；最后在 dispatch.md 追加一节「T41 W15 → Claude：交付」写清结论（VERIFIED/INFERRED）、补丁生成方式（可复现脚本 scripts/make_120.py 优先）、测试数、开放问题、建议的 8 卡验证方案（用 scripts/pod/jobs/dev_template.sh 的方式）。
