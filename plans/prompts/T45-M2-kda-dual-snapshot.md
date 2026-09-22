你是 Codex worker W19，任务 T45（主线 M2：KDA 双点 fp32 快照），产出补丁 140，替代 101/105 的"拆分预填充"做法。
仓库 /workspace/Agentic_science_challenge。先读：AGENTS.md、rule.md §2/§4、research/claude/base/00-summary-mainline.md（第一节 3、第二节 M2）、research/claude/base/03-hybrid-cache.md、04-model-kernels.md、patches/101-d1v12-on-base.md、patches/105-role-split-single-partial.md、notes/findings.md F62、refs/ 下 vLLM #56960（KDA 中途快照 exporter + 卷积历史 + 验证方法）。
基线 = build/base_exact + 000→101→105→110→111→112（fuzz=0，照 base_exact）。

问题：多轮 agent 链每轮在上一轮末尾追加少量 token，但真正可复用的前缀止于"角色边界"（<|user|>=154827 / <|observation|>=154829），末尾 reminder 等内容下一轮会变。底包每次 extend 只记一个 KDA 状态（分叉点优先于末尾，schedule_batch.py:2836-2853），中途状态取自 bf16 的 h（chunk_delta_h.py:349）；101 通过把预填充拆成两段拿到边界处的 fp32 状态，代价是多一轮调度、限制准入（并曾引发双 partial 崩溃，F62）。
目标：补丁 140（+ .md + 生成器 scripts/make_140.py），开关 SGLANG_AX_KDA_DUAL_SNAPSHOT=1 开启时：
1. 一次 extend 在 kernel 内（chunk_delta_h 或等价位置）对"最后一个角色边界所在 chunk 对齐点"输出 fp32 快照（及对应卷积历史 conv state），同时保留末尾 fp32 状态；边界对齐按 KDA chunk(64)/mamba checkpoint grid，给出对齐规则与不对齐时的处理（回退到 101 行为或只存末尾）；
2. 把边界状态作为额外节点写入 UnifiedRadixCache（状态槽分配、引用计数、淘汰顺序：reminder 之后的末尾状态优先淘汰），下一轮请求命中边界节点后从该状态续算；
3. 开关开启时 101 的 tail/admit 拆分不再触发（或 140 取代 101 在栈中的位置，二选一并论证），调度不再多一轮；
4. 开关关闭 = 栈中原行为逐字节一致。
验证：
- 数值（GPU 开发机 2×A100，/sjtu/linhang/arena/ 下，先 source env.sh、nvidia-smi 看占用）：随机权重、真实维度（64 头×128、chunk 64），对比"一次 extend 导出的边界状态"与"只 prefill 到边界得到的状态"逐元素误差（fp32 应≈bit 级或 <1e-5），以及从边界状态续算 N 个 token 的输出与全量重算一致；
- CPU：缓存树插入/命中/淘汰/引用计数/flush 真清的单测；与 101 行为对照的命中长度对比（用 s1-dev 数据集的真实链做离线回放估算命中 token，给出 on/off 数字）；
- 000→…→112→140→120→130 全部 fuzz=0 可打，py_compile 通过。
不碰 bohr/Trisol/pod，不起 8 卡，不打镜像不提交。
落盘：notes/dispatch.md T45 行 accepted→in-progress→done/blocked（阻塞立即写）；证据 evidence/T45/；最后追加「T45 W19 → Claude：交付」节（VERIFIED/INFERRED、数值表、离线命中估算、开放问题、8 卡 A/B 方案）。
