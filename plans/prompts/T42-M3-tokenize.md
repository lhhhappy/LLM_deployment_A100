你是 Codex worker W16，任务 T42（M3：分词移出事件循环 + 路由键）。仓库 /workspace/Agentic_science_challenge。
先读：AGENTS.md、rule.md §4（红线）、research/claude/base/00-summary-mainline.md（第一节 4）、research/claude/base/01-request-path.md、llm-challenge-arena-v1/task.md。
基线代码 = build/base_exact/ + patches 000、101、110、111（顺序打到 base_exact/sglang 副本，`patch -p3 --fuzz=0`）。

问题：tokenizer_manager.py:926-938 在单个 asyncio 事件循环里同步分词、无缓存；负载是多轮长上下文 agent 链（单请求可达 10–25 万 token，每轮只追加少量 token），长分词阻塞整个事件循环，所有请求的首包/流式输出都被推迟（影响 TTFT 与 tpot）。
目标：补丁 `patches/130-async-tokenize.patch`（+ .md），可用 SGLANG_AX_ASYNC_TOKENIZE=0 关回原行为：
1. 大输入分词放到线程池/进程池（HF fast tokenizer 在 Rust 中释放 GIL 与否需实测说明），不阻塞事件循环；
2. 可选：按会话前缀缓存分词结果（chat template 渲染后的前缀字符串→token ids，校验前缀逐字节相同才复用，边界 token 合并问题须处理或保守回退），给出正确性论证；
3. 把请求头 X-S1-Routing-Key / X-S1-Session-ID（若存在）接到请求的 routing_key 字段（供后续调度用），不改变其他行为。
红线：prompt_tokens 必须与原分词结果逐 token 一致（写对照测试）；meta_info 时间戳如实；不改 harness。
验证：CPU 测试——用 s1-dev/glm_tok 分词器对 s1-dev 数据集的真实对话，比较新旧路径 token ids 完全一致；测量大输入分词耗时与事件循环阻塞时间（前后对比）。不要起 GPU、不碰 Trisol/bohr、不打镜像、不提交。
落盘：notes/dispatch.md T42 行 accepted→in-progress→done/blocked；证据 evidence/T42/；最后追加「T42 W16 → Claude：交付」节（VERIFIED/INFERRED、测试数、开放问题、8 卡验证方案）。
