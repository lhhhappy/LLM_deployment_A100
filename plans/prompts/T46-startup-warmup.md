你是 Codex worker W20，任务 T46：补丁 150「启动期预热」，消除服务期 Triton 编译/autotune（F59：首次新形状 TTFT 58.9s/67.8s，日志 "Triton kernel ... took N s to compile after serving started"）。
仓库 /workspace/Agentic_science_challenge。先读：AGENTS.md、rule.md §4、notes/findings.md F59、build/base_exact/sglang/srt/entrypoints/warmup.py（@warmup 注册表，由 --warmups 在 http_server lifespan 中、服务 Up 之前执行）、http_server.py:383-420、kernels/ops/attention/fla/chunk_intra.py 与 kda.py 的 triton.autotune key、patches/112/113 的 kernel（scripts/kernels/sm80_indexer_11{2,3}.py）、patches/101/140 说明。
基线 = base_exact + 000→101→105→110→111→112→113→140→120→130（fuzz=0）。
目标：补丁 150（+ .md + scripts/make_150.py）注册 @warmup("ax_shapes")，启动命令加 `--warmups ax_shapes` 即启用：
1. 用 tokenizer_manager.generate_request 直接发 input_ids（不经分词），覆盖所有服务期会遇到的 kernel 变体：冷短（~600）、前缀命中续算（+1000、+7000）、>8192 的分块续算（冷 20000 与命中 +12000）、含角色边界 token 154827/154829 的请求（触发 101 拆分或 140 快照路径）、并发 decode（6–32 路、不同长度，覆盖 CUDA graph 外的 batch 形状）、MTP 如启用则跳过或单列；max_new_tokens 小、temperature 0；
2. 结束后调用与 /flush_cache 相同的真清流程（红线：flush 必须真清），并断言池已恢复；预热请求不得出现在后续 metrics 的 cached_tokens 里；
3. 通过枚举 autotune key 的方式论证覆盖完整性：列出每个 autotune/JIT kernel 的 key 维度与我们 warmup 覆盖到的取值；给出无法覆盖的项；
4. 可选：把 Triton cache 目录（TRITON_CACHE_DIR）与 FLA autotune 结果缓存指到持久路径，说明镜像内预置的可行性（镜像构建机无 GPU，缓存需在 A100 上生成后拷入）。
验证：CPU 单测（mock tokenizer_manager：请求序列、flush 调用、异常时不阻塞启动而是记录并继续/或失败的策略论证）；GPU 开发机只能跑算子级（2 卡放不下模型），可用 112/113/KDA 算子在开发机上验证"预热后同形状不再编译"（Triton cache 命中计数）；全栈 fuzz=0 可打、py_compile 通过。不碰 bohr/Trisol/pod，不起 8 卡，不打镜像不提交。
落盘：notes/dispatch.md T46 行 accepted→in-progress→done/blocked；证据 evidence/T46/；最后追加「T46 W20 → Claude：交付」节。
