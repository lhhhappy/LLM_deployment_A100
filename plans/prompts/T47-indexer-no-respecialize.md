你是 Codex worker W21，任务 T47：修复 112/113 sm80 indexer kernel 的"按形状重编译"问题（阻塞上线，优先级最高）。
仓库 /workspace/Agentic_science_challenge。先读：AGENTS.md、rule.md §2/§4、patches/112-*.md、patches/113-*.md、scripts/kernels/sm80_indexer_112.py、scripts/kernels/sm80_indexer_113.py、scripts/make_112.py、scripts/make_113.py、scripts/test_sm80_indexer_11{2,3}.py、verify/summarize 脚本、evidence/T46/README.md 与 kernel_keys.md（W20 实测：113 NQ 600→601 新增 2 次编译）。
问题：_paged/_ragged/_prefill/_unpack_prefill 把 NQ、NK、R、P、S 以及各 stride（QB/QN/QH/QD/WB/WH/CB/CN/TB/TP/KB/QQ/KK/KD/SS/WQ/KSS/KES/XR/XH/XD 等）声明为 tl.constexpr，服务中每个新的 prompt 长度/上下文页数都会触发 Triton 重新编译（秒级），直接毁掉 TTFT。
要求：
1. 形状与 stride 全部改为运行时 int 参数（可保留 Triton 默认的 16 整除特化；如某 stride 恒为 1 可显式 constexpr 并断言），只把真正固定的量保留为 constexpr：H、D、PAGE、BQ、BK、HH、DD、CLEAN、GROUP、LOOP、BLOCK 等 tile/模型常量。循环上界用运行时值（tl.cdiv 等），mask 保持正确。
2. 以最小改动更新 112 与 113 的 kernel 源与生成器，重新生成 patches/112-sm80-indexer-kernels.patch 与 patches/113-sm80-prefill-indexer.patch（保持文件名，旧版保留到 patches/drafts/ 以便回溯），并在两个 .md 里追加"v2：去除形状特化"变更记录。
3. 验证（GPU 开发机 /sjtu/linhang/arena/，source env.sh，nvidia-smi 看占用；不碰 bohr/Trisol/pod）：
   - 重跑 112/113 全部数值用例与 graph 用例（与 110 oracle 对照，阈值同原任务），全过；
   - 性能：重跑原性能表（decode B6 32k/190k graph；prefill 8192×{32k,95k,190k} causal/ragged），与 v1 对比，退步不得超过 5%（超过需说明并尝试 tile 调整）；
   - 新增"无重编译"测试：预热后依次跑 50 个随机不同的 NQ（1..16384）、NK（1..200000）、P、batch，统计 Triton 实际编译次数（JIT cache miss）应为 0（或只与 CLEAN/GROUP 等枚举常量相关的常数次），并给出计数日志；
   - 全栈 000→101→105→110→111→112→113→140→120→130→150 fuzz=0 可打、py_compile 通过。
落盘：notes/dispatch.md T47 行 accepted→in-progress→done/blocked（阻塞立即写）；证据 evidence/T47/；最后追加「T47 W21 → Claude：交付」节（数值、性能对比表、编译计数、开放问题）。
