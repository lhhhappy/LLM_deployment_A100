你是 Codex worker W22，任务 T48（主线 M4：MTP / NEXTN 投机解码在 A100 上可用），目标是降低 TPOT（同档排名依据；第一名 tpot_mean 0.0273s）。
仓库 /workspace/Agentic_science_challenge。先读：AGENTS.md、rule.md §2/§4、HANDOFF.md、research/claude/base/00-summary-mainline.md（第一节 5：MTP 需 `--speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models`，开启后 max_running_requests 被设为 48）、research/claude/base/04-model-kernels.md、notes/findings.md F56–F68、patches/110–113、140 说明。
模型：GLM-5.3-Flash，num_nextn_predict_layers=1，index_share_for_mtp_iteration=true；45 层（KDA + DSA 混合），TP8，A100 sm80（无 fp8 MMA、无 DeepGEMM、fa3 仅 Hopper；DSA 用 tilelang；MoE 走 Marlin W8A16）。
基线代码 = base_exact + 000→101→105→110→111→112→113→140→120→130→150（fuzz=0）。
任务：
1. 静态走查 NEXTN 在底包中的完整路径（draft 模型构建、MTP 层的 DSA/indexer/MoE/KDA 调用、eagle/nextn worker、verify 与 KDA/mamba 状态回滚缓冲、CUDA graph、topk 共享 index_share_for_mtp_iteration、调度与 max_running_requests=48 的来源），逐一列出在 sm80 上会走到的不兼容点（DeepGEMM、fp8 Triton、fa3、flashmla 等），给出 文件:行号；
2. 对每个不兼容点给出修复（补丁 160，+ .md + scripts/make_160.py），尽量复用 110–113 已有的 sm80 实现；与 101/140（角色边界状态）与 120（调度）的交互写清楚：spec decode 下 KDA 状态缓冲/回滚是否与 140 的快照槽冲突，需要时在 NEXTN 开启时禁用冲突功能并说明；
3. 在 GPU 开发机（2×A100，/sjtu/linhang/arena/，source env.sh，nvidia-smi 看占用；**模型放不下，只能算子级/随机小权重**）尽可能验证：MTP 路径涉及的每个 kernel 在 sm80 上可编译运行、数值对照、CUDA graph 可捕获；
4. 准备 8 卡验证任务脚本 scripts/pod/jobs/dev_b160_mtp_n6.sh（参照 scripts/pod/jobs/dev_b113_n6.sh 格式、源码名唯一、显式 --max-running-requests 取值理由），以及一个"接受率/每步 token 数"采集方法（从 server.log 的 spec 统计行提取）；
5. 估算：若接受长度 ~1.6–2.0，TPOT 与吞吐的预期，和对 N@SLO（KDA 状态槽、max_running_requests）的影响，标 INFERRED。
全栈 fuzz=0 可打、py_compile 通过。不碰 bohr/Trisol/pod，不起 8 卡，不打镜像不提交。
落盘：notes/dispatch.md T48 行 accepted→in-progress→done/blocked（阻塞立即写）；证据 evidence/T48/；最后追加「T48 W22 → Claude：交付」节。
