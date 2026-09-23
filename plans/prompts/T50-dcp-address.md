你是 Codex worker W24（astra/xhigh），任务 T50：让 DCP（`--dcp-size`）在我们的 A100 补丁栈上**带前缀缓存命中**时正确运行。
仓库 /workspace/Agentic_science_challenge。先读：AGENTS.md、rule.md（§4 第 7 条：pod 只读 pread + pexec_codex，严格不许停）、HANDOFF.md、notes/findings.md 最新几条（DCP 探针与 025 崩溃）、research/codex/R18_cache_loss_and_capacity.md §8.3、patches/112/113/114/115 的 .md。
背景：8 卡上 `--dcp-size 8` + 114 + 115 冷启动探针正常（逻辑 KV ×7.8、长预填充 −19%、能力 12/12），但 dev N10 在一次"新 576 token、命中 77,312"的 extend 后 CUDA illegal memory access。日志：`scripts/pod/pread grep ... /tmp/ax/runs/025-ladder_dcp/server.log`。
任务：
1. 读底包 DCP 路径（dcp/comm.py、dsa_backend 在 dcp 下的 page_table/topk 索引、dsa_indexer_kpool 在 dcp 下的 K 读取与 kpool plan、tilelang 稀疏注意力的 indices），确定每卡 KV 分片的寻址协议，找出 110/112/113/114/115 中哪些路径按"全量 KV"寻址（文件:行号）。
2. 在开发机（2×A100，/sjtu/linhang/arena，gjob 跑长任务）用 TP2+DCP2 复现：可用 scripts/analysis/extend_check.py 的思路（one_batch + 缩小版模型 `scripts/analysis/make_rank_model.py`），构造"先预填充长前缀、再带命中续算"的两次 extend，先复现越界，再修复，并与非 DCP 的 logits 对比（误差门 1e-2）。开发机 NCCL 需 `--disable-custom-all-reduce`、`NCCL_CUMEM_ENABLE=0`、较低 mem-fraction（见 findings）。
3. 产出补丁 116（+ .md + 生成器），全栈 000→101→105→106→110→…→115→116→140→120→130→150→160 fuzz=0 可打；给出 8 卡验证方案（Claude 执行）。
不打镜像不提交，不碰服务生命周期；8 卡 pod 只能用 pread / pexec_codex。
落盘：notes/dispatch.md T50 行 accepted→in-progress→done/blocked；证据 evidence/T50/；最后追加「T50 W24 → Claude：交付」。
