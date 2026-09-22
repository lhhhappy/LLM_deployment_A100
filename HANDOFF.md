# HANDOFF：交接给下一个 session（2026-09-23 早，Claude）

先读：`CLAUDE.md` → `AGENTS.md` → **本文件** → `research/README.md` → `research/claude/base/00-summary-mainline.md`。
版本库：工作区已 `git init`（大只读目录排除），每个里程碑一个提交；`git log --oneline` 看历史。

## 0. 一句话现状
底包已能在真实 8×A100 上启动并通过功能探测（F59）。一批性能/机制补丁已在 CPU/开发机验证完毕、等 8 卡验证。
**8 卡服务 `lh-arena-sess-a` 已被旧守护进程误删（见 notes/decisions.md「8 卡服务被守护进程误删后的处置」，idle_hold 配置未生效）**；已重建 `lh-arena-sess-b`
（service id `2102486579267252224`），在 Trisol 准入队列里。GPU 机 tmux `arena-daemons:boot` 运行 `scripts/pod/autostart.sh`：
准入后自动 `scripts/pod/bootstrap`（重建 pod 的 worker/补丁/开发集/工具）并按序排队任务。日志 `/sjtu/linhang/arena/runs/autostart.log`。
**用户指示：先把测评跑通并记录，暂不正式提交。**

## 1. 必守规则（用户明确要求）
- 8 卡服务不停、不删、不释放，停必须用户同意。旧守护进程 `trisol_test_daemon.py` 已停且**不再启用**（它按启动时配置 3h 空闲释放）。
- 调研优先、大胆读源码改源码；调参放最后。每次压缩/新会话先重读 `research/`。
- 每处改动可回溯：补丁编号文件 + 说明 + 生成器 + 证据；git 提交；L3 提交记录 commit + 补丁清单 + 镜像 tag。
- 跑通修复是前提，**优化修复更重要**（用户 09-23）。
- 省 token：后台任务用 Monitor/后台命令唤醒，不轮询；不做"每 25 分钟自我唤醒"（缓存 TTL 1h，事件会唤醒）。
- 路线保密；中文沟通；正式提交每天 2 次（目前暂停）。

## 2. 补丁栈（全部 `patch -p3 --fuzz=0` 按序可打：000→101→105→110→111→112→113→140→120→130→150）
| 补丁 | 作用 | 验证 |
|---|---|---|
| 000 | 接口合规 | 已有 |
| 101 / 105 | 角色边界拆分 / 修双 partial 崩溃（F62） | 105：CPU；8 卡待测 |
| 110 / 111 | sm80 DSA indexer shim / FP8 MoE→Marlin W8A16（F58） | 8 卡启动+探测通过（F59） |
| 112 / 113（v2） | sm80 indexer 融合 kernel：decode ~4.5–5×；prefill 再 ~6×；v2 去形状特化（200 随机形状 0 编译，T47） | 开发机 A100 PASS；8 卡待测 |
| 120 | M1 调度保护链中间请求（W15/T41），`SGLANG_AX_SCHED_PROTECT`/`_COLD_CAP` | CPU 27 测；8 卡 A/B 待测 |
| 130 | 分词线程池+路由键（W16/T42），`SGLANG_AX_ASYNC_TOKENIZE` | 722/722 token 一致 |
| 140 | M2 KDA 双点 fp32 快照（W19/T45），`SGLANG_AX_KDA_DUAL_SNAPSHOT=1` | 数值逐位一致；离线 extend 轮数 -20% |
| 150 | 启动期预热（W20/T46），`--warmups ax_shapes` | CPU+算子；8 卡待测 |

启动参数固定：`--dsa-prefill-backend tilelang --dsa-decode-backend tilelang`（fa3 仅 Hopper，F57），env `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`、`SGLANG_OPT_USE_TOPK_V2=0`、`SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829`。

## 3. 8 卡待跑队列
autostart 已排：b113 探测 → b113 N6 → 120on N6 → 120cap4096 N6 → b113 N10 → 120on N10 → 120cap4096 N10 → b113 N14。
- 之后手动追加：`dev_b140_n6/n10`（140 on）、150 启动探测（加 `--warmups ax_shapes`，核对 ready 后无 "compile after serving started"）、130 A/B。
- 每个 dev 结果：读 `summary.json`（`build/podtools/summ.sh <job>`），记 TTFT p95 四门、tpm、n_at_slo → `notes/experiments.md`；
  跑完用 `build/podtools/dstat.sh` 看 decode ms/step，必要时 `/start_profile`（`build/podtools/prof.sh` + `profsum.py`）找下一个瓶颈。
- 已知：110 torch 版时 decode 每请求每步 +10ms、indexer 占 ~60% GPU（F63，evidence/T43）；112/113 应大幅改善——**第一件事就是验证这一点**。

## 4. pod 工作方式（`scripts/pod/`，在 GPU 机上运行）
`pexec`/`ppush`（保留相对路径！dest 是目录）/`pstatus`/`podq init|submit|ls|log|pause|resume|cancel`/`bootstrap`/`autostart.sh`。
`podq init` 推送全部 `patches/NNN-*.patch`。任务模板 `scripts/pod/jobs/dev_template.sh`、`dev_b120_template.sh`（变体用不同源码名避免误复用引擎）。
小心：`pkill -f <模式>` 会杀掉包含该模式的 bexec 外壳，用 PID；前台 `sleep` 链式等待会被拦截，用 Monitor。

## 5. Codex 分工
worker W15–W21 均已结束并经 Claude 核验（见 board.md 实例表、notes/dispatch.md T41–T47）。主 Codex 会话 `01a0c731-…`（T37）未再活动。
下一批候选任务：prefill 融合 topk / 输出降精度（113 仍写全 fp32 logits）；MTP（M4）；A100 MoE Marlin 调优；按 8 卡 profile 定。
