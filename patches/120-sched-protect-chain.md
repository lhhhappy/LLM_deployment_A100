# 120 — 调度保护链中间请求（T41，W15）

状态：**CPU VERIFIED，待 Claude 交叉审阅与 GPU 验证；未加入 RELEASE**。
基线唯一为 `build/base_exact/sglang` 的副本，顺序叠加 **000→101→110→111→120**，均 `patch -p3 --fuzz=0`。
仅改 `srt/managers/scheduler.py` 和 `srt/managers/schedule_policy.py`。不改模型计算、请求内容、输出预算、thinking、tools、时间戳、token 计数或 flush 路径。

## 假设与目标

普通 TP8、PP=1、LPM、page=64、extra_buffer、chunked prefill=8192、HiCache/mixed/DP-attention/CP 关闭，是本次目标组合。默认非 NEXTN；NEXTN 保持其原 decode 入口，但实际组合尚未验证。

底包 `scheduler.py:3476` 的 prefill 优先与 `add_chunked_req` 的整轮预算占用已由源码确认。题面 fast_intra 为冻结未命中输入 ≤4096 的链中请求，目标 3s；overall_intra 5s，turn_start 15s，chain_start 30s，tpot_p95 ≤0.10s/token；正式 TTFT 门还采用置信区间口径。**运行时实际缓存命中并不是冻结 phase 标签**，120 不读取隐藏分类，也不把“命中短请求”等同于正式 fast_intra。

预期：允许短命中与冷启动同轮 prefill，且在存活请求之间插入 decode，减少连续 prefill 导致的停顿。**INFERRED**：这可能改善 intra TTFT 与 TPOT；CPU 轮数不代表毫秒或 SLO 通过。

## 机制与取舍

1. **有工作的 decode 交替**：在原 `get_next_batch_to_run` 完成 stash/last-extend merge 后，若上一轮为 prefill 且存在未完成 decoder，本轮使用原 `update_running_batch` 路径。每次 prefill 后至少有一次 decode 调度机会；无 decoder 时继续 prefill，不插入 idle。已有显式 `prefill_decode_interval>0` 优先，保留其原行为，不叠加一次额外 decode。
2. **先保留续算额度，短请求使用剩余批预算**：续算仍先入 `PrefillAdder`、占原请求行并按原 KV/Mamba 逻辑记账，但最多使用 C token；整个批的 8192 token 预算不被缩成 C。默认 C=2048，剩余最多 6144 token 按原 LPM 顺序接完整短命中请求。只有设备前缀非空、无需 host load-back、实际未命中 `0 < tokens <= 4096` 且按页计费能完整放下的请求可加入。
3. **有界跳过不合适的队首**：有旧/新 partial 时，长请求、冷 miss 或放不下的短请求返回 OTHER；执行原拒绝清理（含 Mamba COW/clear 元数据和非 session 新槽释放）后继续扫描余下队列。LPM 的排序方法与队列相对顺序不变。NO_TOKEN、请求槽位耗尽等硬门仍走原停止行为。
4. **首次冷/长准入也受 C 限制**：没有活动 partial 时，保留 LPM 首次选择；完整短命中可使用正常批预算，其他请求的 chunk 上限为 C。每批至多一个 partial，即使 101 关闭亦如此。
5. **101 原逻辑保留**：`_role_split_len`、角色 token 配置、branch 冲突、checkpoint 选择和预算函数 AST 未改；101 可以把最后一个 chunk 再切一次。续算存在时，完整短请求不再产生第二个角色 partial。

与建议的“先接完短请求，续算吃剩余”相比，使用**先保留有限续算额度**：不移动底包必入续算及其槽位/KV 生命周期，短请求持续到来时也无法夺走续算进度。物理 `can_run_list` 仍把续算放前面；同批共享 GPU forward，不把数组位置描述成执行优先级。这里的短请求优先体现在：续算不能再占满全批，剩余额度留给短命中，其他长请求不会挡住它们。

## 开关与支持范围

环境变量在进程内首次读取后缓存；**修改后须重启引擎**。

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_AX_SCHED_PROTECT` | `1` | 精确值 `0` 关闭所有 120 决策；关闭时不解析其余两个变量 |
| `SGLANG_AX_SCHED_COLD_CAP` | `2048` | 正整数；冷/长新准入和续算单轮上限 |
| `SGLANG_AX_SCHED_SHORT_TOKENS` | `4096` | 正整数；完整设备命中短请求的实际新增 token 阈值（包含等号） |

`G = lcm(page_size, truncation_align_size or 1, mamba_checkpoint_grid(tree_page))`（非 Mamba 略去最后项）。配置 cap 向下对齐到 G；小于 G 时向上钳到 G，以保证进度；再受 chunk 与 max_prefill 总预算约束。默认 8192 批预算下 cap=2048/4096 都能为 4096 短请求留空间。若总预算小于一个 G，预算保护回退原路径；decode 保护仍可工作。若原生可用 KV 只够不足 G 的续算，保持原资源限制，不虚造内存；此时不适用下面的 G/C 进度假设。

特殊模式不启用 120：无 chunked prefill、禁用前缀缓存、PP>1、需要 DP MLP 同步、prefill CP/DSA CP、PD 分离、DLLM、mixed chunk、hybrid SWA、HiSparse、HiCache/FlexKV、LoRA、priority preemption、prefill delayer。用相同 predicate 同时保护 cadence 与预算；这些组合保留原模式合约，不宣称支持。

## 不饿死的上界与适用边界

设冷请求**已准入**，尚余 L token；无 abort/retract，每轮有足够资源容纳一个对齐单位，原显式 prefill-decode 间隔为 d（默认 0）。本补丁的有效间隔 `d_eff=max(1,d)`，无存活 decoder 时为 0。

- 续算始终在短请求准入前保留额度。只要该轮能运行 prefill，就不能被等待队列跳过，因此无论后续短请求来多少，都不会使它的可运行轮次变成零。
- 若每轮可用额度至少 C，除最后一轮外至少推进 C；101 只在完整尾块上选最后角色边界，可额外增加至多一轮：`P <= ceil(L/C)+1`（101 不触发则无 `+1`）。从准入开始，含 decode 的调度轮数 `R <= (d_eff+1)*P`。
- 更保守地，每轮至少推进 G，101 的切分也至少 G，则 `P <= ceil(L/G)`；最终不足 G 的尾部完整执行。前缀对齐与资源条件必须成立。
- CPU 对抗测试不断注入短命中：L=100000、C=2048，无角色边界为 49 prefill/97 总轮；有尾部角色边界为 50/99。仅冷启动为连续 49 prefill，没有空转。

**不把该上界扩大成无限过载下的全队列公平承诺**：原 LPM 就可能让尚未准入的零命中请求在无限热流中一直排队；120 未重排 LPM，因此不修复这个既有问题。有限等待集合、有限输出且资源足够时，短请求完成并释放槽位，后续冷请求可依原策略准入并获得上述进度。内存压力/retraction、delayer 或无限到达条件下不能证明绝对请求完成时间。若需要“任意无限输入下每个尚未准入请求也有轮数上界”，必须另加 aging/FIFO 保底，会改变严格 LPM 的优先关系，需单独设计。

## 风险、既有缺陷与回滚

- C=2048 把 100k 冷输入从约 13 批变成约 49 批，增加 launch/调度开销，可能恶化 chain_start 与吞吐；decode 与较长 prefill 同轮间隔仍可能超过 0.10s。没有 GPU 时延保证，也不保证 SLO 通过。
- 本补丁还限制长命中请求的 chunk；仅冷启动也保留 cap，保证开关行为清晰。若 chain_start 退步，先在相同配置对比 cap=4096，不能通过改 token 计数或输出长度补救。
- 改变分块与 batch 形状可能改变浮点累加/greedy 输出。CPU 不验证 logits、KDA 快照数值、实际 overlap buffer 生命周期、Mamba 回收或 NEXTN；需真实 GPU 检查。
- 跳过长 waiter 增加 match/COW 的 CPU 开销。保留原释放逻辑并测了普通槽与 session；真实静态 Mamba 池压力仍待测。
- **既有 101 双 partial 反例 VERIFIED**：已有 chunk 在最终 1536 token 的第 1024 token 处角色切分，剩余预算可让另一个长请求成为新 partial；原 101 只检查 `adder.new_chunked_req`，随后 scheduler 的 `assert self.chunked_req is None` 失败。120 开启时提前拒绝第二个 partial；关闭时保持原断言。此为回退保真，不是把该反例当成功运行。
- 101 自己的额外 deterministic alignment 与 role grid 兼容性未在 120 改写，不能据此关闭此前记录的数值/对齐风险。

回滚：引擎启动前设 `SGLANG_AX_SCHED_PROTECT=0` 并重启，或在 000→101→110→111 副本上执行 `patch -R -p3 --fuzz=0 < 120-sched-protect-chain.patch`。无需改 flush/cache 语义。不要把开关写到 Trisol 对外 env；由镜像内部中性 profile 设置。

## 复现与证据

在仓库根目录，纯 CPU：

```bash
python3 scripts/make_120.py
python3 -B -m unittest discover -s tests -p test_sched_protect_chain.py -v
python3 scripts/verify_120.py
python3 scripts/check_records.py
```

生成器只重建自有 `build/p120/baseline` 和 `candidate`；验证器只重建 `verified`。三个副本均不修改 `build/base_exact`。补丁头为 `a/python/sglang/...`，可在副本 `sglang` 根下 `patch -p3 --fuzz=0`。

- **27/27 CPU 测试**：使用实际生产 scheduler/PrefillAdder/LPM/101 方法 AST，mock ScheduleBatch、缓存池和模型 forward；不完整 import SGLang/CUDA。不等价于 GPU 功能通过。
- 24 组成对 off 序列序列化字节一致：22 组各正常 40 轮；2 组在同一原生断言失败（方法、语句、时点一致），另有定向 regression。不是 24 组都跑完 40 轮。
- 16 组随机 on 序列各 30 轮，至多一个 partial、chunk/input 预算非负；另测冷/短交错、纯冷、纯 decode、D1 admission/tail/branch、对齐与显式间隔、槽位与 KV 拒绝、Mamba 清理与 session 保留、default on/off。
- `evidence/T41/cpu_tests.log`、`off_parity.json`、`interleave.json`、`starvation_bound.json`。
- `validation.json`：实际 -p3/fuzz=0 贴补丁，4684 文件与生成候选一致；仅上述两文件改变；3617 个 `.py` 加 3 个工具/测试文件 py_compile 通过，验证期间只读底包哈希不变。
- 初次失败日志是 CPU 夹具搭建错误（缺依赖 stub、把未执行 chunk 误置成已执行），保留在 `cpu_tests_initial.log` / `cpu_tests_fixture_debug.log`；已修正夹具。基线断言单列如上。

## 建议 8 卡验证（仅交付方案，未入队/未运行）

已提供 `scripts/pod/jobs/dev_b120_template.sh`，照当前 `dev_template.sh` 使用 `prepare_src` + `ensure_engine` + 原 `run_dev.py`；只由 Claude 在获得相应授权和完成交叉审阅后安排。

1. 前置：110/111 的 sm80 启动/正确性门通过。普通 TP8/LPM/extra_buffer、无 NEXTN 先做 off/on 小用例：1 个 100k 冷请求 + 已在 decode 请求 + 持续缓存命中短请求。确认无错误、输出 token 数原样、单 partial、缓存计数真实，完成后 flush 恢复池并重打同 prompt 命中为零。单独记录 chunk/decode trace；时延评测不用重日志。
2. 在同一模型与原始开发集按 **off → on → off** 对照 N=6/10，成功后 14/18/22；保留全部 raw/run/summary/server.log 与补丁 SHA，使用仓库 scorer，报告所有正式门及样本数，不把 dev 临界 N 外推正式 N。
3. 主比较 `AX_P120_VARIANT=off`（中性源码名 b120a）与 `on`（b120b）；若 chain_start 或吞吐明显退步，再做 `cap4096`（b120c）。三者使用不同源码名，避免现有 `ensure_engine` 签名**不含 120 环境变量**导致错误复用；模板固定其他 knob。切换配置必须真正重启。
4. 测四个 TTFT 门与 tpot_p95/mean；冷启动完成轮数同时对账。长 prompt/热 prefix/分叉角色混合压力后查 KV/Mamba/request pool，无泄漏/无持续 retract 才晋级；greedy/logits/能力抽检与同形状 off 对照，不能仅凭 token 数认定数值正确。
5. 最后单独验证 NEXTN（若要采用）和默认 overlap；CPU 通过不代替这两项。

模板在 pod 内由队列注入 `AX`、`RUN_DIR`、`N`，并设置 `AX_P120_VARIANT=off|on|cap4096`。本轮未调用 pod/Trisol/bohr，没有 8 卡、镜像或提交动作；外部服务名字、command、env 仍遵守中性命名规则。


## 版本记录（Claude，2026-09-23）
- v1（W15）：续算与新冷请求始终封顶 cap（2048）；8 卡 N6：intra 排队 6.4→0.36s，但 chain_start 尾部变差、tpot_p95 0.104。存 patches/drafts/120-sched-protect-chain-v1.patch。
- v2：仅在有等待请求时封顶（Fable 建议），chunk 16384 / cap 8192 / short 8192；8 卡 N10：TTFT 四门全过，但 tpot_p95 0.1315 超门。存 drafts/…-v2.patch。
- v3（当前）：有等待**或有请求在 decode** 时封顶，只有完全空闲才用大块；配合 `--prefill-decode-interval 3`、cap 2048 → 梯子 027。
