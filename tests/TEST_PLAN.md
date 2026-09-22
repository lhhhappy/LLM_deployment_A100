# TEST_PLAN.md — 测试用例总表（作者：Claude，2026-09-22；Codex 审阅）

**原则**（借鉴 harness-template-cn："能机械检查的就别只停留在口头"）
- 每个用例有 ID、运行环境、数据、命令、**机械可判的通过标准**、负责人、状态。
- 运行环境：**CPU**＝本地或 GPU 机 CPU；**L2**＝2 卡替身（随机权重小 Kimi-Linear，SGLang v0.5.20）；**T8**＝Trisol 8 卡、官方底包、完整 GLM-5.3-Flash。
- 数据：`cases/*.json`（内部用例集，W4 生成）只用于机制测试；**正式对比一律用原版开发集和 `run_dev.py`**（task.md 明文要求"别改题"）。
- 状态：`todo` / `wip` / `pass` / `fail` / `blocked`。结果写进 `notes/experiments.md`，并把状态回填到本表。改代码的人负责让相关用例重新变成 `pass`。
- 优先级：P0 不过就不能提交；P1 不过就不能宣称该优化有收益；P2 为诊断。

**T19 工具已补齐（W6，CPU/mock 验证）**：IF-02/04/05/06/08 忙碌路径/09/10/11/12、D1-04、D1-07、CAP-01/02；命令与运行前提见 [T19_USAGE.md](T19_USAGE.md)。D1-07 CPU 15/15 pass；工具就绪不代表 live 通过。E2已用独立raw工具执行D1-04（fail）；T29补测IF-08 busy/等待、D1-08/10，限定L2配置通过，D1-05统计总门仍未通过。E1 stock文本响应失败保留。API logits工具比较top-k并集上的归一化logprobs，不冒充全词表raw。历史ID映射见 [实验台账 T28](../notes/experiments.md#t28--e1e2-测试-id-与证据回填codex-main2026-09-22)，后续补测见同台账T29/F47。

## A. 接口合规（D0；task.md「服务接口规范」与「提交前自查」）
| ID | P | 环境 | 用例 | 通过标准 | 覆盖工具 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| IF-01 | P0 | L2/T8 | `GET /v1/models` | 200，且列表含 `model_name`（default） | preflight_8gpu（已覆盖） | todo（E1/E2无专门的models响应及期望模型名校验收据；服务启动不算通过） |
| IF-02 | P0 | T8 | `/v1/chat/completions`，model=default，一道 aime 风格题 | 200；`choices[0].message.content` 非空且含最终答案；思考过程在 `reasoning_content`；没有被截断（finish_reason≠length，除非真到上限） | `if_checks.py --cases IF-02`（preflight `--extended-if`） | todo |
| IF-03 | P0 | L2/T8 | `/generate` SSE 事件结构 | 每个事件都有 `meta_info.{completion_tokens,prompt_tokens,cached_tokens}`；completion_tokens 单调不减；首个 >0 的事件带 `prefill_finished_time`；末事件带 `request_received_ts`；以 `data: [DONE]` 结束 | preflight_8gpu + `if_checks.py --cases IF-04`（严格 DONE）；IF-08同样调用events/validate_stream | pass（T29 IF-08的4096-token live流经W6严格DONE及逐事件计数/首末时间戳检查；仅L2 D0+D1 v1.1，无MTP） |
| IF-04 | P0 | L2/T8 | `ignore_eos` 精确长度：max_new_tokens ∈ {1, 2, 240, 4096}（T8 上 MTP 开关各测一次） | 末事件 completion_tokens == max_new_tokens | `if_checks.py --cases IF-04` | todo（E1/E2回放输出4、数值检查32；未跑规定的1/2/240/4096矩阵） |
| IF-05 | P1 | L2/T8 | 增量 text | 相邻事件的 text 是增量而不是累计（不重复前缀） | `if_checks.py --cases IF-05` | todo |
| IF-06 | P0 | T8 | 不重复套模板：随机抽 20 条开发集 prompt | 服务端 `prompt_tokens` == 冻结的 `glm_tokens`，逐条相等 | `if_checks.py --cases IF-06 --seed 19`（原 Renderer） | todo |
| IF-07 | P0 | L2/T8 | 缓存计数如实 | 同一 prompt 连打两次，第二次 `cached_tokens` > 0 且上涨；flush 之后归零 | preflight_8gpu；E1 replay --smoke | pass（L2 stock E1：0→18624→0；T8/D0候选仍todo） |
| IF-08 | P0 | L2/T8 | flush 语义 | 空闲时 200 + `{"success":true}`；有在途请求且 `timeout=0` 时 400 + `{"success":false}`；带 `?timeout=N` 时等请求结束后成功 | `if_checks.py --cases IF-08` + preflight/ladder（严格 D0 JSON） | pass（T29 D0 v1.1/L2 TP1：busy400/false，等待54.409s后200/true，再打cached=0；evidence/T29/first/if08.json。E1 stock仍FAIL，DP/T8未测） |
| IF-09 | P1 | T8（DP>1 时） | flush 汇总所有 worker | 人为让一个 rank 忙，flush 必须返回 false；成功后每个 rank 上的 cached_tokens 都是 0 | `if_checks.py --cases IF-09 --dp-ranks 0,1`；不可控时按 T19_USAGE 手工步骤 | todo |
| IF-10 | P1 | L2 | 未知 `X-S1-*` 头 | 请求照常成功 | `if_checks.py --cases IF-10` | todo |
| IF-11 | P1 | L2 | rid 复用（F23） | 同一 rid 在前一个请求结束后再发，成功；客户端中途断开后，同一 rid 仍可复用 | `if_checks.py --cases IF-11` | todo |
| IF-12 | P0 | L2/T8 | 时间戳诚实 | 同机客户端：`request_received_ts − 客户端发送时刻` ∈ [0, 50ms]；`prefill_finished_time ≥ request_received_ts`；`ttft_source=server` | `if_checks.py --cases IF-12 --same-host` | todo |

## B. D1 角色边界快照（`patches/001`；E1/E2）
| ID | P | 环境 | 数据 | 用例 | 通过标准 | 覆盖工具 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1-01 | P0 | L2 | cases/smoke | 不设环境变量 = stock | 逐请求 cached_tokens 与未打补丁的 stock 完全相同 | `compare_e2.py` / `compare_e2b.py` | pass（v1.1四集合215项相同；T26/v1.2栈unset smoke6/6等stock、tail split=0；N1） |
| D1-02 | P1 | L2 | cases/reminder_heavy | 收益机制（v1.1缺冷请求最后chunk快照；v1.2/004补此路径） | 打开后逐请求 cached_tokens ≥ stock；与 F24 role_conservative 预测的差 ≤ 64 token（边界对齐误差）的比例 ≥ 95%；fast_intra 实际未命中 p95 下降 | `compare_e2.py` / `compare_e2b.py` | pass（T26 v1.2：70/70精确预测，fast p95 8135→2439；vs控制27好/43同/0差。旧v1.1仍FAIL，仅45/70且p95未降；只限TP1/N1替身） |
| D1-03 | P0 | L2 | cases/strict_append | 回归保护 | 没有任何请求的 cached_tokens 低于 stock | `compare_e2.py` / `compare_e2b.py` | pass（v1.1及T26/v1.2均67/67相同；v1.2扩展三组143项0退步，仅N1 extra_buffer） |
| D1-04 | P0 | L2 | 从 reminder_heavy 取 3 对相邻请求 | 数值正确性 | 下一轮 prompt 的冷算与从 b 快照恢复对比：首 token logits 最大绝对差 ≤ 容差（按 bf16 基线噪声标定：同一冷算重复两次得到的差 ×2）；greedy 32 token 一致 | `logits_check.py`（API logprobs）；`e2_serve_trace.py` + `e2_raw_logits.py`（全词表raw） | fail（E2 v1.1：3对确认b恢复但全超零容差，1对greedy不同；off普通恢复也2对失败，不能直接定位D1损坏；F45） |
| D1-05 | P0 | L2 | cold_heavy 与 reminder_heavy 混合，N=4 并发 | 单 partial 不变量与稳定性 | 没有断言失败或崩溃；每轮 partial 数 ≤ 1（打日志统计）；ROLE_BOUNDARY_STATS 各项之和 = 准入次数 | `e2_completion_checks.py` + scheduler逐轮trace | wip（T29：105请求0error/306轮partial≤1通过；统计和105=新准入，但≠含续跑334或尝试215。严格总commit门FAIL，分母待明确，不标整体pass；context262144/KV524288） |
| D1-06 | P1 | L2 | smoke | 关闭 chunked prefill（`--chunked-prefill-size -1`） | 不崩溃；计数 `skipped_no_chunked_prefill` > 0 | — | todo |
| D1-07 | P1 | CPU | 合成请求 | 没有边界、短 prompt、不对齐 prefix、branch、active/new chunk、无 chunk budget、正常 split、scan window 等 | 跳过返回 None 且相应计数 +1；正常 split 长度正确、max_new_tokens=0、is_chunked=True；不改请求预算 | `test_role_boundary_split.py` / `test_d1_final_chunk.py`（AST实际方法） | fail（v1.2扩展10pass+1额外alignment缺陷expectedFailure，不能标全过；原v1.1仍15/15 pass；evidence/T26/cpu_004.log、004审阅§2） |
| D1-08 | P0 | L2 | reminder_heavy 跑完之后 | 无泄漏 | flush 后 mamba 空闲槽数与 KV 空闲量回到启动基线 | `e2_completion_checks.py` / `compare_e2b.py` + startup/flush池trace | pass（T29 v1.1 N4大配置恢复KV524288/Mamba512/request8；T26/v1.2 N1三case后28次flush全恢复131072/512/8；不同profile，不外推并发/DP） |
| D1-09 | P1 | L2 | reminder_heavy / strict_append | `extra_buffer_lazy` 变体 | 同 D1-02 和 D1-03 | — | todo |
| D1-10 | P0 | L2/T8 | 任意 | flush 清掉角色快照 | flush 后重打同一 prompt，cached_tokens = 0 | `e2_completion_checks.py` + scheduler split/flush trace | pass（T29 v1.1/L2 TP1：实际上一轮role深度90624，目标命中90624，flush后同prompt为0；evidence/T29/first。HiCache/DP/T8未测） |
| D1-11 | P1 | T8 | 开发集 | MTP 开启时 | D1-03、D1-04 的 T8 版本通过 | — | todo |

## B2. D2 shortest-prefill-first（T20；patches/002）

CPU 命令：`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_spf_scheduling.py -v`（本机16/16；GPU机env/sgl完整导入同16/16）；模拟器回归用 `-p 'test_sim*.py'`。负责人 W7；设计与完整复现见 `patches/002-spf-scheduling.md`。模拟器行只证明模型一致性。

| ID | P | 环境 | 用例 | 通过标准 | 状态 |
|---|---|---|---|---|---|
| D2-01 | P0 | CPU | uncached 排序、output 长度、到达 tie、临时降优先级 | 实际 SchedulePolicy 顺序逐项符合 #40024，work 下限1 | pass（16项生产测试之一组） |
| D2-02 | P0 | CPU | long continuation + 两个完整短 waiter | 4096预算=2560+512+1024；只有原 continuation 未完成；记账不超支 | pass |
| D2-03 | P0 | CPU | 续跑最小进度与 page/DSA 对齐 | 正常预留后≥1页；较大 truncation_align_size 保留≥1共同单位；不匹配/预算太小不预留 | pass |
| D2-04 | P0 | CPU | 初次准入/host miss 第二 partial | old/new partial 均拒绝；首次拒绝不触发delayer/H2D；host miss不commit | pass |
| D2-05 | P0 | CPU | D1+D2共享守卫 | 活跃 continuation 时D1跳过切分；D1切分后完整请求可入、第二partial不入；采样预算不变 | pass |
| D2-06 | P0 | CPU | KV、Mamba gap、page debit、tile gate | 原KV拒绝保留；gap同时扣两池预算与槽位；tile拒绝无commit；关键方法AST与base相同 | pass |
| D2-07 | P0 | CPU | default FCFS 回归与001→002组合 | 56组paired范围/预算/verdict快照字节一致；fuzz0组合结果等于build/d2/b；002单独失败明示 | pass |
| D2-08 | P1 | CPU | simulator #40024 与原spf语义 | 新spf-upstream 500随机轮与生产准入一致；slot反例区分原模型；47模拟器tests通过 | pass |
| D2-09 | P1 | CPU | R10校准候选交叉验证 | 7候选×5策略×2缓存×8档完整产出，标MODEL OUTPUT；原/新SPF14组ceiling一致 | pass（560 model runs） |
| D2-10 | P2 | CPU | stock HRRN/LPM近似评估 | token-aging/rid/zero-work、LPM absolute-prefix、>128 fallback与源码一致；报告同候选梯子 | pass（模型中HRRN=FCFS，LPM无增益） |
| D2-11 | P0 | L2 | D1+D2真实共批、cache恢复、取消/flush | 每轮partial≤1；cold/resume数值与D1-04标准一致；取消/flush后无KV/Mamba泄漏 | todo（CPU不替代live） |
| D2-12 | P1 | T8 | 同引擎FCFS/HRRN/LPM/SPF + D1对照 | 原开发集同档全部硬门、TPOT、cold尾与能力门通过后才宣称收益 | todo（需后续授权） |

## C. 性能与梯子（T8；首次 8 卡会话计划）
| ID | P | 用例 | 通过标准 / 产出 | 状态 |
| --- | --- | --- | --- | --- |
| PF-01 | P0 | stock 基线：原版开发集，按正式方式爬坡（ladder_search official） | 得到台账：各档 10 门 + tpot 门，以及估计的统计余量；作为模拟器校准数据（R9 §8） | todo |
| PF-02 | P1 | D1 与 stock 对比：基线临界档及其上一档 | 同一 N 下 fast_intra 和 overall_intra 的 p95 下降，且没有门变差；最好临界档 +4 | todo |
| PF-03 | P2 | cases/formal_like 固定 N 回放，对比 stock 和 D1 | 内部诊断：未命中与 TTFT 分布 | todo |
| PF-04 | P1 | cases/cold_heavy 压力 | chain_start p95 与 tpot_p95 不劣化 | todo |
| PF-05 | P0 | 所有通过档位的 tpot_p95 ≤ 0.10 | 由 score_formal 判定 | todo |

## D. 能力（T8；能力门两科都要 > 90）
| ID | P | 用例 | 通过标准 | 覆盖工具 | 状态 |
| --- | --- | --- | --- | --- | --- |
| CAP-01 | P0 | 公开 AIME 题 10 道，stock 与候选各跑一遍 | 答案在 content 里；候选的正确数 ≥ stock − 1（小样本容差） | `cap_spot_check.py --suite aime --count 10` | todo |
| CAP-02 | P0 | 公开 GPQA-Diamond 20 道 | 同上 | `cap_spot_check.py --suite gpqa --count 20` | todo |
| CAP-03 | P0 | 输出预算没有被压：一道长推理题 | 没有被服务端截断；输出长度与 stock 同一量级 | — | todo |

## E. 提交包（CPU / T8）
| ID | P | 环境 | 用例 | 通过标准 | 状态 |
| --- | --- | --- | --- | --- | --- |
| SUB-01 | P0 | CPU | `check_submission.py --final` | 0 error（digest 已钉，四个字段合规） | todo |
| SUB-02 | P0 | CPU | Dockerfile | ≤ 64 KiB；嵌入的 payload sha256 与补丁一致 | pass（SA2） |
| SUB-03 | P0 | T8 | 补丁能否打在**底包**里的 sglang 上 | 容器内 `patch --dry-run -p3 --fuzz=0` 通过 | todo |
| SUB-04 | P0 | T8 | 用**提交的原文** command 和 env 起服务 | argv 执行成功，`/v1/models` 200（与 IF-01 同一流程） | todo |
| SUB-05 | P0 | CPU | 占位轨迹 | `check_submission.py --trace` 与 `playground trace validate` 都通过 | pass（SA2） |
| SUB-06 | P0 | CPU | T22 审批、STOP、快照/最小 outputs、两级预检 | 无审批绝不提交；变更/撤回拦截；预检失败终止该项；CLI 参数合约通过 | pass（W9 mock；T22 59/59） |
| SUB-07 | P0 | CPU | T22 上海日配额、台账/API 双在途、并发锁 | 默认每天≤2、在途≤1；手工尝试计数/去重；API 异常关闭提交门 | pass（W9 mock） |
| SUB-08 | P0 | CPU | T22 提交超时/崩溃/重启 | write-ahead 占位不丢失；未知提交阻断且不自动重试；台账损坏不清空 | pass（W9 mock） |
| SUB-09 | P0 | CPU | T22 取分、终态摘要与秘密处理 | 全部指定字段正确；双源状态一致才完成；摘要/事件去重；token 不入日志/argv | pass（W9 mock + all_att 样本） |

## F. 工具自身（CPU）
| ID | 用例 | 通过标准 | 状态 |
| --- | --- | --- | --- |
| TL-01 | `scripts/test_ladder_search.py` | 29/29 | pass |
| TL-02 | `scripts/test_sim_closed_loop.py` | 30/30 | pass（W3） |
| TL-03 | `s1-dev/harness/test_s1_harness.py` | 29/29（GPU 机 CPU 上跑；T8 容器内再跑一次） | pass（CPU） |
| TL-04 | `scripts/check_records.py` | 0 error | pass |
| TL-05 | W4 工具测试（test_t16_tools、test_replay_chains 等） | 57/57 | pass（W4） |
| TL-06 | T19 HTTP/mock 工具测试 + D1 AST 单测 + preflight 回归 | 53/53（32 新工具 + 15 D1 + 6 preflight） | pass（W6；evidence/T19/tools_validation.log） |
| TL-07 | T22 提交队列/daemon，`test_submit_daemon.py` | 59/59，全离线 mock；无真实提交、无 APPROVED 文件、无 daemon 启动 | pass（W9；evidence/T22/submit_daemon_tests.log） |

## B3. I6 SLO-aware prefill scheduling（T25 / W12，patches/003）

CPU command: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_slo_scheduling.py -v`。模型复现 `python3 scripts/check_slo_calibration.py`；报告 `research/codex/R12_slo_aware_scheduling.md`。CPU 测试不代表 live 性能/数值通过。

| ID | P | 环境 | 用例 | 通过标准 | 状态 |
|---|---|---|---|---|---|
| SLO-01 | P0 | CPU | CP 下界/允许超时数，61s p95 构造 | 独立60位Decimal CDF匹配n=1/20/65/100/314/388/808/9023；k允许边界及k+1正确；808时51过52不过；空桶不可评估 | pass（test_slo_scheduling，统计3项） |
| SLO-02 | P0 | CPU | header → GenerateReqInput → tokenizer → scheduler；session namespace/flush | 真实helper不改native session/测量时间；3字段AST透传齐全；首次/续轮/缺头/namespace/LRU/requeue正确；实际flush方法忙时保留、空闲真reset后清tracker | pass（evidence/T25_slo/unit_tests.log） |
| SLO-03 | P0 | CPU | EDF/least-slack/weight、缓存阈值与冻结 | 4096/4097分配3/5s；首会话30s；后续partial不改目标；old cold可恢复优先；weight只改调度deadline；不读冻结phase/uncached_expected | pass（实际003 helper/Policy） |
| SLO-04 | P0 | CPU | 对active partial预留完整更紧急waiter | token预算不扩张；页与DSA LCM对齐；最少aligned续跑；不合格/不适配停止；450随机排序/预算与模型一致 | pass（3策略×150随机轮） |
| SLO-05 | P0 | CPU | D1/host miss/ordinary shared partial guard | 实际Adder在3新策略下通过W7现有/新partial、host-miss重选、D1 split+full等对抗场景；最多1partial且不改输出预算 | pass（18组复用对抗场景；unit_tests.log） |
| SLO-06 | P0 | CPU | 003补丁组合、关闭回归、预算未改 | clean→001→002→003及加000均fuzz=0；字节/语法检查；56 FCFS paired场景一致；除opt-in guard外Adder方法AST与002一致 | pass（unit_tests.log；16原D2+15原D1也pass） |
| SLO-07 | P1 | CPU | 固定R10的dev/formal-mix八臂对照，strict与estimated分开 | 896 MODEL OUTPUT runs，完整输出预算/源hash/每档失败门；改冻结评分标签不改变调度trace；CLI支持3策略/weight；旧5策略trace不变 | pass（calibration/comparison.json；47模拟回归+29评分+21加权工具pass） |
| SLO-08 | P0 | L2 | 完整导入/真实并发/取消重排/flush/KDA logits/计数 | 真实依赖可导入；每轮partial≤1；无KV/Mamba泄漏；冷暖logits与基线容差内；输出/工具/thinking/计数/时间戳诚实；busy flush保留状态，成功真清 | todo（CPU fake/AST不替代此项） |
| SLO-09 | P1 | T8 | 同引擎SPF/EDF/EDF+D1/weighted ladder | 原版dev、完整输出和相同参数；逐档真实TTFT/TPOT、超时数/桶数；全部硬门通过且复测收益才启用；不能拿模型当容量证据 | todo（W12未起服务；需协调方另行安排） |

### T24 自动提测工具（CPU/mock；W11）

命令：`python3 -B -m unittest discover -s scripts -p test_trisol_test_daemon.py -v`。
协议、预算单位与 nohup 启停：[TRISOL_TEST_DAEMON.md](TRISOL_TEST_DAEMON.md)。
PF-01/PF-02 的 T8 实测状态不因这些 mock 测试改变。

| ID | 环境 | 用例 | 通过标准 | 状态 |
| --- | --- | --- | --- | --- |
| TQ-01 | CPU/mock | APPROVED、STOP、不可变候选、dry-run | 未批准/暂停无 Trisol；不写 APPROVED；候选变更拒绝；dry-run 无进程/写入 | pass（T24） |
| TQ-02 | CPU/mock | 单实例锁、本人配额、FIFO/接管 | 同时最多一个服务，已有其他服务阻止创建；接管核对8卡/镜像/挂起命令；两个项 create/delete 严格交替 | pass（T24） |
| TQ-03 | CPU/mock | admission/每日GPU时长/硬截止 | 排队≥2h不误算运行；12 GPU·h=8卡90min；UTC跨日拆分；每日用尽不创建 | pass（T24） |
| TQ-04 | CPU/mock | 超时/失败/停止/删除确认 | 各失败路径尝试拉取后删除；delete accepted不解锁；未知创建隔离；替换ID不误删 | pass（T24） |
| TQ-05 | CPU/mock | 重启与独立watchdog | 先清理遗留项、不重复实验；回收自有runner进程组、按PID starttime防复用；watchdog从不排新项 | pass（T24） |
| TQ-06 | CPU/mock | 回传/评分/事件/台账 | collect→score→双落点→RESULT→delete；PF-01/02引用；RESULT去重；真实scorer合成数据仍标estimated | pass（T24） |
| TQ-07 | CPU/mock | 与T22提交队列衔接 | N门/逐ID P0/精确候选/digest任一缺失不入队；达标调用W9 writer且不批准 | pass（T24） |
| TQ-08 | CPU/mock | CLI协议/路径/敏感输出 | 旧CLI缺参数时服务动作前阻止；真实runner help通过扩展契约；路径逃逸拒绝；bohr仅mine全分页；默认公网名中性 | pass（T31真实help+argv+mock端到端；线上schema仍待验证；evidence/T31/trisol_test_daemon_tests.log） |
| TQ-09 | CPU/mock | keep-alive 生命周期 | 两个批准项复用同一服务；空队列 idle-hold 后释放；永不超过一个服务 | pass（T27） |
| TQ-10 | CPU/mock | unlimited 与自动恢复 | daily_gpu_hours=0 忽略历史日用量；服务丢失后下一项重新创建 | pass（T27；mock） |
| TQ-11 | CPU/mock | simulator/ledger/events | 每项 score_formal 后生成 sim_vs_real、RESULT、SIMCHECK 与实验台账；SIMCHECK 去重 | pass（T27；mock） |
| TQ-12 | CPU/mock | forever/status/queue depth | STOP wrapper、target depth 单次 QUEUE_LOW、status 少于30行且不调用模型 | pass（T27；shell/CPU） |


## T23 / Session A 自动化（W10；仅 CPU，不等于 T8 用例通过）

命令：`PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s scripts/session_a -p test_session_a.py -v`。
证据：原 `evidence/T23/session_a_tests.log`（26/26）；T31扩展回归 `evidence/T31/session_a_tests.log`（42/42）、`trisol_test_daemon_tests.log`（57/57）、`queue_dry_run.json`（5项无BLOCKED）；用法 `scripts/session_a/README.md`。

| ID | 环境 | 覆盖 / 通过标准 | 状态 |
|---|---|---|---|
| SA-01 | CPU | CLI JSON framing拒绝歧义；不完整/错N结果不能计为SLO失败；十门+TPOT齐全 | pass |
| SA-02 | CPU | N6/N10保底、±4、N2下界、非单调/上界不冒充临界N | pass |
| SA-03 | CPU | monotonic全局/步骤预算及final reserve；成对matrix准入，预算不足显式incomplete | pass |
| SA-04 | CPU | chunk argv≤64KiB、真实本地base64/tar roundtrip及坏SHA拒绝；路径穿越/link拒绝；增量snapshot hashes | pass |
| SA-05 | CPU | 本地参考源码临时副本，真实000→001→002 fuzz=0 dry-run/apply；原参考文件不变；offline报告 | pass（不是SUB-03底包验证） |
| SA-06 | CPU/mock | 完整batch顺序、每档同步、preflight失败无ladder、copy失败仍stop、缺002仍D1、PID复用拒绝、远端timeout清理 | pass |
| SA-07 | CPU/mock | dry-run零subprocess/network/write；small CAP保持registry非pass；missing metrics不造0 | pass |
| SA-08 | T8 | 真实CLI传输/底包/引擎/逐档双落点校验与final cleanup | todo（本任务禁止进入服务） |
| SA-09 | CPU/mock | daemon实际argv往返、真实wrapper help预检；service ID/name别名、profiles字节/SHA/远端参数、精确matrix（含v1.2/HRRN） | pass（T31；session_a_tests.log + trisol_test_daemon_tests.log） |
| SA-10 | CPU/mock | baseline保底与P/P+4；无baseline三种梯子、levels保序、fast hint、max-n/max-levels、显式缺补丁失败 | pass（T31；evidence/T31/session_a_tests.log） |
| SA-11 | CPU/mock | 已有profiles输出根可用、stale run拒绝；零预算只拉取/重建manifest、SHA损坏修复、无launch/stop、候选非精确不提升P0 | pass（T31；evidence/T31/session_a_tests.log） |

关联工具回归：TL-01 29/29（测试环境 `ARENA_FLUSH_ATTEMPTS=1 ARENA_FLUSH_SERVER_WAIT_S=0`），
T19+preflight 38/38；T31合并重跑67/67，见 `evidence/T31/dependency_regression.log`（原T23证据保留）。
IF/CAP/PF/SUB-03 的 live 状态没有因为这些自动化单测而升级。


## M3 — T42 / 130 分词线程池与路由键（W16）

CPU：`python -B scripts/test_async_tokenize.py --real`；原样长输入性能：加 `--benchmark-real-only`。
需 transformers 5.12.1 / tokenizers 0.22.2 / jinja2；基线临时副本按 000→101→110→111→130 应用。
生产方法经 AST 抽取运行；并非完整 GPU 服务导入。证据 `evidence/T42/`，设计 `patches/130-async-tokenize.md`。

| ID | 环境 | 覆盖 / 通过标准 | 状态 |
|---|---|---|---|
| M3-01 | CPU | 补丁链 fuzz=0、五个改动文件语法、原始 downstream token/time 逻辑不变；回退/fast/slow/batch/pair/EOS/dynamic 策略一致 | pass（14 项单测中的相关项，cpu_tests.log） |
| M3-02 | CPU | 真实 glm_tok、完整 dev bodies：原版/开启/关闭的 token IDs 逐项相同；冻结 glm_tokens 对齐 | pass（722/722，34416777 输入 tokens；token_comparison.jsonl；最大 256733） |
| M3-03 | CPU | 取消运行与排队请求、错误恢复、关闭；同一线程串行且无取消积压 | pass（单测；7 adversarial +21 并发对照 +3 batch/pair 真 tokenizer 对照通过） |
| M3-04 | CPU | body 键优先、Routing-Key→Session-ID fallback、无/空头、batch 子请求、真实 handler SSE/接收时间透传 | pass（生产 helper/handler 抽取 + transport stub；不是 live HTTP） |
| M3-05 | CPU | 大文本同步/线程对照；1ms 心跳期间仍可运行，报告耗时和最大 loop lag；实测 Rust GIL 行为 | pass（real_results.json；96k/250k 派生压力文本各3组交替；原样对话另见 real_prompt_benchmark.json） |
| M3-06 | L2/T8 | 实际服务导入、原始 dev 开关 A/B、SSE/TTFT/TPOT 全门、取消/RSS/flush/输出与时间戳诚实 | todo（未起 GPU/Trisol/镜像/提交；由 Claude 审阅安排） |


## M1. 调度保护链中间请求（T41 / 120；底包线）

命令：`python3 scripts/make_120.py`；`python3 -B -m unittest discover -s tests -p test_sched_protect_chain.py -v`；`python3 scripts/verify_120.py`。
真实调度/准入方法 AST + CPU mock ScheduleBatch/池/forward；不等同于 GPU 正确性或性能通过。说明：`patches/120-sched-protect-chain.md`；证据 `evidence/T41/`。

| ID | P | 环境 | 用例 / 通过标准 | 状态 |
|---|---|---|---|---|
| P120-01 | P0 | CPU | 冷长 + 在跑 decode + 新短命中：prefill/decode 交替；短命中同批加入；续算 cap 生效 | pass（27/27 测试，interleave.json） |
| P120-02 | P0 | CPU | 仅冷启动无空转，100k/2048 在 49 prefill 完成；仅 decode 与基线一致；存活判断及显式 interval 不重复叠加 | pass（cpu_tests.log） |
| P120-03 | P0 | CPU | 短请求完整适配、设备命中/阈值/host 守卫、LPM 原排序 AST 不变；长队首不挡短命中；至多一 partial | pass（cpu_tests.log；16×30 随机轮次） |
| P120-04 | P0 | CPU | 000→101→110→111 与 120-off 24 组决策 JSON 字节一致；断言方法/语句/时点一致 | pass（off_parity.json；22组40轮正常，2组原生断言；off_decision_traces.json） |
| P120-05 | P0 | CPU | 无限短到达下已准入冷请求完成，验证条件轮次上界；101 admission/tail/branch 保留 | pass（starvation_bound.json；100k无role49/97，有role50/99；不承诺未准入LPM公平） |
| P120-06 | P0 | CPU | 页/KV/input 预算、请求行保留、COW 拒绝释放/session槽保留、unsupported回退、配置/对齐；101已存在chunk的双partial反例保护 | pass（cpu_tests.log） |
| P120-07 | P0 | CPU | 真实 -p3/fuzz=0 应用只改2文件，全部4684文件匹配生成器；3617 Python +3工具 py_compile；只读底包不变 | pass（validation.json；patch_apply.log；compile.log） |
| P120-08 | P0 | L2/T8 | 普通TP8/default overlap、cold+hot+decode、101分叉：原计数/输出/时间戳、无双partial/崩溃，flush后KV/Mamba/request池回收，数值对照通过 | todo（本轮仅CPU；Claude安排） |
| P120-09 | P0 | T8 | 原dev harness off→on→off N6/10→14/18/22；全部TTFT门与TPOT/错误率不退步；cap4096按需消融 | todo（dev_b120_template.sh仅方案，未入队/未执行） |
| P120-10 | P1 | T8 | 若拟采用NEXTN，单独重复稳定性/数值/真实token与SLO验证；CPU不能替代 | todo |

## T43 — 112 sm80 indexer融合kernel（W17）

命令：`python3 scripts/make_112.py`、`python3 scripts/verify_112.py`；GPU开发机 `scripts/test_sm80_indexer_112.py --mode all`，详见补丁说明；摘要 `scripts/summarize_112.py`。
直接对照未修改110；随机激活的真实模型形状，非真实模型激活/非L2服务。用户明确授权算子微基准，不作为8卡SLO。

| ID | 环境 | 用例 / 通过标准 | 状态 |
|---|---|---|---|
| P112-01 | GPU算子 | 软件解码全部256种e4m3fn编码，254有限值含有符号0逐bit一致，2NaN类别一致 | pass（final_all.log） |
| P112-02 | GPU算子 | B1/6/32×N1/2×ctx1024/32000/190000；[B]/[B,N]；逐行L∞相对误差<1e-2、topk2048集合≥99.5% | pass（36组；最终全部88组最大相对L∞1.1185e-5，topk最低99.9512%） |
| P112-03 | GPU算子 | 0/1/63/64/65/1025尾部、负页、表宽外/输出短于context、strides、3D q；clean两模式、空/反向/越界区间、空维与8/16/64头；次正规数和大幅值输入 | pass（另36组数值及空维；torch bf16 reduction默认True） |
| P112-04 | GPU算子 | 两入口CUDA graph捕获，decode共享/独立ctx，prefill clean两模式；重放更改q、ctx、bt、ks/ke；与eager逐bit一致并对110过数值门 | pass（4种×3次=12动态重放，final_all.log） |
| P112-05 | GPU微基准 | decode B6×32k/190k与prefill8192×32k/190k，同数据新旧ms/加速比；大prefill每行数值/topk同门 | pass（4大矩阵对照+8性能行；graph decode4.75×/4.54×，prefill2.01–2.95×；见summary.json） |
| P112-06 | CPU/GPU编译 | 全8补丁栈fuzz0；确定生成、编译、reverse全文件字节相等、只读底包未改；实际sm80 bf16 MMA | pass（stack_receipt.json：3623 py_compile；compiler/：代表形状0 spills，sm80/bf16，无fp8指令） |
| P112-07 | L2/T8 | Claude审阅后实际模型/服务/TP8/NEXTN整合、能力、原dev TTFT/TPOT/错误门与flush | todo（本任务未触碰pod/Trisol/镜像/提交） |

## P113 — T44 预填充 indexer（W18）

| ID | 环境 | 通过标准 | 状态 |
|---|---|---|---|
| P113-01 | GPU算子 | 原P112-01/02/03全部用例，对110及112逐行相对L∞<1e-2、topk≥99.5%；decode源不改 | pass（final_all.log：复用所有112用例并分别对110/112；全222组max相对L∞3.956824e-5、topk≥99.9512%；decode源码/PTX相同） |
| P113-02 | GPU算子 | 原P112-04全部动态graph + 新tile边界，capture/replay逐bit同eager | pass（10组graph×3次动态重放，逐bit同eager；包括q/K/scale变化、增长/缩短/清空） |
| P113-03 | GPU微基准 | 8192×32k/95k/190k causal/ragged全行数值与topk过门；旧新ms与有效TFLOPS≥100，未达给profile；decode不退步 | pass（六档131.4–185.7等效TFLOPS、6.19–6.69×；全行对两版数值/topk通过，decode graph0.1024/0.5925ms，见performance_table.md） |
| P113-04 | CPU/GPU编译 | 9补丁全栈fuzz0、确定生成、编译/反向字节还原/base_exact未改；编译与profile收据 | pass（stack_receipt.json：9补丁fuzz0、3623编译/确定生成/反向字节还原/base未改；compiler/和profile/已绑定最终SHA） |
| P113-05 | L2/T8 | Claude交叉审阅后完整模型/服务/能力与SLO | todo（用户限定开发机算子） |
