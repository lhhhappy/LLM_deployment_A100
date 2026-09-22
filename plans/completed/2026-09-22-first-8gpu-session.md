# 首次 8 卡会话：看底包 + stock 基线 +（可选）D1 A/B

> 2026-09-22 归档（dropped）：基于 v0.5.20 线 / SPF / 旧自测设计，已被决策 27–29、tests/L2.md、plans/active/102-role-track.md 取代。

> **T31 / W14 CLI对接更新**：runner现接受daemon完整参数（候选profiles、精确matrix、三种ladder、out、零预算collect-only），见 `scripts/session_a/README.md`。显式matrix有baseline才跑N6/N10+基线搜索及P/P+4；无baseline时每个配置直接按指定ladder跑；v1.2组合支持004。显式缺补丁失败，旧默认自动矩阵才允许跳过。保留逐配置IF/CAP、strict flush、逐档回传与deadline。`--collect-only`仅拉取/重建manifest，不启动或停止引擎；正常finalization/watchdog仍负责清理。T31仅CPU/mock验证，SA-08仍todo。以下T23描述保留为默认路径与历史背景。


> **T23 / W10 执行勘误（用户本轮明确要求，R11 优先于下文旧顺序）**：会话 A 由
> [`scripts/session_a/run_session_a.sh`](../../scripts/session_a/README.md) 驱动；
> inspection → 原 stock 仅 CAP 对照 → **D0-only + D1 unset + FCFS** 的 IF/TL-03/CAP 前置门 →
> **N6、N10 必测**，再±4找相邻通过/失败档 → P/P+4 的 SPF、SPF+D1、D1 矩阵（可选 HRRN）→
> 最佳实测组合 CAP → 回传/停自有引擎/再回传。000 失败不能用文本 flush 的 stock 冒充合规梯子；
> 002 缺失/不兼容时跳过 SPF。每档及长步骤每120秒都校验回传到本仓库与 GPU arena，删除由 Claude 在确认收据后执行。
> runner 必须显式传 `--minutes`；240分钟不保证容纳完整矩阵，时间不足只报告实测边界/缺项。
> **当前只完成 CPU/mock/dry-run 自动化验证，没有进入 Trisol 服务，也没有 T8 测量。**
> 下文原计划保留为历史设计；会话 A 实际参数、时间盒及限制以链接文档和本轮用户指令为准。

- 状态：active（**等待用户批准和配额**）　负责人：Claude（执行），Codex（审阅计划）　创建：2026-09-22
- 关联：D6、D0、D1，pipeline §2，F33（每人最多 8 张 A100），T12/T16 工具

## 目标
一次拿到 8 卡，就把三件事做完：
1. 看清底包的 sglang 版本和 sm80 实现；
2. 拿到 stock 配置在开发集上的临界 N 和各门数值；
3. 如果时间够，在同一镜像上做 D1 开和关的对比。
不为同一个目的反复排队。

## 范围
- 包含：官方 SGLang 底包（160721）原样起服务；在容器里打我们的补丁（手动，不打镜像）；开发集梯子；容器自检；小规模能力抽检。
- 不包含：打镜像、提交、D2/D3、HiCache。

## 前置条件与约束
- 用户批准占用 8 卡（约 3–5 小时，按 `scripts/plan_8gpu_session.py` 估算）。
- 先删掉 1 卡的挂起 pod（它占用我们 8 卡额度中的 1 张）。
- 红线：只用官方底包；时间戳和 flush 如实；用完立即删除。

## 风险与缓解
| 风险 | 缓解 | 回滚 |
|---|---|---|
| 排队很久 | 同时提交，并预先准备好所有脚本；拿到卡后按清单执行 | — |
| 补丁打不上底包 | 先对比底包 sglang 与 v0.5.20；打不上时只跑 stock 基线，并记录差异 | 退回 stock |
| 服务起不来或 OOM | 按 task.md 示例的参数起步；记录日志 | 删除服务 |
| 占卡超时影响他人 | 按规划器时间盒执行，超时就停在已完成的档位 | 立即删除 |

## 里程碑（按优先级，时间不够就截断）
1. 起挂起服务（command=`python3 -m http.server 8000`），用 `trisol inference exec -i` 进容器：`pip show sglang`、对比源码、核对 DSA/indexer 的 sm80 实现、`nvidia-smi`、主机内存大小 → 写 F（D6 结论）。
2. 在容器内用 candidate-01 的 stock 参数起引擎（不带 D1 环境变量），跑 `scripts/preflight_8gpu.sh`（harness 自测 + 提交前自查）。
3. `scripts/ladder_search.py`：stock 正式爬坡（从 N=10 起）或快速模式；每档之前严格 flush；得到 stock 基线（E3）。
4. 补丁能打上的话：设置 `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS` 后重启，在基线临界档和它上一档做 D1 对比（E4）。
5. 能力抽检（几道 aime/gpqa），确认答案在 content 里。
6. 删除服务；数据回传到 `/sjtu/linhang/arena/runs/`，摘要写进 experiments。

## 验证方式
- 命令：`scripts/preflight_8gpu.sh`、`scripts/ladder_search.py`（台账 JSON）、`scripts/score_formal.py`（开发集 10 门 + tpot 门 + 统计余量估计）。
- 通过标准：flush 返回 JSON 且 cached_tokens 归零；各档台账完整；`cached_tokens` 与 F24 预测对账。

## 进度记录
- [x] 工具就绪：ladder_search / score_formal / check_submission / build_image（Dockerfile 预演）
- [x] 工具就绪：preflight_8gpu / plan_8gpu_session / 用例集（W4 T16：CPU/HTTP stub 验证；尚未执行 live 容器自检；用法与限制见 experiments「Tools」）
- [ ] 用户批准
- [ ] 1 卡挂起 pod 处理（等它启动后完成里程碑 1，或删掉它，并入本次）

## 决策记录
- 2026-09-22：D6 与 stock 基线合在一次 8 卡会话里做，原因是每人最多 8 卡且排队长（F33）。

### 2026-09-22 · Codex T17 交叉审阅（建议；不构成启动/提交授权）

**结论：认可顶部T23勘误的基线与取证顺序；D1进入性能矩阵/候选晋升前仍缺正确性门，不能按当前自动选择逻辑直接批准。** 本次仅读文档/源码，没有运行服务、runner、镜像或评测。已核对AGENTS、全部active计划、R11、TEST_PLAN、E2/F42/F45、T23 README及关键实现；不修改执行者代码。

审阅快照：本文件追加前SHA256=`05d055a897f6b8609979d9abf005296188f920397b910c473cb3bad50aeb77cb`；`scripts/session_a/runner.py`=`bdcf0c7be36ae67bbdc53cbb628c667cc73dcc2a14e4fb4c543c8b8adb72c05a`。runner仍在迭代，下面的实现判断仅对应此版本。

#### 1. 已解决的旧计划问题与当前阻断项

- **VERIFIED**：顶部勘误已明确原stock仅CAP对照、D0-only/FCFS/D1 unset作为合规基线、N6/N10先测、相邻P/P+4比较、先回传再清理。认可这些修正；下文旧里程碑2/3/5/6及风险表里的“打不上只跑stock梯子”“删除后回传”均不再是执行指令。此处P+4指更高并发档，避免“上一档”歧义。
- **VERIFIED / major**：`runner.py:380–382`仅用001/002是否能应用决定D1/SPF矩阵；`startup(config)`默认不做preflight/CAP（344–350、413），`best_measured`按性能选出后才做小样本CAP（417–429）。没有读取D1-04或D2-11资格。这是兼容性门，不是正确性门。当前E2 D1-04失败，D1-05/08/10与D2-11仍待测；F45还表明零冷噪声门无法单独定位错误，**既不能放宽阈值追认通过，也不能说D1已安全**。
- **建议 / 执行前关闭缺口**：给每个profile分别记录`diagnostic_eligible`和`promotion_eligible`，绑定源码/补丁/配置哈希。未过正确性门的D1/组合默认不进入自动晋升、正式候选或“最佳安全配置”。若协调方明确批准在T8继续定位数值问题，只做单独、有界的诊断阶段，标`correctness_unresolved`；小样本CAP不解除该状态。D0基线取证不必被D1失败一并阻塞；D2也须自己的live前置检查，不能以CPU单测替代。

#### 2. 建议执行顺序（在T23勘误上增加资格门）

1. **占卡前**：明确本次获准的service ID/负责人、测试范围、时长及结束后的delete/keep-alive策略；仅检查/处理自己已确认的1卡服务，不因文档里旧名字盲删。准备并固定bundle、patch、candidate、cohort/tokenizer/CAP题目哈希与离线依赖；公开服务名/描述/command/env按rule §4.1中性化。审批状态以本轮实际收据为准，不从README或看板的旧文字推断。
2. **底包取证**：保存image digest、实际包路径/版本/源码差异、CUDA/driver/kernel依赖、GPU与主机内存、挂载模型/tokenizer版本；在干净副本依次dry-run并实际应用前驱000→001→002，再验证下一补丁。源码中的sm80 dispatch只是线索，需真实长prompt执行/引擎日志佐证；不能由`pip show`或`--dry-run`宣布DSA可运行，也不要拿本地CUDA兼容包结论替代官方底包证据。
3. **质量/接口先于性能**：原stock仅保留配对CAP/接口诊断；D0-only、D1显式unset、FCFS通过IF/TL-03和CAP冒烟后才进行原始warmup→严格flush→测量。candidate-01自身带D1开关，不能原样叫stock；`common.py:124–145`及`remote.py:250`已显式清开关，这点认可。TP8/DP1不冒充IF-09 DP归约已测。
4. **先取得基线**：原版cohort、Renderer、完整输出预算、gap与run_dev不变；在预算允许且前置门通过时先N6校准、N10留出，再±4寻找相邻通过/失败档。只能报告“最大已观测通过档”；未测高一档、遇到非单调、超时或工具错误时都不能称找到了临界N。不能把工具失败计成一档SLO FAIL。
5. **再做有资格的变体**：优先完成D2检查与同档对照；每次改代码/开关/模型参数后先做该profile的快速IF、输出预算、flush/CAP检查，失败不爬坡。D1分支遵循第1节资格门。若比较001 v1.1与004 v1.2，004依赖002，须增加`000+001+002、FCFS、D1 on`控制，不能把新增002与004的变化合算为004收益。003/004不能因为打包在tar里就默认已执行；当前T23只应用000/001/002。
6. **收尾/晋升分开**：有效profile的同档原始数据与完整门值先归档，再挑“已满足资格的最佳实测配置”。2–3题CAP只是冒烟，不等于CAP-01/02的10/20题，更不证明正式两科严格>90。正式提交仍需独立的原文command/env/image复核、双方审阅与用户批准；Session A的私有端口/临时PYTHONPATH运行不是SUB-04通过收据。

#### 3. 回滚必须说明退到哪里，不能只写“退回stock”

| 触发 | 保留与停止 | 允许的下一步 |
|---|---|---|
| 000不兼容/flush不真清 | 保存底包差异、接口body与cached_tokens；停止合规梯子 | 原stock仅取证/短诊断，不让文本200或假清缓存通过严格评分 |
| 001/002/004应用或live检查失败 | 保存具体版本与失败请求/引擎日志，停止该profile | 从保留的干净副本重建D0-only基线；不要在同一棵已打补丁源码上叠加/反向乱退补丁 |
| 数值或能力退化 | 保存cold/warm、恢复深度、完整答案/finish_reason，撤销该profile晋升资格 | 继续已通过门的独立配置，或单独获准的原因诊断；不削输出、不截历史、不事后放宽容差 |
| OOM/崩溃/在途请求超时 | 中止该档、保存启动和峰值池/显存证据，停止自有进程树 | 若要调容量/参数，作为新配置重跑前置门与同档对照，不拼接到旧曲线 |
| 回传失败/控制端断开/预算截止 | 有界重试或collect-only，独立执行自有引擎清理；保留已验证副本和缺失清单 | 协调方按预先批准的服务保留策略处理pod；不无限占卡等待同步，不因一次copy异常跳过停引擎 |

**VERIFIED / 认可现有实现的部分**：`remote.py:226–266`由原stock副本重建profile，并记录argv/env/源码路径；`runner.py:finish`将pull/stop/pull分别try，copy失败不抑制停机；pod侧watchdog与PID/start-time检查用于控制端故障。**未验证**这些机制在真实Trisol失败场景的效果，不能仅凭源码标live PASS。

**重要资源区别**：停止引擎、nvidia-smi空闲不等于释放Trisol pod或8卡配额。当前runner保留挂起HTTP服务并只打印delete命令；因此计划必须指定最后负责释放或批准继续keep-alive的人/时限。后续T27的keep-alive策略若获批并落地，需要显式更新本计划，不能和旧“用完立即删除”同时作为有效指令。本次审阅不删除、接管或续期任何服务。

#### 4. 时间、对照和证据的验收边界

- **VERIFIED**：T23 README给完整示例约550分钟的保守step timeboxes（另加传输/回传/可能重启），不是旧计划的3–5小时必能完成。`--minutes`只是上限；启动/JIT/CAP/transfer可能先耗尽预算，N6/N10也只能在前置门与剩余预算允许时完成。先排最小有效取证包，预留独立清理时间；不足时诚实交付partial，不自动加时或借用提交配额。
- **建议**：每档结果绑定image/model/tokenizer/harness/patch SHA与完整resolved参数（包括chunk/page/SSM/MTP/DP/slots）；每个profile新引擎、同样warmup、每档含run_dev内部warmup后真flush。候选与基线同档比较；条件允许时末尾复测一个基线档检测机器漂移，不能把跨模型/跨参数/跨日的TTFT差直接归因补丁。
- **VERIFIED**：F42已证明F24理想checkpoint模型不覆盖v1.1的实际行为；不能以“cached_tokens与F24逐项相等”作为T8总门。应保留实际逐请求cache/TTFT/TPOT、分桶超时数/样本数、队列/forward/池占用与预测残差，缺失指标标unavailable；尤其区分驻留FULL-KV、branch与冷首缺口。
- dev原10门+TPOT门与`score_formal`统计估计并列保存，不把estimated verdict当官方分数；case子集/4-token替身测试不可替代正式原cohort。新精度阈值应先由独立基线预注册，再检验候选，不能用候选已观察误差倒推“通过”标准。
- **记录维护待协调方同步**：首页仍写E1进行中、quality仍写D1零运行/D2无代码，与E2/T20矛盾；只按已绑定的结果/状态做资格判断。F25为Codex KDA研究、Claude条目已移F33，本轮不复用编号。后续选定上述建议后，由负责人同步两份执行计划及runner；本审阅不冒充已实现修复。
