# SGLang 组合验证与短预热（2026-09-24）

用户确定：本地只推进一个组合——正式 A + mem0.87 + 新版180 + 修复122。
对应已上传的正式 attempt 46174 / 镜像0924d / 引擎759a6ebb8e31723519ad5daf438e26e24b32501a。
本轮任务067，入口 `scripts/pod/jobs/official_b_full_n30_shortwarm.sh`。实际队列状态见 `notes/queue.md`。

## 改什么

- A：已有正式N14成绩的整套SGLang部署基线，保留MTP与120调度保护。
- 0.87：静态显存比例参数，涵盖权重和缓存/状态池；不是GPU利用率或单独KV占比。
- 180：GPU/CPU分层缓存，完整搬运MLA、indexer、KDA及草稿相关状态；修复HiCache意外关闭120/122。
- 122：按解码节奏控制prefill预算；修复无法入批的短命中反复预留、续块退化到64 token。

该轮只能判断组合整体。122 CPU 15项、120/HiCache调度32项通过；180已有单卡CUDA池往返证据。
组合真实TP8恢复、稳定性及性能仍待这轮验证，不能提前称为已证实优化。

## 预热规则

原harness按四个维度分桶：请求阶段、总输入长度、预期未命中长度、输出预算。
贪心选整条链，直到覆盖全部已出现桶，再补足至少16条链；16是下限。
在当前冻结集上是59链/1326请求/182桶，跳过gap但生成完整输出；不是全部5601请求。

本地诊断固定使用rep16-v1：8条不同链各取2个相邻请求，共16个。首请求输入长度依次靠近
8192、16384、32768、65536、98304、131072、196608、245760；每个原始输出预算不大于1024，
按距离、两请求输出预算之和、chain_id、链内位置排序，保证选择确定。请求正文、完整历史、tools、
输出预算、thinking都原样传递；没有截断任何请求。只在独立的不计分预热阶段选子集。
第一请求从已有完整正文执行，不补跑之前整条链，第二请求用于触达同链缓存复用。
选择与实跑receipt记录SHA、请求ID、耗时、错误；16个均完成且无错误才能进入flush。
它不声称覆盖全部形状；没有按时间强杀，也不承诺固定几分钟。

启动时的模型加载/CUDA graph捕获仍由引擎完成；短预热只替换原runner的warmup子进程。
preflight仍为原先的一条链；随后真实flush必须2xx且success=true，失败阻断测量。
正式平台如何预热由主办方控制，当前不重交镜像或修改正式提交。

## 一轮收什么、什么时候作判断

一轮固定N30、311链/5601个测量请求，保持冻结正文、顺序、输出和gap；不再70分钟截准入。
第一次测量请求发出为t0，启动、preflight、16请求预热的耗时单独记录。

| 时点/数据 | 内容与用途 |
| --- | --- |
| 每15分钟快照（含30/60分钟） | 已完成请求数、各桶样本量/超标量/统计余量、TTFT p95、TPOT均值/p95、错误；实时窗口保持open，不漏算慢请求后宣称通过 |
| 每请求raw，共5601条 | 唯一ID、阶段、发送/接收/准入/执行/首token/结束时间、prompt/cached/output tokens、TTFT/TPOT、错误。服务端字段缺失就明确标缺，不补造 |
| 每10秒metrics、每5秒GPU采样 | 排队/运行量、KV/KDA压力、显存与利用率；结合服务日志观察prefill块长、MTP接受长度和恢复行为 |
| 全量完成 | cohort每请求恰好一次、runner成功、本轮flush证据、原评分器+统计余量+TPOT门；缺记录INVALID，有效但任一门失败为FAIL |
| 失败归因 | 按TTFT四桶和TPOT保留全部失败ID及逐请求CSV；分解等待与执行，核查缓存重算、调度碎块、恢复和输出异常 |

11门：coverage、harness_data、harness_render、engine_error、infra_error、四道TTFT、
gated_phases_have_samples、tpot_p95。TPOT p95≤0.10，无TTFT统计余量。
TPM遵循原harness的固定稳态窗口与有效性检查，不用全程墙钟/预热/首轮输出替代。
启动后30分钟是诊断检查点，不是整档通过结论；完整集VALID也只是本地结果，不换算正式N@SLO。

原harness及数据目录只读。新增wrapper和任务均在scripts/；默认原预热路径不变。
CPU回归覆盖真实冻结集选择、原loadgen执行16请求且预算不变、只替换warmup阶段、
预热失败阻断flush与测量、原评分和清缓存路径。

证据：`evidence/L067-official_b_full_n30_shortwarm/config-verification.json`、
`expected-short-warmup-plan.json`；运行后的真实receipt须与计划SHA核对。
