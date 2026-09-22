> 2026-09-22 归档：v0.5.20 线，打不上底包（决策 29）；仅作 L1 替身参考。

# 004：最后一个 prefill chunk 的角色边界快照 — T26 审阅 / E2b

审阅：Codex main，2026-09-22。范围仅用户恢复的今日收尾：004 审阅和同 qfull 的 E2b；不构建镜像、不提交、不启动新任务。004 原文未修改。

## 1. 版本与结论

- 底版本：SGLang v0.5.20 / `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。
- 004 SHA256：`7a6fd39aa3b26bfaba20dfdf1e199b75fca3b29ca99c86e24ea0bc42cc1e6a2c`。
- 控制臂：000+001+002，FCFS、角色开关开启；候选：同上再加004。没有003。
- 两条栈在开发机独立 clone 上 `patch -p1 --batch --fuzz=0` 和 py_compile 通过。控制/候选 schedule_policy.py 分别为 `6ea708d247c0bd057b4c0aaa613ae56e98a47c522ee17087275b727237a52706` / `e5fd71c550670c6b298272573bb8bf38269a3e60bf0dec01647ecd5d20c5bd2b`，与本地 CPU 测试源逐字一致。
- **有条件允许当前 L2、无 DSA、无额外 truncation alignment 配置做缓存机制 A/B；不能据此通过镜像 B 的正确性门。** 下述 major 仍未修复，E2 的 D1-04 数值门失败也没有被本轮解除。

## 2. Major：续跑路径读不到实际 truncation alignment

004 `_role_boundary_final_chunk_len` 读取 `self._arena_truncation_align`，但这个字段仅在 `add_one_req` 写入。`Scheduler._get_new_batch_prefill_raw` 每轮新建 `PrefillAdder`，**先调用 `add_chunked_req`，再遍历等待队列并把 `self.truncation_align_size` 传给 `add_one_req`**。因此真实续跑路径取默认1，LCM 实际只剩 checkpoint/page grid。

可复现反例：page64、prefix8192、剩余2048、角色位置9000、scheduler 要求512对齐。004 在全新的 adder 上选768，下一段从8960开始，`8960 % 512 = 256`；若手工注入512才选512。`scripts/test_d1_final_chunk.py::test_real_fresh_adder_does_not_receive_scheduler_alignment` 是 **expectedFailure，记录尚未满足的要求，不是通过**。

影响：deterministic inference 的 triton 对齐可为4096；DSA kpool>1 时 scheduler 也会设置/合并该值。实际 GLM 配置若要求的单位已整除page64则此反例不触发，不能无证据宣称 GLM 必然崩溃；但也不能宣传通用 DSA 对齐安全。建议后续修复时在创建 adder 时显式传入对齐，或让 `add_chunked_req` 接受该参数；两条 D1 路径共享同一来源。**本次审阅不私自改004。** 当前 E2b 的 qfull 无DSA/未启 deterministic，运行 trace 另记录实际值。

## 3. 已确认的路径性质及002守卫

- 004仅在原本能结束的最终chunk追加一次切分；设置 `truncated=True`，原有记账不为这段预留decode的 max_new_tokens，原请求的采样预算不改。
- `add_chunked_req` 返回同一 `req`；scheduler继续持有同一个 `self.chunked_req`，不产生 `adder.new_chunked_req`。切后下轮同一角色位置距prefix<grid，不会在同一点反复切，仍能完成尾段。CPU实际方法验证了进度和预算。
- 原非最终chunk、开关关闭路径不改形状；六种长度的关闭差分与001+002控制逐项相同。
- branch位于当前extend时仍优先branch，保持role_conservative。代价是这类请求不补role；不能将F24理想预测当严格引擎oracle。page/scan window/无边界/短尾/不对齐prefix都有检查。
- **确认001 §10.2的host-miss漏洞由002覆盖**：初次选择与host-load失败后的重选都把 `has_chunked_req` 传给 `_select_prefill_admission`；partial分支在SPF或D1开关启用时调用 `_can_start_partial_prefill`；D1 admission-time切分也调用它。004继续持有chunk时，完整waiter仍能入，第二partial在commit之前拒绝。新增004测试实际调用了host-miss重选。这个结论是该准入漏洞已覆盖，不是HiCache整个组合验收通过。
- extra_buffer下可借extend-end tracking保存角色state；额外buffer、lazy、DSA、MTP、HiCache、TP/DP、003组合的完整行为仍需独立验证。本次不扩大验证范围。

## 4. 测试与统计注意事项

CPU：`test_d1_final_chunk.py` 11项中10通过、1已知对齐缺陷expectedFailure；原002测试16/16、原001测试15/15通过。日志 `evidence/T26/cpu_*.log`。第一次测试的“尾部同一grid”fixture误设成整页结束（实际仍可合法切一页），已改成1030长度/9220位置；旧失败日志保留并明确为测试fixture错误。

004增加 `tail_*`，与001新准入计数混合后不能再用“所有ROLE_BOUNDARY_STATS之和=新准入次数”。新准入、续跑尝试、实际tail切分应分别计数。D1-05既有统计口径未决项不因E2b变成pass；N=1也不替代并发不变量测试。

## 5. E2b 预注册与证据

同一 qfull 权重/原E2参数、page64/chunk8192/extra_buffer、TP1、N1、context131072/KV131072；原版Renderer完整prompt、random stand-in仅输出4token。三套：smoke6、reminder_heavy70、strict_append67。当前控制与004候选各跑三套；004栈开关关闭另跑smoke，对照旧stock/v1.1原始逐请求收据。对齐ID、prompt SHA、token数及预测后才计算结果。

有效批次远程证据根：控制组 `/sjtu/linhang/arena/runs/E2b_T26_20260922_v2/`，候选/关闭组 `.../E2b_T26_20260922_v3/`；本地摘要：`evidence/T26/`。两臂采用同一只读trace记录startup/flush池、tail切点，故本轮只解释cache机制，不作TTFT/SLO收益声明。脚本 `run_e2b_batch.py` 串行启动并在finally中停止它创建的进程组；不操纵已有daemons。

首次`E2b_T26_20260922/`遗漏了回放器的`make_case_sets.py`依赖，smoke尚未发请求便报import错误，服务由finally停止。原证据保留；v2补齐并先用`--plan-only`成功验证依赖/渲染，再启动新批次，没有覆盖初次失败。

v2控制组三套全部完成，但批次收尾检查器误要求flush字典严格等于`{"success":true}`，把D0合法的额外`message`字段误判；实际HTTP成功，trace有successful flush，服务已自动停止。修复只接受JSON布尔`success is True`、允许可选字段，新增正反例测试。没有重跑/覆盖控制组；v3仅续跑candidate/off。比较器显式记录两个run root并逐请求核对，不将中断的批次误称完整。

## 6. E2b完成（10:11:56 UTC）

`evidence/T26/comparison.json`：三套143个不同请求的控制cached_tokens与旧v1.1完全一致。004相比控制，smoke3好/3同/0差、reminder27好/43同/0差、strict0好/67同/0差；候选143/143精确符合role_conservative预测。fast实际未命中p95分别6315→562、8135→2439、3747→3747。关闭开关smoke6/6等于stock；候选trace确认27次tail split。

D1-01/02/03通过；候选28次flush均恢复KV131072/Mamba512/request8，D1-08仅本TP1/N1配置通过。完整用例映射、失败重试保留、命令和证据见experiments E2b。最终55 CPU tests中54通过、1已知alignment缺陷expectedFailure；两卡已释放、31000无监听。

**审阅结论不变**：004补齐了当前替身cold-chain头的checkpoint缺口，但§2 major未修，D1-04旧FAIL本轮未重测/未解除，003/DSA/DP/HiCache/lazy/MTP也未验证。因此这是缓存机制证据，不是镜像B的完整正确性批准。后续是否修补/启用由协调方决定，本轮到此idle。
