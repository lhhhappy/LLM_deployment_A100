# bf6b66fa 夜间单参数臂（独立只读，2026-09-28）

基线是 [0928c 冻结服务配置](../../evidence/submission-0928-kda/submission.json)，对照采用已闭合的 [eznq N34](../../evidence/L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N34/timed_verdict.json)；先重复同条件基线以估计波动。该 40 分钟 DRAINED 窗口的 TPOT p95=103.322 ms，四个 TTFT 估计条数门仍过；51/102 条慢 TPOT 在前 5 分钟派发。候选只改以下各自 **一个** 参数，不叠加，也不碰已否决的 133。

| 单参候选 | 当前→试验 | 冻结源码中的实际触发与作用 | chain 代价 / 既有证据 |
|---|---|---|---|
| 较保守：CLI `--prefill-decode-interval` | **2→3** | `bf6b66fa:engine/sglang/srt/managers/scheduler.py` 的 `_arm_prefill_decode_interval` 在一次 extend 后默认 arm 3 轮 decode；125 relieved 时仍被 `BACKLOG_INTERVAL=0` 覆盖，131 判断 cold 风险时仍 `min(3,1)=1`。`ax_deadline.chain_risk_config` 校验 `1<3`，可启动。只在未 relieved 且未被 131 判中的 prefill 间隙新增一次 decode。 | 可能拖长开场未被 131 识别的冷链首（131 还要求剩余≥32768、预测位于预算±8秒）；chain-first 排队优先级不保证解码 cadence。旧 L037 把 interval 提至16 曾严重损伤 chain，但非此配置、非3；未找到当前 bf6b66fa 上的 2→3 同条件试验。 |
| 更直接：env `SGLANG_AX_BACKLOG_INTERVAL` | **0→1** | `bf6b66fa:engine/sglang/srt/managers/ax_deadline.py` 的 `backlog_config` 接受1（配置 interval2）；`scheduler.py` 在 `_ax_backlog_relieved` 时改用1，新增每个 relieved prefill 后一轮 decode。eznq 的 [server.log](../../evidence/L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N34/server.log) 记录累计 **172 relieved prefill rounds**，证明不是无效开关。冷块 cap 仍16384，别的机制不变。 | chain 风险较高：131 设为1，`min(1,1)=1`，无法在 relieved 冷链首上再次减小间隔；开场长 prefill 可能等 decode。旧 v3/N26 L105 曾用 interval1+cap8192，L109 又一起改为更激进的0/更大 cap；两者非同配置单变量，只能作方向参考，不能当本候选已验证。 |

**不列第三臂 `--max-mamba-cache-size 400→320`。** `bf6b66fa:engine/sglang/srt/mem_cache/kv_cache_configurator.py` 中显式池大小会改变静态 Mamba 状态占用与 KV 预算，确实能命中；但 [N34 metrics](../../evidence/L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N34/metrics.jsonl) 在前 5 分钟 30 个采样的 `token_usage` 峰仅0.78、≥0.93为0，而慢 TPOT 已有51条。池400相对无此上限的 N30 同ID chain 12→7，是已保留的核心收益；320未见当前配置下测试。它不是今晚开场 TPOT 风险的优先旋钮，不能由全窗高 KV 水位 3.15%→8% 推断减池会救开场。

两臂的成功条件都先看同 ID chain 修复/新坏和 fast，再看同窗 TPOT p95、>100ms 数量、前5分钟/10分钟后分段及 125 relieved rounds。当前 47043 正式 N30 / 47266 待回与本地 N34 DRAINED 不混作同一分数。未改代码、队列或 GPU。
