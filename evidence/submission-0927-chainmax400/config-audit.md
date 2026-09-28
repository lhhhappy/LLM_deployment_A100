# 46676 → chain-max 16k / Mamba400：逐项配置核对

**离线检查 PASS；正式未上传。TP8 候选机制行和 N30 结果尚待运行。**

引擎 `20a58da99737c70f21c801ff587e94d8f54022be`；镜像 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0927a`，构建 `166567`。

与复核过的 N30 job：性能参数一致；DCP=1 与本地续算关闭显式固定。正式关闭逐请求前缀决策日志、HTTP access INFO，保留启动机制行、30 秒摘要和错误；未改正文、thinking 或输出预算。

`SHORT_TOKENS=2048` 是准入阈值，126 关闭，**不是保证预留 2048 token**。

| 类别 | 字段 | 46676 原始值 | 候选值 | 变化 |
|---|---|---|---|---|
| command | `--chunked-prefill-size` | `<unset>` | `16384` | 变更 |
| command | `--cuda-graph-max-bs` | `32` | `32` | 相同 |
| command | `--dcp-size` | `<unset>` | `1` | 变更 |
| command | `--dsa-decode-backend` | `tilelang` | `tilelang` | 相同 |
| command | `--dsa-prefill-backend` | `tilelang` | `tilelang` | 相同 |
| command | `--enable-hierarchical-cache` | `flag present` | `flag present` | 相同 |
| command | `--enable-metrics` | `flag present` | `flag present` | 相同 |
| command | `--hicache-size` | `64` | `64` | 相同 |
| command | `--hicache-write-policy` | `write_through` | `write_through` | 相同 |
| command | `--host` | `0.0.0.0` | `0.0.0.0` | 相同 |
| command | `--incremental-streaming-output` | `flag present` | `flag present` | 相同 |
| command | `--kv-cache-dtype` | `bfloat16` | `bfloat16` | 相同 |
| command | `--linear-attn-backend` | `triton` | `triton` | 相同 |
| command | `--linear-attn-verify-backend` | `triton` | `triton` | 相同 |
| command | `--log-level-http` | `<unset>` | `warning` | 变更 |
| command | `--mamba-radix-cache-strategy` | `extra_buffer` | `extra_buffer` | 相同 |
| command | `--max-mamba-cache-size` | `<unset>` | `400` | 变更 |
| command | `--max-running-requests` | `32` | `32` | 相同 |
| command | `--mem-fraction-static` | `0.87` | `0.87` | 相同 |
| command | `--model-path` | `/mnt/models` | `/mnt/models` | 相同 |
| command | `--page-size` | `64` | `64` | 相同 |
| command | `--port` | `8000` | `8000` | 相同 |
| command | `--prefill-decode-interval` | `2` | `2` | 相同 |
| command | `--reasoning-parser` | `glm45` | `glm45` | 相同 |
| command | `--schedule-policy` | `lpm` | `lpm` | 相同 |
| command | `--served-model-name` | `default` | `default` | 相同 |
| command | `--speculative-algorithm` | `NEXTN` | `<unset>` | 变更 |
| command | `--speculative-draft-model-path` | `/mnt/models` | `<unset>` | 变更 |
| command | `--speculative-eagle-topk` | `1` | `<unset>` | 变更 |
| command | `--speculative-num-draft-tokens` | `4` | `<unset>` | 变更 |
| command | `--speculative-num-steps` | `3` | `<unset>` | 变更 |
| command | `--tool-call-parser` | `glm47` | `glm47` | 相同 |
| command | `--tp-size` | `8` | `8` | 相同 |
| env | `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS` | `154827,154829` | `154827,154829` | 相同 |
| env | `SGLANG_AX_ASYNC_TOKENIZE` | `0` | `0` | 相同 |
| env | `SGLANG_AX_BACKLOG_COLD_CAP` | `8192` | `16384` | 变更 |
| env | `SGLANG_AX_BACKLOG_GATE` | `0.10` | `0.10` | 相同 |
| env | `SGLANG_AX_BACKLOG_HIGH_S` | `15` | `15` | 相同 |
| env | `SGLANG_AX_BACKLOG_INTERVAL` | `0` | `0` | 相同 |
| env | `SGLANG_AX_BACKLOG_LOW_S` | `5` | `5` | 相同 |
| env | `SGLANG_AX_BACKLOG_MAX_SLOW` | `80` | `80` | 相同 |
| env | `SGLANG_AX_BACKLOG_RELIEF` | `1` | `1` | 相同 |
| env | `SGLANG_AX_CHAIN_RISK_CHUNK` | `<unset>` | `0` | 变更 |
| env | `SGLANG_AX_CHAIN_RISK_INTERVAL` | `<unset>` | `1` | 变更 |
| env | `SGLANG_AX_DCP_LOCAL_EXTEND` | `<unset>` | `0` | 变更 |
| env | `SGLANG_AX_DEADLINE_CHAIN_FIRST` | `<unset>` | `1` | 变更 |
| env | `SGLANG_AX_DEADLINE_FAMILY` | `<unset>` | `0` | 变更 |
| env | `SGLANG_AX_DEADLINE_FREEZE_CLASS` | `<unset>` | `1` | 变更 |
| env | `SGLANG_AX_DEADLINE_LOAD` | `<unset>` | `1.05` | 变更 |
| env | `SGLANG_AX_DEADLINE_TIERS` | `1` | `1` | 相同 |
| env | `SGLANG_AX_DEADLINE_WARM_S` | `15` | `15` | 相同 |
| env | `SGLANG_AX_DSA_SPARSE_TRITON` | `<unset>` | `0` | 变更 |
| env | `SGLANG_AX_INDEXER_ROW_SHARD` | `1` | `1` | 相同 |
| env | `SGLANG_AX_KDA_DUAL_SNAPSHOT` | `0` | `0` | 相同 |
| env | `SGLANG_AX_KDA_FUSE_PROJ` | `0` | `0` | 相同 |
| env | `SGLANG_AX_MOE_FUSE_SWIGLU` | `0` | `0` | 相同 |
| env | `SGLANG_AX_PACE_TPOT` | `0` | `0` | 相同 |
| env | `SGLANG_AX_PREFIX_PRODUCER` | `<unset>` | `1` | 变更 |
| env | `SGLANG_AX_PREFIX_TRACE_ROUNDS` | `<unset>` | `0` | 变更 |
| env | `SGLANG_AX_PREFIX_TRACE_S` | `<unset>` | `0` | 变更 |
| env | `SGLANG_AX_SCHED_COLD_CAP` | `6144` | `16384` | 变更 |
| env | `SGLANG_AX_SCHED_COLD_CAP_MAX` | `<unset>` | `0` | 变更 |
| env | `SGLANG_AX_SCHED_PROTECT` | `1` | `1` | 相同 |
| env | `SGLANG_AX_SCHED_SHORT_TOKENS` | `8192` | `2048` | 变更 |
| env | `SGLANG_AX_SM80_FP8_MOE_HUMMING` | `1` | `1` | 相同 |
| env | `SGLANG_AX_SM80_FP8_MOE_MARLIN` | `1` | `1` | 相同 |
| env | `SGLANG_AX_SM80_INDEXER` | `1` | `1` | 相同 |
| env | `SGLANG_MAMBA_SSM_DTYPE` | `float32` | `float32` | 相同 |
| env | `SGLANG_OPT_DEEPGEMM_HC_PRENORM` | `0` | `0` | 相同 |
| env | `SGLANG_OPT_FUSED_KDA_VERIFY` | `0` | `0` | 相同 |
| env | `SGLANG_OPT_USE_TOPK_V2` | `0` | `0` | 相同 |

未设置不是推测为零：46676 未显式钉 Mamba 池和 chunk；其 8k chunk 等有效默认值应以对应运行日志为准。去 MTP 后 101 角色边界机制恢复，旧 MTP 路径会清除此 env；这也属于候选组合差异。


文件：`submission.json` 是唯一候选配置；`baseline-46676.json` 是已上传 46676 配置副本；`n30-job-reviewed.sh` 是 fable 已复核任务快照；`expected-mechanisms.txt` 仅是待核验期望。
