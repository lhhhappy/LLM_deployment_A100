# a / b / c 方案独立复核（2026-09-24）

已读 Claude 的 80abc38「待执行」计划，接收当前用户转交。061s已停、063s已撤（pread核查）。本轮122关闭，修复由Claude单独进行。

## 结论与修正

1. 停122有根据。061s采样服务日志反复出现64-token单请求prefill、25条排队、仅4条运行。冻结c92的 `_ax_pace_limits` 会把短命中预算累计到整个budget，随后冷续块只剩grid=64。CPU反例中8192-token短命中连续三步未入批，冷请求0→64→128→192；8128-token对照可以入批。不能把平均预算8080误当成冷请求实际块长。
2. 以上证据来自**预热**：snapshot 的 flush_success=false、runner_rc=null。它足以定位调度故障，但不是正式70分钟测量成绩。347份5秒GPU采样最高70446MiB，不证明采样之间或后续长期没有更高峰值。
3. 同提交c92、正式A参数、122关闭的a是迁移源码参数基准。已知180关闭时仍改变少量未对齐KDA检查点行为，不能称为旧镜像逐路径完全原样；这不破坏同提交a/b/c之间的参数对照。
4. 参数逐项核对通过：a→b只加mem-fraction-static=.87，b→c只加HiCache三个参数；三项env和提交一致、MTP保留，N30/全量311链5601请求/4200秒准入后排空一致。
5. 额外显存不能全部除以13.9KB算KV：`kv_cache_configurator.py::_handle_max_mamba_cache` 同时分配KDA主状态，MTP中间池按请求上限计算。报告中+45万token不是已证实的容量增量；以065实际KV容量、KDA槽和运行高水位为准。
6. b/c作为正式校准候选；启动/机制核对/前20分钟无错误是提交前可观测条件，不是长期无OOM、恢复数值正确或N22/N26的保证。20分钟从真正测量开始计，不能把启动和预热算进去。按已有用户授权推进，记录已知未验证项，不恢复逐字生成一致性门。

## 证据

- `evidence/L061s-official_a_122_full_n30_70m/memory-audit/snapshot.json`
- 同目录 `short-reserve-counterexample.json`
- c92acd5 的 scheduler.py `_ax_pace_limits`
- engine/sglang/srt/mem_cache/kv_cache_configurator.py `_handle_max_mamba_cache`
- scripts/pod/jobs/official_a{,_mem087,_180_mem087}_full_n30_70m.sh

队列064(a)、065(b)、066(c)；当前状态以 notes/queue.md 和pread为准。
