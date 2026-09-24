# 061s 预热资源复核（2026-09-24）

独立采集：[snapshot.json](snapshot.json)，采集代码：[collect.py](collect.py)。
原始来源 `/tmp/ax/runs/061s-official_a_122_full_n30_70m/` 的 `server.log`、`N30/gpu_util.csv`、`metrics.jsonl`、`flush_evidence.json`、`warmup.log`。
采集时仍在预热，正式测量清缓存尚未执行（`flush_success=false`；不是清缓存失败）。
CPU反例：[short-reserve-counterexample.json](short-reserve-counterexample.json)，使用冻结引擎c92acd5真实调度器与假请求/池。

## 已核实

- 13:24:33–13:53:29 UTC，8卡各347个约5秒间隔采样；各卡显存采样峰值69946–70446 MiB，跨卡最大70446 MiB=68.80 GiB。以81920 MiB计，采样时最少剩11.21 GiB；不能写成整个运行从未使用11.2 GiB。
- KV池容量1,036,288 token，采样不可淘汰峰值1,034,880。KV available/evictable/used要在同一时刻相加，不能把峰值相加。
- KDA池总321槽。不可淘汰used峰值84、evictable峰值299发生在不同时刻；采样末为used20、evictable299、available2，合计321。
- 实际 `mem_fraction_static=.7885`、`mamba_full_memory_ratio=.9`、`max_running_requests=32`、MTP草稿4。无retraction采样计数，不代表无淘汰或无重算。
- 采样末运行4、排队25，冷prefill连续64 token。冻结代码反例：8192预算、等待命中需8192，旧122先预留8192、却保留64续块，命中自己放不进批；下一轮重复。8128命中能与64续块一起入批。

## 显存预算推算（不是实测容量）

源码 `kv_cache_configurator.py::_profile_available_bytes/_handle_max_mamba_cache` 会把预算分给KV与KDA，MTP中间状态另计。
按载权前可用约77.73–77.87 GiB、KDA槽约17.6 MiB、slot/request比5、draft=4、最大请求数32已饱和的当前分支：

- .7885→.87：新增静态预算约6.33–6.35 GiB；同时增加约97个状态槽，留给KV的增量约4.67 GiB，按约13.9KB/token粗估36万token。
- .7885→.89：新增约7.89–7.90 GiB，同时增加约120个状态槽，KV粗估45万token。

数值有页对齐、启动分配与实现字节口径误差，以新启动日志的实际KV/状态池为准。保留的动态空间是否足够，须看测量阶段和更大形状，不能由预热的5秒采样保证。

## 处理

061s已停止，063s未启动即撤下；两项均无70分钟测量成绩。
122两类预留缺陷的修复在759a6eb，CPU通过；此次准备的064/065/066仍冻结c92acd5且关闭122。
