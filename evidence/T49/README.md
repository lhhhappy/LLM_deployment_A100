# T49 本地只读审计证据

- 报告：`research/codex/R18_cache_loss_and_capacity.md`。
- `local_audit.json`：原方法AST执行结果、cohort五条链首定位、每槽/每token字节、固定预算ratio扫算、N26情景、graph预算算术、输入SHA256。
- T49-01：直接取底包 `SchedulerPoolStatsObserver._get_mamba_token_info`，以SimpleNamespace只替代pool接口。KV free=0/evictable=471680/capacity=943360，Mamba free=0/evictable=560/capacity=584；实际方法输出usage=0.5/0.04109589。
- T49-02：直接取底包 `ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`，只去类型标注并替代scalar配置、返回结构和`.item()`。prefix=36864、extend=8136、page=64、普通extra_buffer，branch=None与38912分别输出tracked深度44992与38912。未执行GPU或整个cache生命周期，不能冒充真实请求复现。
- T49-03：读取原requests.jsonl/cohort；五条prompt长度与N6摘要唯一对应，均cohort idx=0。算术使用943360 KV token、584+1状态槽、34 KDA/11 DSA、SSM fp32、conv bf16、KV bf16。
- 本轮未改变底包/引擎/补丁/数据集；无服务、GPU、8卡或网络访问。日志/JSON是分析产物，不是性能实验。
- 尚缺：任务012 raw/run、实际启动与measurement完整日志、有效配置；T49账本已请Claude取回。没有原始19条ID时，不生成猜测的逐请求归因表。
