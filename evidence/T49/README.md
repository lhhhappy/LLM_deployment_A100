# T49 本地只读审计证据

- 报告：`research/codex/R18_cache_loss_and_capacity.md`。
- `local_audit.json`：原方法AST执行结果、cohort五条链首定位、每槽/每token字节、固定预算ratio扫算、N26情景、graph预算算术、输入SHA256。
- T49-01：直接取底包 `SchedulerPoolStatsObserver._get_mamba_token_info`，以SimpleNamespace只替代pool接口。KV free=0/evictable=471680/capacity=943360，Mamba free=0/evictable=560/capacity=584；实际方法输出usage=0.5/0.04109589。
- T49-02：直接取底包 `ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`，只去类型标注并替代scalar配置、返回结构和`.item()`。prefix=36864、extend=8136、page=64、普通extra_buffer，branch=None与38912分别输出tracked深度44992与38912。未执行GPU或整个cache生命周期，不能冒充真实请求复现。
- T49-03：读取原requests.jsonl/cohort；五条prompt长度与N6摘要唯一对应，均cohort idx=0。算术使用943360 KV token、584+1状态槽、34 KDA/11 DSA、SSM fp32、conv bf16、KV bf16。
- 本轮未改变底包/引擎/补丁/数据集；无服务、GPU、8卡或网络访问。日志/JSON是分析产物，不是性能实验。
- Claude随后回传`remote/012_{dev_raw__.jsonl,dev_run__.json,server.log,job.log}`。本会话只读本地文件，没有访问pod。原日志不改；包含104个CR进度更新，分析引用按LF行号（与`rg -n`一致）。
- T49-08：`remote_analysis.json`核对722 raw、完整measurement、19对前驱；原Renderer与tokenizer重渲染36个prompt，长度全部匹配。19为18 intra+1 turn_start，按实际prompt LCP仍>4096的18条；详细事实与INFERRED分类在R18§4.3。`previous_prefill_slices.txt`是相关前驱的批日志候选窗口，含同秒其它请求，不能冒充逐rid树trace。
- `boot_memory_audit.json`：载权39.15GiB、两池21.225GiB、实际graph1.31GiB及bs116；f=0.77805、VLM折减后的graph64/状态200条件算术、完整N6峰值外推。它修订`local_audit.json`中不含VLM的通用graph算式与早期50%压力情景；保留原算式收据，不作配置实测。
- `remote_pairs.json`保留最初按`前prompt长度−cached>4096 && phase=intra`筛出的23条诊断候选；该口径不扣实际prompt分叉，最终采用`remote_analysis.json`的19对及真实LCP，不能把23称为最终异常数。
- `final_validation.json`：源码/输入SHA不变、19对/18条实际差值、5个chunk网格例、LF日志引用、显存账闭合及78个显式文件:行号存在性检查通过；不替代GPU验证。
- 尚缺：逐节点eviction/track/role-slot日志、运行中激活高水位。T49-04…07是交Claude的8卡验证方案，本轮未执行。源方法CPU夹具和重渲染不等于完整服务/数值/SLO通过。
