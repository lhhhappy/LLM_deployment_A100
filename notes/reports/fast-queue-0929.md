# Fast 五小时参数队列（2026-09-29）

用户授权五小时排队，强调清晰鲁棒记录。固定47266引擎bf6b66fa、rot150冻结数据、N34每组3600秒准入后排空；不自动N38，总测量5h，预计含启动/预热/排空5.5–6h，异常可能改变时长。MTP关闭、DCP1、池400、118/分片/MoE/KDA维持。只有TP8一组资源，串行。

|顺序|任务|相对47266改动|
|---|---|---|
|130eznw1|130eznw1-tail_rot150_n34_fast5h_short4096.sh|{"SGLANG_AX_SCHED_SHORT_TOKENS=2048": "SGLANG_AX_SCHED_SHORT_TOKENS=4096"}|
|130eznw2|130eznw2-tail_rot150_n34_fast5h_pdi1risk0.sh|{"--prefill-decode-interval 2": "--prefill-decode-interval 1", "SGLANG_AX_CHAIN_RISK_INTERVAL=1": "SGLANG_AX_CHAIN_RISK_INTERVAL=0"}|
|130eznw3|130eznw3-tail_rot150_n34_fast5h_cold12k.sh|{"SGLANG_AX_SCHED_COLD_CAP=16384": "SGLANG_AX_SCHED_COLD_CAP=12288"}|
|130eznw4|130eznw4-tail_rot150_n34_fast5h_short4096_cold12k.sh|{"SGLANG_AX_SCHED_SHORT_TOKENS=2048": "SGLANG_AX_SCHED_SHORT_TOKENS=4096", "SGLANG_AX_SCHED_COLD_CAP=16384": "SGLANG_AX_SCHED_COLD_CAP=12288"}|
|130eznw5|130eznw5-tail_rot150_n34_fast5h_control.sh|{}|

第二组是普通间隔与131风险间隔联动的组合，不是单参数；normal1/risk1会报错，已验证normal1/risk0/relief0通过原配置解析。第四组显式测试交互；第五组同条件基线，用来识别运行波动。各组继续产生promotion.json，但LADDER只有34，因此任何结果都不会启动38。

参考为eznu一小时N34的339个chain ID、6条超标，源raw SHA=8a4481841d62a73d50a76013a3f97dd1db73ab3459ae16380352c3494fd43f9a。提取核对原收据桶数、超标数、完整行数、ID唯一性和SHA；参考内容冻结到各job。若缺参考ID，配对结论不成立，不能冒称无新增。最终还须与第五组逐ID复算fast修好/新坏，不能直接比较不同完成量的超标总数。

每组先核固定commit、有效机制和参数日志，真实权重冒烟，再rep16预热、真实flush、原runner和评分器。timed_verdict保留完成数/错误/四桶超标允许数/TPOT/源raw哈希；promotion保留chain新增/修好/缺ID。退出码0只可能表示诊断闭合，不代表性能门通过；不得仅按done判定成功。失败组退出后继续独立下一组；引擎/证据无效与性能未过分别记录。基于现有runner不另造控制器。

发布使用与上一批逐字相同的冻结runtime；仅替换job和chain参考，未改引擎、评分或数据。发布器先暂停领取，publish断言无运行/陌生pending，成功后恢复；发布失败也尝试恢复，日志持久保存到GPU机runs/jobs/queue-0929-fast5h.log。后台tmux不依赖本地联网。

发布前容量快照：/tmp/ax真实路径/dev/shm/arena-runtime/ax，tmpfs空闲596GB，memory.failcnt=0，8卡空闲0%；不新增Pod、模型或第二套底包。仍遵守20Gi临时存储限额。
