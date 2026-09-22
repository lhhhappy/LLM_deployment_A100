# 160 — GLM NEXTN sm80兼容配置与可加权spec统计（T48/W22）

完整路径/行号/不兼容清单、缓存交互、INFERRED性能与容量模型、L2闸门：[`R17_nextn_sm80.md`](../research/codex/R17_nextn_sm80.md)。原始证据：[`evidence/T48/`](../evidence/T48/README.md)。

## 应用与实现

只在只读底包的副本上生成；顺序：000→101→105→110→111→112(v2)→113(v2)→140→120→130→150→**160**。`python3 scripts/make_160.py`生成确定性补丁；`python3 scripts/verify_160.py`核对全栈fuzz0、全Python编译、再次生成相同、应用树一致、完整反向还原与base_exact未改。

新文件`srt/arg_groups/ax_mtp_sm80.py`，在EAGLE参数auto_params解析之后调用。仅GLM target、sm80、NEXTN别名EAGLE触发：

- draft path必须显式给出，topk=1；不改max_running_requests、spec步数、接受阈值、采样策略或用户输出。
- 声明两DSA后端tilelang、KV bfloat16、KDA prefill/decode/verify Triton。
- 开启110/111 sm80分支，112/113从110代理自动复用；关闭DeepGEMM HC prenorm和TOPK V2计划优化。
- 启动期关闭101角色IDs和140双点快照，防止140的spec守卫拒绝以及未经验证的状态所有权组合；原extra_buffer、真实prefix缓存、verify tracking/rollback不变。140槽与spec scratch独立，不能称为已经证实别名冲突。
- 原生spec日志新增`spec tokens: <int>, spec rounds: <int>`，取清零前同窗口原始计数；其余计数/metrics/采样不改。非spec日志不新增字段。

无需复制新的sm80计算kernel：110已经覆盖spec多pool FP8写入，111覆盖draft专家，112/113覆盖所有MQA代理调用。此补丁的收益来自让NEXTN正确选路；算子加速贡献仍归110–113，不归160。

## 启动/采集

未执行的完整任务：`scripts/pod/jobs/dev_b160_mtp_n6.sh`。唯一源码名`b160_mtp_s3_k1_d4_mr32_n6`，显式steps3/topk1/D4、MR32/graph32。MR32为后续N22/26保留cap空间，主池足够且MR起效时相比默认48少约1.10GiB/rank verify scratch；不保证N@SLO。120/130固定off；150 warmup对spec跳过，所以不传ax_shapes。未加入RELEASE、镜像构建或队列。

```
python3 scripts/extract_spec_stats_160.py server.log --draft-tokens 4 --out spec.json
```

`accept len`包含每request-round保证的1个target token；`accept rate`只计正确draft。加权长度=`Σspec_tokens/Σspec_rounds`，D4接受率=`(Σtokens−Σrounds)/(3Σrounds)`。旧日志无原始计数时仅保留逐窗口值，不算伪精确平均。默认只取TP0/无rank行；每个日志应只含一个服务进程/一次测量，重启/多个测量先裁剪。`--start-line`辅助切片；不能用最后一行的#running-req替代窗口round计数。未打印的最后窗口不可恢复。任务脚本输出`spec_harness.log`含warmup，纯measurement需按harness时间进一步裁剪。

## 验证与限制

CPU策略/日志测试、真实ServerArgs解析及draft ModelConfig/quant mapper检查、A100随机算子数值/graph、12补丁栈收据见证据索引。没有加载完整checkpoint、运行TP8 collective、完整EAGLE三runner、真实模型输出/能力、overlap压力、flush服务或N@SLO。C++ AOT库版本与L3需Claude复核。

首轮L2需同栈non-spec对照，两边101/140/120/130同为off；先验证权重与KDA状态、pool/flush和能力，再比较N6→22全部SLO。120 on另做消融，不能把接受长度1.6–2.0当作1.6–2.0倍实测加速。

回滚：重启时移除NEXTN参数恢复普通解码（同时手动恢复需要的101/140 env）；或在副本反向撤160。解析过程改的是进程env，不能热切。源码补丁撤回不会替正在运行的父进程恢复env。

统计口径：`accepted_tokens_including_target`是引擎spec接受计数（含target保证token），不是HTTP实际输出token数；EOS/stop截断可能使两者不同。meta_info仍由原服务逻辑报告。
