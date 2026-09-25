# 078配置正式提交：attempt46364

用户明确选择078并授权立即提交。2026-09-25 15:18 UTC已上传成功，attempt **46364**、job **24605**；首次状态查询为queued，尚无能力或压测成绩。只创建一次attempt。

- 镜像沿用registry.dp.tech/dptech/dp/native/prod-4727808/4650601/lh-img:0925a；源码759a6ebb8e31723519ad5daf438e26e24b32501a。
- 相对正式46251唯一配置变化：SGLANG_AX_SCHED_COLD_CAP从4096改为6144。镜像、command、其余env逐项一致，未加入123、120对齐修复或124/125。
- 目的：观察本地增加冷prefill份额的chain/fast/TPOT取舍能否在线上重现，优先为后续迭代取得可解释的信息。
- 平台仅返回最高通过档指标；不从最高通过档猜测失败档的具体失败门。若档位仍为N22，可与46251同档指标比较；若不同档，不直接拿p95差值归因。

[实际配置](candidate.json)、[单变量核对](config-audit.json)、[打包审计](bundle-audit.json)、[上传收据](upload-receipt.json)、[当前状态](submission-state.json)、[首次平台查询](official-status-initial.txt)。三份包内引擎配置副本逐字节一致，配置SHA256 d87f0ba80f43cb8df94c3ef54f5d0a25a8ec60d1806d75fb9368462df03f1348。

075–080短测已全部结束，开发Pod服务保留，队列running/pending均为空；未追加081。0925b/165463排序镜像仅备用，没有提交。
