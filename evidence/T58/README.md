# T58：实验035 N22独立CPU重评分

结论：按题目TTFT统计余量估算，11道门10过1败；唯一失败项TPOT p95=0.2961931135342048秒/token，要求≤0.10。TPOT均值0.10395581354618791，205/722条超过0.10。

| 门 | p95秒 | 超时数/允许数 | 估算判定 |
|---|---:|---:|---|
| fast_intra | 3.045406 | 17/23 | PASS |
| overall_intra | 5.104129 | 20/27 | PASS |
| turn_start | 7.815638 | 0/3 | PASS |
| chain_start | 38.627189 | 18/22 | PASS |
| TPOT | 0.296193 | 无统计余量 | FAIL |

其它6门：coverage=100%、harness_data=0、harness_render=0、engine_error<1%、infra_error<1%、gated_phases_have_samples，均通过。这里是开发集估算，不是线上正式成绩；原dev直接p95门仍有三项TTFT失败，原报告完整保留在score_formal.json的dev字段。

## 复现

在仓库根目录执行：

```bash
python3 scripts/score_formal.py --raw evidence/L035/raw_dev-combined-v1_N22_1790184432.jsonl --run evidence/L035/run_dev-combined-v1_N22_1790184432.json --out evidence/T58/score_formal.json
python3 -B evidence/T58/verify_integrity.py
```

输入是本轮已同步的L035文件，原文件未改。raw为922260字节，SHA256 `3f8bfa0d683b5ade8caba09d6829754693042589f607a15e7fc3fb7a21a7bd9b`。全部输入和scorer/harness源码SHA在integrity.json；评分器复用只读s1_score.evaluate及原分桶，并加TPOT硬门和TTFT统计估算。

## 完整性核对

722行/722唯一ID与原cohort的ID、chain_id、idx_in_chain逐项一致；只有measure namespace，无预检/预热行；attempted/dispatched均722。所有输出数量等于各自预算，prompt/cached计数范围合法，四类错误和error字段全空；722条都有有限的TPOT、完整服务端TTFT及单调时间戳，TTFT与first−recv重构差为0。

TPOT使用原始tpot_s：它来自首/末SSE的perf_counter。client_finish_at_s记录的是call_engine返回后的epoch时间，不能作为末SSE精确时间；初版核对脚本误要求两者相等而失败，读s1_loadgen.py:169、365–387后改为仅报告差异，不修改tpot_s或评分阈值。两种口径最大差0.000206612秒/token，epoch代理p95约0.296203，不影响结论。并非发现引擎计时bug。

最后完成请求为biomaster:canon:FWyXLCJ4mffNsaYz6LvcE:llm:0，prompt58788、输出2803，于2026-09-23 17:48:08.192 UTC完成，TPOT约0.011586；用户17:48:05的单请求decode快照是收尾。

验证只证明现有文件能重评分，不能补出未保存的逐SSE或清缓存响应；本轮没有GPU实验、引擎请求、队列操作、镜像或提交。
