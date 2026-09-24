# 000 — 接口合规：现行实现

已实现，现有正式配置与本地8卡实验均包含。依据只读 `build/base_exact/sglang/` 和 [task.md](../llm-challenge-arena-v1/task.md)。当前补丁SHA256：`e0d7924989ebb538e331110f4458ccd4e95a0bfa3cbd87dfb8324ac0328960f6`。

## 实现

| 项目 | 当前行为 |
|---|---|
| `/generate` | SGLang原生路由，引擎进程内处理；没有新增代理，不使用vLLM endpoint插件 |
| token计数 | 原生metadata提供prompt_tokens、cached_tokens与累计completion_tokens，不从文本长度估算 |
| 请求入口计时 | ASGI middleware在消费JSON body之前保存perf_counter到私有scope；typed handler传给原生时间统计，覆盖客户端received_time。应用入口不等于网络首字节 |
| 首token计时 | 保留原生prefill_finished_time；启动启用`--enable-metrics`，不扣排队时间 |
| SSE | 启动启用`--incremental-streaming-output`；累计completion_tokens语义保留 |
| 输出长度 | 原生ignore_eos；实际输出必须满足请求max_new_tokens |
| `/flush_cache` | 真调用引擎清缓存；成功200+JSON success:true，失败400+success:false |
| flush结果归并 | 收到非空结果且每个worker都成功才返回成功；任一失败不清tokenizer本地多模态预处理缓存；空结果算失败 |

不清代码缓存和CUDA graph。引擎忙碌时仍拒绝清理；timeout参数允许等待空闲，不能把失败或超时当作清理成功。worker错误字符串中的编号是响应到达顺序，不能当作真实rank。

## 验证与记录

最新实现、047/048逐请求复核和证据范围统一见[接口状态报告](../notes/reports/codex-interface-status.md)。047/048各722请求、coverage100%、错误0；prompt计数与冻结值、output计数与预算均逐条相等，全部TTFT采用服务端时戳。两档仍有TTFT性能失败，不因接口通过而视为晋档。

早期[T29清理收据](../evidence/T29/first/if08.json)验证忙时400、等待空闲后200、清理后冷请求cached_tokens=0；[长前缀记录](../evidence/T29/first/completion_report.json)验证实际复用和清空。它们限TP1功能诊断，不能冒充当前TP8全部配置验证。[CPU反例](../evidence/T29/cpu_regression.log)包含全成功、混合失败、空结果与body前计时。

原harness保留preflight→warmup→flush→正式测量。checked runner现严格要求2xx且JSON success为true，失败中止；本轮补齐了此前过宽的客户端回包检查。通信器仍依赖响应收齐，客户端有独立HTTP截止；未完成DP故障注入或HiCache/L3专项验收，不启用这些存储层。
