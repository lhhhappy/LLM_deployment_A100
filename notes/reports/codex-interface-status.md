# 当前接口实现与证据

2026-09-24，主会话按 task.md 和原始记录复核。当前为 SGLang 原生 `/generate` 与 `/flush_cache`，在引擎进程内修补；不使用 vLLM endpoint 插件，不加前置代理。

## 已实现

- 原生 `/generate` 直接接收 text / input_ids；prompt、cached、completion token 计数来自引擎原生 metadata，不从输出字符串或 TTFT 反推。
- 000 补丁在 ASGI 入口、消费 body 前记录 monotonic 接收时刻，通过 request scope 交给原生时间统计；覆盖客户端传入的 received_time。启动启用 metrics，输出服务端首 token 时刻；不扣去排队时间。
- 000 将 `/flush_cache` 改为 JSON：成功200且 `success:true`，失败400且 `success:false`。返回结果必须非空且全部worker响应成功，不只看第一个响应。没有伪造成功或绕过实际 radix cache reset。
- 启动启用 `--incremental-streaming-output`；输出累计 token 计数和原生 ignore_eos 行为保留。
- 原 `s1-dev/run_dev.py` 顺序是 preflight → warmup → flush → 正式测量。外层 checked runner 只加固 flush 失败即终止，未修改只读原harness或负载。

源代码：[000补丁](../../patches/000-interface-compliance.patch)、[启动器](../../scripts/pod/lib.sh)、[checked runner](../../scripts/pod/verify/run_dev_checked.py)。

## 已有实测记录

主会话本轮直接读取047/048完整raw，重新逐条核数：

| 检查 | 047现有正式配置 | 048现有配置+122 |
|---|---:|---:|
| 请求条数 / 唯一请求 | 722 / 722 | 722 / 722 |
| coverage | 100% | 100% |
| 请求错误 | 0 | 0 |
| prompt_tokens等于冻结glm_tokens | 722 / 722 | 722 / 722 |
| output_tokens等于max_output_i | 722 / 722 | 722 / 722 |
| TTFT来自服务端 | 722 / 722 | 722 / 722 |

[逐项复核汇总](../../evidence/interface-audit-20260924/047-048-recheck.json)、[047原评分](../../evidence/L047-official_a_n22/N22/score_formal.json)、[048原评分](../../evidence/L048-official_a_122_n22/N22/score_formal.json)。这说明接口记录通过这些检查，两个N22档仍因TTFT门失败，不是接口检查通过就算晋档。

缓存功能另有早期T29记录：连续长前缀请求cached_tokens从0增加到90112/90176/90624，清缓存返回200/true，之后冷请求cached_tokens=0；忙时400、等待空闲后200也有收据。该记录是TP1功能诊断、前缀递增长链，并非当前TP8上同一完整prompt两次的专项复测，不混写范围。[缓存记录](../../evidence/T29/first/completion_report.json)、[忙碌/等待清理记录](../../evidence/T29/first/if08.json)。

## 本轮修正与边界

客户端此前会接受未明确拒绝的2xx回包，检查比题面宽；已改为必须解析JSON且 `success is True`，空正文、空JSON、纯文本、数字1、字符串"true"均拒绝。checked runner保存HTTP状态与JSON回包；相关14项评测工具CPU回归通过。新056/057已同步此版本。旧047/048收据只保存flush_success和时间，没有保存回包全文，不追写历史证据。

代码/graph缓存不清。TP8的混合缓存池由引擎实际reset负责；不能仅凭单条请求cached_tokens=0证明每一种DP/HiCache/L3配置都清理正确。当前配置没有启用这些替代存储层，早期多worker失败归并有CPU反例检查，未宣称完成所有多worker故障注入。

生成内容允许变化。用户已取消主会话自设的token/logprob重复一致性门；接口计数、清缓存、请求完整性和赛题评分继续检查。056仅加171、057仅加172均已完成原开发集N22：各722/722接口计数与服务端TTFT检查通过、0请求错误，flush收据HTTP200/JSON success=true，均为TTFT性能门VALID FAIL。[056完整复核](../../evidence/L056-official_a_171_n22/N22/comparison-verification.json)、[057完整复核](../../evidence/L057-official_a_172_n22/N22/comparison-verification.json)。
