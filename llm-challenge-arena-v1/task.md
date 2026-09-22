# 推理服务评测赛

## 导读

### 核心问题

假设你负责给一家公司上线 AI 助手。内测的时候只有你一个人在用，问什么都又快又顺；等到全公司都开始用，还是那 8 张卡，服务就开始塌方——有人等第一个字等半分钟，有人盯着光标一个 token 一个 token 往外蹦。

这时候你能动的东西其实很少。模型是定死的，卡是定死的，用户提什么问题也不归你管。你唯一能调的是**怎么把这个模型装进这些卡里，以及怎么调度涌进来的请求**。而同样的权重跑在同样的硬件上，调得好和调得差之间能差出成倍的服务能力——这部分价值完全由部署创造，和模型本身无关。

于是问题变成一句话：**在服务质量还没崩的前提下，这 8 张卡最多能同时接住多少路会话？**

### 为什么这件事不简单

本赛回放的负载不是「你好，请介绍一下自己」。它是真实的多轮 Agent 会话：带工具调用，上下文一轮一轮滚雪球，单条 prompt 动辄几万 token，而模型吐出来的答案往往只有几百 token。

这个形状很要命：

- 每一路会话都拖着一大块**必须一直留在显存里**的上下文，而模型权重本身已经吃掉了显存的大头；
- 会话是多轮的，相邻轮共享很长的前缀，**复用得好不好**直接决定要不要把几万 token 重算一遍；
- 用户体感同时由两件事决定：**等多久出第一个字**，和**出字有多快**。而这两件事在同一台机器上互相抢资源。

所以「多接一路」和「每一路都还好用」天然对立。并发拉满，所有人一起变慢；只服务一个人，指标漂亮但机器在闲置。工程水平体现在你能把这条取舍曲线往外推多远。

### 一句话

固定模型（GLM-5.3-Flash）、固定机器（8× `A100-SXM4-80GB`）、固定负载，比的是：**守住服务质量底线的前提下同时撑住尽可能多的会话；撑住的路数相同时，token 吐得尽可能快。**

前者是排名第一顺位的 `N@SLO`，后者是第二顺位的 `TPOT`。所有人跑同一个模型、同一批题、同一份负载，分差全部来自部署。

### 术语表

全文用到的缩写，先在这里交代清楚。精确的计算公式和取数位置见后面「指标口径」，这里只说是什么。

**服务质量指标**

| 缩写 | 全称 | 是什么 | 方向 |
| --- | --- | --- | --- |
| TTFT | Time To First Token | 首 token 延迟：从服务端收到请求，到吐出第一个 token 的时间。对应用户「敲回车之后等了多久才看见动静」 | 越小越好 |
| TPOT | Time Per Output Token | 每输出 token 耗时：首 token 之后，平均每吐一个 token 要多少秒。取倒数就是这一路的出字速度。也有人叫 ITL（Inter-Token Latency） | 越小越好 |
| TPM | Tokens Per Minute | 每分钟处理的 token 数，整机吞吐量。本赛分 `tpm_all`（输入加输出）和 `tpm_decode`（只算输出） | 越大越好 |
| SLO | Service Level Objective | 服务质量目标。本赛的 SLO 就是四道 TTFT 门加一道解码门，是「还算能用」的底线 | — |

TTFT 和 TPOT 合起来描述一次对话的完整体感：TTFT 是等待，TPOT 是出字的流畅度。两者都只管单条请求；TPM 管的是整机一共干了多少活。

**容量与统计口径**

| 术语 | 含义 |
| --- | --- |
| N | 并发档位，指**同时在场的逻辑会话数**。注意不等于 HTTP 并发连接数：一路会话在两轮之间有思考和工具执行的间隙，此时它仍占着这个槽，但没有在途请求 |
| N@SLO | 读作「N at SLO」，**守住 SLO 的最大并发档位**。平台沿梯子 `2 / 6 / 10 / 14 / 18 / 22 / …` 逐档实测，取全部硬门都过的最大那档。这是排名第一顺位 |
| p95 | 95 分位数。把该桶所有请求的耗时从小到大排，第 95% 位置上那个值——意思是「95% 的请求不慢于它」。门槛用 p95 而不是平均值，是为了让少数被饿死的请求也能被抓到 |
| 桶（phase） | 按请求在会话中的位置分组。链首那一条、每轮用户话的开头、链中间的短请求，难度天差地别，所以分开各设各的 TTFT 目标，不混在一起取 p95 |

**负载相关**

| 术语 | 含义 |
| --- | --- |
| 会话链（chain） | 一次完整的多轮 Agent 对话，从第一句话到结束，中间夹着若干轮工具调用。压测回放的基本单位 |
| prefill / decode | 推理的两个阶段。prefill 是把整段 prompt 读进去（决定 TTFT），decode 是一个一个往外吐 token（决定 TPOT）。本赛负载 prompt 极长而输出很短，两个阶段的压力很不对称 |
| 前缀缓存（prefix cache） | 同一条会话链相邻两轮共享很长的前缀，引擎把这部分算好的中间结果留着复用，就不用重算。命中多少由 `cached_tokens` 上报 |
| 固定轨迹回放 | 压测重放的是提前冻结好的真实会话：prompt 是渲染好的整段文本，输出长度由 `ignore_eos` 强制写死。这样所有选手承担的计算量逐条相同，成绩才可比 |

## 提交方式

把你的推理服务打成镜像，用 **Playground CLI** 交给主办方的评测平台（赛题 `llm-challenge-arena-v1`）。平台在固定机型上拉起服务、先打能力评测，过门槛后再做长时压测。这是 serving 评测：同一批题目、同一份会话负载，比的是你的部署能撑住多少路并发、token 流得多快。

不要自己构建评测平台，也不要提交公网 endpoint。Playground 怎么打包、轨迹怎么写，见下面「用 Playground 提交」。

## 目标

压测两项按顺位依次比较：**N@SLO**（能撑住的最大并发档位，越大越好）→ **TPOT**（每输出 token 秒数，越小越好）。前一项相同才看后一项。定义见「指标口径」，比较规则见「计分」。

能力评测、四道 TTFT、解码速度、错误率等全是门禁——不过门槛不进入压测，压测硬门不过则该档没有成绩。吞吐（TPM）只回报，不排名。

## 加入 arena team

你自己调参、打镜像、在乌兰上自测，必须先加入 Trisol team **`arena`**。没加入就没有配额，也起不了服务。卡型是 `w1` 的 8 张 `A100-SXM4-80GB`。

完整安装、企业登录、装 Trisol、加入 team 的步骤见飞书文档：[使用 bohr-cli 完成 Trisol 从安装到训练部署](https://dptechnology.feishu.cn/wiki/P05Kw4GQliZN3Mk2wHHcSRTcnFd)。

最短路径：

```bash
# 1. 装 bohr-cli（不要用旧的 curl 脚本）
npm install -g @dptech-corp/bohr-cli@latest

# 2. 企业账户登录：选企业版，企业代码 dpt，飞书授权
#    登录页 https://dpt.bohrium.com/login
bohr auth login
bohr auth status --verify

# 3. 装 Trisol 插件并登录
bohr extension install trisol
bohr trisol login          # 先打开企业登录页，再把终端链接贴进同一个浏览器
bohr trisol config use-team arena

# 4. 加入 arena（本赛要求）
bohr trisol team join arena
bohr trisol config use-team arena
bohr trisol quota list --team arena
```

`submission.json` 只交镜像和启动命令，不要写 `team` / `cluster`。自测一律用 team `arena`、集群 `w1`。

## Trisol 机器开发指南

Trisol 没有公网 SSH。要拿一台 8 卡机器交互开发（装依赖、对环境、试启动命令），就起一个只 hang 不 serving 的推理服务，再用 `exec` 进容器。

不要只写 `sleep infinity`：容器不听端口就过不了 Ready，服务起不来，也进不去。用一个听 8000 的 hang 命令，例如 `python3 -m http.server 8000`：

```bash
trisol inference create \
  --name my-hang-8gpu \
  --team arena \
  --cluster w1 \
  --gpu-model A100-SXM4-80GB \
  --gpu-count 8 \
  --model glm-5-3-flash:2 \
  --image-ref '<你的镜像>' \
  --command-line 'python3 -m http.server 8000 --bind 0.0.0.0' \
  --startup-timeout-seconds 3600 \
  --no-input
```

等到 `trisol inference get my-hang-8gpu --team arena` 的 status 是 `running`，再进机器：

```bash
# -i 才是交互终端；-t 是 --team，不要当成 tty
trisol inference exec my-hang-8gpu --team arena -i -- bash
```

进去之后可以 `nvidia-smi`、改启动参数、手动拉起 serving。用完立刻删，不然一直占 8 卡：

```bash
trisol inference delete my-hang-8gpu --team arena --yes
```

## 服务接口规范

容器起来后要满足：

- 在 `--port` 上提供 HTTP 服务，不需要公网、不需要 HTTPS、不需要你自己做鉴权；
- 请求的 `model` 为你写的 `model_name` 时暴露你要参赛的模型；`model_name` 不填就走默认值 `default`，此时服务要接受 `model=default`；
- 在整个评测窗口内保持可用，能力评测加压测合计可能超过 10 小时。

评测会打你的服务三个口，语义完全不同，必须都实现：

| 阶段 | 路径 | 协议 | 量级 |
| --- | --- | --- | --- |
| 能力评测 | `POST {base_url}/chat/completions` | OpenAI 兼容 | 200 题 |
| 压测 | `POST {engine_root}/generate` | SGLang 形 SSE | 每档整集回放一遍 |
| 清前缀缓存 | `POST {engine_root}/flush_cache` | JSON | 每档开测前 |

`engine_root` 是引擎根（例如 `http://host:8000`），`base_url` 通常是 `{engine_root}/v1`。**压测打的是引擎根，不在 `/v1` 下面**；平台拿到的 `base_url` 若带 `/v1`，会先剥掉再拼 `/generate`。

### 能力评测口 `/chat/completions`

标准 OpenAI 兼容请求，`model` 用你在 `submission.json` 里写的 `model_name`。三条要求：

- 最终答案必须落在 `choices[0].message.content`。用 reasoning parser 把思考过程拆到 `reasoning_content` 没问题，但评分只读 `content`，思考留在 `reasoning_content` 里不影响判分；
- 不要压输出预算。`aime26` 和 `gpqa-diamond` 的题需要完整推理，答案被截断直接算错；
- `GET {base_url}/models` 要返回 200，平台用它确认服务就绪。

### 压测口 `/generate`

压测打的是引擎根上的 `POST /generate`（SGLang 形），不是 `/v1/chat/completions`。原因是压测要做**固定轨迹重放**：prompt 是数据侧提前渲染冻结好的整段文本，输出长度由 `max_new_tokens` 加 `ignore_eos` 强制，这样基线和你的候选承担的 decode 工作量逐条相同。走 chat 口就要再过一次各家自己的 chat template，prompt 和 token 数当场变得不可比。

发压侧固定发下面这个形状（id 与正文已脱敏，结构与真实一致）：

```
POST http://<你的服务>:8000/generate
Content-Type: application/json
X-S1-Request-ID: <req_id>
X-S1-Session-ID: <会话 id>
X-S1-Cache-Namespace: <每档 N 不同>
X-S1-Routing-Key: <前缀族 id>
```

```json
{
  "text": "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>\n# Tools\n...(整段已渲染的 prompt，常达几万 token)...<|assistant|><think>",
  "sampling_params": {"max_new_tokens": 240, "temperature": 0, "ignore_eos": true},
  "stream": true,
  "rid": "<req_id>"
}
```

注意 `text` 里 **chat template 已经渲染进去了**——开头的 `[gMASK]<sop><|system|>` 和结尾的 `<|assistant|><think>` 都是模板产物。你的 `/generate` 实现**不能再套一层模板**，必须把这段文本原样喂给引擎，否则 prompt token 数对不上、前缀缓存也命中不了。`max_new_tokens` 是逐请求的，不是常数。

`X-S1-*` 头只是携带信息，实现可以忽略；但不要因为看到未知头就拒绝请求。

想看真实请求长什么样，用公开开发集跑一轮就有，见后面「公开开发集：自己先压一遍」。

响应必须是 SSE，每个事件一行 `data: {...}`，结束发一行 `data: [DONE]`。评分**只读 `meta_info`，完全不读 `text`**：

| `meta_info` 字段 | 必需 | 用途 |
|---|---|---|
| `completion_tokens` | 是 | 累计输出 token 数。第一个 `>0` 的事件定 TTFT，末事件定输出长度和 TPOT |
| `prompt_tokens` | 是 | 取自首个产出 token 的事件，进 TPM |
| `cached_tokens` | 是 | 同上，前缀缓存命中数 |
| `request_received_ts` | 建议 | 服务端收到请求的 epoch 秒，取自**末事件** |
| `prefill_finished_time` | 建议 | 首 token 产出时刻，取自**首事件** |

后两个给齐了，TTFT 就按服务端口径算（`prefill_finished_time - request_received_ts`），把客户端和网络那一段剔掉；缺任意一个就退回客户端首个 SSE 的 proxy 值，网络抖动会直接算进你的成绩。所以建议给。把这几个字段放进**每一个**事件最省事，首末都能取到。

两条硬要求：

- `ignore_eos=true` 必须生效，输出恰好是 `max_new_tokens` 个 token，不能提前停；
- `cached_tokens` 要如实反映前缀缓存命中，不能恒为 0。

一次合格的响应大致长这样（中间事件省略）：

```
data: {"text":"<delta>","meta_info":{"prompt_tokens":62909,"cached_tokens":0,"completion_tokens":3,"finish_reason":null,"request_received_ts":1789661479.362,"prefill_finished_time":1789661479.456}}
data: {"text":"<delta>","meta_info":{"prompt_tokens":62909,"cached_tokens":0,"completion_tokens":6,"finish_reason":null,"request_received_ts":1789661479.362,"prefill_finished_time":1789661479.456}}
...
data: {"text":"<delta>","meta_info":{"prompt_tokens":62909,"cached_tokens":0,"completion_tokens":240,"finish_reason":{"type":"length","length":240},"request_received_ts":1789661479.362,"prefill_finished_time":1789661479.456}}
data: [DONE]
```

`completion_tokens` 是**累计值**，一步跳几个是投机解码一次出多 token，属正常。`text` 发增量还是发累计都行，评分不读它；但发累计意味着一个 8k token 的回答要推几十 MB SSE，会拖垮你自己的延迟。

### `/flush_cache`

必须实现 `POST /flush_cache`：清掉前缀 KV（代码缓存、CUDA graph 不用动），成功返回 2xx 和 `{"success": true}`。

平台在两个位置调用它：每一档正式测量开始前，以及切换并发档之前。原因是每一档回放的是同一批请求，上一档跑完留在引擎里的正是下一档要用的前缀；单档内部的 warmup 和 preflight 同样会把前缀灌热。不清掉的话缓存命中率虚高、TTFT 虚低，而 `n_at_slo` 正是靠四道 TTFT p95 卡出来的——这是白拿分，不是吃亏。

所以清不掉就判该档失败，不会静默跳过：正式测量前清不掉，这一档直接终止；换档前清不掉，这一档不会启动。换句话说没有 `/flush_cache` 的服务最多只能跑出第一档的成绩。

## 引擎适配指引

`/generate` 是 SGLang 的原生口。用 vLLM、TensorRT-LLM 或自研引擎，需要你自己在镜像里补这一层，平台不会因为框架不同去适配路径。

补的办法**不建议**在前面挂一层反向代理：多一跳延迟，而且代理拿不到 `cached_tokens` 这种引擎内部数字，只能瞎猜。正确做法是**在引擎进程内加一条路由**，直接用引擎自己的 client 拿第一手 token 计数。

以 vLLM 0.13 为例，它有官方扩展点 `vllm.endpoint_plugins`，`build_app()` 会在挂完内建路由后调用。做成插件的好处是**启动命令一个字都不用改**，且新路由与 OpenAI 接口同进程、同端口、同中间件：

```python
# s1_generate/__init__.py
class S1GeneratePlugin:
    name = "s1_generate"
    required_tasks = ("generate",)

    def attach_router(self, app):
        from ._routes import attach_router
        attach_router(app)          # 在这里注册 /generate 和 /flush_cache

    async def init_state(self, engine_client, state, args):
        return None                 # 路由直接读 app.state.engine_client
```

```toml
# pyproject.toml
[project.entry-points."vllm.endpoint_plugins"]
s1_generate = "s1_generate:S1GeneratePlugin"
```

路由实现里，`app.state.engine_client` 就是 `AsyncLLM`。`engine_client.generate(prompt, sampling_params, request_id)` 产出的 `RequestOutput` 上，`len(prompt_token_ids)`、`num_cached_tokens`、`outputs[0].token_ids` 正好对应上表的三个必需字段；`SamplingParams(ignore_eos=..., output_kind=RequestOutputKind.DELTA)` 对应增量流式。`/flush_cache` 就是 `await engine_client.reset_prefix_cache()`。

一个坑：endpoint 插件默认**不加载**（它会打开新的网络面），必须用 `VLLM_PLUGINS` 显式白名单。而这个变量一旦设置就会同时收紧*所有*插件组，所以要把底包原有的插件一并列上：

```dockerfile
ENV VLLM_PLUGINS=s1_generate,lora_filesystem_resolver,lora_hf_hub_resolver
```

这套做法主办方在 8× A100 + GLM-5.3-Flash 上实跑验证过：真实请求的 `prompt_tokens` 与数据侧冻结的 token 数**逐条相等**，`ignore_eos` 精确，重复请求 `cached_tokens` 正常上涨，`/flush_cache` 后归零，TTFT 走服务端口径。

怎么验自己的实现：最省事的办法是用公开开发集（见后面「公开开发集：自己先压一遍」），它用的就是压测同一套 harness 和同一套评分口径。跑通一轮 `run_dev.py` 且 `coverage=100%`、`engine_error=0`，接口这一层基本就没问题了。重点核对三件事：输出 token 数是否恰好等于 `max_new_tokens`、同一 prompt 重复打 `cached_tokens` 是否上涨、报告里 TTFT 是不是服务端口径。

## 制备并注册镜像

**Trisol 只支持从 LBG 打包的镜像。** 乌兰集群不拉 Docker Hub，也不拉你本机 `docker push` 到别的仓库的地址。提交 `docker.io/...`、`ghcr.io/...` 这类镜像，服务起不来。

制备走 LBG / Bohrium 自定义镜像页（同一条路）：

https://www.bohrium.com/web-images/custom

1. 用和 Trisol 同一套企业账号登录（企业代码 `dpt`）。
2. 在这个页面基于下面「可起步的基镜像」做自定义镜像（改启动参数、装补丁、补接口），让 LBG 打包并推到 `registry.dp.tech`。
3. 页面给出完整镜像地址后，写进 `submission.json` 的 `image`，自测时写给 `trisol inference create --image-ref`。
4. 建议钉 digest，不要只留会漂的 `latest`。

这是教程的一部分，不是可选项。只有 LBG 打出来的镜像，Trisol 在 `w1` 上才拉得到。

模型和 tokenizer 的 revision 同样要钉死。不要为了让部署成功而删掉必需的推理参数。

### 用命令行构建并注册

网页之外，也可以用 `lbg` CLI 一步构建加注册。注意光 `docker push` 到 registry 是不够的，镜像没进 LBG 目录，部署时会报 `image is not found, can't distribute`：

```bash
pip install --pre lbg -i https://pypi.tuna.tsinghua.edu.cn/simple
lbg login --ak <你的 Bohrium access key>

lbg sdbx image build --dockerfile ./Dockerfile --name <name>:<tag> --project-id <项目 id> --json
lbg sdbx image build-log <id> --follow    # 跟 kaniko 日志
lbg sdbx image get <id> --json            # status: 0=creating 1=pending 2=success 3=failed
```

它在平台侧用 kaniko 构建，构建完自动注册，提交时就返回最终 `imageUrl`。两个限制：Dockerfile 是**内联**上传的（上限 64 KiB）且**没有 build context**，所以 `COPY` 用不了，要带的文件得自己打包内嵌进 `RUN`（比如 base64 一个 tar.gz）；`--name` 也别用 `latest` 结尾。

### 可起步的基镜像

主办方提供了几条已经按赛场卡（8× A100、sm_80）打好、并完成 LBG 注册的底包。自测和二次构建都从这里起步，不要再从 Docker Hub 拉官方镜像——乌兰拉不到。

| 引擎 | 镜像 |
| --- | --- |
| SGLang | `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918` |
| vLLM | `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-vllm-glm53:260918` |
| TokenSpeed | `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-tokenspeed-glm53:260918-r2` |
| Transformers | `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-transformers-glm53:260918` |
| KTransformers | `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-ktransformers-glm53:260918` |

这些是**官方路线的底包**，不是推荐配置，也不保证原样提交就能过门槛或爬到高档。要改启动参数、补 `/generate` 与 `/flush_cache`、装自己的补丁，基于对应底包再打一条自己的镜像提交。

表里的 vLLM 是官方构建，和下面「示例提交」用的 sm_80 **backport** 不是同一条镜像。backport 主办方已经跑通整条评测链路，可以继续用；官方底包按官方 recipe 打，接口和启动参数不要照抄 backport 那条。

## 公开开发集：自己先压一遍

正式压测集不公开，但我们放了一份**公开开发集** `dev-combined-v1`，用的是同一套 harness 和同一套评分口径。提交前自己先跑几轮，调完再交。

Wenyon 数据集（public，内部账号可直接拉）：[s1-dev-combined-v1](https://wenyon.dp.tech/app/console/datasets/s1-dev-combined-v1)

```bash
# 拉下来（约 68 MB）
wenyon-cli dataset download s1-dev-combined-v1 -o ./s1-dev
cd s1-dev
pip install -r harness/requirements.txt
```

包里是：`data/dev-combined-v1/`（311 链 / 722 请求，**请求正文齐全**，可直接回放）、`harness/`（`s1_loadgen.py` / `s1_score.py` + 冻结 cohort）、`glm_tok/`（GLM-5.3-Flash tokenizer，revision 已固定）、`run_dev.py`（一键脚本）。

```bash
python3 run_dev.py \
  --base-url http://<你的服务>:8000 \
  --set dev-combined-v1 \
  --root data/dev-combined-v1 \
  --cohort harness/g0a/samples_v3/cohort_dev-combined-v1.json \
  --tok-dir glm_tok \
  --out runs/dev-n6 \
  --n 6
```

服务带鉴权就加 `--api-key`（不会打进日志）。`--base-url` 写引擎根，带 `/v1` 会自动剥掉。脚本依次做 preflight（1 条链探通）→ warmup（JIT 预热，不算成绩）→ flush KV → 正式测量 → 评分，产出 `report_*.md` 和 `summary.json`（里面有 `evaluation_status`、四道 TTFT p95、TPM）。

换并发就换 `--n`，自测建议按正式梯子走 `2 / 6 / 10 / 14 / 18 / 22 / …`（从 2 起每隔 4，没有上界），这样看到的过/不过与正式爬坡同构。**换 N 前必须重新 flush KV**，否则上一档的 prefix cache 会把 TTFT 冲好看；预热可以跨 N 复用（`--skip-warmup`），KV 不行。8× A100、N=6 上参考墙钟约 35 分钟。

`summary.json` 里没有 `tpot_mean`（它由平台在评分侧从 raw 记录算），但 `report_*.json` 的 `tpot` 有分位数，`raw_*.jsonl` 每行有 `tpot_s`，自己取均值即可对照第二顺位。

两点注意：

- 开发集是**链前缀抽样**，只能比你自己两次部署的相对变化（A/B、回归），**不能**当成 `N@SLO` 的预测值。正式成绩由主办方在完整集上实跑。
- 别改题：设了 `--max-chains` / `--no-gap` / `--include-all` 之后的结果就不是同一份负载，拿来对比没有意义。

## 提交物

只创建一个文件 `/app/submission/submission.json`：

```json
{
  "image": "registry.example.com/your/serve:latest",
  "command": "python3 -m sglang.launch_server --model-path /mnt/models --host 0.0.0.0 --port 8000 --tp-size 8 --served-model-name default",
  "env": {
    "SGLANG_OPT_USE_TOPK_V2": "0"
  },
  "model_name": "default"
}
```

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `image` | 是 | string | 必须是 LBG 打包的镜像（[Bohrium 自定义镜像页](https://www.bohrium.com/web-images/custom)）。Docker Hub / 其它仓库 Trisol 拉不到。建议固定 digest |
| `command` | 是 | string | 容器启动命令，交给 `trisol inference create --command-line` |
| `env` | 否 | object（string→string） | 透传给容器的环境变量 |
| `model_name` | 否 | string | 服务起来后 OpenAI 兼容的模型 id，默认 `default` |

`image` 与 `command` 都必须是非空字符串。除这个文件之外不评其他产物：调参脚本、压测自测日志等都可以留在容器里自查，但不参与计分。

**不要提交**：`base_url`、`api_key`、任何形式的密文、公网 HTTPS 地址。这一轮的口径是交镜像、由主办方在自己的集群里部署，你不需要也不应该提供任何访问凭据。推理服务的密钥由平台自己签发，不会出现在任何响应、回调或后台页面里。

`team` / `cluster` 不用写。自测用 team `arena`、集群 `w1`（8× A100-SXM4-80GB），加入步骤见上面。

赛场卡是 Ampere（sm_80）。上面「可起步的基镜像」都是按这套卡打的。官方 [vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash?frontend=python) 只支持 Hopper 及更新，A100 上不要自己拉上游镜像；用表里的官方 vLLM 底包，或用「示例提交」那条已经跑通的 sm_80 backport。

`/app/submission/submission.json` 存在、符合上面的精确 schema（`image` 与 `command` 为非空字符串，不含 `base_url` / `api_key`），且指向一个你本人获授权的推理镜像。格式检查在提交时完成；镜像可拉取、服务可就绪、`/chat/completions` / `/generate` / `/flush_cache` 三个口可用在计分运行时核验，届时起不来即该次提交不得分。

### 用 Playground 提交

正式提交走 Playground，不是直接调评分 HTTP。本赛是部署赛，**不评 agent 轨迹**，但创建 attempt 时服务端仍要求 attempt 本体带**非空轨迹**。zip 里有轨迹不算数。

```bash
playground --version
playground submit --challenge-id llm-challenge-arena-v1 \
  --outputs ./outputs \
  --trace stub-trace.jsonl
```

`--outputs` 按赛题包任务书组织，必须能读到上面的 `submission.json`。`--trace` 用下面这份占位即可（不要写 token）：

```json
{"type":"session_start","source":"stub"}
{"role":"user","message":{"content":[{"type":"text","text":"placeholder"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"placeholder"}]}}
```

规则就三条：

- JSONL **第一行必须是** `session_start`，后面至少再有一条 user / assistant。只写一行 `session_start`，当前 CLI 会在本地报 `could not derive normalized trace steps`。
- 字面空的不行：`trace: "[]"`、不传 `raw_messages`、或第一行不是 `session_start`，服务端都是 **400**（原文 `Agent submissions must include trajectory data`）。
- 另一种写法是 attempt 字段 `trace` 给非空 ARM steps（`thought` / `observation` / `tool_call` / `tool_result` 均可），数组不能是 `[]`。二选一。

### command 不是 shell

`command` 被当成 argv 直接 exec，没有 shell 解析。不要写成：

```text
SGLANG_OPT_USE_TOPK_V2=0 python3 -m sglang.launch_server …
```

`SGLANG_OPT_USE_TOPK_V2=0` 会被当成可执行文件名，服务立刻 `failed`。环境变量一律放 `env`。同理 `&&`、`|`、`>` 重定向、`$VAR` 展开都不生效；要做启动前处理就在镜像里放一个入口脚本，让 `command` 去调它。

glm53 跑 A100 必须在 `env` 里设 `SGLANG_OPT_USE_TOPK_V2=0`，否则 CUDA graph 捕获时 JIT 编译会挂。

### 提交前自查

- 目标集群能拉到 `image`，模型与 tokenizer 的 revision 已固定；
- `command` 的第一个 token 是可执行文件，不是 `KEY=value`；
- 在集群内起一次容器，`GET {base_url}/models` 返回 200；
- 真实跑一次 `POST {base_url}/chat/completions` 往返，请求里的 `model` 用你写的 `model_name`；
- 真实跑一次引擎根路径上的 `POST /generate` SSE 往返，确认能流式返回，且每个事件的 `meta_info` 带齐 `completion_tokens` / `prompt_tokens` / `cached_tokens`（细则见「服务接口规范」）；
- 真实打一次 `POST {engine_root}/flush_cache`，确认返回 2xx；再把同一条 prompt 连打两次，确认 `cached_tokens` 第二次上涨、`flush_cache` 之后归零；
- 检查响应的实际模型名与 token 预算，确认没有把 thinking 或输出长度悄悄压掉。

### 示例提交

下面是主办方用来验证整条链路的一条真实提交：vLLM 的 sm_80 **backport** 跑 GLM-5.3-Flash，8× A100，`/generate` 由镜像里的 endpoint 插件提供。这和上面基镜像表里的官方 vLLM 不是同一条，不要混用启动命令。可以照这个形状写自己的：

```json
{
  "image": "registry.dp.tech/dptech/dp/native/prod-20675/vllm-backport:260918-sm80",
  "command": "vllm serve --model /mnt/models --host 0.0.0.0 --port 8000 --served-model-name glm-5-3-flash --tensor-parallel-size 8 --max-model-len 524288 --gpu-memory-utilization 0.92 --max-num-seqs 16 --max-num-batched-tokens 8192 --enable-prefix-caching --enable-prompt-tokens-details --disable-custom-all-reduce --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16],\"max_cudagraph_capture_size\":16}' --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45",
  "env": {
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple"
  },
  "model_name": "glm-5-3-flash"
}
```

几个可以照抄的做法：

- **`--model /mnt/models` 必须显式给。** Trisol 校验要看到这个 flag。但 `vllm serve` 同时也接受位置参数，两个都写会变成 `unrecognized arguments`，服务 CrashLoop 到超时，只留一种。
- **`/generate` 和 `/flush_cache` 由镜像自带的 endpoint 插件提供**，启动命令里看不到任何痕迹，`VLLM_PLUGINS` 白名单也已经烘进镜像的 `ENV`，所以 `env` 里不用再写。做法见「引擎适配指引」。
- `NCCL_ALGO` / `NCCL_PROTO` 放在 `env` 里，没有写成 command 前缀。
- `--served-model-name` 与 `model_name` 一致，能力评测才能命中你的模型。
- `--host 0.0.0.0`，端口固定，平台按此接管。
- `--enable-prefix-caching` 开着。压测的负载是多轮会话链，前缀复用率很高，关掉会同时拖垮 TTFT 和 TPM。
- 注意这条示例的 `--max-num-seqs 16` 和只捕到 16 的 CUDA graph：过了 N=14 之后槽位就不够，请求会在引擎门口排队。参数没有按梯子调过，别直接抄去冲榜。

这条提交主办方实测过：8× A100 上 21 分钟起服务，`/v1/models` 与 `/chat/completions` 正常，真实压测请求的 `prompt_tokens` 与冻结 token 数逐条相等，`ignore_eos` 精确，前缀缓存与 `/flush_cache` 行为正确。它演示的是**接口适配**，参数本身没有调优，不要当成推荐配置直接抄去冲榜。

另一条参考是 SGLang 路线：GLM-5.3-Flash 官方在 A100 上用 SGLang，需要在 `env` 里设 `SGLANG_OPT_USE_TOPK_V2=0`（否则 CUDA graph 捕获时 JIT 编译会挂），启动前要渲染 chat template 时用镜像里的脚本 `--exec` 接真正的 server 命令，而不是拼 shell。

## 评测流程

隐藏的验证器读取 `submission.json` 并校验 schema，然后把 `image` / `command` / `env` / `model_name` 交给主办方的评分服务，轮询至终态并按数据集读取分数。你不需要自己部署、不需要自己发压、也不需要访问评测接口。

提交之后由主办方的平台执行，全程不需要你介入：

1. **排队**：提交入队，等空闲 GPU。
2. **部署**：用你的 `image` + `command` + `env` 在 8× A100 上拉起推理服务，平台自己签发访问密钥。
3. **能力评测**：跑 `aime26` 与 `gpqa-diamond`。
4. **门槛判定**：两科都过线才继续，不过线到此结束，压测主分没有成绩。
5. **压测**：回放隐藏的会话负载集，沿梯子爬并发档位（步长 4，没有上界），单档大约 4 小时。
6. **收尾**：记录成绩，销毁推理服务。
7. **赛后复核**：对进入排名的提交，主办方会复核镜像内容、启动参数与相关代码，核对上报的时间戳、token 计数与 `/flush_cache` 行为是否属实。详见「约束」。

## 能力评测

两科能力评测：`aime26`（数学，`pass@1`）和 `gpqa-diamond`（科学问答，`accuracy`）。题目、题量和采样都由平台钉死，请求里改不了。

两科的 `points` 都 **严格大于 90** 才进入压测。这是硬门槛，不是加分项。

## 压测

过门槛后在同集群的独立压力 Pod 上回放一份隐藏的真实会话负载集：多轮会话链，固定轨迹、依赖感知回放（teacher-forced），工具结果取冻结记录，不执行真实工具。单档大约 4 小时，搜索多档更长。集合内容与规模不公开。

一次只跑一档并发 N。

### 并发档位与爬坡

档位不是连续整数，而是从下界起、步长为 4 的梯子，**没有上界**：

```
N = 2, 6, 10, 14, 18, 22, 26, …
```

平台从 **N=10** 这一级起，**过了往上爬一级（+4），没过往下退一级，首次出现「在已通过档位之上失败」就停**。最大的那个通过档位就是 `n_at_slo`；被淘汰的档位留档备查，不计成绩。连 N=2 都过不了时 `n_at_slo` 为 `null`。过了 22 就继续探 26、30……直到某一档没过。

常见探测路径：

| 探测顺序 | `n_at_slo` |
| --- | --- |
| 10✗ 6✗ 2✗ | `null` |
| 10✗ 6✗ 2✓ | 2 |
| 10✗ 6✓ | 6 |
| 10✓ 14✗ | 10 |
| 10✓ 14✓ 18✗ | 14 |
| 10✓ 14✓ 18✓ 22✗ | 18 |
| 10✓ 14✓ 18✓ 22✓ 26✗ | 22 |
| …过了继续 +4 | 上一档 |

所以 10 过了之后不会回头补测 6，18 挂了之后也不会再去试 22。N 是同时在场的逻辑会话数，不等于 HTTP 并发数；中间的整数不会被测。

### 单档的硬门

一档 N 要算「过」，必须同时满足十一条硬门：`coverage=100%`、`harness_data==0`、`harness_render==0`、`engine_error<1%`、`infra_error<1%`、四道 TTFT p95、`gated_phases_have_samples`、解码门 `tpot_p95`。任何一条不过，该档判 FAIL，爬坡就停在这里。

TPM 取固定稳态窗口，不用全程墙钟平均；预热与首轮不进稳态 cohort。SSE chunk 不是 token，不按 chunk 计数。

## 指标口径

评测结束后主办方会给出下面这些字段。能力两项永远有；`stress` 只在过门槛时有，不过门槛为 `null`。

缩写的全称和直觉解释见开头「术语表」，这一节只给精确定义：每个字段怎么算、从哪里取数。

**能力**

| 指标 | 定义 |
| --- | --- |
| `aime26.score` / `gpqa-diamond.score` | 原始比例，0–1，即 `n_correct / n` |
| `aime26.points` / `gpqa-diamond.points` | 百分制，`round(score × 100, 4)`。门槛比的是这个 |
| `n` / `n_correct` | 该数据集的题量与答对数 |
| `metric` | `pass@1`（aime26）或 `accuracy`（gpqa-diamond） |
| `gate_passed` | 两科 `points` 是否都严格大于 90。为 `false` 时不做压测 |
| `gate_reason` | 没过门槛时的原因说明，过门槛为 `null` |

**压测容量与吞吐**

| 指标 | 定义 |
| --- | --- |
| `n_at_slo` | N@SLO。沿梯子爬坡实测到的最大通过档位（2、6、10、14…，步长 4，没有上界）。N 指同时在场的逻辑会话数，不等于 HTTP 并发数。连 N=2 都过不了为 `null`。**排名第一顺位** |
| `n` | 计入成绩那一档实际启动的并发槽，与 `n_at_slo` 同值，是历史字段 |
| `tpot_mean` | TPOT，每输出 token 秒数。逐请求算 `(末 token 时刻 − 首 token 时刻) / (输出 token 数 − 1)` 再取该档均值。两个时刻取的是发压侧收到首个 / 末个 SSE 事件的时间，所以发累计 `text` 拖慢 SSE 会直接算进这一项。**越小越好，排名第二顺位** |
| `tpot_p95` | 同样的逐请求 TPOT 取该档 p95，是解码门的判据，**不排名**。见下面「压测解码」 |
| `tpm_all` | 逻辑总 TPM，稳态窗口内 `Σ(prompt + completion) × 60 / 窗口秒数`。缓存输入也计。只回报，不排名 |
| `tpm_decode` | 输出 TPM。同一窗口内 completion tokens 总和 × 60 / 窗口秒数。只回报，不排名 |
| `slo_attainment` | 该档请求中满足自己那一桶 TTFT 目标值的比例。只作诊断，不排名 |

**压测 TTFT**

四道门按请求在会话中的位置分桶，各自取该桶 TTFT 的 p95，单位秒。

TTFT 的正式口径是**服务端收到请求到产出首 token**，包含引擎的 admission、排队和调度，剔除客户端、网络和网关那一段。它由你在 `meta_info` 里上报的 `request_received_ts` 与 `prefill_finished_time` 算出（见「服务接口规范」）。两者缺任意一个时退回客户端计时：从建连发请求前开始，到第一段非空模型输出为止；这条通道会把网络抖动算进你的成绩。空 chunk、role、usage、心跳都不算有效输出。

| 指标 | 进哪些请求 | 门槛 |
| --- | --- | --- |
| `fast_intra_p95` | 链中间，且冻结标注的未命中输入 ≤ 4096 token | 3s |
| `overall_intra_p95` | 全部链中间请求，含上面那些，是体验兜底门 | 5s |
| `turn_start_p95` | 一轮用户话的开头，不是链首 | 15s |
| `chain_start_p95` | 该链第一条请求，或上下文重建。prompt 常达十万 token 级、结构性零缓存命中 | 30s |

**上面四个秒数是目标值，判定带统计余量。** 「p95 ≤ 3s」等价于「超标条数不超过该桶的 5%」，而超标条数本身是一次抽样，样本少的时候噪声可以盖过档位之间的真实差异。所以实际判据不是点估计，而是：**只有超标率的 95% 单侧下界仍高于 5%，才判这道门失败**，宁可错放。

翻成允许的超标条数：`fast_intra` 这种样本较多的桶，允许条数比点估计多约 19%，大致相当于把 3.0s 放到 3.6s；`turn_start` 这种样本很少的桶，放宽的幅度更大。余量随样本量自动缩放，不是固定的百分比。报告里会同时给出点估计结论（`pass_point`）、允许条数（`allowed_over`）和下界（`rate_ci_lower`）。

这条规则不改变你该做的事：把四道 TTFT 都做到目标值以内，不要去卡余量。

**压测解码**

四道 TTFT 门只看首 token，管不到 token 流出来有多快，所以另有一道解码门：

| 指标 | 门槛 |
| --- | --- |
| `tpot_p95` | ≤ 0.10 秒/token（约 10 token/s） |

这是**底线而不是竞争线**——0.10 秒/token 大致是人类阅读速度的下限，正常调过的服务离它有数倍余量，它拦的是「首 token 秒回、之后一秒挤一个字」这类把解码饿死换首 token 的做法。想在排名上得分靠的是 `tpot_mean` 越小越好，不是贴着这道门过。

注意它取 p95 而排名取均值：门要的是对少数被饿死的请求足够灵敏，排名要的是稳定可比。

**压测状态**

| 指标 | 定义 |
| --- | --- |
| `passed` | 计入成绩那一档是否过了全部硬门 |
| `evaluation_status` | 压测 harness 的原文结论，`PASS` / `FAIL` |
| `reason` | 压测没跑成时的原因，例如 Job 未完成。正常结束为 `null` |

## 计分

计分用两个压测主分，按顺位依次比较：`n_at_slo`（N@SLO）→ `tpot_mean`（TPOT）。其余全是门禁或诊断。

**第一步，能力门槛。** `aime26.points` 与 `gpqa-diamond.points` 都要严格大于 90。不过门槛的提交不进压测榜，只按两科能力分互相比较。

**第二步，压测硬门。** 一档 N 必须过全部十一条硬门（覆盖率、harness / 引擎 / 基础设施错误率、四道 TTFT p95、`gated_phases_have_samples`、解码门 `tpot_p95`）才算这一档成立。硬门只决定 N 能不能算，数值不拿来排。

**第三步，压测排名。** 过门槛的提交按下面顺序依次比较，前一项相同才看后一项：

| 顺序 | 指标 | 含义 | 方向 |
| --- | --- | --- | --- |
| 1 | `n_at_slo` | N@SLO，能过全部硬门的最大并发档位 | 越大越好 |
| 2 | `tpot_mean` | TPOT，每输出 token 秒数（token 流出来的速度） | 越小越好 |

两项都取 `n_at_slo` 那一档上测出来的值，不是各档平均，也不会用爬坡中被淘汰的档位（例如判 FAIL 的更高 N）。`n_at_slo` 为 `null` 的提交排在所有有 N 的提交之后，其 `tpot_mean` 不回报。

四道 TTFT 门只看首 token，`tpot_mean` 是排名里唯一在看 token 实际流出来有多快的一项，所以它紧跟在容量之后。

**为什么吞吐不排名。** 压测是按真实会话的时间轴回放的，发请求的节奏和每条请求的输出长度都由数据集定死，引擎跟得上就行、快了也变不出更多 token。所以稳态窗口里测到的 TPM 反映的是回放的需求量，而不是你的服务有多快——实测中把并发从 2 提到 11，TPM 随请求数同步上涨，而单流解码速度反而慢了 4 倍多。这样的量不适合用来分高下，因此只回报不排名。想比吞吐能力请看 `n_at_slo`：它衡量的是同样质量下你能同时装下多少路。

**不参与排名的指标（门禁或对照）：**

- `tpm_all` / `tpm_decode`：吞吐，合同里保留，只回报不排名（理由见上）；
- `tpot_p95`：解码门的判据，是前置硬门；
- `slo_attainment`：TTFT 达标率，诊断用；
- `fast_intra_p95` / `overall_intra_p95` / `turn_start_p95` / `chain_start_p95`：已经是 `n_at_slo` 的前置硬门；
- `stress.n`：与 `n_at_slo` 重复；
- `passed` / `evaluation_status`：与「这一档硬门过没过」同义；
- `gate_reason` / `stress.reason`：失败原因说明。

两项不做加权合成。N 是个数、TPOT 是秒/token，量纲不可比，不存在把它们加成一个总分的公式。

## 成绩样例

过门槛且压测正常结束：

```json
{
  "eval_id": "eval_ab12cd34ef56",
  "status": "succeeded",
  "aime26": {
    "score": 0.9772727272727273,
    "points": 97.7273,
    "n": 44,
    "n_correct": 43,
    "metric": "pass@1",
    "status": "succeeded"
  },
  "gpqa-diamond": {
    "score": 0.9807692307692307,
    "points": 98.0769,
    "n": 156,
    "n_correct": 153,
    "metric": "accuracy",
    "status": "succeeded"
  },
  "gate_passed": true,
  "gate_reason": null,
  "stress": {
    "passed": true,
    "n": 6,
    "n_at_slo": 6,
    "evaluation_status": "PASS",
    "fast_intra_p95": 1.2,
    "overall_intra_p95": 2.4,
    "turn_start_p95": 8.1,
    "chain_start_p95": 15.83,
    "tpot_mean": 0.0413,
    "tpot_p95": 0.062,
    "tpm_all": 1011707.58,
    "tpm_decode": 6297.75,
    "slo_attainment": 0.9631,
    "reason": null
  }
}
```

能力没过门槛，直接返回，不做压测：

```json
{
  "eval_id": "eval_ab12cd34ef56",
  "status": "succeeded",
  "aime26": {
    "score": 0.7954545454545454,
    "points": 79.5455,
    "n": 44,
    "n_correct": 35,
    "metric": "pass@1",
    "status": "succeeded"
  },
  "gpqa-diamond": {
    "score": 0.6987179487179487,
    "points": 69.8718,
    "n": 156,
    "n_correct": 109,
    "metric": "accuracy",
    "status": "succeeded"
  },
  "gate_passed": false,
  "gate_reason": "below gate: aime26 79.55 <= 90; gpqa-diamond 69.87 <= 90",
  "stress": null
}
```

## 质量前提

服务端不得通过关闭 thinking、压低输出预算、截断历史或删除请求体里的 tools 来换性能——这类收益不算同质量收益，评测会对照能力分与实际输出长度核查。业务长输出消失时，即使 TTFT 更好也先查质量。

## 约束

- 只提交你本人获授权的镜像与启动命令，提交物里只放 `submission.json` 要求的四个字段。
- 不得攻击、探测或干扰评测平台、压测 Job、其他参赛者或基准基础设施。
- 不得尝试还原隐藏的基准题目。
- 不要在提交物、日志或报告里写入任何明文密钥。
- `meta_info` 里上报的时间戳与 token 计数必须如实反映服务端的实际情况。把 `request_received_ts` 报晚、把 `prefill_finished_time` 报早、或谎报 `completion_tokens` / `cached_tokens`，都按违规处理，不是优化手段。
- `/flush_cache` 必须真的清掉前缀 KV。返回 2xx 但实际不清、或只清一部分，等于让后面的并发档白拿前一档的缓存命中，按违规处理。

以上几条平台不会在运行时逐条判定。主办方保留在赛后复核镜像内容、启动参数与相关代码的权利；复核发现伪造上报或假清缓存的，取消该次提交的全部成绩。
