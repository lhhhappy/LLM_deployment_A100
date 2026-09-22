# R3 — D6 底包访问路径：不拉取镜像的调查

作者：Codex；核对时间 2026-09-22 UTC。状态：**仅调研路径，没有拉镜像、登录 registry、读取凭据、启动节点或请求构建**。

## 1. 结论

**VERIFIED（官方文档 / 本地 CLI help）**：平台内使用的 `registry.dp.tech` 与本地 Docker 访问入口不是同一个认证入口。Bohrium 文档指定本地使用 `registry.bohrium.dp.tech`；本地 bohr 2.7.7 的 `image pull --help` 进一步说明，它会改写入口并使用 Bohrium access key，而不是要求用户掌握 Aliyun ACR 凭据。[Bohrium 镜像中心](https://bohrium-doc.dp.tech/docs/userguide/image/)

**VERIFIED（文档限制）**：上述本地通道文档只承诺公共镜像和本人自定义镜像，不支持他人分享的镜像。**INFERRED**：比赛底包可被 arena/Trisol 部署，不代表当前账号在本地有 pull 权限；`registry.dp.tech` 的 401 也不能单独证明镜像不存在或不可在赛场使用。

**建议顺序（INFERRED）**：先取得授权目录中的镜像 ID、Dockerfile、build provenance / digest 等小型元数据；如目录权限不足，请主办方提供脱敏源码差异及 manifest。实际拉取或起节点留待用户批准。当前无需消耗 GPU 配额来确认访问路径。

## 2. 精确目标，不能混用

**VERIFIED（赛题 `task.md:304–318,438`）**：

| 目标 | 比赛给出的 image ref |
|---|---|
| SGLang | `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918` |
| 官方 vLLM 底包 | `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-vllm-glm53:260918` |
| 已跑通链路的 vLLM sm80 backport | `registry.dp.tech/dptech/dp/native/prod-20675/vllm-backport:260918-sm80` |
| TokenSpeed | `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-tokenspeed-glm53:260918-r2` |

题面另有 Transformers / KTransformers，非本轮优先对象。官方 vLLM 与示例 backport 是两条镜像；不能互抄版本、插件或启动参数。路径中的 `20675` 等编号不能未经证明当成 image catalog ID。

**VERIFIED（题面）**：Trisol w1 使用 LBG 已注册镜像；仅 docker push 不等于进入可分发目录。赛题给出 [Bohrium 自定义镜像页](https://www.bohrium.com/web-images/custom) 和 LBG 构建/注册流程。**INFERRED**：外部拉取地址改写只适用于外部访问，不能擅自改掉提交物中主办方给出的 image ref。

## 3. 零镜像下载的访问路线

| 路线 | 可得到什么 | 当前证据 / 限制 | 本轮是否执行 |
|---|---|---|---|
| 题面与公开文档 | 精确 refs、入口、注册要求 | 已直接阅读 | 是 |
| 本地 `bohr … --help` | CLI 支持的只读入口及认证方式说明 | bohr 2.7.7，commit `f4868539d66dffeb6fb2ef217c36342155e97eda` | 是，仅 help/version |
| 授权镜像目录的 list/search/get | ID、目录可见性、可能的详情 | private/public 范围不同；可能看不到主办方共享镜像 | 否，当前只列路径 |
| 已知有权访问的 image Dockerfile/build-log | 来源镜像、安装版本、patch 下载地址 | 目录 image ID 与 sandbox/build ID 是否通用须确认；不是拿 ref 当 ID | 否 |
| 主办方提供 digest、SBOM、diff 或源码归档 | 最直接解决 D6、移植基线问题 | 需用户/协调者联络，本轮未代发消息 | 否 |
| Registry manifest/config 元数据 | immutable digest、平台、layer 清单、labels | 需正确认证和该 ref 权限；没有这些不猜 token endpoint | 否 |
| 外部 Docker pull 或平台内起容器 | 实际安装内容、源码/依赖 | 明确超出本轮，只能获准后进行 | 否 |

以下为 **已从本地 help 确认存在、但没有执行的业务命令**：

```text
bohr image search <精确镜像名关键字>        # 公共目录，不搜索私人项目
bohr image list                          # 缺省为本人 private catalog
bohr image get <已确认的 catalog-id>
bohr image dockerfile <已授权的 build-image-id>
bohr image build-log <已授权的 build-image-id>
```

这里只列候选读取路线，不声称该账号已经能读这些比赛镜像。Dockerfile / build-log 也可能含历史凭据；将来读取时应先脱敏，不原样存到团队报告。不要打印 auth config 或把 Playground token 当作 registry 密码。

**VERIFIED（本地 help）**：`bohr image pull` 是 `[write]`，要求本地 Docker daemon；当前容器 PATH 中没有 `docker`、`skopeo`、`crane`。本轮没有安装任何工具，也没有执行 pull、build、login、team join 或推送。

**INFERRED（后续如获准）**：外部 pull 要先确认共享镜像限制和企业 dpt 的认证适用范围；不能认为 SSO 已登录就一定具备 registry 权限。若外部路线不通，可以由主办方提供小型源码/版本收据，或另行批准平台内检查；不需要先复制一个镜像来“试着绕过”共享权限。

## 4. D6 最小证据包

**INFERRED — 每个目标独立收集，优先 SGLang，再 vLLM backport：**

1. 原始 ref、immutable digest、manifest 平台、LBG 目录 ID 与可见性。
2. Dockerfile / build log 中的 FROM digest、源码 commit、patch 来源与 hashes；版本字符串不够。
3. 安装的引擎源码相对 upstream 的文件差异；尤其模型、indexer、attention dispatch、scheduler、KV pool、接口插件。
4. torch/CUDA/Triton/TileLang/DeepGEMM/FlashMLA 版本或 fork commit；A100 路径可能在依赖层。
5. sm80 下 prefill、decode、indexer、top-k 各自实际选择哪个 kernel，target/MTP draft 是否一致。
6. #36884/#36885、#39156 等修复有无等价实现；是否有主办方已验证的 DP 配置。
7. source hashes / 软件清单作为可复核收据；不要附密钥、完整环境变量、模型数据或私有用户数据。

`pip show sglang`、镜像日期 tag、上游发布日相同均不能替代上述核对。“包含私有补丁”继续标为未知，而非已证实事实。

## 5. 本轮交付边界与当前未决

**VERIFIED（行动记录）**：只读了题面、公开技术文档、CLI help/version；没有发起带账号的镜像目录请求，没有重新探测 registry 认证，没有镜像层/模型下载、构建或 GPU 服务。

访问路线已调研清楚；**具体 refs 的权限、digest、实际引擎版本和 sm80 实现仍未确认**。下一步最省资源的信息请求是“提供底包的目录 ID、digest、Dockerfile 和脱敏 patch/dependency provenance”，无需用户在聊天里给密码。此项不阻碍继续做 D1/D2 源码设计，但阻碍声称已能在真实底包干净移植。
