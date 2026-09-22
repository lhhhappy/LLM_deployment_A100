# pipeline.md — 复现评测、打镜像、提交的全流程（作者：Claude，2026-09-22；待 Codex 审阅）

依据：`llm-challenge-arena-v1/task.md`、`s1-dev/README.md`、Codex R3（底包访问）、本地 `bohr image --help`。
标注：**V** = 已核实；**I** = 推断或待核实。

## 1. 正式评测由什么组成，我们能复现到什么程度

| 正式评测环节 | 主办方做法 | 我们的复现手段 | 复现度 |
|---|---|---|---|
| 部署 | 用 `image` + `command` + `env` 在 8×A100 上起服务 | 在 Trisol arena/w1 上用**完全相同**的三项起服务（`trisol inference create --image-ref … --command-line … `，env 同样传入） | 高（同一集群、同一机型）V |
| 能力门槛 | aime26（44 题，pass@1）、gpqa-diamond（156 题，accuracy），两科都要 > 90；题目和采样参数由平台钉死 | 自写脚本打 `/v1/chat/completions`，用公开的 GPQA-Diamond 和 AIME 题目；采样参数用模型 `generation_config`（temperature 1.0，top_p 0.95） | 中：题集和采样与正式不完全一致，只能做**回退检测**（改动前后对比），不能当成绩预测 I |
| 压测 | 隐藏整链集，harness 与开发集相同（v7 口径），沿 2/6/10/14/18/22 梯子爬坡，每档约 4 小时 | `s1-dev/run_dev.py`（同一套 harness，开发集 311 链/722 请求，N=6 约 35 分钟），按梯子逐档跑、每档前 flush | 中：流程和口径一致；数据是链前缀抽样，只能做 A/B，**不能预测正式 N@SLO**（题面明言）V |
| 判定 | 11 道硬门；四道 TTFT 门按超标率 95% 单侧下界放宽；另有 `tpot_p95 ≤ 0.10` | dev harness 只有 10 道门、用点估计、没有 tpot 门（F10）。另写 `scripts/score_formal.py` 读 raw 结果：补上 tpot 门，并按题面规则估算带统计余量的结论（Codex 提醒：方法未经主办方确认前，不能标成 "formal PASS"） | 中 I |
| 复核 | 赛后核对时间戳、token 计数、flush 是否真清 | D0 验收清单（`patches/000` §4） | 高 |

**本地 2 卡能复现的**：接口合规、缓存命中行为、调度正确性（替身模型，E1/E2）。**只能在 8 卡复现的**：性能数字、N@SLO 趋势、能力分。

### 1.1 竞品数据给的校准
正式结果（`data/all_att.json`）里，N=18 档的 chain_start p95 约 45s 仍判 PASS，说明正式判定确实带统计余量；而 dev harness 用点估计会判 FAIL。自测时两种结论都要看。

## 2. 在 8 卡上自测一个配置的步骤（需要用户批准，占 8 卡配额）

前置（尚未完成）：
1. `bohr extension install trisol` → `bohr trisol login`（需要用户在浏览器完成企业登录）→ `bohr trisol team join arena` → `bohr trisol config use-team arena` → `bohr trisol quota list --team arena`。
2. 知道要用的镜像 ref（底包或我们自建的镜像）。

步骤：
1. **起服务**：`trisol inference create --name <n> --team arena --cluster w1 --gpu-model A100-SXM4-80GB --gpu-count 8 --model glm-5-3-flash:2 --image-ref '<镜像>' --command-line '<与 submission.json 相同的 command>' --startup-timeout-seconds 3600 --no-input`，env 同样传入（具体 flag 待装好 trisol 后看 `--help`，I）。
   - 调参阶段也可以用题面给的"挂起"模式：command 设为 `python3 -m http.server 8000`，再用 `trisol inference exec <n> --team arena -i -- bash` 进容器手动起引擎，改参数不用重建服务。
2. **接口验收**：`GET /v1/models`、`/chat/completions` 往返一次、`/generate` SSE、`/flush_cache` 返回 JSON 以及 `cached_tokens` 上涨后归零（D0 §4）。
3. **压测**：在能访问服务的地方（挂起容器内部，或者集群内另起一个 pod）跑 `run_dev.py --n 6/10/14/18/22`，每档之间 flush；然后用 `scripts/score_formal.py` 复评。
4. **能力抽检**：跑能力脚本（全量或抽样），对比基线。
5. **用完立刻删**：`trisol inference delete <n> --team arena --yes`。

## 3. 打镜像

**规则**（task.md，V）：Trisol 只拉 LBG 注册过的镜像（`registry.dp.tech/...`）。Docker Hub、ghcr 不行；光 `docker push` 也不行，镜像没进 LBG 目录就会报 `image is not found`。

**两条等价路线**：
- 网页：https://www.bohrium.com/web-images/custom （企业账号 dpt 登录），基于底包做自定义镜像。
- 命令行：本地已有 `bohr image build --dockerfile ./Dockerfile --name <name:tag> --project-id <id> --wait -o json`（V：`bohr image build --help`）；题面写的 `lbg sdbx image build` 是同一套机制，在平台侧用 kaniko 构建、自动注册，返回 `imageUrl`。

**限制与做法**（task.md，V）：
- Dockerfile 是**内联上传**的（上限 64 KiB），**没有 build context**，`COPY` 用不了。我们的补丁要么把 tar.gz 做 base64 后嵌进 `RUN`，要么在 RUN 里从 GitHub 拉取固定 commit（需要确认构建环境能否联网，I）。补丁本身很小（D0 + D1 + D2 估计几百行），base64 放得下。
- `--name` 不要以 `latest` 结尾；拿到镜像后**钉 digest**。
- 模型和 tokenizer 的 revision 要钉死（平台用 `--model glm-5-3-flash:2` 挂载到 `/mnt/models`，I）。

**Dockerfile 草图**（I，待底包内容确认后定稿）：
```dockerfile
FROM registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918
# 内嵌补丁（base64 of patches.tar.gz）→ 应用到已安装的 sglang 包目录
RUN echo '<BASE64>' | base64 -d | tar xz -C /tmp/arena && \
    cd "$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')/.." && \
    patch -p1 --forward < /tmp/arena/000-interface.patch && \
    patch -p1 --forward < /tmp/arena/001-role-boundary.patch && \
    python3 -c "import sglang; print(sglang.__version__)"
```
关键前提：**补丁必须能打在底包里的 sglang 上**。底包是不是 v0.5.20、有没有针对 sm80 的改动，都还不知道（D6/R3）。所以打镜像之前要先拿到底包内容，可以用 `bohr image get/dockerfile/build-log`（R3 列出的只读路线），或在 8 卡挂起容器里 `pip show sglang` 再 diff。

## 4. 提交

1. `submission/submission.json`（只有四个字段，V）：
```json
{
  "image": "registry.dp.tech/...@sha256:<digest>",
  "command": "python3 -m sglang.launch_server --model-path /mnt/models --host 0.0.0.0 --port 8000 --tp-size 8 --served-model-name default --enable-metrics --incremental-streaming-output ...",
  "env": {"SGLANG_OPT_USE_TOPK_V2": "0"},
  "model_name": "default"
}
```
   - `command` 是 argv，不经过 shell；环境变量一律放 `env`。
   - `--served-model-name` 要与 `model_name` 一致。
2. 占位轨迹 `submission/stub-trace.jsonl`：第一行是 `session_start`，后面至少跟一条 user 和一条 assistant（task.md 给了模板）。
3. `source env.sh && playground submit --challenge-id llm-challenge-arena-v1 --outputs submission/ --trace submission/stub-trace.jsonl`。先用 `playground trace validate` 检查轨迹。
4. `playground status --attempt-id <id>` 查询进度。部署加能力评测加压测可能超过 10 小时，爬坡每档约 4 小时。
5. **需要用户明确批准才提交**（rule.md 红线 6）。

## 5. 当前阻塞项（按顺序）
1. Trisol 插件安装、登录、加入 arena team：需要用户在浏览器授权。
2. Bohrium `project-id`：打镜像时要用，需要用户提供或查询。
3. 底包内容（D6）：决定补丁的基线。
4. 本地 E1/E2 结论：决定 D1 值不值得带进镜像。
