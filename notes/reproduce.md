# 克隆、核验与复现入口

2026-10-03，Codex 整理。当前发布为 [47798 / CAP](submissions.md)，引擎 `ca5d646c`，配置 [official-0930-CAP.json](../submission/official-0930-CAP.json)。这里区分本地资料核验、真实服务启动和正式评测。

## 1. 不需要 GPU：核验发布

```bash
git clone https://github.com/lhhhappy/LLM_deployment_A100.git
cd LLM_deployment_A100
python3 scripts/verify_release.py
python3 scripts/check_submission.py submission/official-0930-CAP.json \
  --sglang-src engine/sglang/srt --final
```

`verify_release.py` 检查 18 份资料的 SHA256、当前引擎 Git tree、两份实际上传 ZIP 中的配置，以及官方终态。检查通过证明档案与已提交版本相符；重新运行隐藏评测仍由平台完成。

CPU 合同测试不需要模型权重：

```bash
PYTHONPATH=tests python3 -B -m unittest \
  test_ax_deadline test_prefix_producer test_chain_risk_scheduler \
  test_fp8_humming_tuning_117 test_kda_prefill_cpu_length
```

这组测试检查准入、共享前缀、chain-risk、多形状回退与 KDA 元数据约束。它不运行完整模型，也不证明 GPU 数值或吞吐。GPU 探针沿用 [tests/gpu/](../tests/gpu/) 与对应 [机制说明](../engine/README.md)。

## 2. 理解源码版本

```bash
git log --reverse --oneline engine-base..ca5d646c -- engine/sglang
git show 6b629210 -- engine/sglang engine/docs
git diff image-lh-img-0928c image-lh-img-0930a -- engine/sglang
git diff ca5d646c HEAD -- engine/sglang
```

最后一个 diff 在此发布中为空。底包、早期官方 A 和历次镜像都有原始标签；[阅读索引](read-history.md) 把重要提交和实验接起来。查看历史源码使用独立 worktree：

```bash
git worktree add build/worktrees/inspect-0928 image-lh-img-0928c
```

## 3. 有平台镜像与 8×A100：启动原始服务

冻结配置的 `image` 给出原始 `lh-img:0930a` 镜像；模型按题目挂载到 `/mnt/models`，服务端口为 8000，TP size 为 8。镜像已包含所需引擎与依赖。完整服务命令及环境变量均在 JSON 中，不另维护一份手写命令。

在**已进入该镜像并挂载正确模型的环境**中，将冻结 JSON 放在当前目录，可用以下方式执行原命令：

```bash
python3 - official-0930-CAP.json <<'PY'
import json, os, shlex, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
argv = shlex.split(config['command'])
os.execvpe(argv[0], argv, {**os.environ, **config['env']})
PY
```

需要平台对该镜像和模型的访问权限；Git 克隆本身不提供这些资源。要重建增量镜像，先读 [scripts/build_image.sh](../scripts/build_image.sh)；该脚本生成 Dockerfile 并核对补丁，不自动构建或上传。09-30 实际 Dockerfile 与构建收据保留在 [提交归档](../evidence/submission-0930-execution/README.md)。引擎裸源码目录沿用底包的包结构，不能将它当作另一个完整 pip 源码发行版。

## 4. 做实验和读取成绩

- 正式赛规、接口和官方本地 harness 命令：[task.md](../llm-challenge-arena-v1/task.md)。
- 本地数据、完整性与判分：[evaluation.md](evaluation.md)、[scripts/longchain/](../scripts/longchain/README.md)。
- 最新本地负载的生成与派生：[可执行配方](../scripts/longchain/recipes/README.md)、[参数 JSON](../scripts/longchain/recipes/0930.json)。Git 保存方法，原始输入和生成产物留在外部或本地。
- 既有共享 Pod 的取证、配置核对和队列：[scripts/pod/README.md](../scripts/pod/README.md)。
- 持有平台凭据时，只读查正式成绩：`scripts/official_status.sh 47798 47800`。

模型、缓存、完整本地数据与大型日志按项目约定在本地或开发机归档，不随 Git 克隆提供；本地窗口的派发范围、raw SHA256 和比较边界在实验记录及原始收据中。正式隐藏数据没有公开的逐请求回放，不能通过本地合成集承诺重新得到相同 N@SLO。
