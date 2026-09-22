> **现行入口：[L2.md](L2.md)**（`scripts/l2.py`）。本文件是守护进程的设计说明。

# T24 自动提测守护进程（W11）

T27 在 T24 基础上加入 keep-alive、forever watchdog 和 simulator cross-check。本次仍只做 CPU/mock 和 dry-run，没有启动 daemon、创建 APPROVED 或操作 Trisol 服务。

## 执行前的接口条件

**T31 runner CLI 已对齐**：`scripts/session_a/README.md` 记录候选匹配、精确矩阵、三种梯子、`--out` 与零预算 `--collect-only`；真实wrapper help + daemon→runner mock端到端已通过（`evidence/T31/`）。daemon仍在任何服务动作前检查本地help；此验证不是live服务/底包/引擎通过。

`tests/trisol_test_config.json` 的 `gpu_product_id` 默认 `null`。已安装 `bohr trisol inference create --help` 要求这个产品 ID；board/plans 只给了 A100 型号，没有保存原创建 ID。本轮不查询线上服务，请协调方从原创建收据补齐。接管已有服务不需要此 ID，但必须通过实际 RequestSpec 一致性校验。默认 team=arena、cluster=w1、model=glm-5-3-flash:2、8×A100-SXM4-80GB、replicas=1、startup timeout=3600；挂起命令按本任务明确要求使用 `python3 -m http.server 8000`。

列表必须返回 `scope=mine`、完整 `items`/`services` 和 `id/name`，使用 `list --scope mine --all`。生命周期优先读取 observed `phase` / `runtime_status`，再读 `status`；不认识的值保持等待并保守计费。协议不匹配、所有权不明、配额/预算不确定时阻止下一项；不会查询或操作其他选手服务。实际线上 JSON schema 尚未通过 live 验证。

## 队列与配置

队列按数字前缀排序：`tests/queue/NN-slug/spec.json`。非批准项跳过。`APPROVED` 由 `scripts/l2.py add` 在入队时写入（用户已把自测审批委托给 Claude，2026-09-22），守护进程只读；入队助手保存所有候选原文件 SHA256，候选后来改变则必须重新入队并重新批准。执行前再验证 spec/候选未变，实际 runner 使用 `runs/NN-slug/profiles/` 下冻结的候选副本。

```bash
cd /workspace/Agentic_science_challenge
python3 scripts/l2.py add img_a   # 旧的 queue_test.sh 已由 l2.py 取代
python3 -B scripts/trisol_test_daemon.py --dry-run
```

以上只是未来的入队用法，本轮没有执行入队。`--dry-run` 只打印计划，不跑子进程、不查凭据/Trisol、不写 state/lock/日志。单项预览：

```bash
python3 -B scripts/trisol_test_daemon.py --dry-run --spec tests/queue/01-run-a/spec.json
```

字段：

| 字段 | 语义 |
| --- | --- |
| `image_ref` | 这次被测镜像，创建或接管时核对 |
| `profiles` | repo 内 candidate-b*.json 路径列表；四字段候选 JSON |
| `ladder` | `{"mode":"levels","levels":[6,10,14]}`，或 `{"mode":"official-climb","max_n":30}`，或 `{"mode":"fast","hint":18,"max_levels":6}`；N 必须是 2+4k |
| `matrix` | baseline / spf / d1 / spf_d1 / hrrn / d1v12 / spf_d1v12 的非空列表，按原顺序交给 runner |
| `time_budget_minutes` | 从分配资源开始的硬上限，含启动、实验、拉取、删除预留；每日余额不足会缩短 |
| `max_admission_wait_minutes` | 可选排队上限，默认 1440；WaitingForAdmission 支持等待数小时 |
| `service_name` | 可选，若本人已有同名服务则验证并接管，例如 lh-arena-sess-a；默认使用中性哈希名 lh-t24-xxxxxxxxxxxx，不公开内部 slug/优化方向 |
| `notes` | 可选说明，只存本地，不写 Trisol 描述 |
| `promote` | 可选 `{"min_n":18}`，达到严格条件才生成**未批准**的提交队列项 |

全局配置来自 `tests/trisol_test_config.json`。`daily_gpu_hours=0` 表示不设每日 GPU-hour cap；每项 time budget 和 admission deadline 仍有效。默认 keep-alive=true：同一 8 卡服务连续执行批准项，空队列保留默认 3h 后释放；若存在 `lh-arena-sess-a` 则优先核验并接管。服务消失后下一批准项自动重新创建并继续 FIFO。`target_queue_depth=5` 以下只发一次 `QUEUE_LOW`，daemon 不生成测试项。

只要同一账号 arena 范围存在另一服务（含 1 卡检查服务、排队、停止或删除中），就不创建新服务。全局 flock 锁 + `data/trisol_tests.json` 维护所有会话，8 卡最多一个。必须保留 state；删除 state 不是重置每日预算的方法。运行期间配置固定，改配置需先安全停止再重启。

WaitingForAdmission 的 `queue_wait_s` 单列，不收 GPU 时长。开始分配、拉镜像、启动及状态不明均保守计费，计至删除确认；从最后一次未分配轮询时刻开始收费，最多多算一个轮询间隔。已在运行的接管项按 allocated/running/started/created timestamp 计入旧成本；无可用时间戳拒绝接管。UTC 跨日按区间拆分；不会因重启或跨日重置会话截止时间。

## runner CLI / 输出合约（交接 T23）

```text
scripts/session_a/run_session_a.sh \
  --service-id ID --profiles /repo/runs/NN-slug/profiles/candidate-b0.json ... \
  --matrix baseline,spf,d1,spf_d1 --ladder-mode official-climb \
  --max-n 30 --budget-minutes MINUTES --run-id NN-slug --out /repo/runs/NN-slug
```

levels/fast 模式分别传 `--levels 6,10` / `--hint 18`，可有 `--max-levels`。`--out` 已有 profiles 子目录，runner 不得因此报目录存在。`--budget-minutes` 是扣除了 daemon 清理预留的可用时间，runner 必须整批遵守它。

失败/超时后同一组参数加 `--collect-only`，此时 budget 可为 0；只拉取已有文件并重建manifest，不启动实验或停止引擎；引擎清理由原runner finalization/远端deadline watchdog负责（T31用户要求）。常规运行要逐档落盘并回传。daemon 的 collect 进程有独立时间盒。回传路径必须是本地 `runs/NN-slug/` 与 GPU 机 `/sjtu/linhang/arena/runs/NN-slug/`；daemon 会再次用固定 SSH 配置/代理与 rsync 同步，且在服务删除后补传最后回收收据。

`session_result.json`（应在每个已完成档位后原子更新，以便失败回收）：

```json
{
  "run_id": "01-run-a",
  "results": [{
    "profile_sha256": "sha256-of-original-candidate-bytes",
    "matrix": "baseline",
    "N": 18,
    "raw": "baseline/level_N18/raw_measure.jsonl",
    "run": "baseline/level_N18/run_measure.json",
    "candidate_exact": false,
    "p0": {"IF-01": "pass", "IF-08": "pass"}
  }]
}
```

raw/run 是 out 内的相对路径，拒绝逃逸或外部 symlink；run metadata 的 config.N 必须一致。每个 profile SHA 必须是当前快照。daemon 对每条记录调用真实 `scripts/score_formal.py --raw ... --run ... --out ...`；成功执行不等于 gate 通过，结果一律标 **estimated**，不宣称正式 N@SLO。空/缺 manifest、评分错误、拉取/镜像失败不能触发提交。

`candidate_exact` 只能在运行的 image/command/env 确与候选一致且有证据时为 true。会话 A 在官方底包临时打补丁、改 port/policy/env 的结果应 false。`p0` 逐测试 ID 回填，不接受笼统的 all_pass=true；只做 CAP 2题 smoke 不等于 CAP-01/02 全部通过。自动提交入队要求最优已通过档 N≥min_n、候选精确一致、当前 TEST_PLAN 所有 P0 ID 均 `pass`、候选 image 等于被测镜像且钉 sha256 digest、会话正常结束且已确认删除。调用 W9 的 create_queue_item，使用 submission.json + stub-trace.jsonl + notes.md，不生成 APPROVED、不替换候选 image、不自行打镜像。

## 启停命令（只提供，没有执行）

在上述接口和配置补齐、队列已有用户批准后（本任务不执行）：

```bash
cd /workspace/Agentic_science_challenge
mkdir -p logs data tests/queue
nohup bash scripts/run_forever.sh > logs/trisol_test_forever.log 2>&1 < /dev/null &
echo $! > data/trisol_test_forever.launcher.pid
```

它持续轮询，完成一项并确认服务删除后立即处理下一项。`--once` 是**真实执行一项或恢复清理**，不是 dry-run。PID 由进程取得锁后自己写进 `data/trisol_test.pid`。

停止（STOP 会让 daemon 释放 keep-alive 服务并让 watchdog 不再重启）：

```bash
touch tests/queue/STOP
```

恢复队列并查看状态：

```bash
rm -f tests/queue/STOP
python3 scripts/test_status.py
```

安全停止后台进程（SIGTERM 会走回传/清理）：

```bash
python3 - <<'PY'
import os, pathlib, signal
pid = int(pathlib.Path('data/trisol_test.pid').read_text())
cmd = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert any(x.endswith(b'/trisol_test_daemon.py') or x == b'scripts/trisol_test_daemon.py' for x in cmd)
assert b'--watchdog' not in cmd
os.kill(pid, signal.SIGTERM)
PY
```

事件：`tail -f logs/trisol_test.events`；每项一行 JSON `event=RESULT`（去重 result_id），包含 best、等待秒数、estimated 标签和 PF-01/PF-02。摘要行追加 `notes/experiments.md`；最终回收状态以 `data/trisol_tests.json` 和 `runs/NN-slug/daemon_result.json` 为准。RESULT 在删除前写入，删除完成会另写 DELETED；不要把 RESULT 当作已释放资源。

## 失败与恢复

- 子进程超时杀进程组；失败也先尝试 collect-only / 本地评分 / 双落点同步，随后一定进入删除路径。超时或回传失败不能无限留卡，事件会显式记失败，不伪造产物成功。
- delete 返回 accepted/deleting 不算完成。只有完整本人列表中 ID/name 消失才释放本地槽位；查询失败/删除不确认进入 cleanup，阻止下一项并持续重试。
- create 前持久化意图，超时后按本人同名服务对账并删除，不重复 create。若响应丢失且暂时找不到服务，保留 cleanup 隔离，不能认为创建失败就再开下一项；需协调方从平台确认后修复该条记录。
- 独立本地 watchdog 记录控制 PID+Linux starttime，防 PID 复用。控制进程意外退出时，它拿同一把锁，只恢复未完成项的回收；超出期限则终止失去响应的控制进程，并停止已记录的遗留 runner 进程组。它从不排新测试。
- 本机/网络整体故障、平台删除接口失效时，本地程序无法保证云资源按时消失。恢复后优先清理；保留独立外部平台监控。平台 cleanup 延迟可能使实际记账超过预算，全部如实记入，不篡改为上限。
- T31已修改runner并完成CLI/manifest对接；显式矩阵不静默降级，缺少所需补丁直接失败。profile由内容匹配，v1.2叠004，HRRN可基于baseline派生；实验manifest保守标记candidate_exact=false/p0={}，不能据此自动提交。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s scripts -p test_trisol_test_daemon.py -v
python3 -B scripts/trisol_test_daemon.py --dry-run
python3 scripts/check_records.py
```

证据：`evidence/T24/trisol_test_daemon_tests.log`。mock 审批在内存注入，测试也不创建 APPROVED；mock bohr/runner、临时目录、合成 raw 的真实评分，无线上服务调用。
