# vllm 000 — 启动时打印 `[ax] vllm mechanisms:` 行

调度器初始化时打印一行，写明底包、叠在底包上的引擎提交和各机制状态，任务据此核对 `G_EXPECT`：

```
[ax] vllm mechanisms: base=vllm-backport-v0.13.1@cde54e8e commit=<12 位提交号>
```

- 提交号优先取环境变量 `AX_ENGINE_COMMIT`（开发机运行包设置），否则读提交镜像写入的 `/opt/ax/vllm_engine_commit`，都没有时为 `none`。
  文件名与 SGLang 镜像已有的 `/opt/ax/engine_commit` 分开：在 SGLang 镜像的 8 卡机上运行 vLLM 时不会误读成 SGLang 的提交。
- 以后每个机制在这一行追加自己的状态（如 `101=on:...`/`101=off`）。
- 除这一行日志外行为与底包相同。

| 层级 | 内容 | 结果 |
|---|---|---|
| CPU | `tests/test_ax_mechanisms.py`：环境变量优先、文件兜底、缺省 `none`、整行格式 | 未运行（按用户安排，先做不依赖运行的准备） |
