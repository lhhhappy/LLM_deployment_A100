# engine/vllm — vLLM 路线的底包与开发规则

2026-09-25 起，Claude 负责。正式提交只能在主办方底包镜像上加一个小补丁（平台构建：内联 Dockerfile ≤ 64 KiB、没有构建上下文），
所以 vLLM 路线的底包就是**主办方 vLLM 镜像里的那份 vLLM**。SGLang 路线（`engine/sglang/`、`engine/docs/NNN-*.md`）与这里互不共用编号和 tag。

## 底包

| 项目 | 值 |
|---|---|
| 主办方镜像 | `registry.dp.tech/dptech/dp/native/prod-20675/vllm-backport:260918-sm80`（task.md「示例提交」） |
| 其中的 vLLM | 公开 [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) 标签 `v0.13.1` = 提交 `cde54e8ed390aef9e7d0670474365ae033b38a40`；公开镜像 `lazymio/vllm-backport:v0.13.1-sm80`（amd64 清单 `sha256:8094fcba…`） |
| 主办方加的 | `s1_generate` 端点插件（`/generate`、`/flush_cache`，`s1_generate_adapter-1.0.0`）与 `ENV VLLM_PLUGINS=s1_generate,lora_filesystem_resolver,lora_hf_hub_resolver` |
| 运行环境 | Ubuntu 24.04、glibc 2.39、`/usr/bin/python3` 3.12.3、CUDA 13.0.3；vLLM 装在 `/usr/local/lib/python3.12/dist-packages/vllm` |
| 本仓库 | `engine/vllm` = 该提交的源码包（`codeload.github.com/wtdcode/vllm-backport/tar.gz/<提交>`，SHA256 `b3118448…`）去掉 `docs/`、`.buildkite/`、`.github/`；tag `vllm-base-backport-v0.13.1` |

核对（实测，证据 [vllm-backport-identity-20260925](../../../evidence/vllm-backport-identity-20260925/README.md)）：
主办方镜像的 vLLM 安装记录（RECORD，4190 行）与公开镜像逐字节相同；公开镜像里安装的 `vllm/**/*.py` 与该提交源码两边都有的 2466 个逐字节相同，
只在镜像里的 185 个是构建时生成或下载的（`_version.py`、`third_party/`）。上游 `.gitignore` 挡住的 5 个上游跟踪文件用 `git add -f` 保留。

此前基于官方 vLLM `main` a811738a6 的工作（接口插件、A100 移植、101 初版）保存在 tag `vllm-mainline-final`，只作参考：
那条线的 A100 移植与接口在主办方镜像里都已有现成实现。

## 机制

| 编号 | 机制 | 开关 | 状态 |
|---|---|---|---|
| — | 暂无 | — | 101（角色边界 KDA 检查点）按本底包重新实现中 |

## 开发规则

- 只改 `engine/vllm/`。每个机制一个或一组 `engine vllm NNN:` 提交，说明写 `engine/docs/vllm/NNN-*.md`；修正并入所属机制，不另开编号。
- 补丁只改 `vllm/` 包内的 Python 文件：镜像里只有安装后的包（`dist-packages/vllm`），没有源码树里的 `csrc/`、`tests/` 等；改到编译产物的机制不能用补丁交付。
- 机制默认关闭，关闭时等于底包；不支持的组合启动时拒绝，不静默绕开。启动时打印一行 `[ax] vllm mechanisms:`，任务写明期望值并在测量前核对。
- 共享工作区有他人未提交的改动：只按路径暂存自己的文件，不 `git add -A`。

## 提交镜像与开发环境

- 提交镜像：`FROM` 主办方镜像，把 `vllm-base-backport-v0.13.1..<提交>` 的 `vllm/` 补丁打进 `dist-packages`，打之前逐个核对被改文件的 SHA256 等于底包（`scripts/vllm/build_image.sh`，编写中）。
- 开发机：主办方镜像的完整根文件系统副本在 `/sjtu/linhang/arena/vllm-backport/260918-sm80/`（`scripts/vllm/image_export.sh` 经 Bohrium CPU 沙箱导出，分片 SHA256 在沙箱内计算）。
- 8 卡机上的运行方式另行商定；在此之前不向 8 卡机安装或启动。
