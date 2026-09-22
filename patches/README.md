# patches/

- 现行（打进提交镜像）：`000-interface-compliance`、`101-d1v12-on-base`；清单见 `RELEASE`。
  写法：照 `build/base_exact/`（= L3 实际代码，F54）生成 `a/python/sglang/...` 形式、`patch -p3 --fuzz=0` 可打。
  新补丁编号 1xx，叠在 000 之上。
- `drafts/`：未定稿（102a 单点版，存在 strict-append 退步与 bf16 精度问题，见 plans/active/102-role-track.md）。
- `v0520/`：v0.5.20 线（001/002/003/004），打不上底包，只作 L1 替身参考。
