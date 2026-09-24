# 061r 启动检查误判

061r 引擎160秒后就绪，TP0记录120=on、122=on、180=off、spec=EAGLE、dcp=1。
任务却要求spec=NEXTN，因此在preflight/warmup/测量前退出，没有测量raw。

这是任务期望值错误。底包speculative_hook.py的真实别名解析函数会把NEXTN改成EAGLE。
启动参数仍保持正式A的NEXTN；只将两个任务G_EXPECT中的预期值改为EAGLE。
063r包含同一错误检查，已停止，避免继续消耗启动时间；没有性能结果。

CPU回归使用真实别名解析函数与061r实际机制日志；两份任务均通过，关闭122或关闭推测解码仍拒绝。
测试文件tests/test_queue_bundle.py共5项通过。

用061s/063s新目录重启同一c92acd5引擎提交、同全量N30、70分钟准入后排空；清缓存不变。
