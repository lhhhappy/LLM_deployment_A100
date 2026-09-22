# 105 — 101 角色切分：有续算分块时不再截断新请求（修复双 partial 崩溃，F62）
- 位置：`schedule_policy.py` 截断分支，101 原保护之后插入 `if _role_boundary_token_ids() and has_chunked_req: return OTHER`。
- 仅在角色切分开启时生效；stock 下该路径因续算吃满预算本就到不了（trunc_len<=0），所以对 stock 行为无影响。
- 叠加顺序：000 → 101 → 105 → 110 → 111 → 120 → 130，fuzz=0 全部可打。
- 验证：8 卡 dev N6/N10（pod 任务 013/016，b112），以"无 AssertionError、引擎跑完仍存活"为通过标准。
