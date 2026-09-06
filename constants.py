"""V5 记忆系统共享常量 (2026-08-20 audit fix).

历史问题:
  - `PROMOTE_WEIGHT` / `PROMOTE_ACCESS` 在 importance.py 和 reflect/registry.py
    两处各自定义 (0.55 / 2), 加 registry.py 拼成 `PROMOTE_ACCESSES`, 改一个忘改
    另一个就会出怪 bug。
  - 现在所有 promote/archive 阈值只在这里定义; 改值只动一处。

用法:
    from memory_v5.constants import PROMOTE_WEIGHT, PROMOTE_ACCESS, PROMOTE_EI, ARCHIVE_WEIGHT
"""

# 晋升/归档阈值 (单一口径 — lifecycle.py / reflect/registry.py 都从这里取)
PROMOTE_WEIGHT = 0.55
PROMOTE_ACCESS = 2
PROMOTE_EI = 0.6
ARCHIVE_WEIGHT = 0.45