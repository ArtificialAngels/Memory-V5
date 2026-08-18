"""统一记忆重要性 (P2 融汇) — 2026-08-14.

把分散在 store.upsert (写时强化) / memory_retrieval._score_items (检索排序) /
lifecycle.retention_pass (生命周期归档) 三处的"重要性"概念, 收敛到**单一 EI 公式**:

    EI = weight × (1 + reinforcement×0.5) × log2(access_count+1) × 0.5^(days/30)

  - 写侧: upsert 合并 +reinforcement → EI 上升 (被合并越多越重要)
  - 检索侧: EI 作为 signals.ei 透出, 供上层/LLM 解释重要性
  - 生命周期: retention_pass 用 EI 做 promote/archive 判定

三处共用本模块的 effective_importance / memory_importance, 不再各算各的。
"""

from __future__ import annotations

# 晋升/归档阈值 (与旧 lifecycle 一致)
PROMOTE_ACCESS = 2
PROMOTE_WEIGHT = 0.55
PROMOTE_EI = 0.6
ARCHIVE_WEIGHT = 0.45


def effective_importance(
    weight: float,
    access_count: int,
    last_accessed: float,
    now: float,
    reinforcement: float = 0.0,
) -> float:
    """有效重要性 EI (纯函数, 可单测; mnemon 借鉴 + Ikaros reinforcement)."""
    w = max(0.0, min(1.0, float(weight)))
    # access_factor: 访问越多越重要 (log2(access+1)+1)
    access_factor = max(1.0, (int(access_count) + 1).bit_length())
    # decay_factor: 30 天半衰期; 无访问史 (last_accessed<=0) 不衰减
    if last_accessed and float(last_accessed) > 0:
        days = max(0.0, (now - float(last_accessed)) / 86400.0)
        decay = 0.5 ** (days / 30.0)
    else:
        decay = 1.0
    # reinforcement: 被合并/强化越多越重要 (封顶 +0.5)
    rein = 1.0 + min(0.5, float(reinforcement) * 0.5)
    return w * rein * access_factor * decay


def memory_importance(row, now: float | None = None) -> float:
    """从 sqlite3.Row / dict / store.Memory 计算 EI (统一入口).

    row 需含 weight / access_count / last_accessed / reinforcement。
    """
    import time as _t

    now = _t.time() if now is None else now

    def _g(key, default=0.0):
        try:
            return row[key]
        except Exception:
            try:
                return getattr(row, key, default)
            except Exception:
                return default

    return effective_importance(
        _g("weight", 0.5), _g("access_count", 0),
        _g("last_accessed", 0.0), now, _g("reinforcement", 0.0),
    )
