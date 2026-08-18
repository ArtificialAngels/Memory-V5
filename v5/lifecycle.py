"""统一记忆生命周期 (mnemon 借鉴) — V5.7 (2026-08-14).

把原先分散在 promote / cleanup / memory_promote 三个反思 op 里的生命周期逻辑,
收敛到一个 EI (Effective Importance, 有效重要性) 公式 + 一个 retention_pass,
消除三个 op 阈值打架的机制性根源 (见 AGENTS.md "promote/memory_promote 打架")。

EI 公式 (mnemon, 适配 Ikaros 字段):
    EI = weight × (1 + reinforcement×0.5) × access_factor × decay_factor
      access_factor = log2(access_count + 1) + 1   (访问越多越重要)
      decay_factor  = 0.5 ^ (days_since_access / 30)  (30 天半衰期; 无访问史不衰减)

retention_pass 单事务做: demote(90 天冷记忆降级) → promote(高频/强化/老/EI 高
晋升) → archive(conversation 7d / decision 30d / 低 weight 归档), 与旧 op 阈值
一致, 但只用一轮 SQL 批写, 不再有 promote 后立刻被回收的窗口。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("ikaros.v5.lifecycle")

# 灵魂核心类型不归档 (与 cleanup 一致)
_SOUL_TYPES = {"identity", "axiom", "rule"}

# P2 融汇 (2026-08-14): EI 公式与阈值收敛到 v5.importance (单一口径,
# store.upsert 写时强化 / _score_items 检索 signals.ei / retention 生命周期共用)。
from v5.importance import (  # noqa: F401
    effective_importance,
    memory_importance,
    PROMOTE_ACCESS,
    PROMOTE_WEIGHT,
    PROMOTE_EI,
    ARCHIVE_WEIGHT,
)


def retention_pass(now: float | None = None) -> dict:
    """统一生命周期一轮: demote → promote → archive, 单事务批写。

    返回 {"promoted": n, "demoted": n, "archived": n}。纯算法, 无 LLM。
    """
    import time as _t
    from v5 import store

    now = now or _t.time()
    promote_ids: list[int] = []
    demote_ids: list[int] = []
    archive_ids: list[int] = []

    try:
        with store.conn() as c:
            rows = c.execute(
                "SELECT id, type, weight, access_count, last_accessed, created, "
                "long_term, short_term, archived, reinforcement FROM memory "
                "WHERE archived = 0"
            ).fetchall()
    except Exception as exc:
        logger.debug("retention: scan failed (%s)", exc)
        return {"promoted": 0, "demoted": 0, "archived": 0}

    for r in rows:
        ei = effective_importance(
            r["weight"], r["access_count"], r["last_accessed"], now,
            r["reinforcement"],
        )
        age_days = (now - float(r["created"])) / 86400.0 if r["created"] else 0.0
        days_acc = (now - float(r["last_accessed"])) / 86400.0 if r["last_accessed"] else None

        # demote: long_term 且有访问史但 90 天未访问 → 降 short
        if (r["long_term"] and r["access_count"] == 0
                and days_acc is not None and days_acc > 90):
            demote_ids.append(r["id"])
        # promote: short → long (高频/强化/30 天老/EI 高)
        elif (r["short_term"] and (
                r["access_count"] >= PROMOTE_ACCESS
                or r["weight"] >= PROMOTE_WEIGHT
                or r["reinforcement"] >= 1.0
                or age_days > 30
                or ei >= PROMOTE_EI)):
            promote_ids.append(r["id"])
        # archive (cleanup 规则; 排除灵魂核心)
        if r["type"] not in _SOUL_TYPES and (
            (r["type"] == "conversation" and age_days > 7)
            or (r["type"] == "decision" and age_days > 30)
            or r["weight"] < ARCHIVE_WEIGHT
        ):
            archive_ids.append(r["id"])

    # 去冲突: 已晋升/已降级的不归档 (避免同轮 promote 又 archive)
    keep = set(promote_ids) | set(demote_ids)
    archive_ids = [i for i in archive_ids if i not in keep]

    try:
        with store.conn() as c:
            if demote_ids:
                c.executemany(
                    "UPDATE memory SET long_term = 0, short_term = 1 WHERE id = ?",
                    [(i,) for i in demote_ids])
            if promote_ids:
                c.executemany(
                    "UPDATE memory SET short_term = 0, long_term = 1 WHERE id = ?",
                    [(i,) for i in promote_ids])
            if archive_ids:
                c.executemany(
                    "UPDATE memory SET archived = 1, archived_at = ? WHERE id = ?",
                    [(now, i) for i in archive_ids])
            c.commit()
    except Exception as exc:
        logger.debug("retention: write failed (%s)", exc)
        return {"promoted": 0, "demoted": 0, "archived": 0}

    result = {"promoted": len(promote_ids), "demoted": len(demote_ids),
              "archived": len(archive_ids)}
    if any(result.values()):
        logger.info("retention: promoted=%d demoted=%d archived=%d",
                    result["promoted"], result["demoted"], result["archived"])
    return result
