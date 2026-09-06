"""集群新鲜度计数器 + 水印 (OpenViking freshness_policy 借鉴) — F5.

问题: vector_sync / rule_entity_extract 等反思 op 每 6h 全表扫描 memory 找待处理
记忆, 规模一大就慢 (AGENTS.md 记 "rule_entity_extract 2871 一次补齐" 历史欠账)。
OpenViking 的 freshness 计数器: 每父节点带 total/pending, pending/total >= ratio
才刷新父摘要, 避免每写必刷(贵)与 cron 刷(陈旧窗口)两极端。

Ikaros 适配: 按 bucket (= memory.type, 或 v5_project:<proj>) 维护水印 last_refreshed_at
+ pending 计数。待处理 op (vector_sync / rule_entity_extract 等) 用 watermark(bucket)
做 `WHERE created > ?` 增量扫描, 不再全表扫。本模块只提供基础设施 + 一个 reconcile op;
现有 op 增量采用 watermark 是独立渐进改造。

纯算法 + SQLite (落 v5.db), 无 LLM。fail-open。
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("ikaros.v5.freshness")

DEFAULT_REFRESH_RATIO = 0.10  # pending/total >= 10% 才 due (OpenViking 默认 0.10)
DEFAULT_MIN_PENDING = 5        # 或 pending 绝对值 >= 5 (小桶也能触发一次)

_FRESHNESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS cluster_freshness (
    bucket TEXT PRIMARY KEY,
    total_entries INTEGER NOT NULL DEFAULT 0,
    pending INTEGER NOT NULL DEFAULT 0,
    last_refreshed_at REAL NOT NULL DEFAULT 0
);
"""


def _ensure_schema(c) -> None:
    c.executescript(_FRESHNESS_SCHEMA)


def _bucket_for_type(mem_type: str) -> str:
    """type → bucket key (默认按 type 分桶)."""
    return (mem_type or "default").strip().lower() or "default"


def record_write(bucket: str) -> None:
    """写记忆时调: total+1, pending+1 (幂等写, 失败静默)."""
    from memory_v5 import store
    try:
        with store.committed() as c:
            _ensure_schema(c)
            c.execute(
                "INSERT INTO cluster_freshness (bucket, total_entries, pending, last_refreshed_at) "
                "VALUES (?, 1, 1, 0) "
                "ON CONFLICT(bucket) DO UPDATE SET "
                "  total_entries = total_entries + 1, "
                "  pending = pending + 1",
                (bucket,),
            )
    except Exception as exc:
        logger.debug("freshness: record_write failed for %s (%s)", bucket, exc)


def watermark(bucket: str) -> float:
    """取该 bucket 的水印 (上次刷新时间). 供增量扫描: WHERE created > watermark."""
    from memory_v5 import store
    try:
        with store.conn() as c:
            _ensure_schema(c)
            row = c.execute(
                "SELECT last_refreshed_at FROM cluster_freshness WHERE bucket = ?",
                (bucket,),
            ).fetchone()
            return float(row["last_refreshed_at"]) if row else 0.0
    except Exception:
        return 0.0


def set_watermark(bucket: str, ts: float, *, pending: int = 0) -> None:
    """刷新完成: 更新水印 + 重置 pending (失败静默)."""
    from memory_v5 import store
    try:
        with store.committed() as c:
            _ensure_schema(c)
            c.execute(
                "INSERT INTO cluster_freshness (bucket, total_entries, pending, last_refreshed_at) "
                "VALUES (?, 0, ?, ?) "
                "ON CONFLICT(bucket) DO UPDATE SET "
                "  pending = ?, last_refreshed_at = ?",
                (bucket, pending, ts, pending, ts),
            )
    except Exception as exc:
        logger.debug("freshness: set_watermark failed for %s (%s)", bucket, exc)


def reconcile_totals() -> dict:
    """按 type 重建 total_entries (修漂移; 计 pending = total - 已处理).

    返回 {bucket: {total, pending, due}}. due = pending/total >= ratio 或 pending >= min.
    本 op 不真处理 (那是 vector_sync/rule_entity_extract 的事), 只对账 + 标 due。
    """
    from memory_v5 import store
    try:
        with store.conn() as c:
            _ensure_schema(c)
            # 真实 total (按 type)
            real = c.execute(
                "SELECT type, COUNT(*) AS n FROM memory WHERE archived = 0 GROUP BY type"
            ).fetchall()
            real_by_type = {r["type"] or "default": int(r["n"]) for r in real}
            # 现有水印
            rows = c.execute(
                "SELECT bucket, total_entries, pending, last_refreshed_at FROM cluster_freshness"
            ).fetchall()
            cur = {r["bucket"]: dict(r) for r in rows}
    except Exception as exc:
        logger.debug("freshness: reconcile scan failed (%s)", exc)
        return {}

    now = time.time()
    out: dict = {}
    # 合并: 真实 type 桶 + 已记录的非 type 桶 (如 v5_project:xxx)
    all_buckets = set(real_by_type) | set(cur)
    due_updates: list[tuple] = []
    for b in all_buckets:
        real_total = real_by_type.get(b, cur.get(b, {}).get("total_entries", 0))
        prev = cur.get(b, {})
        last_ts = float(prev.get("last_refreshed_at", 0))
        # pending = 真实总数 - 自上次刷新以来已知的处理量. 用 watermark 估:
        # 自 last_ts 后新建的数量 (archived=0)
        try:
            with store.conn() as c2:
                if last_ts > 0:
                    r = c2.execute(
                        "SELECT COUNT(*) AS n FROM memory WHERE archived = 0 "
                        "  AND created > ? AND LOWER(COALESCE(type,'default')) = ?",
                        (last_ts, b.lower()),
                    ).fetchone()
                    pending = int(r["n"]) if r else 0
                else:
                    pending = real_total  # 从未刷新 → 全部 pending
        except Exception:
            pending = real_total
        ratio = (pending / real_total) if real_total > 0 else 0.0
        due = (real_total > 0 and ratio >= DEFAULT_REFRESH_RATIO) or pending >= DEFAULT_MIN_PENDING
        out[b] = {"total": real_total, "pending": pending, "ratio": round(ratio, 3),
                  "last_refreshed_at": last_ts, "due": due}
        if due:
            due_updates.append((b,))
    return out


def due_buckets() -> list[str]:
    """返回需要刷新的 bucket 列表 (供 op 决策)."""
    stats = reconcile_totals()
    return [b for b, s in stats.items() if s.get("due")]


def stats_summary() -> dict:
    """供 v5 工具/面板: 全桶新鲜度概览."""
    return reconcile_totals()
