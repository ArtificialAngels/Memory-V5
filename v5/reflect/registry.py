# 详细说明见 docs/scripts/core/v5/v5/reflect/registry.md

from __future__ import annotations

import logging
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent.parent  # v5/
sys.path.insert(0, str(V5_ROOT.parent))

from v5.reflect.scheduler import (  # noqa: E402
    DEFAULT_CLEANUP_INTERVAL,
    DEFAULT_NARRATIVE_INTERVAL,
    DEFAULT_CONSOLIDATE_INTERVAL,
    DEFAULT_DEDUP_INTERVAL,
    DEFAULT_DISTILL_INTERVAL,
    DEFAULT_PROMOTE_INTERVAL,
    DEFAULT_REFLECT_INTERVAL,
    DEFAULT_VECTOR_SYNC_INTERVAL,
    ReflectOp,
    ReflectScheduler,
    ScheduleState,
)

logger = logging.getLogger("ikaros.memory.v5.registry")


# ─── Op factory ──────────────────────────────────────────────

def make_consolidate_op() -> ReflectOp:
    """对话整合: 1h, 小模型提取 + 大模型验证."""
    from v5.reflect import consolidate

    def _fn() -> int:
        result = consolidate.consolidate_conversations()
        return result.get("consolidated", 0)

    return ReflectOp(
        name="consolidate",
        fn=_fn,
        interval_sec=DEFAULT_CONSOLIDATE_INTERVAL,
        last_run_key="last_consolidate",
    )


def make_dedup_op() -> ReflectOp:
    """去重: 6h, 小模型判断关系 (V3 兼容, V4 暂未单独实现, 用 v3 路径)."""
    def _fn() -> int:
        # V4 暂未独立实现 dedup, 直接返 0 (待 Phase 3.5)
        logger.debug("dedup: V4 待实现, 跳过")
        return 0

    return ReflectOp(
        name="dedup",
        fn=_fn,
        interval_sec=DEFAULT_DEDUP_INTERVAL,
        last_run_key="last_dedup",
    )


def make_promote_op() -> ReflectOp:
    """短期 → 长期晋升: 12h, 纯算法."""
    from v5 import store

    PROMOTE_WEIGHT = 0.7
    PROMOTE_ACCESSES = 3

    def _fn() -> int:
        with store.conn() as c:
            cur = c.execute(
                "UPDATE memory SET short_term = 0, long_term = 1 "
                "WHERE short_term = 1 "
                "  AND weight >= ? "
                "  AND access_count >= ?",
                (PROMOTE_WEIGHT, PROMOTE_ACCESSES),
            )
            n = cur.rowcount
        if n:
            logger.info("promote: %d memories promoted to long-term", n)
        return n

    return ReflectOp(
        name="promote",
        fn=_fn,
        interval_sec=DEFAULT_PROMOTE_INTERVAL,
        last_run_key="last_promote",
    )


def make_distill_op() -> ReflectOp:
    """灵魂蒸馏: 24h, 小模型 (V4 已有 distill.distill)."""
    from v5.reflect import distill

    def _fn() -> int:
        result = distill.distill()
        return result.get("distilled", 0)

    return ReflectOp(
        name="distill",
        fn=_fn,
        interval_sec=DEFAULT_DISTILL_INTERVAL,
        last_run_key="last_distill",
    )


def make_reflect_op() -> ReflectOp:
    """灵魂层反思: 3h, 大模型 (V4 已有 distill.reflect).

    哥哥 7-12 要求提升自省频率: 从记忆反推"我是谁 / 我怎么变了".
    3h 一次, 容许云 API 消耗增加.
    """
    from v5.reflect import distill

    def _fn() -> int:
        result = distill.reflect()
        return result.get("reflections", 0)

    return ReflectOp(
        name="reflect",
        fn=_fn,
        interval_sec=DEFAULT_REFLECT_INTERVAL,  # 3h
        last_run_key="last_reflect",
    )


def make_cleanup_op() -> ReflectOp:
    """自动清理: 6h, 删除低 weight / 过期 conversation / 过期 decision."""
    from v5 import store
    import time

    def _fn() -> int:
        now = time.time()
        seven_days = 7 * 86400
        thirty_days = 30 * 86400
        deleted = 0
        with store.conn() as c:
            # conversation > 7d
            cur = c.execute(
                "DELETE FROM memory WHERE type = 'conversation' AND created < ?",
                (now - seven_days,),
            )
            deleted += cur.rowcount
            # 低 weight 非灵魂
            cur = c.execute(
                "DELETE FROM memory WHERE weight < 0.4 "
                "AND type NOT IN ('identity', 'axiom', 'rule')"
            )
            deleted += cur.rowcount
            # decision > 30d
            cur = c.execute(
                "DELETE FROM memory WHERE type = 'decision' AND created < ?",
                (now - thirty_days,),
            )
            deleted += cur.rowcount
        if deleted:
            logger.info("cleanup: deleted %d memories", deleted)
        return deleted

    return ReflectOp(
        name="cleanup",
        fn=_fn,
        interval_sec=DEFAULT_CLEANUP_INTERVAL,  # 6h
        last_run_key="last_cleanup",
    )


# ─── V5.2 新增 op: 反思晋升、过期指令清理、反重复语料压缩 ─────

def make_reflection_promote_op() -> ReflectOp:
    """反思晋升: 6h, 自动晋升证据充分的反思到 persona.

    扫描所有 CONFIRMED 状态且 3 天以上的反思,
    满足 evidence_version >= 3 则触发 promote_to_persona.
    """
    from v5.reflections import auto_promote_stale
    REFLECTION_PROMOTE_INTERVAL = 6 * 3600  # 6h

    def _fn() -> int:
        count = auto_promote_stale("")
        if count:
            logger.info("reflection_promote: %d reflections promoted", count)
        return count

    return ReflectOp(
        name="reflection_promote",
        fn=_fn,
        interval_sec=REFLECTION_PROMOTE_INTERVAL,
        last_run_key="last_reflection_promote",
    )


def make_expire_directives_op() -> ReflectOp:
    """过期指令清理: 6h, 标记所有已过期指令为 inactive."""
    from v5.user_directives import expire_old
    EXPIRE_DIRECTIVES_INTERVAL = 6 * 3600  # 6h

    def _fn() -> int:
        count = expire_old()
        if count:
            logger.info("expire_directives: %d directives expired", count)
        return count

    return ReflectOp(
        name="expire_directives",
        fn=_fn,
        interval_sec=EXPIRE_DIRECTIVES_INTERVAL,
        last_run_key="last_expire_directives",
    )


# ─── 默认 scheduler (V5.2) ─────────────────────────────────

def make_default_scheduler(state: ScheduleState | None = None) -> ReflectScheduler:
    """构造 V5.2 scheduler, 包含所有原始 op + V5.2 新增 op."""
    s = ReflectScheduler(state=state)
    s.register(make_consolidate_op())
    s.register(make_dedup_op())
    s.register(make_promote_op())
    s.register(make_distill_op())
    s.register(make_reflect_op())
    s.register(make_cleanup_op())
    s.register(make_vector_sync_op())
    s.register(make_narrative_op())
    s.register(make_self_discovery_op())
    # V5.2 新增
    s.register(make_reflection_promote_op())
    s.register(make_expire_directives_op())
    return s


def make_vector_sync_op() -> ReflectOp:
    """向量回填/校正: 24h, 确保 v5.db 每条记忆都有 Chroma 向量 (A2 修复).

    幂等: 直接 upsert 全部记忆, 缺失/失败的单独计数。
    A1 已让 store() 写时同步, 此 op 作崩溃恢复 + 历史回填的安全网。
    chromadb / :8587 不可用时静默返 0, 不阻塞其他反思 op。
    """
    from v5 import store as _store
    from v5.search import VectorIndex

    def _fn() -> int:
        try:
            idx = VectorIndex()
        except Exception as e:
            logger.warning("vector_sync: VectorIndex 不可用, 跳过: %s", e)
            return 0
        synced = 0
        failed = 0
        with _store.conn() as c:
            rows = c.execute(
                "SELECT id, content, type, tags, weight FROM memory"
            ).fetchall()
            # 纯 SELECT 后显式提交, 释放 SHARED 读锁, 避免阻塞其他进程写入
            c.commit()
        for r in rows:
            ok = idx.add(int(r["id"]), r["content"], type=r["type"],
                         tags=r["tags"] or "", weight=float(r["weight"]))
            if ok:
                synced += 1
            else:
                failed += 1
        logger.info("vector_sync: synced=%d failed=%d total=%d",
                    synced, failed, len(rows))
        return synced

    return ReflectOp(
        name="vector_sync",
        fn=_fn,
        interval_sec=DEFAULT_VECTOR_SYNC_INTERVAL,
        last_run_key="last_vector_sync",
    )


def make_narrative_op() -> ReflectOp:
    """自我叙事连续性: 30d, 大模型 (V5 #7).

    每月生成连贯的自我叙事 — "这个月我变成了什么样".
    """
    from v5.narrative import generate_narrative

    def _fn() -> int:
        result = generate_narrative()
        if result.get("narrative"):
            return 1
        return 0

    return ReflectOp(
        name="narrative",
        fn=_fn,
        interval_sec=DEFAULT_NARRATIVE_INTERVAL,
        last_run_key="last_narrative",
    )


def make_self_discovery_op() -> ReflectOp:
    """自我认知探索: 3h, Hermes Agent 分析自身架构.

    每 3h 读关键文件 + 调 Hermes 分析项目结构,
    让伊卡洛斯了解自己的真实架构, 而非被写死的描述。"""
    DEFAULT_SELF_DISCOVERY_INTERVAL = 3 * 3600  # 3h

    def _fn() -> int:
        from v5.self_discovery import self_discover
        return self_discover()

    return ReflectOp(
        name="self_discovery",
        fn=_fn,
        interval_sec=DEFAULT_SELF_DISCOVERY_INTERVAL,
        last_run_key="last_self_discovery",
    )
