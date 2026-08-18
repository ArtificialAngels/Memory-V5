# 详细说明见 docs/scripts/core/v5/v5/reflect/registry.md

from __future__ import annotations

import logging
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent  # v5/
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
    """算法去重 (无 LLM): 6h, 归档同类型高相似重复记忆, 保留最强一条.

    2026-08-14 实现 (原空壳返 0): 复用 store._find_similar 的判重思路
    (difflib ratio + 子串包含), 同类型内容高度相似 (ratio >= 0.92 或子串
    包含) 时保留 weight 最高/created 最新的那条, 其余 archived=1 软删
    (与 cleanup "归档不删除" 一致, 可恢复)。不调 LLM, 与决策 A 一致
    (LLM 生成类反思已停用)。排除 conversation (天然相似易误伤)、
    identity/axiom/rule (灵魂核心) 与 v5_key: 结构化记录。
    """
    DEDUP_THRESHOLD = 0.92
    DEDUP_MAX_SCAN = 2000

    def _fn() -> int:
        from v5 import store
        import difflib
        import time as _time
        now = _time.time()
        try:
            with store.conn() as c:
                rows = c.execute(
                    "SELECT id, content, type FROM memory "
                    "WHERE archived = 0 "
                    "AND type NOT IN ('conversation','identity','axiom','rule') "
                    "AND (tags IS NULL OR tags NOT LIKE '%v5_key:%') "
                    "ORDER BY type, weight DESC, created DESC LIMIT ?",
                    (DEDUP_MAX_SCAN,),
                ).fetchall()
        except Exception as exc:
            logger.debug("dedup: scan failed (%s)", exc)
            return 0

        dup_ids: list[int] = []
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(r["type"], []).append(r)
        for _typ, items in groups.items():
            keepers: list[tuple[str, int]] = []  # (规范化 content, id)
            for r in items:
                content = " ".join((r["content"] or "").split())
                is_dup = False
                for kc, _kid in keepers:
                    try:
                        ratio = difflib.SequenceMatcher(None, kc, content).ratio()
                    except Exception:
                        ratio = 0.0
                    if (ratio >= DEDUP_THRESHOLD
                            or (len(kc) >= 8 and kc in content)
                            or (len(content) >= 8 and content in kc)):
                        is_dup = True
                        break
                if is_dup:
                    dup_ids.append(r["id"])
                else:
                    keepers.append((content, r["id"]))

        if dup_ids:
            try:
                with store.conn() as c:
                    c.executemany(
                        "UPDATE memory SET archived = 1, archived_at = ? WHERE id = ?",
                        [(now, i) for i in dup_ids],
                    )
                    c.commit()
            except Exception as exc:
                logger.debug("dedup: archive failed (%s)", exc)
                return 0
        if dup_ids:
            logger.info("dedup: archived %d duplicate memories", len(dup_ids))
        return len(dup_ids)

    return ReflectOp(
        name="dedup",
        fn=_fn,
        interval_sec=DEFAULT_DEDUP_INTERVAL,
        last_run_key="last_dedup",
    )


def make_promote_op() -> ReflectOp:
    """短期 → 长期晋升: 1h (原12h), 纯算法.

    2026-08-02 加速: 原 12h + weight>=0.7 + access>=3 门槛过高, 大量短期记忆
    永远停留 short_term, 崩溃即失。放宽到 weight>=0.55 或 access>=2,
    配合 1h 频率, 短期记忆转存窗口从 12h 缩到 1h。
    """
    from v5 import store

    PROMOTE_WEIGHT = 0.55
    PROMOTE_ACCESSES = 2

    def _fn() -> int:
        with store.conn() as c:
            cur = c.execute(
                "UPDATE memory SET short_term = 0, long_term = 1 "
                "WHERE short_term = 1 AND archived = 0 "
                "  AND (weight >= ? OR access_count >= ?)",
                (PROMOTE_WEIGHT, PROMOTE_ACCESSES),
            )
            n = cur.rowcount
            # 关键: conn() 退出默认 rollback, 写操作必须显式 commit
            # (2026-08-02 修复: 原实现无 commit, 短期→长期转存从未真正落库)
            c.commit()
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
    """自动清理: 6h, 归档低 weight / 过期 conversation / 过期 decision.

    2026-08-02 改归档不删除: 原实现直接 DELETE, 短期记忆被物理抹除无转存——
    进程崩溃或 promote 未跑时低权重记忆永久丢失。现改为标记 archived=1
    (检索默认排除, 数据保留可恢复)。
    """
    from v5 import store
    import time

    def _fn() -> int:
        now = time.time()
        seven_days = 7 * 86400
        thirty_days = 30 * 86400
        archived = 0
        with store.conn() as c:
            # conversation > 7d → 归档 (原 DELETE)
            cur = c.execute(
                "UPDATE memory SET archived = 1, archived_at = ? "
                "WHERE type = 'conversation' AND created < ? AND archived = 0",
                (now, now - seven_days),
            )
            archived += cur.rowcount
            # 低 weight 非灵魂 → 归档 (原 DELETE; 保底 weight 下调到 0.45,
            # 0.4-0.45 的也归档而非删, 进一步降低误删面)
            cur = c.execute(
                "UPDATE memory SET archived = 1, archived_at = ? "
                "WHERE weight < 0.45 "
                "AND type NOT IN ('identity', 'axiom', 'rule') AND archived = 0",
                (now,),
            )
            archived += cur.rowcount
            # decision > 30d → 归档 (原 DELETE)
            cur = c.execute(
                "UPDATE memory SET archived = 1, archived_at = ? "
                "WHERE type = 'decision' AND created < ? AND archived = 0",
                (now, now - thirty_days),
            )
            archived += cur.rowcount
            # 关键: conn() 退出默认 rollback, 写操作必须显式 commit
            # (2026-08-02 修复: 原实现无 commit, 归档从未真正落库)
            c.commit()
        if archived:
            logger.info("cleanup: archived %d memories (no physical delete)", archived)
        return archived

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


def make_memory_promote_op() -> ReflectOp:
    """记忆两档分层桥接 (阶段 2, 借鉴 cognee 快缓存→永久库): 6h.

    高频/强化/长期记忆晋升 long_term=1 (排序有 boost), 反向 90 天冷记忆回收。
    纯 SQL 侧, 无 LLM 成本。
    """
    MEM_PROMOTE_INTERVAL = 6 * 3600  # 6h

    def _fn() -> int:
        from v5 import store
        import time
        now = time.time()
        promoted = demoted = 0
        try:
            with store.conn() as c:
                # 先回收: long_term=1 且 90 天零访问 → 降回 short_term
                # (必须放在晋升前: 同一事务内刚晋升的行若 last_accessed=0 会立即被回收)
                # 2026-08-14 修复: 原条件 `access_count=0 AND (last_accessed=0 OR ...)`
                # 会把从未被访问的历史合并行 (last_accessed=0) 无条件降级,
                # 与 promote_op 打架——promote 901 条 → memory_promote 立刻回收 901 条,
                # 全部卡在 short=0/long=0。只回收"有访问史但 90 天未访问"的行。
                cur2 = c.execute(
                    "UPDATE memory SET long_term = 0 WHERE long_term = 1 AND "
                    "  access_count = 0 AND last_accessed > 0 "
                    "  AND (? - last_accessed > 90 * 86400)",
                    (now,),
                )
                demoted = getattr(cur2, "rowcount", 0) or 0
                # 再晋升: 高频访问 / 强化 / 超过 30 天的记忆 → long_term=1
                cur = c.execute(
                    "UPDATE memory SET long_term = 1 WHERE long_term = 0 AND ("
                    "  access_count >= 3 OR reinforcement >= 1.0 "
                    "  OR (created > 0 AND ? - created > 30 * 86400)"
                    ")",
                    (now,),
                )
                promoted = getattr(cur, "rowcount", 0) or 0
                c.commit()
        except Exception as exc:
            logger.debug("memory_promote: failed (%s)", exc)
            return 0
        if promoted or demoted:
            logger.info("memory_promote: %d promoted, %d demoted", promoted, demoted)
        return promoted + demoted

    return ReflectOp(
        name="memory_promote",
        fn=_fn,
        interval_sec=MEM_PROMOTE_INTERVAL,
        last_run_key="last_memory_promote",
    )


def make_temporal_extract_op() -> ReflectOp:
    """事件+时间戳抽取 (阶段 5, 借鉴 cognee tasks/temporal_graph 的 Event/Interval 模型,
    但不引 graphiti): 24h. 扫描近 24h 新记忆, LLM 抽时间戳写 valid_from;
    抽不出/LLM 失败静默跳过 (fail-open)."""
    TEMPORAL_EXTRACT_INTERVAL = 24 * 3600  # 24h

    def _fn() -> int:
        from v5 import store
        import re as _re
        import time  # 2026-08-14: 缺 import, op 一直 NameError 静默失败
        now = time.time()
        try:
            with store.conn() as c:
                rows = c.execute(
                    "SELECT id, content, type FROM memory "
                    "WHERE created > ? - 86400 AND valid_from IS NULL "
                    "AND type IN ('fact', 'preference', 'emotion') "
                    "LIMIT 50",
                    (now,),
                ).fetchall()
        except Exception as exc:
            logger.debug("temporal_extract: scan failed (%s)", exc)
            return 0
        if not rows:
            return 0

        # LLM 抽取: 返回 ISO 日期 / 相对表达 / NONE
        _PROMPT = (
            "从这段记忆里抽取它发生/成立的时间点。规则:\n"
            "- 有明确时间(如 2026-03-01、昨天、上周、三年前) → 只输出 ISO 日期或今天偏移的"
            "相对日期(YYYY-MM-DD), 不要任何其他文字\n"
            "- 没有时间信息 → 只输出 NONE\n"
            "记忆: {text}"
        )
        updated = 0
        for r in rows:
            try:
                from v5.reflect.llm_client import call_llm
                resp = call_llm(
                    _PROMPT.format(text=(r["content"] or "")[:200]),
                    "", provider="deepseek", max_tokens=16,
                    temperature=0.0, timeout=20,
                )
                answer = (resp.content or "").strip().upper()
            except Exception:
                continue  # LLM 不可用 → 跳过该条, 不阻塞
            ts = None
            m = _re.search(r"\d{4}-\d{2}-\d{2}", answer)
            if m:
                import datetime as _dt
                try:
                    ts = _dt.datetime.strptime(m.group(0), "%Y-%m-%d").timestamp()
                except Exception:
                    ts = None
            if ts is None and answer != "NONE" and answer:
                # 相对日期 (今天/昨天/上周/数字天前) 由 LLM 直接给日期更好; 这里兜底忽略
                continue
            if ts is not None:
                try:
                    with store.conn() as c:
                        c.execute("UPDATE memory SET valid_from = ? WHERE id = ?",
                                  (ts, r["id"]))
                        c.commit()
                    updated += 1
                except Exception:
                    pass
        if updated:
            logger.info("temporal_extract: stamped %d memories", updated)
        return updated

    return ReflectOp(
        name="temporal_extract",
        fn=_fn,
        interval_sec=TEMPORAL_EXTRACT_INTERVAL,
        last_run_key="last_temporal_extract",
    )


def make_retention_op() -> ReflectOp:
    """统一记忆生命周期 (mnemon EI 借鉴): 6h, demote/promote/archive 单轮.

    2026-08-14 落地 (推荐 5): 取代分散的 promote / cleanup / memory_promote
    三个 op, 用单一 EI 公式 + retention_pass 一轮批写, 消除阈值打架。
    """
    RETENTION_INTERVAL = 6 * 3600  # 6h

    def _fn() -> int:
        from v5.lifecycle import retention_pass
        r = retention_pass()
        return r["promoted"] + r["demoted"] + r["archived"]

    return ReflectOp(
        name="retention",
        fn=_fn,
        interval_sec=RETENTION_INTERVAL,
        last_run_key="last_retention",
    )


# ─── 默认 scheduler (V5.2) ─────────────────────────────────

def make_default_scheduler(state: ScheduleState | None = None) -> ReflectScheduler:
    """构造 V5.2 scheduler.

    2026-08-14 决策 A（用户拍板：反思管线 LLM 生成类没用）：
    - 停用 5 个 LLM 生成 op：consolidate / distill / reflect / narrative / self_discovery
      —— 无去重（dedup 从未实现）产生 579 条雷同 user_trait、哲学味叙事、
      思维链泄漏的 emotional_event，且白烧云端 API。op 工厂函数保留，
      需要时可手动调用 consolidate_conversations() / distill() / reflect()。
    - 保留算法类 op：retention（统一生命周期，取代 promote/cleanup/memory_promote）/
      dedup / vector_sync / temporal_extract / reflection_promote / expire_directives
      （记忆生命周期基础设施，与 LLM 生成无关）。
    """
    s = ReflectScheduler(state=state)
    s.register(make_dedup_op())          # 算法去重 (2026-08-14 已实现)
    # V5.7 (2026-08-14): 统一生命周期 retention 取代 promote/cleanup/memory_promote
    # (三者阈值打架, 见 AGENTS.md; 旧工厂函数保留, 默认调度器只用 retention)
    s.register(make_retention_op())
    s.register(make_vector_sync_op())
    # V5.2 新增
    s.register(make_reflection_promote_op())
    s.register(make_expire_directives_op())
    # 阶段 5 新增: 时间戳抽取
    s.register(make_temporal_extract_op())
    return s


def make_vector_sync_op() -> ReflectOp:
    """向量回填/校正: 1h, 确保 v5.db 每条记忆都有 Chroma 向量.

    2026-08-02 改增量: 先取 Chroma 已有 id 集合, 只补缺失的记忆 (原全量 upsert
    会给 hnsw compactor 压力, 且写时同步失败率高——多进程并发写冲突 / chromadb
    缺失环境)。幂等, chromadb / :8587 不可用时静默返 0, 不阻塞其他反思 op。
    """
    from v5 import store as _store
    from v5.search import VectorIndex

    def _fn() -> int:
        try:
            idx = VectorIndex()
        except Exception as e:
            logger.warning("vector_sync: VectorIndex 不可用, 跳过: %s", e)
            return 0
        # 已存在向量 id 集合 (增量判定)
        try:
            existing = set(idx._collection.get(limit=1000000)["ids"])
        except Exception as e:
            logger.warning("vector_sync: 读取已有向量失败, 跳过: %s", e)
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
            if str(r["id"]) in existing:
                continue  # 已有向量, 跳过
            ok = idx.add(int(r["id"]), r["content"], type=r["type"],
                         tags=r["tags"] or "", weight=float(r["weight"]))
            if ok:
                synced += 1
            else:
                failed += 1
        logger.info("vector_sync: synced=%d failed=%d missing=%d total=%d",
                    synced, failed, synced + failed, len(rows))
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
