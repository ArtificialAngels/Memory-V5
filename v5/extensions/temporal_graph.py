"""
temporal_graph.py — V5 时序图谱 + dissonance supersede (骨架 / EXPERIMENTAL)
============================================================================

现状(上一轮已确认)
------------------
V5 已有实体图 (eg_entities / eg_edges / eg_episodic / eg_activations) 与矛盾检测
(dissonance.detect_dissonance), 但:

  1. 缺 fact 级时效窗口 valid_from / valid_to。
     eg_activations.expires_at 只记"最近被访问", 不是"事实何时为真";
     检索 `spreading_activation_search` 用 ORDER BY importance DESC, created_at DESC,
     **根本没用时效过滤**。

  2. dissonance 发现矛盾只 `_record_dissonance`(记一条 type='dissonance' 事件),
     **不自动作废旧事实**。导致"用户住 X"变"用户住 Y"后两条共存, 旧值可能被捞回。

对比 Graphiti: 旧事实 valid_to=now, 新事实 valid_from=now, 检索永远取当前值 ——
这正是它在 LongMemEval 领先 Mem0 的主因 (63.8% vs 49%, 差在"事实随时间变能取当前值")。

本骨架在现有 eg_* 表 (及 memory 事实表) 上加 valid_from/valid_to, 并把 dissonance
从"只记录"升级为"触发 supersede"(检测即更替当前值)。

接入点(详见同目录 EXTENSIONS.md)
---------------------------------
  - 启动时跑 apply_migration() 加列(幂等);
  - 在 dissonance._record_dissonance 之后调用 resolve_dissonance_supersede();
  - 用 retrieve_temporal (现位于 v5.memory_retrieval, 2026-08-14 迁移)
    / spreading_activation_search_temporal 包裹原检索。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger("ikaros.v5.ext.temporal_graph")


# ─── 1) 迁移: 加 valid_from / valid_to ─────────────────────────

def apply_migration() -> None:
    """给 memory + eg 表加时效窗口列 (NULL = 永久 / 未闭合)。幂等。"""
    from v5 import store
    from v5.entity_graph import eg_conn

    # memory 事实主表: dissonance 检测的就是 type='fact'/'preference'
    mem_ddl = [
        "ALTER TABLE memory ADD COLUMN valid_from REAL",
        "ALTER TABLE memory ADD COLUMN valid_to REAL",
    ]
    # 实体图(时序图谱)
    eg_ddl = [
        "ALTER TABLE eg_entities ADD COLUMN valid_from REAL",
        "ALTER TABLE eg_entities ADD COLUMN valid_to REAL",
        "ALTER TABLE eg_edges ADD COLUMN valid_from REAL",
        "ALTER TABLE eg_edges ADD COLUMN valid_to REAL",
        "ALTER TABLE eg_episodic ADD COLUMN valid_to REAL",
    ]
    with store.conn() as c:
        for stmt in mem_ddl:
            try:
                c.execute(stmt)
            except Exception:
                pass  # 列已存在 → 幂等跳过
    with eg_conn() as c:
        for stmt in eg_ddl:
            try:
                c.execute(stmt)
            except Exception:
                pass


# ─── 2) supersede: 矛盾时作废旧事实 ────────────────────────────

def supersede_memory(old_id: str, now: Optional[float] = None) -> bool:
    """把一条旧记忆标记为失效 (valid_to=now)。

    仅当该记忆当前 valid_to 为 NULL(仍生效) 才更新, 避免重复覆盖历史失效时间。
    """
    now = now or time.time()
    from v5 import store
    ok = False
    with store.conn() as c:
        cur = c.execute(
            "SELECT valid_to FROM memory WHERE id=?", (old_id,)
        ).fetchone()
        if cur and cur["valid_to"] is None:
            c.execute("UPDATE memory SET valid_to=? WHERE id=?", (now, old_id))
            # store.conn() 退出时默认 rollback, 必须显式提交写事务
            c.commit()
            ok = True
    if ok:
        logger.info("temporal_graph: superseded memory %s @ %.0f", old_id, now)
    return ok


def supersede_entity_attribute(entity_id: str, now: Optional[float] = None) -> int:
    """(Graphiti 式时序图谱路径, 骨架) 当某实体的属性变更时, 把它的旧出边/实体失效。

    注意: V5 当前 eg_edges 无关系类型(只有 weight), 无法精确区分"住在 X"与"喜欢 Y"
    两条边哪条代表被推翻的属性。因此本函数仅作**粗粒度**失效(该实体全部旧出边),
    生产化需要先在 eg_edges 加 relation_type 列 + 让 consolidate 填它。TODO。
    """
    now = now or time.time()
    from v5.entity_graph import eg_conn
    n = 0
    with eg_conn() as c:
        r = c.execute(
            "UPDATE eg_edges SET valid_to=? "
            "WHERE source_entity_id=? AND valid_to IS NULL",
            (now, entity_id),
        )
        n = getattr(r, "rowcount", 0) or 0
    if n:
        logger.info("temporal_graph: superseded %d edges of entity %s", n, entity_id)
    return n


def resolve_dissonance_supersede(
    new_content: str,
    conflicts: list[dict],
    now: Optional[float] = None,
) -> int:
    """dissonance 检测到 contradiction 后调用: 把每条冲突旧事实作废。

    在 dissonance._record_dissonance() 之后接这一句即可启用 supersede 行为。
    Returns: 实际作废的条数。
    """
    now = now or time.time()
    done = 0
    for cf in conflicts:
        old_id = cf.get("old_id")
        if old_id and supersede_memory(old_id, now):
            done += 1
    if done:
        logger.info("temporal_graph: superseded %d stale facts for new=%r",
                    done, new_content[:60])
    return done


# ─── 3) 时效感知检索 (过滤 / 降权过期事实) ─────────────────────

# _valid_to_map 已迁至 v5.store.valid_to_map; retrieve_temporal 已迁至
# v5.memory_retrieval.retrieve_temporal (2026-08-14 解开
# temporal_graph ↔ memory_retrieval 循环依赖)。


def filter_expired_episodic(memories: list, now: Optional[float] = None) -> list:
    """过滤 spreading_activation_search 结果里的过期 eg_episodic。

    用法: results = filter_expired_episodic(spreading_activation_search(seeds))
    更高效的做法是直接改 entity_graph.py:445 的 SQL 加
    `AND (em.valid_to IS NULL OR em.valid_to > ?)` —— 见 EXTENSIONS.md。
    """
    now = now or time.time()
    if not memories:
        return memories
    from v5.store import valid_to_map
    ids = [m.id for m in memories]
    vt = valid_to_map(ids, "eg_episodic", "id")
    return [m for m in memories
            if vt.get(m.id) is None or vt[m.id] > now]


# ─── 4) (推荐) 直接改 SQL 的时效过滤补丁说明 ──────────────────
#
# entity_graph.py:445 原句:
#   ORDER BY em.importance DESC, em.created_at DESC
# 改为:
#   WHERE ee.entity_id IN (...) AND (em.valid_to IS NULL OR em.valid_to > ?)
#   ORDER BY (em.valid_to IS NULL) DESC, em.importance DESC, em.created_at DESC
# 让"仍生效"的事实排前面、"已过期"的沉底, 配合 top_k 自然剔除。
# 同样 memory_retrieval.retrieve 的 FTS/向量结果在融合后按 id 批量过滤即可。
