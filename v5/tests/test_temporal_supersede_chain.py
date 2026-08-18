"""阶段 2/5 测试: 记忆两档桥接 + dissonance→supersede 闭环 + temporal scope.

验证:
  T1. memory_promote op: 高频/强化/30天记忆晋升 long_term; 90天冷记忆回收
  T2. supersede_memory: 矛盾旧记忆 valid_to 被置, 幂等 (二次调用不覆盖)
  T3. resolve_dissonance_supersede: conflicts 列表逐条作废
  T4. unified_retrieve scope="temporal": 过期事实被过滤
  T5. registry: memory_promote / temporal_extract 已注册进默认调度器
"""
import sys
import tempfile
import time
from pathlib import Path

# 盘符无关: 脚本位置推导 (tests/ -> v5 -> core)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import v5.memory_retrieval as mr
from v5 import store as v5store
from v5.extensions import temporal_graph as tg
from v5.reflect import registry


def _fresh_db():
    """建临时 v5.db 并切换到 store 模块的 DB 路径 (conn() 每次自动建表)."""
    import os
    from pathlib import Path
    tmp = tempfile.mkdtemp(prefix="temporal_test_")
    db = os.path.join(tmp, "v5.db")
    v5store.V5_DB_PATH = Path(db)  # 模块常量直接换 (conn() 用模块级 V5_DB_PATH)
    # 加时效窗口列 (真实环境启动时同样需跑; 幂等)
    tg.apply_migration()
    return db


# ── T1: memory_promote ──
def test_t1_memory_promote_op():
    _fresh_db()
    from v5.reflect import registry as reg
    op = reg.make_memory_promote_op()

    # 高频记忆 → 晋升
    h = v5store.store("高频记忆", type="fact")
    with v5store.conn() as c:
        c.execute("UPDATE memory SET access_count = 5, created = ? WHERE id = ?",
                  (time.time() - 86400, h))
        c.commit()
    # 普通记忆 → 不晋升
    n = v5store.store("普通记忆", type="fact")
    with v5store.conn() as c:
        c.execute("UPDATE memory SET access_count = 0, created = ? WHERE id = ?",
                  (time.time() - 86400, n))
        c.commit()
    # 30 天老记忆 → 晋升
    o = v5store.store("老记忆", type="fact")
    with v5store.conn() as c:
        c.execute("UPDATE memory SET access_count = 0, created = ? WHERE id = ?",
                  (time.time() - 40 * 86400, o))
        c.commit()
    # 90 天冷 long_term 记忆 → 回收
    cold = v5store.store("冷记忆", type="fact")
    with v5store.conn() as c:
        c.execute("UPDATE memory SET long_term = 1, access_count = 0, "
                  "last_accessed = ? WHERE id = ?", (time.time() - 100 * 86400, cold))
        c.commit()

    count = op.fn()
    assert count >= 3
    with v5store.conn() as c:
        lt = dict(c.execute("SELECT id, long_term FROM memory").fetchall())
        st = dict(c.execute("SELECT id, short_term FROM memory").fetchall())
    assert lt[h] == 1 and lt[o] == 1
    assert lt[n] == 0
    assert lt[cold] == 0  # 回收


# ── T2: supersede_memory 幂等 ──
def test_t2_supersede_memory_idempotent():
    _fresh_db()
    mid = v5store.store("旧事实: 用户住北京", type="fact")
    now = time.time()
    assert tg.supersede_memory(str(mid), now) is True
    assert tg.supersede_memory(str(mid), now + 10) is False  # 已失效, 不覆盖
    with v5store.conn() as c:
        vt = c.execute("SELECT valid_to FROM memory WHERE id = ?", (mid,)).fetchone()["valid_to"]
    assert abs(float(vt) - now) < 1e-3


# ── T3: resolve_dissonance_supersede ──
def test_t3_resolve_dissonance_supersede():
    _fresh_db()
    old1 = v5store.store("用户住在北京", type="fact")
    old2 = v5store.store("用户喜欢咖啡", type="preference")
    conflicts = [
        {"old_id": str(old1), "old_content": "用户住在北京", "score": 0.8},
        {"old_id": str(old2), "old_content": "用户喜欢咖啡", "score": 0.7},
    ]
    n = tg.resolve_dissonance_supersede("用户搬去上海了", conflicts)
    assert n == 2
    with v5store.conn() as c:
        rows = c.execute("SELECT id, valid_to FROM memory").fetchall()
        vts = {r["id"]: r["valid_to"] for r in rows}
    assert vts[old1] is not None and vts[old2] is not None


# ── T4: unified_retrieve scope=temporal 过滤过期 ──
def test_t4_temporal_scope_filters_expired(monkeypatch):
    _fresh_db()
    live = v5store.store("当前有效: 用户住在上海", type="fact")
    stale = v5store.store("过期: 用户住在北京", type="fact")
    with v5store.conn() as c:
        c.execute("UPDATE memory SET valid_to = ? WHERE id = ?", (time.time() - 100, stale))
        c.commit()
    # 用真实 DB 走 retrieve_temporal (绕过向量: FTS 命中即可)
    monkeypatch.setattr(mr, "retrieve", lambda q, **kw: [
        {"id": str(live), "content": "当前有效: 用户住在上海", "type": "fact",
         "weight": 0.6, "tags": "", "created": time.time(), "pad_p": 0.0,
         "pad_a": 0.0, "source": "semantic", "score": 0.8},
        {"id": str(stale), "content": "过期: 用户住在北京", "type": "fact",
         "weight": 0.6, "tags": "", "created": time.time(), "pad_p": 0.0,
         "pad_a": 0.0, "source": "semantic", "score": 0.7},
    ])
    out = mr.unified_retrieve("住在哪", scope="temporal")
    ids = [o["id"] for o in out]
    assert str(live) in ids
    assert str(stale) not in ids  # 过期被剔除


# ── T5: registry 注册 ──
def test_t5_registry_has_new_ops():
    sched = registry.make_default_scheduler()
    assert sched.get_op("retention") is not None      # 统一生命周期 (V5.7)
    assert sched.get_op("temporal_extract") is not None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
