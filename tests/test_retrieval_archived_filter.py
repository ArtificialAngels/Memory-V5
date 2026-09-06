"""检索层 archived / 孤儿记忆过滤 (2026-08-30 修复) 回归测试。

背景（实测证据，E:/Ikaros/core/memory_v5/data/v5/v5.db 2026-08-30）：
    库内 1092 条记忆，其中 archived=1 有 753 条（69%）。
    chroma 向量库 1200 条向量里 753 条属于 archived 记忆、110 条是 v5.db 里
    已不存在的孤儿向量。

    archived=1 是 V5 既定的软删语义（retention 淘汰 / dedup 合并 / 过期转存），
    lifecycle、freshness、project_edges、reflect/* 每条读路径都带
    `archived = 0`，唯独检索层没接：
      - 三路融合的 FTS5 走 store.search（自带过滤，安全）
      - **向量路直接查 chroma，从头到尾不看 archived**  ← 本测试守的口子
      - memory_api.search 的结构化 SQL 路径也漏了 archived 条件  ← 一并守

    后果：归档机制在语义检索面前形同虚设 —— 每天被 retention 标死的记忆，
    语义照捞回候选池并顶掉 top_k 名额；孤儿向量召回的 id 在 v5.db 里查不到，
    v5_memory_get 直接空。

设计约束（改这个文件时务必保留）：
    1. fail-open：拿不到存活集（DB 挂/表缺失）必须**不过滤**，绝不能把检索搞挂。
    2. 只拦 id 空间确定是 memory 表主键的路径（fts/vec/time）。graph(eg_*)、
       vault(ThirdSpace) 的 id 不在 memory 表，用存活集过滤会误杀，故不动 _finish。
    3. 过滤发生在 _add 入口而非结果截断 —— 死记忆若只被截断仍占 top_k 名额
       并稀释融合排序。
"""
import contextlib
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # core/ -> import memory_v5

import memory_v5.memory_retrieval as mr  # noqa: E402
import memory_v5.store as store_mod  # noqa: E402
import memory_v5.search as search_mod  # noqa: E402


class _M:
    """store.search 返回的记忆桩（模拟 FTS5 路，自带 archived=0 语义）。"""

    def __init__(self, id, content, type="fact", weight=0.8, created=0.0):
        self.id = id
        self.content = content
        self.type = type
        self.weight = weight
        self.created = created
        self.pad_p = 0.0
        self.pad_a = 0.0
        self.tags = ""


def _vec(items):
    """构造向量路 mock：get_vector_index().search() 返回 items。"""
    class _VI:
        def search(self, q, top_k=5):
            return list(items)
    return _VI()


def _no_threshold(monkeypatch):
    """把融合分下限降到 0，让「是否被过滤」只取决于 archived 过滤本身。

    默认 min_fused_score=0.6，纯 FTS 命中的 raw 分最高才 0.3（fts_weight *
    1/(rank)），会被**评分阈值**正常淘汰，与本文件要测的归档过滤无关。
    不降阈值的话测试断言的是评分逻辑，会掩盖/误报过滤行为。
    """
    base = mr._defaults()
    base["min_fused_score"] = 0.0
    monkeypatch.setattr(mr, "_defaults", lambda: base)
    # 让 retrieve() 读配置的分支失败 → 走 _defaults()（配置层在测试环境不可控）
    try:
        import memory_v5.preprocess_config as pc
        monkeypatch.setattr(pc, "cfg", lambda: (_ for _ in ()).throw(RuntimeError("no cfg")))
    except Exception:
        pass


def _run(monkeypatch, fts, vec, live, threshold=False, **kw):
    """跑一次 retrieve()，注入 FTS/向量结果 + 存活集。"""
    if threshold:
        _no_threshold(monkeypatch)
    # ⚠️ 签名必须收 **kw: retrieve() 用 (query, top_k=, min_weight=, character=)
    #    调用, 少收参数会 TypeError 被 store 的 except 静默吞掉 → FTS 路空,
    #    表现成"结果为空"却没有任何报错, 极易误判成过滤逻辑有问题。
    monkeypatch.setattr(store_mod, "search",
                        lambda q, top_k=5, **kw: list(fts))
    monkeypatch.setattr(search_mod, "get_vector_index", lambda *a, **k: _vec(vec))
    monkeypatch.setattr(mr, "_live_ids", lambda: live)
    mr._RET_CACHE.clear()
    return mr.retrieve("probe", top_k=10, **kw)


# ── S1: 向量路拦截 archived / 孤儿 ────────────────────────────────────

def test_vec_path_drops_archived_and_orphans(monkeypatch):
    """核心回归：archived id + 库中不存在的孤儿 id 一律不进候选池。"""
    live = {"1", "2"}
    vec = [
        {"id": "101", "content": "dead archived", "type": "fact",
         "weight": 0.9, "score": 0.99, "created": 0.0},
        {"id": "999999", "content": "orphan vector", "type": "fact",
         "weight": 0.9, "score": 0.99, "created": 0.0},
        {"id": "1", "content": "live one", "type": "fact",
         "weight": 0.8, "score": 0.9, "created": 0.0},
    ]
    res = _run(monkeypatch, fts=[], vec=vec, live=live)
    ids = [r["id"] for r in res]
    assert "101" not in ids, "archived 记忆泄漏进候选池"
    assert "999999" not in ids, "孤儿向量泄漏进候选池"
    assert "1" in ids, "live 记忆被误杀"


def test_dead_ids_do_not_consume_topk_slots(monkeypatch):
    """死记忆必须在入口剔除，不能只截断结果 —— 否则仍占 top_k 名额。"""
    live = {str(i) for i in range(1, 12)}          # 11 条活记忆
    # 向量路：8 条死记忆分数最高，3 条活记忆垫底
    vec = ([{"id": f"dead{i}", "content": f"d{i}", "type": "fact",
             "weight": 0.9, "score": 0.99, "created": 0.0} for i in range(8)]
           + [{"id": str(i), "content": f"live{i}", "type": "fact",
               "weight": 0.5, "score": 0.4, "created": 0.0} for i in range(1, 4)])
    res = _run(monkeypatch, fts=[], vec=vec, live=live, threshold=True)
    ids = [r["id"] for r in res]
    assert not any(i.startswith("dead") for i in ids)
    # 活记忆全部保留（没被死记忆挤掉名额）
    assert {"1", "2", "3"} <= set(ids)


def test_fts_path_still_passes_live(monkeypatch):
    """FTS5 路本身自带 archived=0，不能被新过滤误伤。"""
    live = {"1", "2"}
    res = _run(monkeypatch,
               fts=[_M(1, "alpha", weight=0.9), _M(2, "beta", weight=0.8)],
               vec=[], live=live, threshold=True)
    assert {r["id"] for r in res} == {"1", "2"}


# ── S2: fail-open 语义（最重要的一条）────────────────────────────────

def test_live_lookup_failure_is_fail_open(monkeypatch):
    """拿不到存活集时必须不过滤 —— 宁可多召回，也不能让检索整体失灵。"""
    vec = [{"id": "101", "content": "unknown status", "type": "fact",
            "weight": 0.9, "score": 0.99, "created": 0.0}]
    res = _run(monkeypatch, fts=[], vec=vec, live=None, threshold=True)
    assert [r["id"] for r in res] == ["101"], "fail-open 失效: 检索被过滤搞挂"


def test_live_ids_returns_none_on_db_error(monkeypatch):
    """_live_ids 自身：DB 抛异常 → 返回 None，异常不外泄。"""
    def boom():
        raise RuntimeError("db gone")
    monkeypatch.setattr(store_mod, "conn", boom)
    mr.invalidate_live_ids()
    assert mr._live_ids() is None


def test_live_ids_caches_and_invalidates(monkeypatch):
    """TTL 缓存生效，且 invalidate_live_ids() 能立刻击穿缓存。"""
    calls = {"n": 0}
    import contextlib

    @contextlib.contextmanager
    def fake_conn():
        calls["n"] += 1
        class _C:
            def execute(self, sql):
                class _R:
                    def fetchall(self_inner):
                        return [(1, 0), (2, 1)]     # id=1 活, id=2 归档
                return _R()
        yield _C()

    monkeypatch.setattr(store_mod, "conn", fake_conn)
    mr.invalidate_live_ids()
    first = mr._live_ids()
    assert first == {"1"}, f"存活集应为 {{'1'}}, 实际 {first}"
    mr._live_ids()
    mr._live_ids()
    assert calls["n"] == 1, f"TTL 缓存未生效, 查询了 {calls['n']} 次"
    mr.invalidate_live_ids()
    mr._live_ids()
    assert calls["n"] == 2, "invalidate_live_ids 未击穿缓存"


# ── S3: memory_api.search 结构化路径 ──────────────────────────────────

def test_structured_search_adds_archived_filter(monkeypatch):
    """结构化路径（v5_project_retrieve / 按 type 筛选走这条）必须带 archived = 0。"""
    seen = {}

    @contextlib.contextmanager
    def fake_conn():
        class _C:
            def execute(self, sql, params=None):
                seen["sql"] = sql
                seen["params"] = params or []

                class _R:
                    def fetchall(self_inner):
                        return []
                return _R()
        yield _C()

    import memory_v5.memory_api as api_mod
    monkeypatch.setattr(api_mod._store, "conn", fake_conn)
    api = api_mod.V5MemoryAPI()
    api.search(domain="project", tags=["v5_project:ikaros"], top_k=5)
    assert "archived = 0" in seen.get("sql", ""), (
        f"结构化检索未过滤 archived: {seen.get('sql')}")


def test_structured_search_falls_back_without_archived_column(monkeypatch):
    """老库没有 archived 列时，去掉该条件重试，不能整体失灵。"""
    import contextlib
    attempts = []

    class _R:
        def fetchall(self_inner):
            return []

    @contextlib.contextmanager
    def fake_conn():
        class _C:
            def execute(self, sql, params=None):
                attempts.append(sql)
                if "archived = 0" in sql:
                    raise Exception("no such column: archived")
                return _R()
        yield _C()

    import memory_v5.memory_api as api_mod
    monkeypatch.setattr(api_mod._store, "conn", fake_conn)
    api = api_mod.V5MemoryAPI()
    out = api.search(domain="project", tags=["v5_project:ikaros"], top_k=5)
    assert len(attempts) == 2, f"应重试一次, 实际执行 {len(attempts)} 次"
    assert "archived = 0" in attempts[0]
    assert "archived = 0" not in attempts[1]
    assert out == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
