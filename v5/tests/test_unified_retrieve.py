"""unified_retrieve 统一检索路由层测试 (阶段 1, 借鉴 cognee recall auto-scope).

验证:
  U1. scope 非法值回退 auto, 空 query 返回 []
  U2. scope="lexical" 仅走 FTS, 空则回退 semantic
  U3. scope="graph" 仅走图扩散, 空则回退 semantic
  U4. scope="tree" 走树域检索 (tree + node_id 注入), tree 缺失降级 auto
  U5. scope="auto" 语义不足时自动补图路 (graph_min_score 过滤)
  U6. 多路命中去重按 id, 取最高分; source 字段正确归一化
  U7. 任一路异常 fail-open, 不阻塞
"""
import sys
from pathlib import Path

# 盘符无关: 脚本位置推导 (tests/ -> v5 -> core)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import v5
import v5.store  # noqa: F401  (确保包属性存在, 供 monkeypatch.setattr(v5, "store", ...))
import v5.memory_retrieval as mr


# ── fixtures: 可注入的假数据源 ──
class FakeMem:
    def __init__(self, mid, content, mtype="fact", weight=0.5, created=1.0e9):
        self.id = mid
        self.content = content
        self.type = mtype
        self.weight = weight
        self.created = created
        self.pad_p = 0.0
        self.pad_a = 0.0


def _semantic(query, **kw):
    return [{
        "id": "m1", "content": "关于 Hermes 路由的旧结论", "type": "fact",
        "weight": 0.5, "tags": "route", "created": 1.0e9,
        "pad_p": 0.0, "pad_a": 0.0, "source": "semantic", "score": 0.7,
    }]


def _semantic_one(query, **kw):
    return [{
        "id": "m1", "content": "唯一一条语义记忆", "type": "fact",
        "weight": 0.5, "tags": "", "created": 1.0e9,
        "pad_p": 0.0, "pad_a": 0.0, "source": "semantic", "score": 0.7,
    }]


def _graph(query, top_k=5, **kw):
    return [{
        "id": "epi:1", "content": "图扩散记忆", "type": "episodic",
        "weight": 0.4, "score": 0.5, "source": "entity_graph", "detail": "",
    }]


def _graph_high(query, top_k=5, **kw):
    return [{
        "id": "epi:9", "content": "图扩散高分记忆", "type": "episodic",
        "weight": 0.4, "score": 0.9, "source": "entity_graph", "detail": "",
    }]


def _noop_graph(query, top_k=5, **kw):
    return []


def _noop_semantic(query, **kw):
    return []


# ── U1: 基本守卫 ──
def test_u1_guard_and_bad_scope(monkeypatch):
    assert mr.unified_retrieve("   ") == []
    monkeypatch.setattr(mr, "retrieve", _semantic)
    out = mr.unified_retrieve("hi", scope="bogus")
    assert out and out[0]["source"] == "semantic"
    assert out[0]["id"] == "m1"


# ── U2: lexical ──
def test_u2_lexical_hit_and_fallback(monkeypatch):
    # 命中: 纯词法结果
    fake_store = type("S", (), {"search": staticmethod(
        lambda q, top_k=5, **kw: [FakeMem("lx1", "词法命中记忆")])})
    monkeypatch.setattr(mr, "retrieve", _noop_semantic)
    monkeypatch.setattr(v5, "store", fake_store)
    out = mr.unified_retrieve("lexical 词", scope="lexical")
    assert out and out[0]["id"] == "lx1" and out[0]["source"] == "lexical"
    # 空 → 回退 semantic
    fake_empty = type("S", (), {"search": staticmethod(lambda q, **kw: [])})
    monkeypatch.setattr(v5, "store", fake_empty)
    monkeypatch.setattr(mr, "retrieve", _semantic)
    out2 = mr.unified_retrieve("fallback", scope="lexical")
    assert out2 and out2[0]["source"] == "semantic" and out2[0]["id"] == "m1"


# ── U3: graph ──
def test_u3_graph_hit_and_fallback(monkeypatch):
    monkeypatch.setattr(mr, "retrieve", _noop_semantic)
    monkeypatch.setattr("v5.search.entity_graph_search", _graph)
    out = mr.unified_retrieve("图", scope="graph")
    assert out and out[0]["id"] == "epi:1" and out[0]["source"] == "graph"
    # 空 → 回退 semantic
    monkeypatch.setattr("v5.search.entity_graph_search", _noop_graph)
    monkeypatch.setattr(mr, "retrieve", _semantic)
    out2 = mr.unified_retrieve("fallback", scope="graph")
    assert out2 and out2[0]["source"] == "semantic"


# ── U4: tree ──
def test_u4_tree_scope(monkeypatch):
    monkeypatch.setattr(mr, "retrieve", _noop_semantic)
    calls = {}

    class FakeTree:
        pass

    def fake_tree_scoped(tree, node_id, query, top_k=5, character=None):
        calls["tree"] = tree
        calls["node_id"] = node_id
        return [{
            "id": "t1", "content": "树域记忆", "type": "fact",
            "weight": 0.5, "tags": "node:n1", "created": 1.0e9,
            "pad_p": 0.0, "pad_a": 0.0, "score": 0.8, "tree_scope": "node",
        }]

    monkeypatch.setattr("v5.extensions.tree_adapter.tree_scoped_retrieve",
                        fake_tree_scoped)
    ft = FakeTree()
    out = mr.unified_retrieve("树", scope="tree", node_id="n1", tree=ft)
    assert out and out[0]["id"] == "t1" and out[0]["source"] == "tree"
    assert calls.get("node_id") == "n1" and calls.get("tree") is ft
    # tree 缺失 → 降级 auto (不崩)
    monkeypatch.setattr(mr, "retrieve", _semantic)
    out2 = mr.unified_retrieve("降级", scope="tree", node_id="n1")
    assert out2 and out2[0]["source"] == "semantic"


# ── U5: auto 补路 ──
def test_u5_auto_graph_backfill(monkeypatch):
    # semantic 只 1 条 → 触发补路 → 图路并入
    monkeypatch.setattr(mr, "retrieve", _semantic_one)
    monkeypatch.setattr("v5.search.entity_graph_search", _graph)
    out = mr.unified_retrieve("auto 补路")
    sources = {o["source"] for o in out}
    assert "semantic" in sources and "graph" in sources
    assert len(out) == 2
    # semantic 足 3 条 → 不补路
    monkeypatch.setattr(mr, "retrieve", _semantic)  # 只有 1 条, 用批量版
    many = _semantic("x") + [{
        "id": f"m{i}", "content": f"记忆 {i}", "type": "fact", "weight": 0.5,
        "tags": "", "created": 1.0e9, "pad_p": 0.0, "pad_a": 0.0,
        "source": "semantic", "score": 0.6,
    } for i in range(2, 4)]
    monkeypatch.setattr(mr, "retrieve", lambda q, **kw: many)
    monkeypatch.setattr("v5.search.entity_graph_search", _graph)
    out2 = mr.unified_retrieve("足量")
    assert all(o["source"] == "semantic" for o in out2)


# ── U6: 去重 + source 归一 ──
def test_u6_dedup_and_norm(monkeypatch):
    # semantic 与 graph 返回同一 id → 去重取高分
    monkeypatch.setattr(mr, "retrieve", _semantic)  # m1 score=0.7
    monkeypatch.setattr("v5.search.entity_graph_search",
                        lambda q, top_k=5, **kw: [{
                            "id": "m1", "content": "关于 Hermes 路由的旧结论",
                            "type": "episodic", "weight": 0.4, "score": 0.9,
                            "source": "entity_graph", "detail": "",
                        }])
    out = mr.unified_retrieve("重复")
    m1s = [o for o in out if o["id"] == "m1"]
    assert len(m1s) == 1
    assert abs(m1s[0]["score"] - 0.9) < 1e-6  # 取高分
    assert m1s[0]["content"]  # 字段归一存在


# ── U7: 异常 fail-open ──
def test_u7_fail_open(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("retrieval exploded")

    monkeypatch.setattr(mr, "retrieve", boom)
    monkeypatch.setattr("v5.search.entity_graph_search", boom)
    monkeypatch.setattr(v5, "store",
                        type("S", (), {"search": staticmethod(boom)}))
    # 全部炸 → 返回空, 不抛
    assert mr.unified_retrieve("all down") == []
    assert mr.unified_retrieve("all down", scope="lexical") == []
    assert mr.unified_retrieve("all down", scope="graph") == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
