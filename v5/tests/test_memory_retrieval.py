"""v5 记忆多路检索单元测试 (R3). 用 monkeypatch 隔离外部依赖."""
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # Ikaros-memory/

import v5.memory_retrieval as mr  # noqa: E402
import v5.store as store_mod  # noqa: E402
import v5.search as search_mod  # noqa: E402


class _M:
    def __init__(self, id, content, type="fact", weight=0.8, created=0.0):
        self.id = id
        self.content = content
        self.type = type
        self.weight = weight
        self.created = created
        self.pad_p = 0.0
        self.pad_a = 0.0


class _Vec:
    """默认向量 mock: 高分中性内容, 各测试可在自己的 scope 覆盖."""
    def search(self, q, top_k=5, min_weight=0.0):
        return [
            {"id": "1", "content": "alpha high", "type": "fact", "weight": 0.8,
             "score": 0.9, "created": 0.0},
            {"id": "2", "content": "beta low", "type": "fact", "weight": 0.6,
             "score": 0.3, "created": 0.0},
        ]


class _VecMatched:
    """向量结果与 fts 同 id/content 高分 (用于类型 boost 验证)."""
    def search(self, q, top_k=5, min_weight=0.0):
        return [
            {"id": "1", "content": "x fact", "type": "fact", "weight": 0.8,
             "score": 0.9, "created": 0.0},
            {"id": "2", "content": "y emotion", "type": "emotion", "weight": 0.8,
             "score": 0.9, "created": 0.0},
        ]


def test_empty_query_returns_empty():
    assert mr.retrieve("") == []
    assert mr.retrieve("   ") == []


def test_retrieve_ranks_and_fuses(monkeypatch):
    monkeypatch.setattr(store_mod, "search",
                        lambda q, top_k=5, min_weight=0.0: [_M(1, "alpha high", weight=0.8)])
    monkeypatch.setattr(search_mod, "get_vector_index", lambda *a, **k: _Vec())
    res = mr.retrieve("alpha", top_k=5)
    assert res, "expected at least one result"
    assert res[0]["id"] == "1"
    assert res[0]["score"] > 0
    assert "alpha" in res[0]["content"]


def test_retrieve_applies_type_boost(monkeypatch):
    # emotion 类型应获 1.2 boost, 在有向量高分时通过阈值并被保留
    monkeypatch.setattr(store_mod, "search",
                        lambda q, top_k=5, min_weight=0.0:
                        [_M(1, "x fact", type="fact", weight=0.8),
                         _M(2, "y emotion", type="emotion", weight=0.8)])
    monkeypatch.setattr(search_mod, "get_vector_index", lambda *a, **k: _VecMatched())
    res = mr.retrieve("x y", top_k=5)
    types = [r["type"] for r in res]
    assert "emotion" in types


def test_retrieve_time_range(monkeypatch):
    now = time.time()
    monkeypatch.setattr(store_mod, "search",
                        lambda q, top_k=5, min_weight=0.0: [])
    monkeypatch.setattr(store_mod, "search_by_time_range",
                        lambda s, e, limit=10: [_M(99, "old event", weight=0.9,
                                                    created=now - 86400 * 100)])
    monkeypatch.setattr(search_mod, "get_vector_index", lambda *a, **k: _Vec())
    res = mr.retrieve("event", top_k=5, time_range=(now - 86400 * 200, now))
    ids = [r["id"] for r in res]
    assert "99" in ids


def test_retrieve_excludes_known(monkeypatch):
    monkeypatch.setattr(store_mod, "search",
                        lambda q, top_k=5, min_weight=0.0: [_M(1, "已知内容ABC", weight=0.9)])
    monkeypatch.setattr(search_mod, "get_vector_index", lambda *a, **k: _Vec())
    res = mr.retrieve("已知内容ABC", top_k=5, exclude=["已知内容ABC"])
    assert all("已知内容ABC" not in r["content"] for r in res)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
