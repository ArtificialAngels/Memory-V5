"""v5.search / v5.memory_retrieval 运行时缓存单元测试 (哥哥优化项).

覆盖:
  - embedding LRU 缓存: 同文本二次调用跳过网络
  - VectorIndex 单例: 同进程复用, refresh/过期重建
  - retrieve 结果短 TTL: 同 query 短期内直接返回, 跳过 embedding+chroma
全用 monkeypatch 隔离真实网络/磁盘.
"""
import sys
from pathlib import Path
from collections import OrderedDict

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # Ikaros-memory/

import v5.search as search_mod


# ── embedding LRU 缓存 ──
def test_embedding_cache_skips_network_on_repeat(monkeypatch):
    calls = {"n": 0}

    def _fake_fetch(text, task="query"):
        calls["n"] += 1
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(search_mod, "_fetch_embedding", _fake_fetch)
    # 清空缓存, 关掉配置开关以防干扰
    monkeypatch.setattr(search_mod, "_EMBED_CACHE", OrderedDict())
    monkeypatch.setattr(search_mod, "_cache_enabled", lambda: True)

    a = search_mod._get_embedding("同一个句子", task="query")
    b = search_mod._get_embedding("同一个句子", task="query")  # 同 key → 命中
    assert a == b == [0.1, 0.2, 0.3]
    assert calls["n"] == 1, "同文本二次调用应跳过网络 (只打 1 次)"

    # 不同 task 前缀 → 不同 key → 再打一次
    c = search_mod._get_embedding("同一个句子", task="document")
    assert calls["n"] == 2


def test_embedding_cache_disabled_returns_uncached(monkeypatch):
    calls = {"n": 0}

    def _fake_fetch(text, task="query"):
        calls["n"] += 1
        return [9.0]

    monkeypatch.setattr(search_mod, "_fetch_embedding", _fake_fetch)
    monkeypatch.setattr(search_mod, "_cache_enabled", lambda: False)

    search_mod._get_embedding("x")
    search_mod._get_embedding("x")
    assert calls["n"] == 2, "缓存关闭时应每次打网络"


# ── VectorIndex 单例 ──
def test_vector_index_singleton_reuses_instance(monkeypatch):
    made = {"n": 0}

    class _FakeVI:
        pass

    def _factory(persist_dir=None):
        made["n"] += 1
        return _FakeVI()

    monkeypatch.setattr(search_mod, "VectorIndex", _factory)
    monkeypatch.setattr(search_mod, "_VI", {"instance": None, "dir": None, "ts": 0.0})
    # 强制开启单例
    monkeypatch.setattr(search_mod, "_cache_cfg",
                        lambda: {"vector_index_singleton": True, "vector_refresh_seconds": 30})

    i1 = search_mod.get_vector_index()
    i2 = search_mod.get_vector_index()
    assert i1 is i2, "单例应复用同一实例"
    assert made["n"] == 1, "单例应只创建一次"


def test_vector_index_singleton_refresh_recreates(monkeypatch):
    made = {"n": 0}

    class _FakeVI:
        pass

    def _factory(persist_dir=None):
        made["n"] += 1
        return _FakeVI()

    monkeypatch.setattr(search_mod, "VectorIndex", _factory)
    monkeypatch.setattr(search_mod, "_VI", {"instance": None, "dir": None, "ts": 0.0})
    monkeypatch.setattr(search_mod, "_cache_cfg",
                        lambda: {"vector_index_singleton": True, "vector_refresh_seconds": 30})

    i1 = search_mod.get_vector_index()
    i2 = search_mod.get_vector_index(refresh=True)  # 强制刷新
    assert i1 is not i2
    assert made["n"] == 2


def test_vector_index_singleton_disabled_creates_each_time(monkeypatch):
    made = {"n": 0}

    class _FakeVI:
        pass

    def _factory(persist_dir=None):
        made["n"] += 1
        return _FakeVI()

    monkeypatch.setattr(search_mod, "VectorIndex", _factory)
    monkeypatch.setattr(search_mod, "_VI", {"instance": None, "dir": None, "ts": 0.0})
    monkeypatch.setattr(search_mod, "_cache_cfg",
                        lambda: {"vector_index_singleton": False, "vector_refresh_seconds": 30})

    search_mod.get_vector_index()
    search_mod.get_vector_index()
    assert made["n"] == 2, "单例关闭时每轮新建 (原行为)"


# ── retrieve 结果短 TTL ──
def test_retrieve_ttl_skips_on_repeat(monkeypatch):
    import v5.memory_retrieval as mr
    import v5.store as store_mod

    vec_calls = {"n": 0}

    class _Vec:
        def search(self, q, top_k=5, min_weight=0.0):
            vec_calls["n"] += 1
            return [{"id": "1", "content": "x", "type": "fact",
                     "weight": 0.8, "score": 0.9, "created": 0.0}]

    monkeypatch.setattr(store_mod, "search", lambda q, top_k=5, min_weight=0.0: [])
    monkeypatch.setattr(search_mod, "get_vector_index", lambda *a, **k: _Vec())
    monkeypatch.setattr(mr, "_RET_CACHE", {})
    monkeypatch.setattr(mr, "_retrieve_ttl", lambda: 20.0)

    Q = "哥哥我们决定用b10000-cuda这个llama-server来跑GPU推理"
    mr.retrieve(Q, top_k=5)
    mr.retrieve(Q, top_k=5)  # 同 query 复读 → TTL 命中
    mr.retrieve(Q, top_k=5)
    assert vec_calls["n"] == 1, "同 query 20s 内应只打 1 次向量检索"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
