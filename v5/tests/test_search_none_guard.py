"""v5.search.VectorIndex.search 对 chroma 返回中的 None 条目防御性处理回归测试.

背景 (2026-07-26): 生产 chroma 在个别结果上可能返回 metadata=None / documents=None /
distances=None (迁移遗留或异常写入). 原实现直接 `meta.get(...)` 会抛
`AttributeError: 'NoneType' object has no attribute 'get'`, 被外层 try 吞掉导致整次向量检索
静默返回 [], 语义召回失效.

本测试确保: 任何单条坏数据都不会让整次查询 abort, 而是跳过坏条目并返回其余有效候选.
"""
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # Ikaros-memory/

import v5.search as search_mod


@pytest.fixture
def stub_embed(monkeypatch):
    """隔离真实 :8587 embedding 服务, 用固定向量.

    用 monkeypatch 保证跑完自动还原 (原实现手动还原是死代码:
    _get_embedding 无 __wrapped__, 还原时把自己赋给自己,
    导致后续测试 (如 test_search_cache) 拿到被污染的 stub).
    """
    monkeypatch.setattr(search_mod, "_get_embedding", lambda text, task="query": [0.1] * 64)
    yield


def _fresh_index():
    d = Path(tempfile.mkdtemp(prefix="vi_none_"))
    return search_mod.VectorIndex(persist_dir=d)


def test_search_skips_none_metadata(stub_embed):
    idx = _fresh_index()
    # 模拟迁移遗留: 有 embedding+documents, 但 metadata=None
    idx._collection.upsert(
        ids=["1"],
        documents=["哥哥喜欢简洁的设计"],
        embeddings=[[0.1] * 64],
        metadatas=[None],
    )
    res = idx.search("哥哥喜欢简洁", top_k=3)
    assert isinstance(res, list)
    assert len(res) == 1, "None-metadata 条目不应让查询 abort, 应正常返回"
    assert res[0]["content"] == "哥哥喜欢简洁的设计"
    assert res[0]["type"] == "fact"  # 缺 metadata 时回退默认


def test_search_skips_none_document_entry(stub_embed):
    idx = _fresh_index()
    idx._collection.upsert(ids=["1"], documents=["正常记忆A"],
                           embeddings=[[0.2] * 64],
                           metadatas=[{"type": "fact", "weight": 0.8}])
    idx._collection.upsert(ids=["2"], documents=[None],
                           embeddings=[[0.1] * 64],
                           metadatas=[{"type": "fact", "weight": 0.8}])
    res = idx.search("正常记忆", top_k=5)
    ids = [r["id"] for r in res]
    assert "2" not in ids, "content=None 的条目应被跳过, 不污染结果"
    assert "1" in ids


def test_search_handles_none_distance(stub_embed):
    idx = _fresh_index()
    idx._collection.upsert(ids=["1"], documents=["距离缺失条目"],
                           embeddings=[[0.1] * 64],
                           metadatas=[{"type": "fact", "weight": 0.8}])
    orig = idx._collection.query

    def _q_none_dist(**kw):
        r = orig(**kw)
        r["distances"] = [[None]]
        return r

    idx._collection.query = _q_none_dist
    res = idx.search("距离缺失条目", top_k=3)
    assert len(res) == 1
    assert res[0]["score"] == 0.0, "distance=None 应安全回退为 1.0 距离 -> score 0"


def test_search_normal_multi_entry(stub_embed):
    idx = _fresh_index()
    for i, c in enumerate(["哥哥喜欢简洁", "CUDA 升级很麻烦", "猫比狗可爱"]):
        assert idx.add(i + 1, c, type="fact", weight=0.7)
    res = idx.search("哥哥喜欢简洁", top_k=3)
    assert len(res) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
