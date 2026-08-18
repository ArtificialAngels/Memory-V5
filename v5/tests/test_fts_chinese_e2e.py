"""真实 SQLite FTS5 中文检索端到端测试 (R9 测试缺口 #2 / L2).

不 mock SQLite/FTS5: monkeypatch store.V5_DB_PATH 指向临时库, 真实建表
+ 真实 FTS5 索引 (memory_fts 由触发器维护) + 真实
store.search / store.search_like / memory_retrieval.unified_retrieve。

验证点 (unicode61 分词语义, 连续中文 = 单个 token):
  - FTS5 对整串连续中文命中 (query 短语 == 整串 token)
  - FTS5 对中文 2-gram 子串 0 命中 (整串分词, 2-gram 不是独立 token)
  - search_like (LIKE %substr%) 能召回 FTS5 漏掉的 2-gram 内容
  - unified_retrieve 整串 miss + 向量空 → 拆词 2-gram LIKE 兜底召回
  - LIKE 通配符 %/_ 被转义

仅屏蔽与 SQLite 无关的旁路 (Chroma 向量 / 事件日志 / 失调检测),
避免测试环境无 Chroma 服务时挂起。
"""
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # core

from v5 import store  # noqa: E402
import v5.search as search_mod  # noqa: E402
from v5.memory_retrieval import unified_retrieve  # noqa: E402


def _noop(*args, **kwargs):
    return None


class _EmptyVec:
    """屏蔽向量路: 返回空 (语义分量只来自 FTS5)."""

    def search(self, query, top_k=5, **kwargs):
        return []


@pytest.fixture
def real_store(monkeypatch, tmp_path):
    """临时库的真实 SQLite/FTS5 store (仅屏蔽旁路)."""
    monkeypatch.setattr(store, "V5_DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "V5_DB_PATH", tmp_path / "v5_test.db")
    monkeypatch.setattr(store, "_sync_vector_best_effort", _noop)
    monkeypatch.setattr(store, "_record_event_best_effort", _noop)
    monkeypatch.setattr(store, "_run_dissonance_detection", _noop)
    return store


def test_fts5_hits_whole_chinese_string(real_store):
    """连续中文 = 单个 FTS5 token: 整串 query 命中 (unicode61)."""
    real_store.store("主力战机选型需要平衡成本与性能",
                     type="conversation", tags="test,fts", weight=0.6)
    hits = real_store.search("主力战机选型需要平衡成本与性能")
    assert hits, "FTS5 应命中整串连续中文 (整串即唯一 token)"
    assert "主力战机选型" in hits[0].content


def test_fts5_phrase_token_within_doc(real_store):
    """带空格的独立 token: FTS5 短语命中文档内片段."""
    real_store.store("主力战机选型 需要平衡成本与性能",
                     type="conversation", tags="test,fts", weight=0.6)
    hits = real_store.search("主力战机选型")
    assert hits, "空格分隔的独立 token 应被 FTS5 命中"
    assert "主力战机选型" in hits[0].content


def test_fts5_misses_2gram_but_search_like_recalls(real_store):
    """FTS5 对 2-gram 0 命中, search_like (LIKE) 100% 召回 — R9 核心断言."""
    real_store.store("哥哥喜欢在周末研究主力战机选型",
                     type="conversation", tags="test,fts", weight=0.6)
    # FTS5 把连续中文当单个 token: MATCH '"主力"' 不是文档 token → 0 命中
    assert real_store.search("主力") == [], \
        "FTS5 对中文 2-gram 应 0 命中 (整串分词)"
    # LIKE %主力% 是字节子串匹配, 不依赖 tokenizer → 召回 FTS5 漏掉的内容
    like = real_store.search_like("主力", top_k=5)
    assert len(like) == 1
    assert "主力战机" in like[0].content


def test_search_like_escapes_like_meta(real_store):
    """LIKE 特殊字符 (%/_ ) 应被转义, 不当通配符."""
    real_store.store("概率是 100% 确定的结论", type="fact",
                     tags="test,like", weight=0.6)
    assert len(real_store.search_like("100%", top_k=5)) == 1, \
        "%% 应转义为字面量"
    assert real_store.search_like("100_", top_k=5) == [], \
        "_ 应转义为字面量"


def test_unified_retrieve_kw_fallback_for_2gram(real_store, monkeypatch):
    """整串 query FTS5 不中 + 向量为空 → 拆词 2-gram LIKE 兜底召回.

    语义路仅剩 FTS5 分量时, unified_retrieve 应经 _keyword_fallback
    (score=0.45 弱信号) 召回 FTS5 漏掉的中文 2-gram 内容。
    """
    real_store.store("哥哥喜欢在周末研究主力战机选型",
                     type="conversation", tags="test,fts", weight=0.6)
    monkeypatch.setattr(search_mod, "get_vector_index",
                        lambda *a, **k: _EmptyVec())
    monkeypatch.setattr(search_mod, "entity_graph_search",
                        lambda *a, **k: [])
    out = unified_retrieve("主力战机", top_k=5)
    assert out, "unified_retrieve 应经 kw 兜底召回 2-gram 内容"
    assert any("主力战机" in r["content"] for r in out)
    # kw 兜底特征: 固定弱信号分 0.45
    assert any(r["score"] == pytest.approx(0.45) for r in out)
