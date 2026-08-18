"""关键词兜底检索测试 (2026-08-10 新增, P2 长句拆词重查).

背景: 实测 "memU 调研学到了什么" 整句 FTS+向量双 miss → 0 命中;
拆成 "memU"/"调研" 后各能命中。retrieve() 命中 <3 时自动拆词补足。

验证:
  K1. 整句 miss 时拆词补足, 命中带 source='kw', score=0.45
  K2. 已命中 >=3 时不触发兜底
  K3. 短 query (单 token) 不触发
  K4. 兜底结果去重 (不与已有命中重复)
  K5. store.search 异常时 fail-open 返回原结果
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import v5
import v5.store  # noqa: F401  (确保包属性存在, 供 monkeypatch.setattr(v5, "store", ...))
import v5.memory_retrieval as mr


class FakeMem:
    def __init__(self, mid, content, mtype="fact", weight=0.6, created=1.0e9):
        self.id = mid
        self.content = content
        self.type = mtype
        self.weight = weight
        self.created = created
        self.pad_p = 0.0
        self.pad_a = 0.0
        self.access_count = 0
        self.reinforcement = 0.0
        self.last_accessed = 0.0
        self.long_term = False
        self.tags = ""


def _kw_store_factory(records):
    """store.search/search_like/count_like 按关键词返回对应记录 (大小写不敏感).

    2026-08-10: fallback 从 FTS MATCH 切到 LIKE 子串查询, mock 同步补
    search_like / count_like。
    """
    def _search(q, top_k=5, **kw):
        ql = q.lower()
        return [m for m in records if ql in m.content.lower()][:top_k]
    def _search_like(substr, top_k=5, **kw):
        return _search(substr, top_k=top_k)
    def _count_like(substr, **kw):
        return sum(1 for m in records if substr.lower() in m.content.lower())
    return type("S", (), {"search": staticmethod(_search),
                          "search_like": staticmethod(_search_like),
                          "count_like": staticmethod(_count_like)})


# ── K1: 拆词补足 ──

def test_k1_long_query_split_fallback(monkeypatch):
    records = [FakeMem(1, "memU 调研记忆"), FakeMem(2, "调研 结论")]
    monkeypatch.setattr(v5, "store", _kw_store_factory(records))
    # 整句 miss → 0 命中 → 触发兜底
    monkeypatch.setattr(mr, "retrieve", lambda q, **kw: [])
    out = mr._keyword_fallback("memU 调研学到了什么", [], tk=5, min_weight=0.0, character="")
    assert len(out) >= 1
    assert all(x["source"] == "kw" for x in out)
    assert all(x["score"] == 0.45 for x in out)


# ── K2: 已命中 >=3 不触发 ──

def test_k2_no_fallback_when_enough(monkeypatch):
    existing = [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}, {"id": "c", "content": "z"}]
    monkeypatch.setattr(v5, "store", _kw_store_factory([FakeMem(9, "memU 额外")]))
    out = mr._keyword_fallback("memU 调研", existing, tk=5, min_weight=0.0, character="")
    assert out == existing  # 原样返回, 不追加


# ── K3: 单 token 不触发 ──

def test_k3_single_token_no_fallback(monkeypatch):
    out = mr._keyword_fallback("memU", [], tk=5, min_weight=0.0, character="")
    assert out == []


# ── K4: 去重 ──

def test_k4_dedup(monkeypatch):
    records = [FakeMem(1, "memU 调研记忆"), FakeMem(2, "memU 落地")]
    monkeypatch.setattr(v5, "store", _kw_store_factory(records))
    existing = [{"id": "1", "content": "memU 调研记忆", "source": "semantic"}]
    out = mr._keyword_fallback("memU 调研落地", existing, tk=5, min_weight=0.0, character="")
    ids = [x["id"] for x in out]
    assert ids.count("1") == 1  # 不重复追加已命中的
    assert "2" in ids


# ── K5: store 异常 fail-open ──

def test_k5_store_error_fail_open(monkeypatch):
    class Boom:
        @staticmethod
        def search(q, **kw):
            raise RuntimeError("boom")
    monkeypatch.setattr(v5, "store", Boom)
    out = mr._keyword_fallback("memU 调研", [], tk=5, min_weight=0.0, character="")
    assert out == []
