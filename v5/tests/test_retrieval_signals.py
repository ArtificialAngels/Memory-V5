# -*- coding: utf-8 -*-
"""意图检测 + 意图驱动加权 + 信号透明 (2026-08-14, mnemon 借鉴).

覆盖:
  S1. detect_intent: WHY/WHEN/ENTITY/GENERAL (中英文)
  S2. _score_items 暴露 signals 信号分量 (mnemon 信号透明)
  S3. 意图驱动加权: WHY → decision/lesson 加权; entity → fact 加权
  S4. 意图关闭时不加权 (向后兼容)
  S5. unified_retrieve 结果带 intent 字段
  S6. store.valid_to_map (循环依赖解开后的公共助手)
  S7. retrieve_temporal 过滤已失效事实
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # core/

import v5.memory_retrieval as mr

BASE = {
    "base_weight_factor": 1.0,
    "type_boost": {"default": 1.0, "decision": 1.0, "lesson": 1.0, "fact": 1.0},
    "type_decay": {"default": {"per_day": 0.05, "floor": 0.2}},
    "situational": {"enabled": False},
    "min_fused_score": 0.0,
    "top_k": 5,
}
NOW = 1000000.0


def _item(mid, raw=0.5, weight=0.6, mtype="fact", created=0.0, tags=""):
    return {"id": mid, "raw": raw, "weight": weight, "type": mtype,
            "created": created, "source": "fts", "tags": tags,
            "access_count": 0, "reinforcement": 0.0,
            "last_accessed": 0.0, "long_term": False}


# ── S1: detect_intent ──
def test_detect_intent_why():
    assert mr.detect_intent("为什么选了 SQLite") == "WHY"
    assert mr.detect_intent("这个报错的原因是什么") == "WHY"
    assert mr.detect_intent("why did we choose sqlite") == "WHY"


def test_detect_intent_when():
    assert mr.detect_intent("上次什么时候改的配置") == "WHEN"
    assert mr.detect_intent("最近发生了什么") == "WHEN"


def test_detect_intent_entity():
    assert mr.detect_intent("什么是 Hermes gateway") == "ENTITY"
    assert mr.detect_intent("关于 omp 便携化配置") == "ENTITY"
    assert mr.detect_intent("讲讲这个项目") == "ENTITY"


def test_detect_intent_general():
    assert mr.detect_intent("帮我看下这段代码") == "GENERAL"
    assert mr.detect_intent("") == "GENERAL"
    assert mr.detect_intent("   ") == "GENERAL"


# ── S2: 信号透明 ──
def test_score_items_exposes_signals():
    out = mr._score_items({"1": _item("1")}, dict(BASE), now=NOW, min_fused=0.0)
    assert out and out[0]["signals"], "应暴露 signals 分量"
    for k in ("fts", "vector", "time", "base_weight", "type_decay",
              "type_boost", "frequency", "situational"):
        assert k in out[0]["signals"], f"missing signal key {k}"
    assert out[0]["intent"] == "GENERAL"


# ── S3: 意图驱动加权 ──
INTENT_CFG = dict(BASE, intent={
    "enabled": True,
    "why": {"decision": 1.5, "lesson": 1.5},
    "when": {"conversation": 1.3},
    "entity": {"fact": 1.5},
    "general": {},
})


def test_intent_why_boosts_decision():
    r = mr._score_items({
        "d": _item("d", mtype="decision"),
        "f": _item("f", mtype="fact"),
    }, INTENT_CFG, now=NOW, min_fused=0.0, intent="WHY")
    assert r[0]["id"] == "d", "WHY 意图下 decision 应加权排前"
    assert r[0]["signals"]["type_boost"] == pytest.approx(1.5)


def test_intent_entity_boosts_fact():
    r = mr._score_items({
        "d": _item("d", mtype="decision"),
        "f": _item("f", mtype="fact"),
    }, INTENT_CFG, now=NOW, min_fused=0.0, intent="ENTITY")
    assert r[0]["id"] == "f", "ENTITY 意图下 fact 应加权排前"
    assert r[0]["signals"]["type_boost"] == pytest.approx(1.5)


# ── S4: 意图关闭时不加权 ──
def test_intent_disabled_no_boost():
    cfg = dict(BASE, intent={"enabled": False, "why": {"decision": 1.5}})
    r = mr._score_items({
        "d": _item("d", mtype="decision"),
        "f": _item("f", mtype="fact"),
    }, cfg, now=NOW, min_fused=0.0, intent="WHY")
    for o in r:
        assert o["signals"]["type_boost"] == pytest.approx(1.0), "意图关闭不应加权"


# ── S5: unified_retrieve 带 intent ──
def test_unified_exposes_intent(monkeypatch):
    monkeypatch.setattr(mr, "retrieve", lambda q, **kw: [{
        "id": "m1", "content": "关于 Hermes 的结论", "type": "fact",
        "weight": 0.5, "tags": "", "created": 1.0e9, "pad_p": 0.0,
        "pad_a": 0.0, "source": "semantic", "score": 0.7,
    }])
    out = mr.unified_retrieve("什么是 Hermes")
    assert out and out[0]["intent"] == "ENTITY"


# ── S6: store.valid_to_map ──
def test_store_valid_to_map():
    import os
    import tempfile
    import time as _t
    import v5.store as store
    tmp = tempfile.mkdtemp(prefix="signals_test_")
    store.V5_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    store.conn()
    with store.conn() as c:  # 手动加 valid_to 列 (真实环境由 apply_migration 加)
        for col in ("valid_from", "valid_to"):
            try:
                c.execute(f"ALTER TABLE memory ADD COLUMN {col} REAL")
            except Exception:
                pass
        # 直接 INSERT (绕过 store.store 的异步向量同步, 避免污染真实 chroma)
        c.execute("INSERT INTO memory (content, type, weight) VALUES ('当前有效记忆', 'fact', 0.6)")
        c.execute("INSERT INTO memory (content, type, weight) VALUES ('已失效记忆', 'fact', 0.6)")
        c.commit()
    with store.conn() as c:
        rows = c.execute("SELECT id, content FROM memory").fetchall()
        ids = {r["content"]: r["id"] for r in rows}
        live, stale = ids["当前有效记忆"], ids["已失效记忆"]
        c.execute("UPDATE memory SET valid_to = ? WHERE id = ?",
                  (_t.time() - 100, stale))
        c.commit()
    vm = store.valid_to_map([str(live), str(stale)], "memory", "id")
    assert vm.get(str(live)) is None
    assert vm.get(str(stale)) is not None


# ── S7: retrieve_temporal 过滤失效 ──
def test_retrieve_temporal_filters_expired(monkeypatch):
    import v5.store as store_mod
    monkeypatch.setattr(mr, "retrieve", lambda q, **kw: [
        {"id": "1", "content": "live", "type": "fact", "weight": 0.6,
         "tags": "", "created": 1.0e9, "pad_p": 0.0, "pad_a": 0.0,
         "source": "semantic", "score": 0.8},
        {"id": "2", "content": "stale", "type": "fact", "weight": 0.6,
         "tags": "", "created": 1.0e9, "pad_p": 0.0, "pad_a": 0.0,
         "source": "semantic", "score": 0.7},
    ])
    monkeypatch.setattr(store_mod, "valid_to_map",
                        lambda ids, table, col="id": {"2": 1.0, "1": None})
    out = mr.retrieve_temporal("q", now=2.0)
    ids = [o["id"] for o in out]
    assert "1" in ids and "2" not in ids, "过期事实应被剔除"


# ── S8: dedup op (算法去重, 原空壳返 0) ──
def test_dedup_op_archives_duplicates():
    import os
    import tempfile
    import v5.store as store
    from v5.reflect import registry
    tmp = tempfile.mkdtemp(prefix="dedup_signals_test_")
    store.V5_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    store.conn()
    with store.conn() as c:
        c.execute("INSERT INTO memory (content, type, weight) VALUES ('哥哥喜欢喝咖啡', 'fact', 0.6)")
        c.execute("INSERT INTO memory (content, type, weight) VALUES ('哥哥喜欢喝咖啡', 'fact', 0.6)")
        c.execute("INSERT INTO memory (content, type, weight) VALUES ('完全不同的内容XYZ', 'fact', 0.6)")
        c.execute("INSERT INTO memory (content, type, weight, tags) "
                  "VALUES ('项目笔记记录', 'fact', 0.6, 'v5_key:note1')")
        c.commit()
    n = registry.make_dedup_op().fn()
    assert n == 1, "应只归档 1 条完全重复的记忆"
    with store.conn() as c:
        rows = c.execute("SELECT content, tags, archived FROM memory ORDER BY id").fetchall()
    assert [r["content"] for r in rows if r["archived"] == 1] == ["哥哥喜欢喝咖啡"]
    assert all(r["archived"] == 0 for r in rows if "v5_key:" in (r["tags"] or "")), \
        "v5_key 结构化记录不应参与去重"


# ── S9: explain_result 可观测性 (P8) ──
def test_explain_semantic_signals():
    from v5.memory_retrieval import explain_result
    item = {"source": "semantic", "intent": "WHY", "signals": {
        "vector": 0.52, "fts": 0.11, "time": 0.0, "ei": 0.9, "type_boost": 1.3}}
    why = explain_result(item)
    assert "向量0.52" in why and "关键词0.11" in why
    assert "意图WHY" in why and "EI=0.90" in why and "类型加权×1.3" in why


def test_explain_graph_relation():
    from v5.memory_retrieval import explain_result
    item = {"source": "graph", "relation": "SOLVES", "kind": "decision"}
    why = explain_result(item)
    assert "图扩散(relation=SOLVES)" in why and "kind=decision" in why


def test_explain_structured_and_kw():
    from v5.memory_retrieval import explain_result
    assert "精确标签命中" in explain_result({"source": "structured", "kind": "decision"})
    assert "关键词兜底" in explain_result({"source": "kw", "signals": {}})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
