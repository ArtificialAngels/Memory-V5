# -*- coding: utf-8 -*-
"""类型化项目知识边 (V5.7, graph-memory 借鉴) 测试.

覆盖:
  P1. auto_link_project_note: decision → pitfall 建 SOLVES 边 (有向)
  P2. auto_link_project_note: convention → pitfall 建 PREVENTS 边
  P3. 关键词重叠不足时不建边
  P4. traverse: 沿边扩散返回类型化邻居 (relation + direction)
  P5. link_project_edge 幂等 (同 relation 覆盖 weight)
  P6. eg_edges relation_type 迁移 (旧库 ALTER 加列)
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # core/

import v5.store as store
import v5.project_edges as pe


def _fresh_db():
    tmp = tempfile.mkdtemp(prefix="project_edges_test_")
    store.V5_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    store.conn()  # 建表 (含 project_edges + eg_edges relation_type)
    return tmp


def _note(content, kind, project="ikaros"):
    """直接 INSERT 项目笔记 (绕过 store.store 的向量同步, 避免污染真实 chroma)。"""
    with store.conn() as c:
        cur = c.execute(
            "INSERT INTO memory (content, type, tags, weight) VALUES (?, 'fact', ?, 0.7)",
            (content, f"v5_project:{project},v5_kind:{kind}"),
        )
        c.commit()
        return int(cur.lastrowid)


# ── P1: decision → pitfall 建 SOLVES 边 ──
def test_decision_solves_pitfall():
    _fresh_db()
    pit = _note("choma 向量索引在多进程并发写时报 hnsw compactor 冲突", "pitfall")
    dec_content = "choma 向量写前加跨进程文件锁解决 compactor 冲突"
    dec = _note(dec_content, "decision")
    n = pe.auto_link_project_note(dec, dec_content, "decision", "ikaros")
    assert n >= 1
    edges = store.get_project_edges(dec)
    solves = [e for e in edges if e["relation"] == "SOLVES"]
    assert solves, "decision 应 SOLVES 相关 pitfall"
    assert solves[0]["source_id"] == dec and solves[0]["target_id"] == pit


# ── P2: convention → pitfall 建 PREVENTS 边 ──
def test_convention_prevents_pitfall():
    _fresh_db()
    _note("直接跑 llama-server 会因缺 CUDA 环境 SIGSEGV", "pitfall")
    con_content = "llama-server 必须经看门狗启动，禁止裸跑避免 SIGSEGV"
    con = _note(con_content, "convention")
    n = pe.auto_link_project_note(con, con_content, "convention", "ikaros")
    assert n >= 1
    edges = store.get_project_edges(con)
    assert any(e["relation"] == "PREVENTS" for e in edges)


# ── P3: 无重叠不建边 ──
def test_no_edge_without_overlap():
    _fresh_db()
    _note("北京烤鸭好吃", "pitfall")
    dec_content = "咖啡因会导致失眠"
    dec = _note(dec_content, "decision")
    n = pe.auto_link_project_note(dec, dec_content, "decision", "ikaros")
    assert n == 0


# ── P4: traverse 沿边扩散 ──
def test_traverse_returns_typed_neighbors():
    _fresh_db()
    pit = _note("choma 向量多进程并发写报 hnsw compactor 冲突", "pitfall")
    dec_content = "choma 向量写前加跨进程文件锁解决 compactor 冲突"
    dec = _note(dec_content, "decision")
    pe.auto_link_project_note(dec, dec_content, "decision", "ikaros")
    links = pe.traverse(pit)  # 从坑出发, 反向找到解决它的决策
    hit = [l for l in links if l["relation"] == "SOLVES" and l["id"] == dec]
    assert hit, "pitfall 应沿 SOLVES 反向找到 decision"
    assert hit[0]["direction"] == "in"


# ── P5: link_project_edge 幂等 ──
def test_link_project_edge_idempotent():
    _fresh_db()
    a = _note("主题甲内容足够长", "decision")
    b = _note("主题乙内容足够长", "pitfall")
    assert store.link_project_edge(a, b, "SOLVES", weight=0.5) is True
    assert store.link_project_edge(a, b, "SOLVES", weight=0.8) is True  # 覆盖 weight
    edges = [e for e in store.get_project_edges(a) if e["relation"] == "SOLVES"]
    assert len(edges) == 1
    assert edges[0]["weight"] == pytest.approx(0.8)


# ── P6: eg_edges relation_type 迁移 ──
def test_eg_edges_relation_type_migration():
    import sqlite3
    import v5.entity_graph as eg
    tmp = tempfile.mkdtemp(prefix="eg_rel_test_")
    db = Path(os.path.join(tmp, "v5.db"))
    eg.EG_DB_PATH = db
    # 先建"旧" eg_edges (无 relation_type), 模拟旧库
    c = sqlite3.connect(str(db))
    c.execute(
        "CREATE TABLE eg_edges (source_entity_id TEXT NOT NULL, "
        "target_entity_id TEXT NOT NULL, weight REAL NOT NULL DEFAULT 0.0, "
        "co_occurrence_count INTEGER NOT NULL DEFAULT 0, "
        "last_seen_at INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY(source_entity_id, target_entity_id))"
    )
    c.commit()
    c.close()
    with eg.eg_conn() as c2:  # eg_conn 应 ALTER 加列
        cols = [r["name"] for r in c2.execute("PRAGMA table_info(eg_edges)").fetchall()]
    assert "relation_type" in cols


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
