# -*- coding: utf-8 -*-
"""推荐 4/5 落地测试 (V5.7, 2026-08-14): graph_rank + lifecycle + graph-scope 并轨.

覆盖:
  G1. personalized_pagerank: seed 扩散 + 距离衰减
  G2. label_propagation: 社区检测
  L1. effective_importance: 访问 boost / 时间衰减 / 强化
  L2. retention_pass: 统一 promote/demote/archive 单轮
  P1. project_graph_search: 项目知识图沿类型化边扩散
"""
import os
import sys
import tempfile
import time as _t
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # core/

from v5.graph_rank import personalized_pagerank, label_propagation
from v5.lifecycle import effective_importance, retention_pass
import v5.store as store
import v5.project_edges as pe


# ── G1: PPR ──
def test_pagerank_seed_diffusion():
    edges = [("a", "b", 1.0), ("b", "c", 1.0), ("b", "d", 0.5)]
    scores = personalized_pagerank(edges, ["a"])
    assert scores["a"] > scores["b"], "seed 应得分最高"
    assert scores["c"] > 0 and scores["d"] > 0, "多跳邻居应可达"
    assert scores["a"] > scores["c"], "距离越远得分越低"


def test_pagerank_dangling_handled():
    # 悬空节点 (c 无出边) 不应导致质量丢失发散
    edges = [("a", "b", 1.0), ("b", "c", 1.0)]
    scores = personalized_pagerank(edges, ["a"], iterations=10)
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.05, f"质量应守恒, got {total}"


# ── G2: Label Propagation ──
def test_label_propagation_two_communities():
    edges = [("a", "b"), ("b", "c"), ("d", "e")]
    labels = label_propagation(edges)
    assert labels["a"] == labels["b"] == labels["c"]
    assert labels["d"] == labels["e"]
    assert labels["a"] != labels["d"], "无连接的两团应分属不同社区"


def test_label_propagation_empty():
    assert label_propagation([]) == {}


# ── L1: effective_importance ──
def test_ei_access_boost():
    now = 1000000.0
    cold = effective_importance(0.6, 0, 0.0, now)
    hot = effective_importance(0.6, 7, now, now)
    assert hot > cold


def test_ei_decay():
    now = 1_800_000_000.0  # 用真实量级 epoch, 避免 60 天前变成负数
    recent = effective_importance(0.6, 0, now, now)
    stale = effective_importance(0.6, 0, now - 60 * 86400, now)
    assert recent > stale


def test_ei_reinforcement():
    now = 1000000.0
    base = effective_importance(0.6, 0, 0.0, now)
    rein = effective_importance(0.6, 0, 0.0, now, reinforcement=1.0)
    assert rein > base


# ── L2: retention_pass ──
def test_retention_pass_promote_demote_archive():
    tmp = tempfile.mkdtemp(prefix="retention_test_")
    store.V5_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    store.conn()
    now = _t.time()
    with store.conn() as c:
        # 高频 → promote
        c.execute("INSERT INTO memory (content, type, weight, access_count, short_term, long_term) "
                  "VALUES ('高频', 'fact', 0.5, 5, 1, 0)")
        # 30 天老 → promote
        c.execute("INSERT INTO memory (content, type, weight, access_count, short_term, long_term, created) "
                  "VALUES ('老记忆', 'fact', 0.5, 0, 1, 0, ?)", (now - 40 * 86400,))
        # 90 天冷 long_term → demote
        c.execute("INSERT INTO memory (content, type, weight, access_count, short_term, long_term, last_accessed) "
                  "VALUES ('冷记忆', 'fact', 0.6, 0, 0, 1, ?)", (now - 100 * 86400,))
        # conversation 8 天 → archive
        c.execute("INSERT INTO memory (content, type, weight, created) "
                  "VALUES ('旧对话', 'conversation', 0.5, ?)", (now - 8 * 86400,))
        # 低 weight → archive
        c.execute("INSERT INTO memory (content, type, weight) VALUES ('低权重', 'fact', 0.3)")
        # 灵魂核心低 weight → 不归档
        c.execute("INSERT INTO memory (content, type, weight) VALUES ('灵魂身份', 'identity', 0.3)")
        c.commit()

    r = retention_pass(now=now)
    assert r["promoted"] == 2, r
    assert r["demoted"] == 1, r
    assert r["archived"] == 2, r

    with store.conn() as c:
        rows = c.execute("SELECT content, short_term, long_term, archived FROM memory").fetchall()
    by = {r["content"]: r for r in rows}
    assert by["高频"]["long_term"] == 1
    assert by["冷记忆"]["short_term"] == 1 and by["冷记忆"]["long_term"] == 0
    assert by["旧对话"]["archived"] == 1
    assert by["低权重"]["archived"] == 1
    assert by["灵魂身份"]["archived"] == 0, "identity 不应归档"


# ── P1: project_graph_search 沿类型化边扩散 ──
def _insert_project(content, kind, project="ikaros"):
    with store.conn() as c:
        cur = c.execute(
            "INSERT INTO memory (content, type, tags, weight) VALUES (?, 'fact', ?, 0.7)",
            (content, f"v5_domain:project,v5_project:{project},v5_kind:{kind}"),
        )
        c.commit()
        return int(cur.lastrowid)


def test_project_graph_search_expands_edges():
    tmp = tempfile.mkdtemp(prefix="pgs_test_")
    store.V5_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    store.conn()
    pit = _insert_project("choma 向量多进程并发写报 compactor 冲突", "pitfall")
    dec_content = "choma 向量写前加跨进程文件锁解决 compactor 冲突"
    dec = _insert_project(dec_content, "decision")
    pe.auto_link_project_note(dec, dec_content, "decision", "ikaros")

    hits = pe.project_graph_search("compactor", top_k=5)
    ids = {h["id"] for h in hits}
    assert str(dec) in ids, "应召回决策本身"
    assert str(pit) in ids, "应沿 SOLVES 边扩散召回坑"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
