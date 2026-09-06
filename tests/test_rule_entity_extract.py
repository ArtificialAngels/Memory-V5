"""rule_entity_extract 实跑验证 (2026-08-19 新增 op).

确认纯规则实体抽取真正填充 eg_* 表 (此前 LLM 抽取从未调度, 图全空),
且幂等 (重跑实体数不重复增长)。

隔离: 用临时 DB 替代真实 v5.db / eg.db, 不污染生产数据。
"""
import sys
import tempfile
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent  # core/memory_v5
sys.path.insert(0, str(V5_ROOT.parent))   # core

import memory_v5.store as store
import memory_v5.entity_graph as eg
import memory_v5.extensions.rule_entity_extract as re_mod

# ── 隔离: 临时 DB, 并禁用会触网的最佳努力调用 ──
_TMP = Path(tempfile.mkdtemp(prefix="v5_eg_test_"))
store.V5_DB_PATH = _TMP / "v5.db"
eg.EG_DB_PATH = _TMP / "eg.db"
store._sync_vector_best_effort = lambda *a, **k: None
store._run_dissonance_detection = lambda *a, **k: None
store._record_event_best_effort = lambda *a, **k: None


def _eg_counts():
    with eg.eg_conn() as c:
        e = c.execute("SELECT COUNT(*) FROM eg_entities").fetchone()[0]
        ed = c.execute("SELECT COUNT(*) FROM eg_edges").fetchone()[0]
        ep = c.execute("SELECT COUNT(*) FROM eg_episodic_entities").fetchone()[0]
    return e, ed, ep


def test_rule_extract_fills_graph():
    # 灌几条带实体 token 的高权重记忆
    store.store("哥哥偏好 direct 沟通, Ikaros 架构用 dsh 底座", type="preference", weight=0.8)
    store.store("决策: 本地 LLM 退役, 记忆检索走 bge-m3 向量", type="decision", weight=0.9)
    store.store("教训: store.conn() 的 finally rollback 必须显式 commit", type="lesson", weight=0.7)

    before = _eg_counts()
    assert before == (0, 0, 0), before  # 抽取前 eg_* 应全空

    stats = re_mod.run_rule_extract(limit=300, min_weight=0.45)
    after = _eg_counts()

    assert stats["entities_created"] > 0, stats
    assert after[0] > 0, after  # eg_entities 被填充
    assert after[1] > 0, after  # eg_edges 共现边被填充
    assert after[2] > 0, after  # eg_episodic_entities 链接被填充


def test_rule_extract_idempotent():
    n1, _, _ = _eg_counts()
    assert n1 > 0, "前置: 上一用例已建实体"
    re_mod.run_rule_extract(limit=300, min_weight=0.45)
    n2, _, _ = _eg_counts()
    # INSERT OR IGNORE: 同一批记忆重跑, 实体数不重复增长
    assert n2 == n1, (n1, n2)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
