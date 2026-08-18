# -*- coding: utf-8 -*-
"""Phase 4 全套加权: 基础权重/类型化衰减/强化/情境 (纯函数 _score_items)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # core/

from v5.memory_retrieval import _score_items

BASE = {
    "base_weight_factor": 1.0,
    "type_boost": {"default": 1.0},
    "type_decay": {"default": {"per_day": 0.05, "floor": 0.2}},
    "situational": {"enabled": False},
    "min_fused_score": 0.0,
    "top_k": 5,
}
NOW = 1000000.0


def _item(mid, raw=0.5, weight=0.6, mtype="fact", created=0.0, source="fts",
          tags="", reinforcement=0.0, access=0, last=0.0, long_term=False):
    return {"id": mid, "raw": raw, "weight": weight, "type": mtype, "created": created,
            "source": source, "tags": tags, "access_count": access,
            "reinforcement": reinforcement, "last_accessed": last, "long_term": long_term}


def test_base_weight_factor():
    cfg = dict(BASE, base_weight_factor=0.5)
    r = _score_items({"1": _item("1", weight=0.9), "2": _item("2", weight=0.4)},
                     cfg, now=NOW, min_fused=0.0)
    assert r[0]["id"] == "1", "基础权重 0.9 应排前"


def test_base_weight_factor_disabled_keeps_old_behavior():
    cfg = dict(BASE, base_weight_factor=1.0)  # 1.0 = 完全忽略基础权重
    r = _score_items({"1": _item("1", weight=0.9), "2": _item("2", weight=0.4)},
                     cfg, now=NOW, min_fused=0.0)
    assert r[0]["score"] == pytest.approx(r[1]["score"]), "bwf=1.0 时权重不影响评分"


def test_type_decay_persona_preserved():
    old = NOW - 30 * 86400
    cfg = dict(BASE, type_decay={
        "conversation": {"per_day": 0.05, "floor": 0.2},
        "user_trait": {"per_day": 0.01, "floor": 0.6},
        "default": {"per_day": 0.05, "floor": 0.2},
    })
    r = _score_items({
        "c": _item("c", mtype="conversation", created=old),
        "u": _item("u", mtype="user_trait", created=old),
    }, cfg, now=NOW, min_fused=0.0)
    assert r[0]["id"] == "u", "30天前 user_trait 应保值排前"
    assert r[1]["score"] < r[0]["score"] / 2, "conversation 衰减应明显更快"


def test_reinforcement_boost():
    cfg = dict(BASE, reinforcement_weight=0.10)
    r = _score_items({
        "x": _item("x", reinforcement=0.5),
        "y": _item("y", reinforcement=0.0),
    }, cfg, now=NOW, min_fused=0.0)
    assert r[0]["id"] == "x", "被强化记忆应排前"


def test_situational_project_boost_when_coding():
    cfg = dict(BASE, situational={"enabled": True, "project_activity_boost": 0.10,
                                  "hour_match_boost": 0.0})
    r = _score_items({
        "p": _item("p", tags="v5_project:ikaros"),
        "n": _item("n", tags=""),
    }, cfg, now=NOW, min_fused=0.0,
        sit_ctx={"activity": "在写代码", "window": "VS Code"}, coding_activity=True)
    assert r[0]["id"] == "p", "写代码时项目记忆应加分排前"


def test_situational_disabled_no_boost():
    cfg = dict(BASE, situational={"enabled": False, "project_activity_boost": 0.10,
                                  "hour_match_boost": 0.0})
    r = _score_items({
        "p": _item("p", tags="v5_project:ikaros"),
        "n": _item("n", tags=""),
    }, cfg, now=NOW, min_fused=0.0,
        sit_ctx={"activity": "在写代码", "window": "VS Code"}, coding_activity=True)
    assert r[0]["score"] == pytest.approx(r[1]["score"]), "关闭情境加权时不应有 boost"


def test_hour_match_boost():
    import datetime as _dt
    now = _dt.datetime(2026, 8, 14, 22, 30).timestamp()
    same_hour = now - 3600 * 2  # 20:30, 与 22:30 差 2h → 不匹配
    close_hour = now - 3600  # 21:30, 差 1h → 匹配
    cfg = dict(BASE, situational={"enabled": True, "project_activity_boost": 0.0,
                                  "hour_match_boost": 0.05})
    r = _score_items({
        "a": _item("a", created=same_hour),
        "b": _item("b", created=close_hour),
    }, cfg, now=now, min_fused=0.0,
        sit_ctx={"activity": "晚上", "window": ""}, coding_activity=False)
    assert r[0]["id"] == "b", "created 小时 ≈ now 的记忆应时段联想加分"


def test_merge_reinforce_increment():
    """upsert 合并时 reinforcement 应累积 (C: 被合并越多越重要)."""
    import os
    import tempfile
    import v5.store as store
    tmp = tempfile.mkdtemp(prefix="weight_test_")
    store.V5_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    store.conn()
    a = store.upsert("合并强化验证唯一内容", type="fact", weight=0.6)
    b = store.upsert("合并强化验证唯一内容，补充细节", type="fact", weight=0.6)
    assert b == a, "相似内容应合并"
    with store.conn() as c:
        r = c.execute("SELECT reinforcement FROM memory WHERE id=?", (a,)).fetchone()
    assert r[0] > 0, f"合并应累积 reinforcement, got {r[0]}"
