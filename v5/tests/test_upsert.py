# -*- coding: utf-8 -*-
"""upsert 写策略 + context_anchor 情境锚 (Phase 1, 2026-08-14)."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # core/

import v5.store as store


@pytest.fixture()
def fresh_db(monkeypatch):
    """临时 v5.db 隔离 (同 test_temporal_supersede_chain 模式)."""
    tmp = tempfile.mkdtemp(prefix="upsert_test_")
    db = os.path.join(tmp, "v5.db")
    monkeypatch.setattr(store, "V5_DB_PATH", Path(db))
    store.conn()  # 建表
    return db


def test_upsert_merges_similar_same_type(fresh_db):
    a = store.upsert("哥哥偏好简短直接的沟通，用'好''知道了'快速切换话题",
                     type="preference", weight=0.7)
    b = store.upsert("哥哥偏好简短直接的沟通，用'好''知道了'等短语快速切换话题，避免深谈",
                     type="preference", weight=0.8)
    assert b == a, "相似同类型应合并到同一 id"
    with store.conn() as c:
        row = c.execute("SELECT weight, access_count FROM memory WHERE id=?", (a,)).fetchone()
    assert row[0] == pytest.approx(0.8), "权重应取高者"
    assert row[1] == 1, "合并应 access_count +1 (首次 INSERT 为 0)"


def test_upsert_new_for_distinct_content(fresh_db):
    a = store.upsert("哥哥喜欢用表情符号表达思考状态", type="user_trait", weight=0.6)
    b = store.upsert("哥哥在讨论技术话题时直接给出结论", type="user_trait", weight=0.6)
    assert a != b, "不同内容应新建"


def test_upsert_does_not_merge_different_type(fresh_db):
    a = store.upsert("哥哥偏好简短直接的沟通", type="preference", weight=0.7)
    b = store.upsert("哥哥偏好简短直接的沟通", type="fact", weight=0.6)
    assert a != b, "同内容不同类型不应合并"


def test_upsert_below_threshold_creates_new(fresh_db):
    a = store.upsert("哥哥在讨论家庭事务时会详细列举具体细节，遇到敏感话题突然沉默",
                     type="user_trait", weight=0.6)
    b = store.upsert("哥哥在深夜用简短回应结束对话，保持克制", type="user_trait", weight=0.6)
    assert a != b, "低于相似阈值应新建"


def test_upsert_merges_content_longer_wins(fresh_db):
    a = store.upsert("哥哥偏好简短直接的沟通", type="preference", weight=0.7)
    b = store.upsert("哥哥偏好简短直接的沟通，说人话比修辞更有效",
                     type="preference", weight=0.7)
    with store.conn() as c:
        content = c.execute("SELECT content FROM memory WHERE id=?", (a,)).fetchone()[0]
    assert "说人话比修辞更有效" in content, "合并后内容应取更长者"


def test_context_anchor_basic():
    from v5.context_anchor import now_context, time_str, time_narrative, weekday_str
    ctx = now_context()
    assert "epoch" in ctx and "time_str" in ctx and "activity" in ctx
    assert ctx["weekday"] in ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
    assert "/" in time_str(), "time_str 应为 2026/8/14 23:05 格式"
    assert time_narrative() and weekday_str()


def test_upsert_reinforcement_cap(fresh_db):
    a = store.upsert("哥哥偏好简短回应", type="preference", weight=0.6, reinforcement=0.9)
    b = store.upsert("哥哥偏好简短回应", type="preference", weight=0.6, reinforcement=0.9)
    with store.conn() as c:
        r = c.execute("SELECT reinforcement FROM memory WHERE id=?", (a,)).fetchone()[0]
    assert r <= 1.0, "reinforcement 不应超过 1.0"
    assert b == a


# ── Phase 2: should_recall 召回决策 ──

def test_should_recall_cues_force_recall():
    from v5.context_anchor import should_recall
    assert should_recall("你好，你还记得上次我们聊的学习效率吗") is True, "线索词应召回"
    assert should_recall("回顾一下我们之前讨论的 V5 架构") is True
    assert should_recall("do you remember the omp setup?") is True


def test_should_recall_trivial_skipped():
    from v5.context_anchor import should_recall
    assert should_recall("你好") is False
    assert should_recall("谢谢") is False
    assert should_recall("晚安") is False
    assert should_recall("ok") is False


def test_should_recall_substantive_recalled():
    from v5.context_anchor import should_recall
    assert should_recall("帮我看看这个代码报错是什么原因") is True, "实质内容应召回"
    assert should_recall("今天想把 omp 的配置再梳理一遍") is True


def test_should_recall_empty_false():
    from v5.context_anchor import should_recall
    assert should_recall("") is False
    assert should_recall("  ") is False


# ── Phase 3: 时间锚定检索 (过期事实默认排除) ──

def test_unified_retrieve_excludes_expired(fresh_db):
    from v5.extensions import temporal_graph as tg
    from v5.memory_retrieval import unified_retrieve
    tg.apply_migration()  # memory 表加 valid_to 列
    live_id = store.store("用户住在上海", type="fact", weight=0.9)
    stale_id = store.store("用户住在北京", type="fact", weight=0.9)
    tg.supersede_memory(stale_id)  # 作废旧事实 (valid_to=now)
    r = unified_retrieve("用户住在", scope="lexical", top_k=10)
    ids = {str(x["id"]) for x in r}
    assert str(live_id) in ids, "有效事实应保留"
    assert str(stale_id) not in ids, "已失效事实不应出现在默认检索"


def test_unified_retrieve_keeps_when_no_migration(fresh_db):
    """未跑迁移 (无 valid_to 列) 时应 fail-open 正常返回, 不报错."""
    from v5.memory_retrieval import unified_retrieve
    store.store("测试记忆保留", type="fact", weight=0.9)
    r = unified_retrieve("测试记忆", scope="lexical", top_k=10)
    assert len(r) > 0, "无 valid_to 列时检索不应被过滤逻辑破坏"
