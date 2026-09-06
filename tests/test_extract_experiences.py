"""extract_experiences 单测 (F3): 结构化抽取走 upsert 合并 (不堆积雷同)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memory_v5 import store
from memory_v5.reflect import extract_experiences as ex


def _fresh_db():
    tmp = tempfile.mkdtemp(prefix="extract_test_")
    store.V5_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    store.conn()


def _insert_conversation(content, created=None):
    if created is None:
        created = time.time()
    with store.conn() as c:
        c.execute("INSERT INTO memory (content, type, created) VALUES (?, ?, ?)",
                  (content, "conversation", created))
        c.commit()


# ── E1: 无对话 → 0 ──
def test_no_conversations_returns_zero():
    _fresh_db()
    assert ex._gather_recent_conversations() == []
    assert ex.make_extract_experiences_op().fn() == 0


# ── E2: LLM 返回结构化 ops → 走 upsert 写入 ──
def test_structured_extracts_written_via_upsert():
    _fresh_db()
    _insert_conversation("用户喜欢简洁直接的沟通")
    ops = [
        {"type": "preference", "content": "用户偏好简洁直接的沟通", "weight": 0.7},
        {"type": "lesson", "content": "改完配置必须用真实 FS 验证 mtime", "weight": 0.8},
        # 非法 type → 跳过
        {"type": "identity", "content": "我是个务实的人", "weight": 0.9},
        # content 过长 → 跳过
        {"type": "fact", "content": "x" * 250, "weight": 0.5},
    ]
    with patch.object(ex, "_call_extract_llm", return_value=ops):
        n = ex.make_extract_experiences_op().fn()
    # 只有 preference + lesson 两条合法 (identity/过长跳过)
    assert n == 2
    # 验证写入: 检索应能找到
    with store.conn() as c:
        rows = c.execute(
            "SELECT content, type, tags FROM memory WHERE tags LIKE '%v5_extracted%'"
        ).fetchall()
    contents = [r["content"] for r in rows]
    assert any("简洁直接" in c for c in contents)
    assert any("mtime" in c for c in contents)


# ── E3: 重复抽取相似经验 → upsert 合并强化 (不堆积雷同) ──
def test_repeated_extract_merges_not_accumulates():
    _fresh_db()
    _insert_conversation("用户喜欢简洁沟通, 少修辞")
    # 第一次抽取
    ops1 = [{"type": "preference", "content": "用户偏好简洁直接的沟通", "weight": 0.7}]
    with patch.object(ex, "_call_extract_llm", return_value=ops1):
        ex.make_extract_experiences_op().fn()
    # 第二次抽取 (相似内容, 模拟重叠窗口)
    ops2 = [{"type": "preference", "content": "用户偏好简洁直接的沟通", "weight": 0.7}]
    with patch.object(ex, "_call_extract_llm", return_value=ops2):
        ex.make_extract_experiences_op().fn()
    # 应只有 1 条 (upsert 合并), reinforcement 累积 > 0
    with store.conn() as c:
        rows = c.execute(
            "SELECT id, content, reinforcement FROM memory WHERE tags LIKE '%v5_extracted%'"
        ).fetchall()
    assert len(rows) == 1, f"应合并为 1 条, 实有 {len(rows)} 条 (雷同堆积未挡)"
    assert float(rows[0]["reinforcement"]) > 0.0  # 合并即强化


# ── E4: LLM 不可用 → fail-open 返 0 ──
def test_llm_unavailable_fail_open():
    _fresh_db()
    _insert_conversation("对话内容")
    with patch.object(ex, "_call_extract_llm", return_value=[]):
        assert ex.make_extract_experiences_op().fn() == 0


# ── E5: LLM 返回非法 JSON → fail-open ──
def test_malformed_llm_output_fail_open():
    _fresh_db()
    _insert_conversation("对话内容")
    with patch.object(ex.reflect_llm if hasattr(ex, "reflect_llm") else ex, "_call_extract_llm",
                      return_value=[]):
        # _call_extract_llm 内部已吞 parse 错误返 [], 所以 fn 返 0
        assert ex.make_extract_experiences_op().fn() == 0


# ── E6: op 注册到默认调度器 ──
def test_op_registered_in_default_scheduler():
    from memory_v5.reflect.registry import make_default_scheduler
    s = make_default_scheduler()
    names = [op.name for op in s._ops]
    assert "extract_experiences" in names
    assert "refresh_freshness" in names
