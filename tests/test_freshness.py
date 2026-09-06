"""freshness 单测 (F5): 水印 + 对账 + due 判定."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memory_v5 import store, freshness


def _fresh_db():
    tmp = tempfile.mkdtemp(prefix="fresh_test_")
    store.V5_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    store.conn()  # 建库


def _insert(content, type="fact", created=None):
    with store.conn() as c:
        if created is None:
            c.execute("INSERT INTO memory (content, type) VALUES (?, ?)", (content, type))
        else:
            c.execute("INSERT INTO memory (content, type, created) VALUES (?, ?, ?)",
                      (content, type, created))
        c.commit()


# ── F1: 水印默认 0 ──
def test_watermark_default_zero():
    _fresh_db()
    assert freshness.watermark("fact") == 0.0


# ── F2: set_watermark + 读取 ──
def test_set_and_read_watermark():
    _fresh_db()
    freshness.set_watermark("fact", 1000.0)
    assert freshness.watermark("fact") == 1000.0


# ── F3: reconcile — 从未刷新 → 全 pending ──
def test_reconcile_never_refresh_all_pending():
    _fresh_db()
    _insert("m1", "fact")
    _insert("m2", "fact")
    _insert("m3", "lesson")
    stats = freshness.reconcile_totals()
    assert "fact" in stats and "lesson" in stats
    assert stats["fact"]["total"] == 2
    assert stats["lesson"]["total"] == 1
    # 从未刷新 → pending = total
    assert stats["fact"]["pending"] == 2
    assert stats["fact"]["due"] is True


# ── F4: 刷新后 pending 归零 ──
def test_after_refresh_pending_resets():
    _fresh_db()
    _insert("m1", "fact")
    freshness.set_watermark("fact", time.time())  # 刷新到现在
    stats = freshness.reconcile_totals()
    # created <= now → pending=0
    assert stats["fact"]["pending"] == 0
    assert stats["fact"]["due"] is False


# ── F5: 新写入后 pending 上升 → 再次 due ──
def test_new_writes_make_due_again():
    _fresh_db()
    old = time.time() - 100
    _insert("old", "fact", created=old)
    freshness.set_watermark("fact", old)  # 刷新到 old 时刻
    # 新写一条 (created=now > watermark)
    _insert("new", "fact", created=time.time())
    stats = freshness.reconcile_totals()
    assert stats["fact"]["pending"] >= 1
    assert stats["fact"]["due"] is True


# ── F6: due_buckets 返回列表 ──
def test_due_buckets_returns_list():
    _fresh_db()
    _insert("m", "fact")
    due = freshness.due_buckets()
    assert isinstance(due, list)
    assert "fact" in due


# ── F7: stats_summary 含 due 标记 ──
def test_stats_summary_has_due_flag():
    _fresh_db()
    _insert("m", "decision")
    s = freshness.stats_summary()
    assert "decision" in s
    assert "due" in s["decision"]
