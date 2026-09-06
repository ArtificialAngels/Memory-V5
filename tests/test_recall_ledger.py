"""recall_ledger 单测 (F1): 跨轮去重 + bare-URI 不冷却 + 剪枝."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memory_v5 import recall_ledger as rl


def _fresh_ledger(tmp_path, session="test", dedup_turns=5):
    """每个测试用独立 temp 目录, 互不串扰."""
    rl._LEDGER_ROOT = Path(tmp_path)
    return rl.RecallLedger(session_id=session, dedup_turns=dedup_turns)


# ── L1: advance_turn 推进 ──
def test_advance_turn(tmp_path):
    lg = _fresh_ledger(tmp_path, "s1")
    assert lg.turn == 0
    assert lg.advance_turn() == 1
    assert lg.turn == 1
    assert lg.advance_turn() == 2


# ── L2: 有正文的记忆在窗口内被冷却 ──
def test_served_with_body_is_cooled(tmp_path):
    lg = _fresh_ledger(tmp_path, "s2", dedup_turns=3)
    lg.advance_turn()  # turn 1
    lg.record_served([("10", True), ("11", True)])  # turn 1 展示了正文
    lg.advance_turn()  # turn 2
    cooled = lg.cooled_ids()
    assert "10" in cooled and "11" in cooled


# ── L3: bare-URI 不冷却 (核心: 输给预算 ≠ 读者看过) ──
def test_bare_uri_not_cooled(tmp_path):
    lg = _fresh_ledger(tmp_path, "s3", dedup_turns=3)
    lg.advance_turn()  # turn 1
    # ("20", False) = bare URI, 无正文
    lg.record_served([("20", False), ("21", True)])
    lg.advance_turn()  # turn 2
    cooled = lg.cooled_ids()
    assert "20" not in cooled  # bare-URI 不冷却
    assert "21" in cooled      # 有正文 → 冷却


# ── L4: 窗口外不再冷却 ──
def test_out_of_window_not_cooled(tmp_path):
    lg = _fresh_ledger(tmp_path, "s4", dedup_turns=2)
    lg.advance_turn()  # turn 1
    lg.record_served([("30", True)])
    lg.advance_turn()  # turn 2
    lg.advance_turn()  # turn 3
    # turn 1 在窗口 [2,3] 之外 → 不冷却
    assert "30" not in lg.cooled_ids()


# ── L5: dedup_turns=0 → 全不冷却 ──
def test_zero_dedup_turns_cools_nothing(tmp_path):
    lg = _fresh_ledger(tmp_path, "s5", dedup_turns=0)
    lg.advance_turn()
    lg.record_served([("40", True)])
    lg.advance_turn()
    assert lg.cooled_ids() == set()


# ── L6: 剪枝 — 超 MAX_LEDGER_URIS 按最旧淘汰 ──
def test_prune_keeps_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "_MAX_LEDGER_URIS", 5)
    lg = _fresh_ledger(tmp_path, "s6", dedup_turns=100)
    lg.advance_turn()
    # 记 10 条 (turn 1), 超 5 条上限
    lg.record_served([(str(i), True) for i in range(10)])
    # 剪枝后 ≤ 5 条
    entries = lg._state["entries"]
    assert len(entries) <= 5


# ── L7: 持久化 — 重开实例恢复 state ──
def test_persistence_across_instances(tmp_path):
    lg = _fresh_ledger(tmp_path, "s7", dedup_turns=5)
    lg.advance_turn()
    lg.record_served([("70", True)])
    # 新实例同 session → 读回
    lg2 = rl.RecallLedger(session_id="s7", dedup_turns=5)
    assert lg2.turn == 1
    lg2.advance_turn()
    assert "70" in lg2.cooled_ids()


# ── L8: 不同 session 互不干扰 ──
def test_different_sessions_isolated(tmp_path):
    a = _fresh_ledger(tmp_path, "sessA", dedup_turns=5)
    b = _fresh_ledger(tmp_path, "sessB", dedup_turns=5)
    a.advance_turn()
    a.record_served([("80", True)])
    a.advance_turn()
    # B 的账本里没有 A 的记忆
    assert "80" not in b.cooled_ids()
    b.advance_turn()
    assert "80" not in b.cooled_ids()


# ── L9: reset 清空 ──
def test_reset_clears(tmp_path):
    lg = _fresh_ledger(tmp_path, "s9", dedup_turns=5)
    lg.advance_turn()
    lg.record_served([("90", True)])
    lg.reset()
    assert lg.turn == 0
    assert lg.cooled_ids() == set()
