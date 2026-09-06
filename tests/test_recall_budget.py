"""recall_budget 单测 (F2): 广度后深度 + per-entry cap + 降级 + dedup + stats."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memory_v5.recall_budget import (
    Candidate,
    plan_entries,
    per_entry_cap,
    estimate_tokens,
)


def _cand(i, content, score=0.5, type="fact"):
    return Candidate(id=str(i), content=content, type=type, score=score)


# ── B1: per_entry_cap = 2× 平均份额 ──
def test_per_entry_cap_is_double_average():
    assert per_entry_cap(1600, 8) == 400  # 1600//8 * 2
    assert per_entry_cap(1600, 1) == 3200  # 单条 = 2× 全预算 (受预算上限兜底)
    # 0 候选: 不除零, 返回 >= 1 (公式 max_tokens//1*2, moot)
    assert per_entry_cap(100, 0) >= 1


# ── B2: 全部放得下 → 全 full ──
def test_all_fit_at_full():
    cs = [_cand(i, f"短记忆{i}", score=1.0 - i * 0.1) for i in range(4)]
    plan = plan_entries(cs, 1600)
    assert len(plan.entries) == 4
    assert all(e.detail == "full" for e in plan.entries)
    assert plan.stats["dropped"] == 0
    assert plan.stats["deduped"] == 0
    assert plan.stats["tier_counts"]["full"] == 4


# ── B3: 长记忆超 cap → 降级到 abstract (不截断 full) ──
def test_oversized_degrades_to_abstract_not_truncated():
    long_content = "x" * 500  # 500 token > cap
    cs = [_cand(1, long_content, score=0.9)]
    # cap = 1600 // 1 * 2 = 3200, 但内容 ~500 < 3200, 不会降级. 用小预算强制降级:
    plan = plan_entries(cs, 100)  # cap = 100//1*2 = 200, full ~501 > 200 → 降级
    assert len(plan.entries) == 1
    e = plan.entries[0]
    assert e.detail == "abstract"
    assert "…" in e.text or len(e.text) <= 121
    assert plan.stats["tier_counts"]["abstract"] == 1


# ── B4: 多条超预算 → 降级优先 (广度: 多条 abstract 胜过 1 条 full) ──
def test_breadth_over_depth_many_degrade():
    # 5 条各 ~300 token, 预算 600 → cap=240, full 放不下 → 全降 abstract (~120)
    # 内容各不同 (相同内容会触发 body-hash dedup, 不是降级路径)
    cs = [_cand(i, f"内容{i}_" + "中" * 295, score=1.0 - i * 0.01) for i in range(5)]
    plan = plan_entries(cs, 600)
    # 至少 3 条放下 (abstract ~120 token × 3 ≈ 360 < 600)
    assert len(plan.entries) >= 3
    assert plan.stats["tier_counts"]["abstract"] >= 1


# ── B5: body-hash dedup → 内容相同的两条只保留首条 ──
def test_body_hash_dedup_identical_content():
    cs = [_cand(1, "完全相同的内容", score=0.9), _cand(2, "完全相同的内容", score=0.8)]
    plan = plan_entries(cs, 1600)
    assert len(plan.entries) == 1
    assert plan.stats["deduped"] == 1


# ── B6: 深度升级 — 剩余预算把 abstract 升回 full ──
def test_depth_upgrade_uses_leftover_budget():
    # 1 条短(10 token, full) + 1 条超 cap(300 token → abstract). 剩余预算够升第2条吗?
    # cap=1600//2*2=1600, 第2条 full 300 < 1600 但首条占后剩余够 → 升 full
    cs = [_cand(1, "短", score=0.9), _cand(2, "y" * 300, score=0.8)]
    plan = plan_entries(cs, 1600)
    # 两条都应 full (预算足够)
    assert all(e.detail == "full" for e in plan.entries)


# ── B7: stats 轨迹完整 ──
def test_stats_trajectory_complete():
    cs = [_cand(i, f"c{i}", score=0.9 - i * 0.1) for i in range(3)]
    plan = plan_entries(cs, 1600)
    s = plan.stats
    for k in ("candidates", "placed", "dropped", "deduped", "tier_counts",
              "used_tokens", "max_tokens", "per_entry_cap", "spare_tokens", "fill"):
        assert k in s, f"stats 缺 {k}"
    assert s["candidates"] == 3
    assert s["placed"] == 3
    assert "full_upgrades" in s["fill"]
    assert "spare" in s["fill"]


# ── B8: 空候选 ──
def test_empty_candidates():
    plan = plan_entries([], 1600)
    assert plan.entries == []
    assert plan.stats["placed"] == 0


# ── B9: estimate_tokens CJK vs ASCII ──
def test_estimate_tokens_cjk_heavier_than_ascii():
    assert estimate_tokens("中文字") > estimate_tokens("abc")
    assert estimate_tokens("") == 1
