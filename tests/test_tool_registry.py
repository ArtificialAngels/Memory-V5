"""memory_v5.tests.test_tool_registry — 工具注册表完整性测试。

存在理由: 2026-08-30 之前工具清单有**两份**真相源 (tools/__init__ 扫出的全集 +
mcp_server 手写的 _TOOL_GROUPS)。2026-08-24 新增 v5_recall 时只改了前者,
后者停在 49 -> tests/test_mcp_tool_groups.py 12 个测试全红 (assert 50 == 49)。

现在两份合成一份 (tools/registry.py), 且**漏登记 = import 即 RuntimeError**。
本测试守住"单一来源"这个不变量不再退化。
详见 docs/v5-mcp-consolidation.md。
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/ -> memory_v5/ -> core/
_CORE = Path(__file__).resolve().parents[2]
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import pytest

from memory_v5.tools import registry


# ─── S1: 三层视图自洽 ────────────────────────────────────────────────

def test_every_tool_has_group_and_tier():
    """分组 / 层级视图与工具全集一一对应 (不再有"注册了但分组表没有")。"""
    assert set(registry.TOOL_GROUPS) == set(registry.ALL_TOOL_NAMES)
    assert set(registry.TOOL_TIERS) == set(registry.ALL_TOOL_NAMES)


def test_all_groups_and_tiers_are_valid():
    assert set(registry.TOOL_GROUPS.values()) <= set(registry.VALID_GROUPS)
    assert set(registry.TOOL_TIERS.values()) <= set(registry.VALID_TIERS)


def test_slim_plus_legacy_only_equals_all():
    """slim 集 + 仅 legacy 集 = 全集, 无重叠无遗漏。"""
    slim = set(registry.SLIM_TOOL_NAMES)
    legacy = set(registry.LEGACY_ONLY_NAMES)
    assert slim & legacy == set()
    assert slim | legacy == set(registry.ALL_TOOL_NAMES)


def test_tier_of_names_matches_views():
    for n in registry.SLIM_TOOL_NAMES:
        assert registry.TOOL_TIERS[n] in ("core", "facade")
    for n in registry.LEGACY_ONLY_NAMES:
        assert registry.TOOL_TIERS[n] == "legacy"


# ─── S2: 门面吸收关系完整 (文档用的映射表不能指空) ────────────────────

def test_facade_absorbs_reference_real_tools():
    """FACADE_ABSORBS 里每个被吸收的旧工具都真实存在, 且确实是 legacy 层。"""
    for facade_name, absorbed in registry.FACADE_ABSORBS.items():
        assert facade_name in registry.SLIM_TOOL_NAMES, f"{facade_name} 不在 slim 集"
        for old in absorbed:
            assert old in registry.ALL_TOOL_NAMES, f"{old} 不存在"
            assert registry.TOOL_TIERS[old] == "legacy", f"{old} 不该是 legacy"


def test_every_legacy_tool_is_accounted_for():
    """每个 legacy 工具都被某个门面或某个 Loop 阶段认领 —— 不留孤儿。

    2026-09-05: v5_content 的 3 个 action (dissonance/narrative/proactive)
    已从 slim facade 移除 (零消费方清理), 底层函数仍在 legacy 中但不再被
    任何 facade action 吸收。这些是"已从 slim 移除"的合法孤儿, 豁免。
    """
    from memory_v5.tools.slim_check import SLIM_REMOVED_LEGACY
    claimed = set()
    for absorbed in registry.FACADE_ABSORBS.values():
        claimed.update(absorbed)
    for absorbed in registry.LOOP_ABSORBS.values():
        claimed.update(absorbed)
    orphans = sorted(set(registry.LEGACY_ONLY_NAMES) - claimed - SLIM_REMOVED_LEGACY)
    assert orphans == [], f"以下 legacy 工具无人认领: {orphans}"


def test_loop_absorbs_phases_are_valid():
    from memory_v5.loop import PHASES
    for phase, absorbed in registry.LOOP_ABSORBS.items():
        assert phase in PHASES, f"未知 phase: {phase}"
        for old in absorbed:
            assert old in registry.ALL_TOOL_NAMES


# ─── S3: 模式选择 ────────────────────────────────────────────────────

def test_tools_for_mode_legacy_is_all():
    names = [f.__name__ for f in registry.tools_for_mode("legacy")]
    assert sorted(names) == sorted(registry.ALL_TOOL_NAMES)


def test_tools_for_mode_slim_is_core_and_facade():
    names = [f.__name__ for f in registry.tools_for_mode("slim")]
    assert sorted(names) == sorted(registry.SLIM_TOOL_NAMES)


@pytest.mark.parametrize("mode", ["", None, "bogus", "SLIM_TYPO"])
def test_tools_for_mode_invalid_fails_open_to_all(mode):
    """未知 mode -> 全量 (fail-open), 与分组过滤同款约定: 宁可多注册。"""
    names = [f.__name__ for f in registry.tools_for_mode(mode)]
    assert sorted(names) == sorted(registry.ALL_TOOL_NAMES)


def test_slim_set_is_the_documented_16():
    """精简面 = 9 热路径 + 7 门面。改这个数字必须同步 docs/v5-mcp-consolidation.md。

    2026-09-05: v5_content 从 slim 移除, facade 8→7, slim 17→16。
    """
    assert len(registry.SLIM_TOOL_NAMES) == 16
    assert len([n for n, t in registry.TOOL_TIERS.items() if t == "core"]) == 9
    assert len([n for n, t in registry.TOOL_TIERS.items() if t == "facade"]) == 7


def test_hot_path_tools_are_core():
    """persona / SOUL / AGENTS 点名的工具必须在 core 层 (slim 模式也可见)。"""
    hot = {
        "v5_memory_store", "v5_memory_search", "v5_recall",
        "v5_project_note", "v5_project_retrieve",
    }
    for n in hot:
        assert registry.TOOL_TIERS[n] == "core", f"{n} 必须在热路径"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
