"""MCP 工具分组 (docs/hermes-tools-scoping.md Option 2) 测试.

覆盖:
- S1: 分组表完整性 — 48 工具全覆盖、无遗漏无多余, 组名合法, 分组与文档规范一致
- S2: env 过滤正确性 — 指定组后注册列表 = 该组工具; 空/未设置 = 全量
- S3: fail-open — 非法组名 / 大小写不符 → 回退全量注册, 不破坏现有行为
- S4: 默认注册 — 未设置 env 时模块级注册全量 48 工具 (真实 FastMCP)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from v5 import mcp_server
from v5.mcp_server import (
    _NEW_V5_TOOLS, _TOOL_GROUPS, _VALID_GROUPS,
    _parse_tool_groups, _register_tools,
)


# ── 规范: docs/hermes-tools-scoping.md 40-53 行分组表 (7 组 48 工具) ──
_DOC_GROUPS: dict[str, list[str]] = {
    "memory": [
        "v5_memory_store", "v5_memory_search", "v5_memory_get",
        "v5_memory_delete", "v5_memory_stats", "v5_dissonance_check",
        "v5_context_compression_stats",
        "v5_directive_add", "v5_directive_list", "v5_directive_deactivate",
        "v5_directive_stats",
        "v5_anti_repeat_record", "v5_anti_repeat_check",
        "v5_anti_repeat_penalty", "v5_anti_repeat_clear",
        "v5_anti_repeat_stats",
        "v5_reflection_synthesize", "v5_reflection_read",
        "v5_reflection_apply_evidence", "v5_reflection_promote",
        "v5_reflection_stats",
    ],
    "self": [
        "v5_analyze_emotion", "v5_emotion_status", "v5_emotion_label",
        "v5_self_model", "v5_self_reflect", "v5_self_discover",
        "v5_latest_thought", "v5_curiosity_check", "v5_subconscious",
        "v5_reflect_run_op", "v5_narrative_generate", "v5_proactive_check",
        "v5_activity_status",
    ],
    "care": ["v5_care_check", "v5_care_status"],
    "vitality": ["v5_vitality", "v5_vitality_tick"],
    "relationship": ["v5_relationship", "v5_relationship_tick"],
    "skill": [
        "v5_skill_write", "v5_skill_list", "v5_skill_get",
        "v5_skill_search", "v5_skill_remove",
    ],
    "project": ["v5_project_note", "v5_project_retrieve", "v5_project_stats"],
}
_DOC_COUNTS = {g: len(names) for g, names in _DOC_GROUPS.items()}
assert sum(_DOC_COUNTS.values()) == 48


class _RecordingMCP:
    """只记录 add_tool 调用名的假 MCP 接收器."""

    def __init__(self) -> None:
        self.added: list[str] = []

    def add_tool(self, fn) -> None:
        self.added.append(getattr(fn, "__name__", str(fn)))


# ── S1: 分组表完整性 ──

def test_table_covers_all_new_tools_no_extra():
    """分组表与 _NEW_V5_TOOLS 双向一致: 无遗漏、无多余."""
    table_names = set(_TOOL_GROUPS)
    tool_names = {fn.__name__ for fn in _NEW_V5_TOOLS}
    assert table_names == tool_names
    assert len(_TOOL_GROUPS) == 48
    assert len(_NEW_V5_TOOLS) == 48
    assert len(set(_TOOL_GROUPS)) == 48  # 无重复键


def test_all_groups_valid_and_match_doc():
    """组名全部合法; 分组与文档 40-53 行规范逐名一致."""
    assert set(_TOOL_GROUPS.values()) <= set(_VALID_GROUPS)
    # 逐组精确相等
    for group, names in _DOC_GROUPS.items():
        actual = sorted(n for n, g in _TOOL_GROUPS.items() if g == group)
        assert actual == sorted(names), f"group {group} names mismatch"


def test_group_counts_match_doc():
    """各组数量与文档一致 (21/13/2/2/2/5/3)."""
    from collections import Counter
    counts = Counter(_TOOL_GROUPS.values())
    assert dict(counts) == _DOC_COUNTS


# ── S2: env 过滤正确性 ──

def _expected_names(group: str) -> list[str]:
    return [fn.__name__ for fn in _NEW_V5_TOOLS if _TOOL_GROUPS.get(fn.__name__) == group]


def test_single_group_registers_only_that_group():
    rec = _RecordingMCP()
    _register_tools(rec, "memory")
    assert rec.added == _expected_names("memory")
    assert len(rec.added) == _DOC_COUNTS["memory"]
    assert "v5_skill_write" not in rec.added  # 其它组不进


def test_every_group_registers_exact_doc_set():
    for group in _VALID_GROUPS:
        rec = _RecordingMCP()
        _register_tools(rec, group)
        assert rec.added == _expected_names(group), f"group {group} filter broken"
        assert len(rec.added) == _DOC_COUNTS[group]


def test_multi_group_and_whitespace():
    rec = _RecordingMCP()
    _register_tools(rec, " memory , self ")
    allowed = {"memory", "self"}
    expected = [fn.__name__ for fn in _NEW_V5_TOOLS
                if _TOOL_GROUPS.get(fn.__name__) in allowed]
    assert rec.added == expected  # 保持 _NEW_V5_TOOLS 顺序, 仅过滤
    assert len(rec.added) == 21 + 13


def test_empty_or_unset_env_registers_all():
    for env in ("", None, "   ", ","):
        rec = _RecordingMCP()
        _register_tools(rec, env)
        assert len(rec.added) == 48
        assert rec.added == [fn.__name__ for fn in _NEW_V5_TOOLS]  # 顺序保持


# ── S3: fail-open ──

@pytest.mark.parametrize("env", ["bogus", "Memory", "memory,bogus", "MEMORY"])
def test_invalid_group_name_falls_back_to_all(env):
    rec = _RecordingMCP()
    _register_tools(rec, env)
    assert len(rec.added) == 48


def test_parse_tool_groups_unit():
    assert _parse_tool_groups(None) is None
    assert _parse_tool_groups("") is None
    assert _parse_tool_groups(" memory , self ") == {"memory", "self"}
    assert _parse_tool_groups("bogus") is None
    assert _parse_tool_groups("memory,bogus") is None


# ── S4: 默认注册 (真实 FastMCP, 模块级 import 已全量注册) ──

def test_default_module_registration_all_48():
    """未设置 env 时 mcp_server 模块级注册全量 48 工具 (行为不变)."""
    tools = mcp_server.mcp._tool_manager.list_tools()
    names = sorted(t.name for t in tools)
    assert len(tools) == 48
    assert names == sorted(_TOOL_GROUPS)
