"""MCP 工具分组 / 注册模式测试。

## 历史 (为什么这个文件被重写)

原版把 7 组 49 个工具的分组表**硬编码**在测试里, 作为"文档规范"去比对
mcp_server._TOOL_GROUPS。2026-08-24 新增 v5_recall 时只改了工具全集,
没同步分组表 -> 全集 50 / 分组表 49 -> 本文件 12 个测试全红。

根因是**两份真相源**。2026-08-30 起分组与层级收进 tools/registry.py,
mcp_server 全部派生。本测试相应改为从注册表派生期望值:
硬编码数字只保留 "slim = 16 (9 core + 7 facade)" 这一处, 且它由
test_tool_registry.py 交叉验证。
# 2026-09-05: v5_content 从 slim 移除, facade 8→7, slim 17→16

覆盖:
- S1: 分组表完整性 — 与工具全集双向一致, 无遗漏无多余
- S2: env 分组过滤正确性 — 指定组后注册列表 = 该组工具
- S3: fail-open — 非法组名 / 非法 mode → 回退全量, 不破坏现有行为
- S4: 默认注册 — 未设置 env 时模块级注册全量 (真实 FastMCP)
- S5: slim 模式 — 只注册 core + facade
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from memory_v5 import mcp_server
from memory_v5.mcp_server import (
    _NEW_V5_TOOLS, _SLIM_V5_TOOLS, _TOOL_GROUPS, _TOOL_TIERS, _VALID_GROUPS,
    _parse_tool_groups, _parse_tool_mode, _register_tools,
)
from memory_v5.tools import registry


def _expected_names(groups) -> list[str]:
    """给定组集合, 期望注册的工具名 (保持 _NEW_V5_TOOLS 的顺序)。"""
    if isinstance(groups, str):
        groups = {groups}
    return [fn.__name__ for fn in _NEW_V5_TOOLS
            if _TOOL_GROUPS.get(fn.__name__) in groups]


class _RecordingMCP:
    """只记录 add_tool 调用名的假 MCP 接收器."""

    def __init__(self) -> None:
        self.added: list[str] = []

    def add_tool(self, fn) -> None:
        self.added.append(getattr(fn, "__name__", str(fn)))


# ── S1: 分组表完整性 (期望值从注册表派生, 不硬编码) ──

def test_table_covers_all_tools_no_extra():
    """分组表与工具全集双向一致: 无遗漏、无多余。

    这是 2026-08-30 修复的核心不变量 —— 加工具只改 registry.py,
    漏登记会在 import registry 时直接 RuntimeError, 不会静默漂移。
    """
    table_names = set(_TOOL_GROUPS)
    tool_names = {fn.__name__ for fn in _NEW_V5_TOOLS}
    assert table_names == tool_names
    assert len(_TOOL_GROUPS) == len(registry.ALL_TOOL_NAMES)
    assert len(set(_TOOL_GROUPS)) == len(_TOOL_GROUPS)  # 无重复键


def test_all_groups_valid():
    assert set(_TOOL_GROUPS.values()) <= set(_VALID_GROUPS)


def test_group_membership_matches_registry():
    """mcp_server 的分组表 == registry 的分组表 (派生, 不是第二份手写)。"""
    assert _TOOL_GROUPS == registry.TOOL_GROUPS
    assert _TOOL_TIERS == registry.TOOL_TIERS


def test_v5_recall_is_present_in_table():
    """回归: 2026-08-24 的 v5_recall 曾只进全集不进分组表 (12 测试全红)。"""
    assert "v5_recall" in _TOOL_GROUPS
    assert any(fn.__name__ == "v5_recall" for fn in _NEW_V5_TOOLS)


def test_loop_group_exists_and_holds_v5_loop():
    """v5_loop 是 loop 组唯一成员; cordis.patch.yml 的 env 必须包含 loop。"""
    assert "loop" in _VALID_GROUPS
    assert _TOOL_GROUPS["v5_loop"] == "loop"


# ── S2: env 分组过滤正确性 ──

def test_single_group_registers_only_that_group():
    rec = _RecordingMCP()
    _register_tools(rec, "memory")
    assert rec.added == _expected_names("memory")
    assert "v5_skill_write" not in rec.added  # 其它组不进


def test_every_group_registers_exact_set():
    for group in _VALID_GROUPS:
        rec = _RecordingMCP()
        _register_tools(rec, group)
        assert rec.added == _expected_names(group), f"group {group} filter broken"


def test_multi_group_and_whitespace():
    rec = _RecordingMCP()
    _register_tools(rec, " memory , self ")
    assert rec.added == _expected_names({"memory", "self"})


def test_empty_or_unset_env_registers_all():
    for env in ("", None, "   ", ","):
        rec = _RecordingMCP()
        _register_tools(rec, env)
        assert rec.added == [fn.__name__ for fn in _NEW_V5_TOOLS]


# ── S3: fail-open ──

@pytest.mark.parametrize("env", ["bogus", "Memory", "memory,bogus", "MEMORY"])
def test_invalid_group_name_falls_back_to_all(env):
    rec = _RecordingMCP()
    _register_tools(rec, env)
    assert len(rec.added) == len(_NEW_V5_TOOLS)


def test_parse_tool_groups_unit():
    assert _parse_tool_groups(None) is None
    assert _parse_tool_groups("") is None
    assert _parse_tool_groups(" memory , self ") == {"memory", "self"}
    assert _parse_tool_groups("bogus") is None
    assert _parse_tool_groups("memory,bogus") is None


@pytest.mark.parametrize("env,expected", [
    ("legacy", "legacy"), ("slim", "slim"), ("SLIM", "slim"),
    (" slim ", "slim"), ("", "legacy"), (None, "legacy"),
    ("bogus", "legacy"),
])
def test_parse_tool_mode(env, expected):
    """mode 解析: 合法值归一化, 空/非法 fail-open 到 legacy。"""
    assert _parse_tool_mode(env) == expected


def test_invalid_mode_falls_back_to_all_tools():
    rec = _RecordingMCP()
    _register_tools(rec, None, "bogus_mode")
    assert len(rec.added) == len(_NEW_V5_TOOLS)


# ── S4: 默认注册 (真实 FastMCP, 模块级 import 已按 env 注册) ──

def test_default_module_registration_is_legacy_full():
    """未设置 V5_MCP_TOOL_MODE 时模块级注册全量 (向后兼容)。"""
    tools = mcp_server.mcp._tool_manager.list_tools()
    names = sorted(t.name for t in tools)
    assert mcp_server._TOOL_MODE == "legacy"
    assert len(tools) == len(registry.ALL_TOOL_NAMES)
    assert names == sorted(_TOOL_GROUPS)


# ── S5: slim 模式 ──

def test_slim_mode_registers_only_core_and_facade():
    rec = _RecordingMCP()
    _register_tools(rec, None, "slim")
    assert sorted(rec.added) == sorted(registry.SLIM_TOOL_NAMES)
    assert len(rec.added) == 16  # 2026-09-05: v5_content 移除, 9 core + 7 facade


def test_slim_mode_excludes_legacy_tools():
    """被门面吸收 / 被 Loop 内化的旧工具在 slim 模式下不注册。"""
    rec = _RecordingMCP()
    _register_tools(rec, None, "slim")
    for old in registry.LEGACY_ONLY_NAMES:
        assert old not in rec.added, f"{old} 不该出现在 slim 面"


def test_slim_mode_hot_path_still_present():
    """热路径 9 个工具在 slim 模式下全部保留。"""
    rec = _RecordingMCP()
    _register_tools(rec, None, "slim")
    for n in ("v5_memory_store", "v5_memory_search", "v5_memory_get",
              "v5_memory_delete", "v5_memory_stats", "v5_recall",
              "v5_project_note", "v5_project_retrieve", "v5_project_stats"):
        assert n in rec.added, f"{n} 是热路径, slim 必须保留"


def test_slim_mode_respects_group_filter():
    """slim + 分组过滤可叠加: 两个维度互不干扰。"""
    rec = _RecordingMCP()
    _register_tools(rec, "project", "slim")
    assert sorted(rec.added) == [
        n for n in registry.SLIM_TOOL_NAMES if _TOOL_GROUPS[n] == "project"
    ]
