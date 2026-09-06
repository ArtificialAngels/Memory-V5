#!/usr/bin/env python3
"""MCP server — 把 V5 记忆引擎以 MCP 协议暴露给外部 agent (dsh 等).

两种注册模式 (V5_MCP_TOOL_MODE):
    legacy  全量 58 个 v5_* 工具 (默认, 兼容外部契约)
    slim    精简 17 个 —— 9 个热路径独立工具 + 8 个冷路径门面
            (一个资源一个工具 + action 分发)

工具清单 / 分组 / 层级的唯一真相源是 ``memory_v5/tools/registry.py``;
本模块只做派生与注册, 不再维护第二份表。

经 stdio transport 与客户端通信。见 docs/v5-mcp-consolidation.md
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure Ikaros-memory/ is on the Python path
_HERE = Path(__file__).resolve().parent  # Ikaros-memory/v5/
_V5_ROOT = _HERE.parent                  # Ikaros-memory/
if str(_V5_ROOT) not in sys.path:
    sys.path.insert(0, str(_V5_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [v5-mcp] %(message)s",
)
logger = logging.getLogger("ikaros.v5.mcp")


# ── MCP Server ─────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

# 精简模式下的工具说明 (legacy 模式追加在公共说明之后)
_SLIM_INSTRUCTIONS = (
    "\n"
    "TOOL SURFACE (consolidated): 9 hot-path tools are standalone — "
    "memory_store / memory_search / memory_get / memory_delete / memory_stats / "
    "recall / project_note / project_retrieve / project_stats. "
    "Everything else is a facade with an `action` parameter:\n"
    "  v5_self(action=model|reflect|thought|curiosity|subconscious|anchor|discover)\n"
    "  v5_state(action=emotion|emotion_update|emotion_label|care|care_check|"
    "vitality|relationship|activity|compression)\n"
    "  v5_content(action=narrative|dissonance|proactive)\n"
    "  v5_skill(action=list|search|get|write|remove)\n"
    "  v5_reflection(action=stats|read|synthesize|apply|promote)\n"
    "  v5_directive(action=list|add|off|stats)\n"
    "  v5_repeat(action=stats|check|penalty|clear)\n"
    "  v5_loop(action=status|run, phase=pre|post|maintenance)\n"
    "Passing an unknown action returns the valid list — no guessing needed.\n"
    "\n"
    "Maintenance actions (vitality/relationship tick, anti-repeat recording, "
    "reflection pipeline) are driven automatically by the Standard Memory Loop "
    "at pre / post / maintenance phases of each turn. Do NOT call them manually; "
    "use v5_loop(action=status) to inspect their last-run times."
)

mcp = FastMCP(
    "Ikaros V5 Memory",
    instructions=(
        "Ikaros V5 Memory System provides long-term memory storage, entity graph retrieval, "
        "emotion tracking, self-reflection, curiosity-driven introspection, care scoring, "
        "vitality management, relationship modeling, narrative generation, dissonance "
        "detection, and proactive self-improvement.\n"
        "\n"
        "Memory tools: memory_store saves facts, preferences, lessons, and emotional events "
        "with PAD emotion fingerprints. memory_search performs dual-path retrieval combining "
        "full-text keyword and semantic vector search. memory_get fetches a single memory by ID. "
        "memory_delete removes a memory by ID. memory_stats returns total count, long-term count, "
        "average weight, and per-type breakdown.\n"
        "\n"
        "Emotion tools: analyze_emotion processes text through the PAD model to extract "
        "pleasure, arousal, and dominance scores, producing a raw dict and a human-readable "
        "mood label. emotion_status returns the current emotional state. emotion_label maps "
        "PAD values to mood labels.\n"
        "\n"
        "Self tools: self_model returns the full self-model JSON including curiosity, vitality, "
        "relationship, and care state. self_reflect triggers a deep self-reflection cycle. "
        "latest_thought returns what Ikaros is currently thinking. curiosity_check returns "
        "curiosity level and whether it crosses the reflection threshold. subconscious returns "
        "subconscious activity state. self_discover generates new self-knowledge by introspecting "
        "on stored memories. reflect_run_op is the background reflection operator.\n"
        "\n"
        "Care tools: care_check evaluates a conversation message for care and concern signals. "
        "care_status returns current care state and trend history.\n"
        "\n"
        "Vitality tools: vitality returns the current vitality meter and decay rate. "
        "vitality_tick advances vitality decay by one increment.\n"
        "\n"
        "Relationship tools: relationship returns the relationship model with brother "
        "including trust, familiarity, and mood trends. relationship_tick advances the "
        "relationship decay cycle.\n"
        "\n"
        "Narrative and proactive tools: narrative_generate synthesizes recent memories "
        "into a coherent daily narrative. dissonance_check detects logical contradictions "
        "among stored memories. proactive_check evaluates whether Ikaros should initiate "
        "conversation proactively.\n"
        "\n"
        "V5.2 Reflection tools: reflection_synthesize creates new reflection entries from facts. "
        "reflection_read queries the reflection database by status/entity. "
        "reflection_apply_evidence applies reinforcement or disputation signals to a reflection, "
        "triggering automatic status transitions (pending→confirmed→promoted→merged). "
        "reflection_promote merges a reflection into the character's self-model persona. "
        "reflection_stats returns aggregate reflection counts by status.\n"
        "\n"
        "V5.2 Anti-repeat tools: anti_repeat_record logs a response's n-grams into the "
        "anti-repetition corpus. anti_repeat_check evaluates a candidate text for repetition "
        "risk using BM25-style scoring. anti_repeat_penalty returns a system prompt hint if "
        "repetition risk exceeds threshold.\n"
        "\n"
        "V5.2 Directive tools: directive_add creates a user directive (banned topic, preference, "
        "behavior rule) with configurable TTL. directive_list returns active directives for a "
        "character. directive_deactivate disables a directive by ID. directive_stats returns "
        "total/active directive counts."
    )
    # 精简模式追加门面说明 (在文件头读取 env, 见 _TOOL_MODE)
    + (_SLIM_INSTRUCTIONS if os.environ.get("V5_MCP_TOOL_MODE", "").strip().lower() == "slim" else ""),
)


# Inline docs: docs/scripts/core/memory_v5/v5/mcp_server.md
# 2026-08-24 P3-25: 工具清单单一来源 — 从 tools.__all__ 派生, 消除 import 块与
# _NEW_V5_TOOLS 显式列表双份维护的漂移风险 (加工具漏一边 → 静默不注册或 NameError)。
#
# 2026-08-30: P3-25 只统一了「工具清单」一边, 分组表 _TOOL_GROUPS 仍是手写第二份,
# 于是 2026-08-24 新增的 v5_recall 进了清单(50) 却不在分组表(49) →
# tests/test_mcp_tool_groups.py 12 个测试全红。现在分组/层级也收进
# memory_v5/tools/registry.py, 本模块全部派生: 漏登记 = 启动即 RuntimeError,
# 不会再有「注册了但分组表没有」的静默漂移。
from memory_v5 import tools as _tools_pkg  # noqa: E402
from memory_v5.tools import registry as _registry  # noqa: E402

# ── 工具分组 / 层级: 全部由 registry 派生, 本模块零手写 ──────────────
_VALID_GROUPS = _registry.VALID_GROUPS          # 8 组 (含 loop)
_TOOL_GROUPS: dict[str, str] = _registry.TOOL_GROUPS
_TOOL_TIERS: dict[str, str] = _registry.TOOL_TIERS

# ── 注册模式 ─────────────────────────────────────────────────────────
# legacy: core + facade + legacy (全量 58, 兼容外部契约)
# slim  : core + facade (17)
_VALID_MODES = ("legacy", "slim")
_DEFAULT_MODE = "legacy"


def _parse_tool_mode(env_value: str | None) -> str:
    """解析 V5_MCP_TOOL_MODE。未设置 / 空 / 非法 → legacy (fail-safe 全量)。

    与分组过滤同款 fail-open 约定: 宁可多注册, 不可静默少注册。
    """
    mode = (env_value or "").strip().lower()
    if mode in _VALID_MODES:
        return mode
    if mode:
        logger.warning(
            "V5_MCP_TOOL_MODE=%r 非法 (可选 %s); 回退 legacy 全量注册 (fail-open)",
            env_value, list(_VALID_MODES))
    return _DEFAULT_MODE


_TOOL_MODE = _parse_tool_mode(os.environ.get("V5_MCP_TOOL_MODE"))


def _parse_tool_groups(env_value: str | None) -> set[str] | None:
    """解析 V5_MCP_TOOL_GROUPS 环境变量 → 允许的工具组集合.

    未设置 / 空 / 含非法组名 → 返回 None (fail-open: 全量注册, 不破坏现有行为).
    返回 None 或全量集合时, 注册循环不做过滤.
    """
    if env_value is None:
        return None
    names = [n.strip() for n in env_value.split(",") if n.strip()]
    if not names:
        return None
    unknown = sorted(set(names) - set(_VALID_GROUPS))
    if unknown:
        logger.warning(
            "V5_MCP_TOOL_GROUPS contains unknown group(s) %s; ignoring filter "
            "and registering all tools (fail-open)", unknown)
        return None
    return set(names)


def _register_tools(mcp_obj, env_value: str | None = None,
                    mode: str = _DEFAULT_MODE) -> None:
    """按 V5_MCP_TOOL_MODE 选工具集、按 V5_MCP_TOOL_GROUPS 过滤, 注册到 mcp_obj.

    env_value=None/空/非法组 → 该模式下全量注册 (行为与旧循环一致).
    mode 非法 → 由 tools_for_mode 内部 fail-open 到全量.
    被过滤的工具跳过 add_tool, 记 debug 日志.
    """
    groups = _parse_tool_groups(env_value)
    for _tool_fn in _registry.tools_for_mode(mode):
        _name = getattr(_tool_fn, "__name__", str(_tool_fn))
        _group = _TOOL_GROUPS.get(_name)
        # 未分组的工具视为全组可见, 不参与过滤 (与文档约定一致)
        if groups is not None and _group is not None and _group not in groups:
            logger.debug("skip tool %s (group %s not in V5_MCP_TOOL_GROUPS=%s)",
                         _name, _group, env_value)
            continue
        try:
            mcp_obj.add_tool(_tool_fn)
        except Exception as _e:  # noqa: BLE001
            logger.warning("failed to register tool %s: %s", _name, _e)


# legacy 全量 (向后兼容: 旧代码 / 测试按 _NEW_V5_TOOLS 取全集)
_NEW_V5_TOOLS = [
    getattr(_tools_pkg, _n) for _n in _tools_pkg.__all__ if _n != "__all__"
]
# slim 精简集 (core 热路径 + facade 门面)
_SLIM_V5_TOOLS = [
    getattr(_tools_pkg, _n) for _n in _tools_pkg.__all_slim__
]

# V5_MCP_TOOL_MODE=legacy|slim; V5_MCP_TOOL_GROUPS=memory,self,... 过滤注册
# 两者未设置/空/非法 → 全量 (fail-open)
_register_tools(mcp, os.environ.get("V5_MCP_TOOL_GROUPS"), _TOOL_MODE)
logger.info("v5 MCP tools registered: mode=%s groups=%s count=%d",
            _TOOL_MODE, os.environ.get("V5_MCP_TOOL_GROUPS") or "*",
            len(mcp._tool_manager.list_tools()))


# ── Entry point ────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        # Hermes Studio transport: SSE on :9877.
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = 9877
        logger.info("v5 MCP server starting (sse) on 127.0.0.1:9877 ...")
        mcp.run(transport="sse")
    else:
        logger.info("v5 MCP server starting (stdio)...")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()