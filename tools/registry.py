# 详细说明见 docs/v5-mcp-consolidation.md
"""MCP 工具单一注册表 — 工具名 -> (分组, 层级) 的唯一权威来源。

## 为什么要它

2026-08-30 之前工具清单存在**两份**真相源:

    1. memory_v5/tools/__init__.py  —— 从各子模块 dir() 扫出的 v5_* 全集
    2. memory_v5/mcp_server.py      —— 手写的 _TOOL_GROUPS 字典 (用于分组过滤)

2026-08-24 的 P3-25 提交把 (1) 改成自动派生, 却漏了 (2)。结果 2026-08-24 新增
的 ``v5_recall`` 进了全集 (50 个) 但不在分组表 (49 个) 里 ——
``tests/test_mcp_tool_groups.py`` 12 个测试全红 (assert 50 == 49)。

本模块把两份合成一份: **分组与层级都在本文件的表里声明, mcp_server 全部派生**。
加工具只改这里, 漏改 = 工具不注册 (启动即见), 不会再出现"注册了但分组表没有"
这种静默漂移。

## 层级 (tier)

    core     热路径, 两个模式都注册。独立工具、无 action 参数, 调用最快。
    facade   冷路径门面 (一个资源一个工具 + action 分发), 两个模式都注册。
    legacy   被门面吸收 / 被 Loop 内化的旧工具。仅 legacy 模式注册。

legacy 模式 (默认, 兼容) 注册 core + facade + legacy = 全量。
slim 模式注册 core + facade = 精简面 (约 17 个)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ─── 分组 ────────────────────────────────────────────────────────────
# 7 个历史分组 (docs/archive/hermes-tools-scoping.md Option 2) + loop。
# 注意: cordis.patch.yml 的 V5_MCP_TOOL_GROUPS 需包含 loop, 否则 v5_loop
# 会被过滤掉 —— 新增分组时务必同步 overlay。
VALID_GROUPS = (
    "memory", "self", "care", "vitality", "relationship",
    "skill", "project", "loop",
)

CORE = "core"
FACADE = "facade"
LEGACY = "legacy"
VALID_TIERS = (CORE, FACADE, LEGACY)


@dataclass(frozen=True)
class ToolSpec:
    """一个 MCP 工具的注册信息。"""
    fn: Callable
    group: str
    tier: str


# ─── 热路径 (core): 独立工具, 无 action 歧义 ──────────────────────────
_CORE: dict[str, str] = {
    # memory 读写 — 外部引用最密集的 5 个 (SOUL / cordis.patch.yml persona 点名)
    "v5_memory_store": "memory",
    "v5_memory_search": "memory",
    "v5_memory_get": "memory",
    "v5_memory_delete": "memory",
    "v5_memory_stats": "memory",
    # 预算感知召回 — persona 直接点名 v5_recall
    "v5_recall": "memory",
    # 项目轨 — AGENTS.md / SOUL.md 点名 v5_project_note / v5_project_retrieve
    "v5_project_note": "project",
    "v5_project_retrieve": "project",
    "v5_project_stats": "project",
}

# ─── 冷路径门面 (facade): 一个资源一个工具 + action 分发 ───────────────
# 定义于 tools/facade.py。
_FACADE: dict[str, str] = {
    "v5_self": "self",            # 3 action: model/reflect/anchor
                                  # (2026-09-05 精简: 移除 thought/curiosity/
                                  #  subconscious/discover 四个零消费方 action)
    "v5_state": "self",           # 6 action: emotion/emotion_update/care/
                                  #          care_check/vitality/relationship
                                  # (2026-09-05 精简: 移除 emotion_label/
                                  #          activity/compression)
    "v5_skill": "skill",          # 5 -> 1: write/list/get/search/remove
    "v5_reflection": "memory",    # 5 -> 1: synthesize/read/apply/promote/stats
    "v5_directive": "memory",     # 4 -> 1: add/list/off/stats
    "v5_repeat": "memory",        # 5 -> 1 (record 内化进 Loop, 剩 4 个 action)
    "v5_loop": "loop",            # 新: 标准记忆循环的入口与状态观测
}
# v5_content 已于 2026-09-05 从 slim 移除 (narrative/dissonance/proactive
# 均零消费方); 底层函数仍在 _LEGACY 中保留, legacy 模式可用。

# ─── legacy: 被吸收 / 被内化的旧工具 ──────────────────────────────────
# 这些函数在 Python 侧**全部保留** (测试、脚本、桥接层继续 import),
# 仅在 slim 模式下不注册为 MCP 工具。
_LEGACY: dict[str, str] = {
    # self 轨 -> v5_self
    "v5_analyze_emotion": "self",
    "v5_emotion_status": "self",
    "v5_emotion_label": "self",
    "v5_self_model": "self",
    "v5_self_reflect": "self",
    "v5_self_discover": "self",
    "v5_latest_thought": "self",
    "v5_curiosity_check": "self",
    "v5_subconscious": "self",
    "v5_context_refresh": "self",        # -> v5_self(action=anchor) / Loop pre
    # 状态轨 -> v5_state
    "v5_care_check": "care",
    "v5_care_status": "care",
    "v5_vitality": "vitality",
    "v5_vitality_tick": "vitality",      # -> Loop post
    "v5_relationship": "relationship",
    "v5_relationship_tick": "relationship",  # -> Loop post
    # 内容生成/检测 -> v5_content
    "v5_narrative_generate": "self",
    "v5_dissonance_check": "memory",
    "v5_proactive_check": "self",
    "v5_activity_status": "self",
    "v5_context_compression_stats": "memory",
    # 反思管线 -> Loop maintenance
    "v5_reflect_run_op": "self",
    # skill 轨 -> v5_skill
    "v5_skill_write": "skill",
    "v5_skill_list": "skill",
    "v5_skill_get": "skill",
    "v5_skill_search": "skill",
    "v5_skill_remove": "skill",
    # reflection 轨 -> v5_reflection
    "v5_reflection_synthesize": "memory",
    "v5_reflection_read": "memory",
    "v5_reflection_apply_evidence": "memory",
    "v5_reflection_promote": "memory",
    "v5_reflection_stats": "memory",
    # directive 轨 -> v5_directive
    "v5_directive_add": "memory",
    "v5_directive_list": "memory",
    "v5_directive_deactivate": "memory",
    "v5_directive_stats": "memory",
    # anti_repeat 轨 -> v5_repeat (record 内化进 Loop post)
    "v5_anti_repeat_record": "memory",
    "v5_anti_repeat_check": "memory",
    "v5_anti_repeat_penalty": "memory",
    "v5_anti_repeat_clear": "memory",
    "v5_anti_repeat_stats": "memory",
}


def _build() -> dict[str, ToolSpec]:
    """装配注册表。三步: 先收集函数, 再按表赋 (group, tier)。"""
    from memory_v5.tools import (
        care_tool, directive_tool, emotion_tool, extra_tool,
        facade, memory_tool, project_tool, recall_tool,
        reflection_tool, relationship_tool, repeat_tool,
        self_tool, skill_tool, vitality_tool,
    )

    modules = (
        emotion_tool, memory_tool, self_tool, care_tool, vitality_tool,
        relationship_tool, extra_tool, reflection_tool, repeat_tool,
        directive_tool, project_tool, skill_tool, recall_tool, facade,
    )

    # 1. 从子模块扫出所有 v5_* callable (行为同原 tools/__init__ 的收集逻辑)
    found: dict[str, Callable] = {}
    for mod in modules:
        for name in dir(mod):
            if not name.startswith("v5_"):
                continue
            fn = getattr(mod, name)
            if callable(fn) and name not in found:
                found[name] = fn

    # 2. 按声明表赋 (group, tier)
    tier_map: dict[str, tuple[str, str]] = {}
    for name, group in _CORE.items():
        tier_map[name] = (group, CORE)
    for name, group in _FACADE.items():
        tier_map[name] = (group, FACADE)
    for name, group in _LEGACY.items():
        tier_map[name] = (group, LEGACY)

    # 3. 装配 + 完整性自检 (漏登记 = 启动即炸, 不留静默漂移)
    specs: dict[str, ToolSpec] = {}
    missing = [n for n in found if n not in tier_map]
    if missing:
        raise RuntimeError(
            f"tools/registry.py 未登记以下工具 (加工具必须同步本文件): {sorted(missing)}"
        )
    ghost = [n for n in tier_map if n not in found]
    if ghost:
        raise RuntimeError(
            f"tools/registry.py 登记了不存在的工具: {sorted(ghost)}"
        )

    for name, (group, tier) in tier_map.items():
        if group not in VALID_GROUPS:
            raise RuntimeError(f"tool {name}: 非法分组 {group!r}")
        if tier not in VALID_TIERS:
            raise RuntimeError(f"tool {name}: 非法层级 {tier!r}")
        specs[name] = ToolSpec(fn=found[name], group=group, tier=tier)
    return specs


TOOL_SPECS: dict[str, ToolSpec] = _build()

# 派生视图 (mcp_server 与测试都从这里取, 不再手写第二份)
TOOL_GROUPS: dict[str, str] = {n: s.group for n, s in TOOL_SPECS.items()}
TOOL_TIERS: dict[str, str] = {n: s.tier for n, s in TOOL_SPECS.items()}

ALL_TOOL_NAMES: list[str] = sorted(TOOL_SPECS)                    # legacy 全量
SLIM_TOOL_NAMES: list[str] = sorted(                              # core + facade
    n for n, s in TOOL_SPECS.items() if s.tier in (CORE, FACADE)
)
LEGACY_ONLY_NAMES: list[str] = sorted(
    n for n, s in TOOL_SPECS.items() if s.tier == LEGACY
)


def tools_for_mode(mode: str) -> list[Callable]:
    """按模式返回待注册的函数列表 (legacy 顺序 = 全量字母序, 保持幂等)。

    mode:
        "legacy" -> core + facade + legacy (全量, 兼容旧行为)
        "slim"   -> core + facade
    未知 mode -> fail-open 返回全量 (同分组过滤的 fail-open 约定)。
    """
    if mode == "slim":
        names = SLIM_TOOL_NAMES
    else:
        names = ALL_TOOL_NAMES
    return [TOOL_SPECS[n].fn for n in names]


# 门面 -> 被吸收的旧工具 (文档 / 迁移提示用)
# 2026-09-05: v5_content 从 slim 移除; v5_self/v5_state 精简 action。
# 被移除的 action 对应底层函数仍在 _LEGACY 中, legacy 模式可直接调用。
FACADE_ABSORBS: dict[str, list[str]] = {
    "v5_self": [
        "v5_self_model", "v5_self_reflect", "v5_context_refresh",
        # 以下为 2026-09-05 从 slim action 移除的零消费方工具, legacy 仍可用:
        "v5_latest_thought", "v5_curiosity_check", "v5_subconscious",
        "v5_self_discover",
    ],
    "v5_state": [
        "v5_analyze_emotion", "v5_emotion_status", "v5_care_check",
        "v5_care_status", "v5_vitality", "v5_vitality_tick",
        "v5_relationship", "v5_relationship_tick",
        # 以下为 2026-09-05 从 slim action 移除的零消费方工具, legacy 仍可用:
        "v5_emotion_label", "v5_activity_status", "v5_context_compression_stats",
    ],
    # v5_content 已从 slim 移除, 但其吸收的工具仍列于此供迁移参考:
    # "v5_content": ["v5_narrative_generate", "v5_dissonance_check", "v5_proactive_check"],
    "v5_skill": [
        "v5_skill_write", "v5_skill_list", "v5_skill_get",
        "v5_skill_search", "v5_skill_remove",
    ],
    "v5_reflection": [
        "v5_reflection_synthesize", "v5_reflection_read",
        "v5_reflection_apply_evidence", "v5_reflection_promote",
        "v5_reflection_stats",
    ],
    "v5_directive": [
        "v5_directive_add", "v5_directive_list",
        "v5_directive_deactivate", "v5_directive_stats",
    ],
    "v5_repeat": [
        "v5_anti_repeat_record", "v5_anti_repeat_check",
        "v5_anti_repeat_penalty", "v5_anti_repeat_clear",
        "v5_anti_repeat_stats",
    ],
}

# 被 Loop 内化的工具 (slim 模式下不再暴露, 由 loop 引擎自动驱动)
LOOP_ABSORBS: dict[str, list[str]] = {
    "pre": ["v5_context_refresh", "v5_recall", "v5_project_retrieve"],
    "post": [
        "v5_vitality_tick", "v5_relationship_tick", "v5_anti_repeat_record",
    ],
    "maintenance": ["v5_reflect_run_op"],
}
