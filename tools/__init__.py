# 详细说明见 docs/v5-mcp-consolidation.md
# 工具清单的**收集**在这里 (从各子模块 dir() 扫 v5_* callable);
# 工具的**分组与层级**在 registry.py 声明, 本模块派生, 不重复维护。

from __future__ import annotations

import sys

# Ensure Ikaros-memory/ is importable (so `import memory_v5` works from anywhere).
# Reuse the single canonical root path defined in utils (no duplicate computation).
from memory_v5.tools.utils import V5_ROOT
if str(V5_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(V5_ROOT.parent))

from memory_v5.tools import care_tool
from memory_v5.tools import emotion_tool
from memory_v5.tools import extra_tool
from memory_v5.tools import memory_tool
from memory_v5.tools import relationship_tool
from memory_v5.tools import self_tool
from memory_v5.tools import vitality_tool
# V5.2: neko migration tools
from memory_v5.tools import reflection_tool
from memory_v5.tools import repeat_tool
from memory_v5.tools import directive_tool
# V5.4: project track (decisions / pitfalls / conventions)
from memory_v5.tools import project_tool
# V5.5: skill track (agent-distilled reusable workflows, Markdown files)
from memory_v5.tools import skill_tool
# F1+F2+F4: 预算感知召回 (OpenViking context_assembler 借鉴)
from memory_v5.tools import recall_tool
# 2026-08-30: 冷路径门面 (一个资源一个工具 + action 分发)
from memory_v5.tools import facade

# Collect every v5_* callable from the submodules into __all__.
__all__: list[str] = []
_SEEN = set()
for _mod in (
    emotion_tool, memory_tool, self_tool,
    care_tool, vitality_tool, relationship_tool, extra_tool,
    reflection_tool, repeat_tool, directive_tool, project_tool,
    skill_tool, recall_tool, facade,
):
    for _name in dir(_mod):
        if _name.startswith("v5_") and _name not in _SEEN:
            _fn = getattr(_mod, _name)
            if callable(_fn):
                globals()[_name] = _fn
                __all__.append(_name)
                _SEEN.add(_name)

__all__.sort()

# registry 必须在所有子模块 import 完之后再导入 —— 它内部会
# `from memory_v5.tools import <submodule>`, 依赖本模块已把子模块挂到包命名空间。
# (放在文件头会撞上半初始化包的循环导入。)
from memory_v5.tools import registry  # noqa: E402

# ── 派生视图 (mcp_server / 测试 / 运维脚本从这里取) ──────────────────
#: slim 模式注册的工具名 (core 热路径 + facade 门面)
__all_slim__: list[str] = list(registry.SLIM_TOOL_NAMES)
#: 仅 legacy 模式注册的工具名 (被门面吸收 / 被 Loop 内化)
__legacy_only__: list[str] = list(registry.LEGACY_ONLY_NAMES)
#: 工具名 -> 分组 (单一来源, 由 registry 派生)
TOOL_GROUPS = registry.TOOL_GROUPS
#: 工具名 -> 层级 (core / facade / legacy)
TOOL_TIERS = registry.TOOL_TIERS

# 保持历史行为: __all__ 末尾带字面量 "__all__" (mcp_server 会过滤掉它)。
if "__all__" not in __all__:
    __all__.append("__all__")
