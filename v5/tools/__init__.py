# 详细说明见 docs/scripts/core/v5/v5/tools/__init__.md

from __future__ import annotations

import sys

# Ensure Ikaros-memory/ is importable (so `import v5` works from anywhere).
# Reuse the single canonical root path defined in utils (no duplicate computation).
from v5.tools.utils import V5_ROOT
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from v5.tools import care_tool
from v5.tools import emotion_tool
from v5.tools import extra_tool
from v5.tools import memory_tool
from v5.tools import relationship_tool
from v5.tools import self_tool
from v5.tools import vitality_tool
# V5.2: neko migration tools
from v5.tools import reflection_tool
from v5.tools import repeat_tool
from v5.tools import directive_tool

# Collect every v5_* callable from the submodules into __all__.
__all__: list[str] = []
_SEEN = set()
for _mod in (
    emotion_tool, memory_tool, self_tool,
    care_tool, vitality_tool, relationship_tool, extra_tool,
    reflection_tool, repeat_tool, directive_tool,
):
    for _name in dir(_mod):
        if _name.startswith("v5_") and _name not in _SEEN:
            _fn = getattr(_mod, _name)
            if callable(_fn):
                globals()[_name] = _fn
                __all__.append(_name)
                _SEEN.add(_name)

__all__.sort()

# 内联说明见 docs/scripts/core/v5/v5/tools/__init__.md（见“内联注释摘录”）
if "__all__" not in __all__:
    __all__.append("__all__")
