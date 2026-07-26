# 详细说明见 docs/scripts/core/v5/v5/tools/vitality_tool.md

from __future__ import annotations

from v5.tools.utils import safe_tool, dumps


@safe_tool
def v5_vitality() -> str:
    """Return Ikaros's current energy / 精力 state."""
    from v5.vitality import Vitality
    v = Vitality.load().tick()
    v.save()
    return dumps({
        "vitality": round(v.vitality, 4),
        "label": v.label(),
        "emoji": v.to_emoji(),
        "total_uptime_sec": v.total_uptime_sec,
    })


@safe_tool
def v5_vitality_tick(conversation: bool = False) -> str:
    """Advance the energy model one step.

    conversation=True costs more energy (a real dialogue turn).
    """
    from v5.vitality import Vitality
    v = Vitality.load().tick(conversation=bool(conversation))
    v.save()
    return dumps({
        "vitality": round(v.vitality, 4),
        "label": v.label(),
        "conversation": bool(conversation),
    })
