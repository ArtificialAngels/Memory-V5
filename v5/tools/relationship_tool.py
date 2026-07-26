# 详细说明见 docs/scripts/core/v5/v5/tools/relationship_tool.md

from __future__ import annotations

from v5.tools.utils import safe_tool, dumps


@safe_tool
def v5_relationship() -> str:
    """Return Ikaros's relationship closeness with哥哥."""
    from v5.relationship import Relationship
    r = Relationship.load()
    return dumps({
        "depth": round(r.depth, 4),
        "warmth": round(r.warmth, 4),
        "stage": r.stage(),
        "closeness": round(r.closeness(), 4),
        "days_known": round(r.days_known(), 1),
        "interaction_count": r.interaction_count,
    })


@safe_tool
def v5_relationship_tick(intensity: float = 0.3) -> str:
    """Record one interaction (intensity 0..1) and return the updated state."""
    from v5.relationship import Relationship
    r = Relationship.load()
    r = r.record_interaction(float(intensity))
    r.save()
    return dumps({
        "depth": round(r.depth, 4),
        "warmth": round(r.warmth, 4),
        "stage": r.stage(),
        "closeness": round(r.closeness(), 4),
    })
