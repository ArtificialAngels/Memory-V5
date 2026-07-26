"""V5.2: Reflection tools for MCP exposure."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("ikaros.v5.tools.reflection_tool")


def v5_reflection_synthesize(content: str, character: str = "",
                              source_fact_ids: str = "",
                              entity: str = "master",
                              relation_type: str = "experience",
                              importance: int = 5) -> str:
    """Synthesize a new reflection from facts.

    Args:
        content: Reflection text content
        character: Character/role name (empty for default)
        source_fact_ids: JSON list of source fact IDs
        entity: master/neko/relationship
        relation_type: preference/habit/identity/opinion/experience/relationship_dynamic
        importance: Importance 1-10

    Returns:
        Reflection ID string
    """
    from v5.reflections import synthesize
    try:
        sids = json.loads(source_fact_ids) if source_fact_ids else None
    except (json.JSONDecodeError, TypeError):
        sids = None
    rid = synthesize(
        character=character,
        content=content,
        source_fact_ids=sids,
        entity=entity,
        relation_type=relation_type,
        importance=importance,
    )
    return rid or "failed"


def v5_reflection_read(character: str = "", status: str = "",
                        limit: int = 10, entity: str = "") -> str:
    """Read reflections from database.

    Args:
        character: Character/role name (empty for all)
        status: Filter by status (pending/confirmed/promoted/merged/denied/archived)
        limit: Max items to return
        entity: master/neko/relationship

    Returns:
        JSON array of reflection objects
    """
    from v5.reflections import read
    refs = read(character=character, status=status or None,
                limit=limit, entity=entity)
    return json.dumps([{
        "id": r.id,
        "content": r.content,
        "entity": r.entity,
        "relation_type": r.relation_type,
        "status": r.status.value,
        "importance": r.importance,
        "reinforcement": r.reinforcement,
        "disputation": r.disputation,
        "created_at": r.created_at,
    } for r in refs], ensure_ascii=False)


def v5_reflection_apply_evidence(reflection_id: str, character: str = "",
                                  delta_rein: float = 0.0,
                                  delta_disp: float = 0.0) -> str:
    """Apply evidence signals (reinforcement/disputation) to a reflection.

    Args:
        reflection_id: Reflection ID
        character: Character/role
        delta_rein: Reinforcement delta
        delta_disp: Disputation delta

    Returns:
        "ok" or error message
    """
    from v5.reflections import apply_evidence
    ok = apply_evidence(character, reflection_id, delta_rein, delta_disp)
    return "ok" if ok else "failed"


def v5_reflection_promote(reflection_id: str, character: str = "") -> str:
    """Promote a reflection to persona (self_model).

    Args:
        reflection_id: Reflection ID
        character: Character/role

    Returns:
        "ok" or error message
    """
    from v5.reflections import promote_to_persona
    ok = promote_to_persona(reflection_id, character)
    return "ok" if ok else "failed"


def v5_reflection_stats(character: str = "") -> str:
    """Get reflection statistics.

    Returns:
        JSON stats dict
    """
    from v5.reflections import stats
    return json.dumps(stats(character), ensure_ascii=False)
