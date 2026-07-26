"""V5.2: User directive tools for MCP exposure."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("ikaros.v5.tools.directive_tool")


def v5_directive_add(character: str, directive_text: str,
                      directive_type: str = "ban_topic",
                      ttl_hours: float = 72.0) -> str:
    """Add a user directive (banned topic, preference, etc.).

    Args:
        character: Character/role name
        directive_text: The directive content
        directive_type: ban_topic / preference / behavior_rule / etc.
        ttl_hours: Time-to-live in hours (0 = never expires)

    Returns:
        Directive ID or "failed"
    """
    from v5 import user_directives
    ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else 0
    did = user_directives.add_directive(
        character=character,
        directive_text=directive_text,
        directive_type=directive_type,
        ttl_seconds=ttl_seconds,
    )
    return str(did) if did else "failed"


def v5_directive_list(character: str, directive_type: str = "") -> str:
    """List active directives.

    Args:
        character: Character/role name
        directive_type: Optional type filter

    Returns:
        JSON array of directive dicts
    """
    from v5 import user_directives
    directives = user_directives.get_active_directives(character, directive_type)
    return json.dumps([{
        "id": d.get("id"),
        "directive_text": d.get("directive_text"),
        "directive_type": d.get("directive_type"),
        "created_at": d.get("created_at"),
        "expires_at": d.get("expires_at"),
    } for d in directives], ensure_ascii=False)


def v5_directive_deactivate(directive_id: int) -> str:
    """Deactivate a directive by ID.

    Args:
        directive_id: Directive ID

    Returns:
        "ok" or "failed"
    """
    from v5 import user_directives
    ok = user_directives.deactivate(directive_id)
    return "ok" if ok else "failed"


def v5_directive_stats(character: str = "") -> str:
    """Get directive statistics.

    Returns:
        JSON stats dict
    """
    from v5 import user_directives
    return json.dumps(user_directives.stats(character))
