"""V5.2: User directives module — migrated from neko UserDirectivesManager.

Stores explicit user-issued directives (banned topics, behavioral preferences, etc.)
with TTL-based expiry. Integrated into system prompt building.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

logger = logging.getLogger("ikaros.v5.user_directives")

# ─── Configuration ──────────────────────────────────────────
DEFAULT_TTL_SECONDS = 3 * 86400    # 3 days (matching neko default)
MAX_DIRECTIVES_PER_CHAR = 50


def add_directive(character: str, directive_text: str,
                  directive_type: str = "ban_topic",
                  ttl_seconds: float = DEFAULT_TTL_SECONDS) -> int:
    """Add a user directive.

    Args:
        character: Character/role name
        directive_text: The directive content
        directive_type: Type of directive (ban_topic, preference, behavior_rule, etc.)
        ttl_seconds: Time-to-live in seconds (0 = never expires)

    Returns:
        The new directive ID, or 0 on failure
    """
    from v5.store import conn
    try:
        expires_at = (time.time() + ttl_seconds) if ttl_seconds > 0 else 0
        with conn() as c:
            # Prune oldest over limit
            c.execute(
                "DELETE FROM user_directives WHERE character = ? "
                "AND id NOT IN (SELECT id FROM user_directives WHERE character = ? "
                "  ORDER BY created_at DESC LIMIT ?)",
                (character, character, MAX_DIRECTIVES_PER_CHAR),
            )
            cur = c.execute(
                "INSERT INTO user_directives (character, directive_text, directive_type, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (character, directive_text, directive_type, expires_at),
            )
            c.commit()
            did = cur.lastrowid
            # Record event
            from v5.store import _record_event_best_effort
            _record_event_best_effort(did, directive_text, "user_directive", character,
                                      "user_directive.created")
            return did
    except Exception as e:
        logger.warning("user_directives.add failed: %s", e)
        return 0


def get_active_directives(character: str, directive_type: str = '') -> list[dict]:
    """Get all active (non-expired) directives for a character.

    Args:
        character: Character/role name
        directive_type: Optional filter by type (empty = all types)

    Returns:
        List of active directive dicts, oldest first
    """
    from v5.store import conn
    now = time.time()
    try:
        with conn() as c:
            if directive_type:
                rows = c.execute(
                    "SELECT * FROM user_directives "
                    "WHERE character = ? AND is_active = 1 AND directive_type = ? "
                    "AND (expires_at = 0 OR expires_at > ?) "
                    "ORDER BY created_at ASC",
                    (character, directive_type, now),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM user_directives "
                    "WHERE character = ? AND is_active = 1 "
                    "AND (expires_at = 0 OR expires_at > ?) "
                    "ORDER BY created_at ASC",
                    (character, now),
                ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("user_directives.get_active failed: %s", e)
        return []


def expire_old(character: str = '') -> int:
    """Mark expired directives as inactive.

    Args:
        character: If empty, expire for all characters

    Returns:
        Number of directives expired
    """
    from v5.store import conn
    now = time.time()
    try:
        with conn() as c:
            if character:
                cur = c.execute(
                    "UPDATE user_directives SET is_active = 0 "
                    "WHERE character = ? AND expires_at > 0 AND expires_at <= ?",
                    (character, now),
                )
            else:
                cur = c.execute(
                    "UPDATE user_directives SET is_active = 0 "
                    "WHERE expires_at > 0 AND expires_at <= ?",
                    (now,),
                )
            c.commit()
            return cur.rowcount
    except Exception as e:
        logger.warning("user_directives.expire_old failed: %s", e)
        return 0


def deactivate(directive_id: int) -> bool:
    """Deactivate a specific directive by ID."""
    from v5.store import conn
    try:
        with conn() as c:
            c.execute(
                "UPDATE user_directives SET is_active = 0 WHERE id = ?",
                (directive_id,),
            )
            c.commit()
            return True
    except Exception as e:
        logger.warning("user_directives.deactivate failed: %s", e)
        return False


def build_directives_block(character: str) -> str:
    """Build a directives text block for system prompt injection.

    Returns empty string if no active directives exist.
    """
    directives = get_active_directives(character)
    if not directives:
        return ""
    lines = ["[用户指令]"]
    for d in directives:
        lines.append(f"- {d['directive_text']} ({d['directive_type']})")
    return "\n".join(lines)


def stats(character: str = '') -> dict:
    """Directives statistics."""
    from v5.store import conn
    try:
        with conn() as c:
            if character:
                total = c.execute(
                    "SELECT COUNT(*) FROM user_directives WHERE character = ?",
                    (character,),
                ).fetchone()[0]
                active = c.execute(
                    "SELECT COUNT(*) FROM user_directives WHERE character = ? AND is_active = 1",
                    (character,),
                ).fetchone()[0]
            else:
                total = c.execute("SELECT COUNT(*) FROM user_directives").fetchone()[0]
                active = c.execute(
                    "SELECT COUNT(*) FROM user_directives WHERE is_active = 1"
                ).fetchone()[0]
            return {"total": total, "active": active, "character": character or "all"}
    except Exception as e:
        return {"error": str(e)}
