"""V5.2: Reflections module — migrated from neko ReflectionEngine.

Manages the reflection lifecycle:
  pending → confirmed → promoted → merged (into self_model)
  pending → denied
  pending → archived

Uses evidence scoring (reinforcement/disputation with half-life decay)
to drive status transitions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("ikaros.v5.reflections")

# ─── Enums ──────────────────────────────────────────────────

class ReflectionStatus(str, Enum):
    PENDING = "pending"           # Freshly synthesized, awaiting evidence
    CONFIRMED = "confirmed"       # Evidence score above threshold
    PROMOTED = "promoted"         # Promoted to near-persona level
    MERGED = "merged"             # Absorbed into self_model
    DENIED = "denied"             # Disputed/rebutted by user
    ARCHIVED = "archived"         # Archived (no longer relevant)
    PROMOTE_BLOCKED = "promote_blocked"  # Dead letter state


class RelationType(str, Enum):
    PREFERENCE = "preference"
    HABIT = "habit"
    IDENTITY = "identity"
    OPINION = "opinion"
    EXPERIENCE = "experience"
    RELATIONSHIP_DYNAMIC = "relationship_dynamic"


class TemporalScope(str, Enum):
    PATTERN = "pattern"    # Recurring behavior/pattern
    STATE = "state"        # Current state
    EPISODE = "episode"    # Single event/episode
    PAST = "past"          # Past event (no longer current)


class Entity(str, Enum):
    MASTER = "master"       # Human user
    NEKO = "neko"           # AI character
    RELATIONSHIP = "relationship"  # Relationship dynamic


# ─── Data class ─────────────────────────────────────────────

@dataclass
class Reflection:
    """A reflection entry with evidence-driven lifecycle."""
    id: str
    character: str
    content: str
    entity: str = Entity.MASTER.value
    relation_type: str = RelationType.EXPERIENCE.value
    temporal_scope: str = TemporalScope.EPISODE.value
    status: ReflectionStatus = ReflectionStatus.PENDING
    importance: int = 5
    source_fact_ids: list[str] = field(default_factory=list)
    reinforcement: float = 0.0
    disputation: float = 0.0
    event_start_at: float | None = None
    event_end_at: float | None = None
    created_at: float = field(default_factory=time.time)
    confirmed_at: float | None = None
    promoted_at: float | None = None
    merged_into: str | None = None
    sub_zero_days: int = 0
    evidence_version: int = 0


# ─── Constants ──────────────────────────────────────────────

EVIDENCE_REIN_HALF_LIFE_DAYS = 14.0
EVIDENCE_DISP_HALF_LIFE_DAYS = 7.0
EVIDENCE_CONFIRMED_THRESHOLD = 10.0   # Sum of (reinforcement * importance)
EVIDENCE_PROMOTED_THRESHOLD = 50.0
EVIDENCE_ARCHIVE_THRESHOLD = -5.0
RETENTION_DAYS = 30  # Max days before archiving stale reflections


def _halflife_decay(value: float, elapsed_days: float, half_life_days: float) -> float:
    if elapsed_days <= 0 or half_life_days <= 0:
        return value
    return value * (0.5 ** (elapsed_days / half_life_days))


def _make_id(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_db() -> tuple:
    """Get connection helper."""
    from v5.store import conn
    return conn()


# ─── CRUD ───────────────────────────────────────────────────

def read(character: str, status: str | None = None,
         limit: int = 100, entity: str = '',
         min_importance: int = 0) -> list[Reflection]:
    """Read reflections from database."""
    from v5.store import conn
    parts = ["SELECT * FROM reflections WHERE character = ?"]
    params: list = [character]
    if status:
        parts.append("AND status = ?")
        params.append(status)
    if entity:
        parts.append("AND entity = ?")
        params.append(entity)
    if min_importance > 0:
        parts.append("AND importance >= ?")
        params.append(min_importance)
    parts.append("ORDER BY importance DESC, created_at DESC LIMIT ?")
    params.append(limit)
    try:
        with conn() as c:
            rows = c.execute(" ".join(parts), params).fetchall()
        result = []
        for r in rows:
            result.append(Reflection(
                id=r["id"], character=r["character"],
                content=r["content"], entity=r["entity"],
                relation_type=r["relation_type"],
                temporal_scope=r["temporal_scope"],
                status=ReflectionStatus(r["status"]),
                importance=r["importance"],
                source_fact_ids=json.loads(r["source_fact_ids"] or "[]"),
                reinforcement=r["reinforcement"],
                disputation=r["disputation"],
                event_start_at=r["event_start_at"],
                event_end_at=r["event_end_at"],
                created_at=r["created_at"],
                confirmed_at=r["confirmed_at"],
                promoted_at=r["promoted_at"],
                merged_into=r["merged_into"],
                sub_zero_days=r["sub_zero_days"],
                evidence_version=r["evidence_version"],
            ))
        return result
    except Exception as e:
        logger.warning("reflections.read failed: %s", e)
        return []


def synthesize(character: str, content: str, source_fact_ids: list[str] | None = None,
               entity: str = Entity.MASTER.value,
               relation_type: str = RelationType.EXPERIENCE.value,
               temporal_scope: str = TemporalScope.EPISODE.value,
               importance: int = 5,
               initial_reinforcement: float = 0.5) -> str:
    """Synthesize a new reflection from facts.

    Args:
        character: Character/role
        content: Reflection text
        source_fact_ids: IDs of source facts
        entity: master/neko/relationship
        relation_type: Type of reflection
        temporal_scope: Time scope
        importance: 1-10
        initial_reinforcement: Initial evidence reinforcement value

    Returns:
        The reflection ID, or '' on failure
    """
    from v5.store import conn
    rid = _make_id(content)
    try:
        with conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO reflections "
                "(id, character, content, entity, relation_type, temporal_scope, "
                "status, importance, source_fact_ids, reinforcement, evidence_version) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, 1)",
                (rid, character, content, entity, relation_type, temporal_scope,
                 importance, json.dumps(source_fact_ids or []), initial_reinforcement),
            )
            c.commit()
        from v5.store import _record_event_best_effort
        _record_event_best_effort(rid, content, "reflection", character,
                                  "reflection.synthesized")
        logger.info("reflections.synthesize: id=%s char=%s imp=%d",
                     rid[:8], character, importance)
        return rid
    except Exception as e:
        logger.warning("reflections.synthesize failed: %s", e)
        return ""


def apply_evidence(character: str, reflection_id: str,
                   delta_rein: float = 0.0, delta_disp: float = 0.0) -> bool:
    """Apply evidence signals to a reflection (reinforcement/disputation).

    After applying, recomputes evidence score and may trigger status transition.
    """
    from v5.store import conn
    try:
        with conn() as c:
            row = c.execute(
                "SELECT * FROM reflections WHERE id = ? AND character = ?",
                (reflection_id, character),
            ).fetchone()
            if not row:
                return False
            new_rein = float(row["reinforcement"]) + delta_rein
            new_disp = float(row["disputation"]) + delta_disp
            new_version = int(row["evidence_version"]) + 1

            # Weighted evidence score
            importance = int(row["importance"])
            weighted = (new_rein - new_disp) * importance
            now = time.time()
            elapsed_days = (now - float(row["created_at"])) / 86400.0
            decayed = _halflife_decay(
                new_rein - new_disp, elapsed_days, EVIDENCE_REIN_HALF_LIFE_DAYS
            ) * importance

            # Status transition logic
            current_status = row["status"]
            new_status = current_status
            sub_zero = int(row["sub_zero_days"])

            if current_status == ReflectionStatus.PENDING.value:
                if decayed >= EVIDENCE_PROMOTED_THRESHOLD:
                    new_status = ReflectionStatus.PROMOTED.value
                elif decayed >= EVIDENCE_CONFIRMED_THRESHOLD:
                    new_status = ReflectionStatus.CONFIRMED.value
                elif decayed < 0:
                    sub_zero += 1
                    if sub_zero >= 3:  # 3 consecutive checks below zero
                        new_status = ReflectionStatus.DENIED.value
                else:
                    sub_zero = 0

            elif current_status == ReflectionStatus.CONFIRMED.value:
                if decayed >= EVIDENCE_PROMOTED_THRESHOLD:
                    new_status = ReflectionStatus.PROMOTED.value
                elif decayed < -EVIDENCE_ARCHIVE_THRESHOLD:
                    new_status = ReflectionStatus.ARCHIVED.value

            elif current_status == ReflectionStatus.PROMOTED.value:
                if decayed < -EVIDENCE_ARCHIVE_THRESHOLD * 2:
                    new_status = ReflectionStatus.ARCHIVED.value

            c.execute(
                "UPDATE reflections SET reinforcement = ?, disputation = ?, "
                "evidence_version = ?, status = ?, sub_zero_days = ?"
                + (", confirmed_at = ?" if new_status == ReflectionStatus.CONFIRMED.value else "") +
                (", promoted_at = ?" if new_status == ReflectionStatus.PROMOTED.value else "") +
                " WHERE id = ? AND character = ?",
                ([new_rein, new_disp, new_version, new_status, sub_zero] +
                 ([now] if new_status == ReflectionStatus.CONFIRMED.value else []) +
                 ([now] if new_status == ReflectionStatus.PROMOTED.value else []) +
                 [reflection_id, character]),
            )
            c.commit()

            if new_status != current_status:
                logger.info("reflections.evidence: %s %s → %s (score=%.2f)",
                             reflection_id[:8], current_status, new_status, decayed)

        from v5.store import _record_event_best_effort
        _record_event_best_effort(reflection_id, "", "reflection", character,
                                  "reflection.status_changed")
        return True
    except Exception as e:
        logger.warning("reflections.apply_evidence failed: %s", e)
        return False


def promote_to_persona(reflection_id: str, character: str, merge_target: str = "") -> bool:
    """Promote a reflection by merging it into the character's persona (self_model).

    Args:
        reflection_id: Reflection to promote
        character: Character/role
        merge_target: If empty, auto-generate from reflection content

    Returns:
        True on success
    """
    reflections = read(character, limit=100)  # 100 to ensure we can find the target
    target = next((r for r in reflections if r.id == reflection_id), None)
    if not target:
        return False

    from v5 import self_model as sm
    try:
        model = sm.SelfModel.load()
        narrative_key = f"self_narrative.{character}" if character else "self_narrative"
        current = model.data.get(narrative_key, "")
        entry = f"[{target.relation_type}] {target.content}"
        if current:
            model.data[narrative_key] = current + "\n" + entry
        else:
            model.data[narrative_key] = entry
        model.save()

        from v5.store import conn
        with conn() as c:
            c.execute(
                "UPDATE reflections SET status = 'merged', merged_into = ? "
                "WHERE id = ? AND character = ?",
                (merge_target or narrative_key, reflection_id, character),
            )
            c.commit()

        from v5.store import _record_event_best_effort
        _record_event_best_effort(reflection_id, entry, "reflection", character,
                                  "reflection.promoted")
        logger.info("reflections.promote: id=%s → %s", reflection_id[:8], narrative_key)
        return True
    except Exception as e:
        logger.warning("reflections.promote_to_persona failed: %s", e)
        return False


def auto_promote_stale(character: str, min_evidence_version: int = 3) -> int:
    """Automatically promote high-evidence reflections that have sat long enough.

    Returns number of promoted reflections.
    """
    reflections = read(character, status=ReflectionStatus.CONFIRMED.value)
    now = time.time()
    promoted = 0
    for r in reflections:
        if r.evidence_version < min_evidence_version:
            continue
        if r.created_at and (now - r.created_at) > 3 * 86400:  # 3 days old
            if promote_to_persona(r.id, character):
                promoted += 1
    return promoted


def build_reflections_block(character: str, max_items: int = 5) -> str:
    """Build a reflection context block for system prompt injection.

    Includes only confirmed and promoted reflections.
    """
    confirmed = read(character, status=ReflectionStatus.CONFIRMED.value,
                     limit=max_items)
    promoted = read(character, status=ReflectionStatus.PROMOTED.value,
                    limit=max_items)
    all_refs = sorted(
        confirmed + promoted,
        key=lambda r: (r.importance * (r.reinforcement - r.disputation)),
        reverse=True,
    )[:max_items]
    if not all_refs:
        return ""
    lines = ["[反思 / 自我认识]"]
    for r in all_refs:
        status_mark = "✓" if r.status == ReflectionStatus.CONFIRMED.value else "★"
        lines.append(f"- {status_mark} [{r.relation_type}] {r.content}"
                     f" (重要性:{r.importance})")
    return "\n".join(lines)


def stats(character: str = '') -> dict:
    """Reflection statistics."""
    from v5.store import conn
    try:
        with conn() as c:
            if character:
                total = c.execute(
                    "SELECT COUNT(*) FROM reflections WHERE character = ?",
                    (character,),
                ).fetchone()[0]
                by_status = c.execute(
                    "SELECT status, COUNT(*) FROM reflections "
                    "WHERE character = ? GROUP BY status",
                    (character,),
                ).fetchall()
            else:
                total = c.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
                by_status = c.execute(
                    "SELECT status, COUNT(*) FROM reflections GROUP BY status"
                ).fetchall()
            return {
                "total": total,
                "by_status": {r[0]: r[1] for r in by_status},
                "character": character or "all",
            }
    except Exception as e:
        return {"error": str(e), "character": character or "all"}
