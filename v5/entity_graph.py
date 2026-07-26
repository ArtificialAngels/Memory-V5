# Entity Graph Memory System for Ikaros V5
# Ported from Innerlife-main (MIT License) entity graph architecture.
# Provides: entity extraction, entity resolution, spreading activation search,
# episodic memory consolidation, and graph-based memory retrieval.

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("ikaros.v5.entity_graph")

EG_ROOT = Path(__file__).resolve().parent.parent
EG_DATA_DIR = EG_ROOT / "data" / "v5"
EG_DB_PATH = EG_DATA_DIR / "v5.db"

ENTITY_TYPES = {"person", "place", "object", "event"}
MERGE_CONFIDENCE_THRESHOLD = 0.75
SPREAD_FACTOR = 0.35
ACTIVATION_TTL_MINUTES = 20
EPISODIC_BATCH_SIZE = 3
STAGE_B_CANDIDATE_LIMIT = 5


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

ENTITY_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS eg_entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    embedding_text TEXT NOT NULL DEFAULT '',
    embedding TEXT NOT NULL DEFAULT '[]',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_updated_at INTEGER,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    last_seen_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_eg_entities_type ON eg_entities(type);
CREATE INDEX IF NOT EXISTS idx_eg_entities_name ON eg_entities(canonical_name);

CREATE TABLE IF NOT EXISTS eg_aliases (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_memory_id TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    last_seen_at INTEGER,
    UNIQUE(entity_id, alias)
);

CREATE INDEX IF NOT EXISTS idx_eg_aliases_entity ON eg_aliases(entity_id);

CREATE TABLE IF NOT EXISTS eg_edges (
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    co_occurrence_count INTEGER NOT NULL DEFAULT 0,
    last_seen_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    PRIMARY KEY(source_entity_id, target_entity_id)
);

CREATE TABLE IF NOT EXISTS eg_episodic (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    source_text TEXT NOT NULL DEFAULT '',
    detail TEXT DEFAULT '',
    entity_text TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS eg_episodic_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    PRIMARY KEY(memory_id, entity_id)
);

CREATE TABLE IF NOT EXISTS eg_activations (
    episodic_memory_id TEXT NOT NULL,
    activated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    expires_at INTEGER NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY(episodic_memory_id)
);
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@contextmanager
def eg_conn() -> Iterator[sqlite3.Connection]:
    EG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(EG_DB_PATH))
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA busy_timeout=5000")
        c.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    c.executescript(ENTITY_GRAPH_SCHEMA)
    c.commit()
    try:
        yield c
    finally:
        try:
            c.commit()
        except Exception:
            pass
        try:
            c.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EntityCandidate:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    episodic_count: int = 0
    similarity: float = 0.0


@dataclass
class EpisodicMemory:
    id: str
    summary: str
    source_text: str = ""
    detail: str = ""
    entity_text: str = ""
    importance: float = 0.5
    created_at: int = 0
    graph_score: float = 0.0
    text_score: float = 0.0


@dataclass
class EntityLink:
    local_entity_id: str
    weight: float


@dataclass
class EpisodicDraft:
    summary: str
    detail: str
    importance: float
    entity_links: list[EntityLink] = field(default_factory=list)


@dataclass
class LocalEntity:
    local_id: str
    surface: str
    entity_type: str
    context_hint: str = ""


# ---------------------------------------------------------------------------
# Stage A: Entity Extraction Prompts
# ---------------------------------------------------------------------------

STAGE_A_SYSTEM = """You are an entity extraction system. Extract entities and episodic memories from the given short-term memory batch.

Output STRICT JSON only. No markdown, no explanation. Top-level format: {"entities":[...], "episodic_memories":[...]}

Entity format: {"local_entity_id": "e1", "surface": "exact text", "type": "person|place|object|event", "context_hint": "brief explanation"}

Episodic format: {"summary": "...", "detail": "...", "importance": 0.0-1.0, "entity_links": [{"local_entity_id": "e1", "weight": 0.3-1.0}]}

Weight rules: 0.8-1.0 = core entity (memory makes no sense without it), 0.5-0.8 = important related, 0.3-0.5 = weak background. Links below 0.3 will be filtered.
Only output episodic memories that have entity_links."""


def build_stage_a_prompt(memories: list[str]) -> str:
    items = "\n\n---\n\n".join(
        f"[{i}] {text}" for i, text in enumerate(memories)
    )
    return f"Memories:\n\n{items}"


# ---------------------------------------------------------------------------
# Stage B: Entity Resolution Prompts
# ---------------------------------------------------------------------------

STAGE_B_SYSTEM = """You are an entity resolution system. Decide whether each local entity should merge into an existing entity or be created as new.

Output STRICT JSON array only. No markdown, no explanation.
Format: [{"local_entity_id": "e1", "action": "merge", "entity_id": "<existing>", "confidence": 0.0-1.0, "alias_to_add": null|"string"}, ...]

Rules:
- action is "merge" or "create_new"
- merge requires confidence >= 0.75
- For merge: if local surface differs from canonical_name and all existing aliases, set alias_to_add to local surface; otherwise null
- For create_new: include "canonical_name" and "type"
- Same scene, same category, similar words, related things are NOT aliases unless they genuinely refer to the SAME entity
- Canonical name should be the most natural, complete form"""


def build_stage_b_prompt(local_entities: list[LocalEntity], candidates: list[list[EntityCandidate]]) -> str:
    lines = ["Local entities to resolve:\n"]
    for ent in local_entities:
        lines.append(f"  {ent.local_id}: surface=\"{ent.surface}\", type={ent.entity_type}, hint=\"{ent.context_hint}\"")

    lines.append("\nCandidate existing entities for each local entity:\n")
    for i, (ent, cands) in enumerate(zip(local_entities, candidates)):
        lines.append(f"\nFor {ent.local_id} ({ent.surface}):")
        if not cands:
            lines.append("  (no candidates)")
        for c in cands:
            lines.append(
                f"  - id={c.entity_id} name=\"{c.canonical_name}\" type={c.entity_type} "
                f"aliases={c.aliases} desc=\"{c.description}\" "
                f"sim={c.similarity:.3f} mem_count={c.episodic_count}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity Graph CRUD
# ---------------------------------------------------------------------------

def create_entity(
    entity_id: str, entity_type: str, canonical_name: str,
    description: str = "", confidence: float = 1.0,
    embedding_text: str = "", embedding: str = "[]",
    embedding_model: str = ""
) -> None:
    now = int(time.time())
    with eg_conn() as c:
        c.execute(
            """INSERT OR IGNORE INTO eg_entities
               (id, type, canonical_name, description, confidence,
                embedding_text, embedding, embedding_model, embedding_updated_at,
                created_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entity_id, entity_type, canonical_name, description, confidence,
             embedding_text, embedding, embedding_model, now, now, now)
        )


def add_entity_alias(entity_id: str, alias: str, confidence: float = 1.0,
                     source_memory_id: str | None = None) -> None:
    import uuid
    alias_id = str(uuid.uuid4())
    now = int(time.time())
    with eg_conn() as c:
        c.execute(
            """INSERT OR IGNORE INTO eg_aliases
               (id, entity_id, alias, confidence, source_memory_id, created_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (alias_id, entity_id, alias, confidence, source_memory_id, now, now)
        )


def upsert_entity_edge(source_id: str, target_id: str, delta: float) -> None:
    now = int(time.time())
    with eg_conn() as c:
        existing = c.execute(
            "SELECT weight, co_occurrence_count FROM eg_edges WHERE source_entity_id=? AND target_entity_id=?",
            (source_id, target_id)
        ).fetchone()
        if existing:
            new_weight = min(1.0, existing["weight"] + delta)
            c.execute(
                "UPDATE eg_edges SET weight=?, co_occurrence_count=co_occurrence_count+1, last_seen_at=? "
                "WHERE source_entity_id=? AND target_entity_id=?",
                (new_weight, now, source_id, target_id)
            )
        else:
            c.execute(
                "INSERT INTO eg_edges (source_entity_id, target_entity_id, weight, co_occurrence_count, last_seen_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (source_id, target_id, min(1.0, delta), now)
            )


def create_episodic_memory(
    memory_id: str, summary: str, source_text: str = "",
    detail: str = "", entity_text: str = "", importance: float = 0.5
) -> None:
    now = int(time.time())
    with eg_conn() as c:
        c.execute(
            """INSERT OR IGNORE INTO eg_episodic
               (id, summary, source_text, detail, entity_text, importance, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, summary, source_text, detail, entity_text, importance, now)
        )


def link_episodic_entity(memory_id: str, entity_id: str, weight: float) -> None:
    with eg_conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO eg_episodic_entities (memory_id, entity_id, weight) VALUES (?, ?, ?)",
            (memory_id, entity_id, weight)
        )


def activate_episodic_memory(memory_id: str, score: float,
                              ttl_minutes: int = ACTIVATION_TTL_MINUTES) -> None:
    now = int(time.time())
    expires = now + ttl_minutes * 60
    with eg_conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO eg_activations
               (episodic_memory_id, activated_at, expires_at, score)
               VALUES (?, ?, ?, ?)""",
            (memory_id, now, expires, score)
        )


# ---------------------------------------------------------------------------
# Entity Search & Matching
# ---------------------------------------------------------------------------

def find_entity_candidates(surface: str) -> list[EntityCandidate]:
    """Match a surface string against existing entities by exact/contains match."""
    with eg_conn() as c:
        # Exact match on canonical_name or alias
        exact_rows = c.execute(
            """SELECT DISTINCT e.id, e.canonical_name, e.type, e.description,
               (SELECT COUNT(*) FROM eg_episodic_entities ee WHERE ee.entity_id = e.id) as ep_count
               FROM eg_entities e
               LEFT JOIN eg_aliases a ON a.entity_id = e.id
               WHERE e.canonical_name = ? OR a.alias = ?
               LIMIT 10""",
            (surface, surface)
        ).fetchall()

        if exact_rows:
            return [
                EntityCandidate(
                    entity_id=r["id"], canonical_name=r["canonical_name"],
                    entity_type=r["type"], description=r["description"] or "",
                    episodic_count=r["ep_count"], similarity=1.0
                )
                for r in exact_rows
            ]

        # Contains match
        candidates: dict[str, EntityCandidate] = {}
        for row in c.execute(
            "SELECT id, canonical_name, type, description FROM eg_entities"
        ).fetchall():
            name = row["canonical_name"]
            score = 0.0
            if surface in name:
                score = len(surface) / len(name)
            elif name in surface:
                score = len(name) / len(surface)

            if score > 0.0:
                candidates[row["id"]] = EntityCandidate(
                    entity_id=row["id"], canonical_name=name,
                    entity_type=row["type"], description=row["description"] or "",
                    similarity=score
                )

        return sorted(candidates.values(), key=lambda x: x.similarity, reverse=True)[:5]


def get_entity_aliases(entity_id: str) -> list[str]:
    with eg_conn() as c:
        rows = c.execute(
            "SELECT alias FROM eg_aliases WHERE entity_id = ?", (entity_id,)
        ).fetchall()
    return [r["alias"] for r in rows]


# ---------------------------------------------------------------------------
# Spreading Activation Search
# ---------------------------------------------------------------------------

def spreading_activation_search(
    seed_entities: list[tuple[str, float]],
    top_k: int = 10
) -> list[EpisodicMemory]:
    """
    Spreading activation from seed entities through the entity graph.
    seed_entities: list of (entity_id, activation_score) pairs.
    Returns episodic memories ranked by combined graph + importance score.
    """
    if not seed_entities:
        return []

    activation: dict[str, float] = {}
    processed: set[str] = set()

    # Initialize activation from seeds
    for eid, score in seed_entities:
        activation[eid] = min(1.0, score)

    # Spread one hop
    with eg_conn() as c:
        seed_ids = [eid for eid, _ in seed_entities]
        placeholders = ",".join("?" * len(seed_ids))

        # Get outgoing edges from seed entities
        edges = c.execute(
            f"""SELECT source_entity_id, target_entity_id, weight
                FROM eg_edges
                WHERE source_entity_id IN ({placeholders})""",
            seed_ids
        ).fetchall()

        for edge in edges:
            src = edge["source_entity_id"]
            tgt = edge["target_entity_id"]
            spread = activation.get(src, 0.0) * edge["weight"] * SPREAD_FACTOR
            if spread > 0.01:
                current = activation.get(tgt, 0.0)
                activation[tgt] = min(1.0, current + spread)

        # Get all episodic memories linked to activated entities
        active_ids = list(activation.keys())
        if not active_ids:
            return []

        active_placeholders = ",".join("?" * len(active_ids))

        mem_rows = c.execute(
            f"""SELECT em.id, em.summary, em.source_text, em.detail,
                       em.entity_text, em.importance, em.created_at,
                       ee.entity_id, ee.weight as link_weight
                FROM eg_episodic em
                JOIN eg_episodic_entities ee ON ee.memory_id = em.id
                WHERE ee.entity_id IN ({active_placeholders})
                ORDER BY em.importance DESC, em.created_at DESC""",
            active_ids
        ).fetchall()

    # Score memories
    memory_scores: dict[str, tuple[EpisodicMemory, float]] = {}
    for row in mem_rows:
        mid = row["id"]
        link_entity = row["entity_id"]
        link_weight = row["link_weight"]
        act_score = activation.get(link_entity, 0.0)

        if mid not in memory_scores:
            mem = EpisodicMemory(
                id=mid, summary=row["summary"], source_text=row["source_text"] or "",
                detail=row["detail"] or "", entity_text=row["entity_text"] or "",
                importance=row["importance"], created_at=row["created_at"]
            )
            memory_scores[mid] = (mem, 0.0)

        mem, current_score = memory_scores[mid]
        contrib = act_score * link_weight
        memory_scores[mid] = (mem, current_score + contrib)

    # Final scoring with importance bonus and entity diversity bonus
    results: list[EpisodicMemory] = []
    for mem, raw_score in memory_scores.values():
        # Count unique entities linked to this memory
        with eg_conn() as c2:
            entity_count = c2.execute(
                "SELECT COUNT(DISTINCT entity_id) FROM eg_episodic_entities WHERE memory_id = ?",
                (mem.id,)
            ).fetchone()[0]

        mem.graph_score = raw_score
        mem.text_score = 0.0
        # Score = graph contribution + importance factor + entity diversity bonus
        final_score = raw_score + 0.15 * mem.importance + 0.1 * max(0, entity_count - 1)
        results.append((mem, final_score))

    results.sort(key=lambda x: x[1], reverse=True)

    # Activate top results
    top_results = results[:top_k]
    for mem, score in top_results:
        activate_episodic_memory(mem.id, score)

    return [mem for mem, _ in top_results]


# ---------------------------------------------------------------------------
# Episodic Consolidation
# ---------------------------------------------------------------------------

def run_episodic_consolidation(
    stm_batch: list[str],
    llm_call: Any
) -> dict:
    """
    Run Stage A + Stage B episodic consolidation on a batch of short-term memories.
    llm_call: function(text, system_prompt) -> str (LLM response text).
    Returns stats dict with entity/edge/episodic counts.
    """
    stats = {"entities_created": 0, "entities_merged": 0, "aliases_added": 0,
             "edges_updated": 0, "episodic_created": 0}

    if not stm_batch:
        return stats

    # Stage A: Extract entities and episodic memories
    prompt = build_stage_a_prompt(stm_batch)
    logger.debug("entity_graph: Stage A LLM call batch_size=%d prompt_len=%d", len(stm_batch), len(prompt))
    try:
        response = llm_call(prompt, system=STAGE_A_SYSTEM)
        parsed = json.loads(response)
    except Exception as exc:
        logger.warning("entity_graph: Stage A failed: %s", exc)
        return stats

    entities_raw = parsed.get("entities", [])
    episodic_raw = parsed.get("episodic_memories", [])

    # Filter episodic memories that have entity links
    valid_episodic = [
        e for e in episodic_raw
        if e.get("entity_links") and len(e["entity_links"]) > 0
    ]

    if not valid_episodic:
        return stats

    local_entities: dict[str, LocalEntity] = {}
    for e in entities_raw:
        lid = e.get("local_entity_id", "")
        if lid:
            local_entities[lid] = LocalEntity(
                local_id=lid,
                surface=e.get("surface", ""),
                entity_type=e.get("type", "object"),
                context_hint=e.get("context_hint", "")
            )

    # Stage B: Entity resolution
    referenced_ids: set[str] = set()
    for ep in valid_episodic:
        for link in ep.get("entity_links", []):
            referenced_ids.add(link.get("local_entity_id", ""))

    local_list = [local_entities[lid] for lid in referenced_ids if lid in local_entities]
    if not local_list:
        return stats

    # Build candidates for each local entity
    candidates_per_entity: list[list[EntityCandidate]] = []
    for ent in local_list:
        candidates = find_entity_candidates(ent.surface)
        candidates_per_entity.append(candidates[:STAGE_B_CANDIDATE_LIMIT])

    # Stage B LLM call
    stage_b_prompt = build_stage_b_prompt(local_list, candidates_per_entity)
    logger.debug("entity_graph: Stage B LLM call entities=%d prompt_len=%d", len(local_list), len(stage_b_prompt))
    try:
        response_b = llm_call(stage_b_prompt, system=STAGE_B_SYSTEM)
        resolutions = json.loads(response_b)
    except Exception as exc:
        logger.warning("entity_graph: Stage B failed: %s", exc)
        # Fallback: create all as new
        resolutions = []
        for ent in local_list:
            resolutions.append({
                "local_entity_id": ent.local_id,
                "action": "create_new",
                "canonical_name": ent.surface,
                "type": ent.entity_type,
                "confidence": 1.0
            })

    # Apply resolutions
    import uuid
    entity_id_map: dict[str, str] = {}  # local_id -> real entity_id

    for res in resolutions:
        lid = res.get("local_entity_id", "")
        if lid not in local_entities:
            continue
        ent = local_entities[lid]

        if res.get("action") == "merge" and res.get("confidence", 0) >= MERGE_CONFIDENCE_THRESHOLD:
            target_id = res.get("entity_id", "")
            if target_id:
                entity_id_map[lid] = target_id
                stats["entities_merged"] += 1
                # Add alias if surface differs
                alias_to_add = res.get("alias_to_add")
                if alias_to_add:
                    add_entity_alias(target_id, alias_to_add)
                    stats["aliases_added"] += 1
                continue

        # create_new
        eid = str(uuid.uuid4())
        canonical = res.get("canonical_name", ent.surface)
        create_entity(eid, ent.entity_type, canonical, description=ent.context_hint)
        entity_id_map[lid] = eid
        stats["entities_created"] += 1

    # Create episodic memories
    for ep in valid_episodic:
        mid = str(uuid.uuid4())
        importance = max(0.0, min(1.0, ep.get("importance", 0.5)))
        summary = ep.get("summary", "")
        detail = ep.get("detail", "")

        entity_ids_in_memory: list[str] = []
        for link in ep.get("entity_links", []):
            lid = link.get("local_entity_id", "")
            real_id = entity_id_map.get(lid)
            if real_id:
                entity_ids_in_memory.append(real_id)
                link_episodic_entity(mid, real_id, max(0.3, min(1.0, link.get("weight", 0.5))))

        if entity_ids_in_memory:
            create_episodic_memory(
                mid, summary,
                source_text="\n---\n".join(stm_batch),
                detail=detail,
                entity_text=", ".join(entity_ids_in_memory),
                importance=importance
            )
            stats["episodic_created"] += 1

            # Update edges for co-occurring entities
            for i in range(len(entity_ids_in_memory)):
                for j in range(i + 1, len(entity_ids_in_memory)):
                    left_w = 1.0
                    right_w = 1.0
                    delta = 0.1 * min(left_w, right_w) * importance
                    upsert_entity_edge(entity_ids_in_memory[i], entity_ids_in_memory[j], delta)
                    stats["edges_updated"] += 1

    logger.info("entity_graph: consolidation done entities_created=%d entities_merged=%d aliases_added=%d edges_updated=%d episodic_created=%d",
                stats["entities_created"], stats["entities_merged"], stats["aliases_added"],
                stats["edges_updated"], stats["episodic_created"])
    return stats


# ---------------------------------------------------------------------------
# Graph stats
# ---------------------------------------------------------------------------

def entity_graph_stats() -> dict:
    with eg_conn() as c:
        entities = c.execute("SELECT COUNT(*) FROM eg_entities").fetchone()[0]
        aliases = c.execute("SELECT COUNT(*) FROM eg_aliases").fetchone()[0]
        edges = c.execute("SELECT COUNT(*) FROM eg_edges").fetchone()[0]
        episodic = c.execute("SELECT COUNT(*) FROM eg_episodic").fetchone()[0]
        active = c.execute(
            "SELECT COUNT(*) FROM eg_activations WHERE expires_at > ?",
            (int(time.time()),)
        ).fetchone()[0]
    return {
        "entities": entities, "aliases": aliases, "edges": edges,
        "episodic_memories": episodic, "active_memories": active
    }
