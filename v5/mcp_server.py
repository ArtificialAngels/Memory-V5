#!/usr/bin/env python3
# See docs/scripts/core/v5/v5/mcp_server.md

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure Ikaros-memory/ is on the Python path
_HERE = Path(__file__).resolve().parent  # Ikaros-memory/v5/
_V5_ROOT = _HERE.parent                  # Ikaros-memory/
if str(_V5_ROOT) not in sys.path:
    sys.path.insert(0, str(_V5_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [v5-mcp] %(message)s",
)
logger = logging.getLogger("ikaros.v5.mcp")


# ── MCP Server ─────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Ikaros V5 Memory",
    instructions=(
        "Ikaros V5 Memory System provides long-term memory storage, entity graph retrieval, "
        "emotion tracking, self-reflection, curiosity-driven introspection, care scoring, "
        "vitality management, relationship modeling, narrative generation, dissonance "
        "detection, and proactive self-improvement.\n"
        "\n"
        "Memory tools: memory_store saves facts, preferences, lessons, and emotional events "
        "with PAD emotion fingerprints. memory_search performs dual-path retrieval combining "
        "full-text keyword and semantic vector search. memory_get fetches a single memory by ID. "
        "memory_delete removes a memory by ID. memory_stats returns total count, long-term count, "
        "average weight, and per-type breakdown.\n"
        "\n"
        "Emotion tools: analyze_emotion processes text through the PAD model to extract "
        "pleasure, arousal, and dominance scores, producing a raw dict and a human-readable "
        "mood label. emotion_status returns the current emotional state. emotion_label maps "
        "PAD values to mood labels.\n"
        "\n"
        "Self tools: self_model returns the full self-model JSON including curiosity, vitality, "
        "relationship, and care state. self_reflect triggers a deep self-reflection cycle. "
        "latest_thought returns what Ikaros is currently thinking. curiosity_check returns "
        "curiosity level and whether it crosses the reflection threshold. subconscious returns "
        "subconscious activity state. self_discover generates new self-knowledge by introspecting "
        "on stored memories. reflect_run_op is the background reflection operator.\n"
        "\n"
        "Care tools: care_check evaluates a conversation message for care and concern signals. "
        "care_status returns current care state and trend history.\n"
        "\n"
        "Vitality tools: vitality returns the current vitality meter and decay rate. "
        "vitality_tick advances vitality decay by one increment.\n"
        "\n"
        "Relationship tools: relationship returns the relationship model with brother "
        "including trust, familiarity, and mood trends. relationship_tick advances the "
        "relationship decay cycle.\n"
        "\n"
        "Narrative and proactive tools: narrative_generate synthesizes recent memories "
        "into a coherent daily narrative. dissonance_check detects logical contradictions "
        "among stored memories. proactive_check evaluates whether Ikaros should initiate "
        "conversation proactively.\n"
        "\n"
        "V5.2 Reflection tools: reflection_synthesize creates new reflection entries from facts. "
        "reflection_read queries the reflection database by status/entity. "
        "reflection_apply_evidence applies reinforcement or disputation signals to a reflection, "
        "triggering automatic status transitions (pending→confirmed→promoted→merged). "
        "reflection_promote merges a reflection into the character's self-model persona. "
        "reflection_stats returns aggregate reflection counts by status.\n"
        "\n"
        "V5.2 Anti-repeat tools: anti_repeat_record logs a response's n-grams into the "
        "anti-repetition corpus. anti_repeat_check evaluates a candidate text for repetition "
        "risk using BM25-style scoring. anti_repeat_penalty returns a system prompt hint if "
        "repetition risk exceeds threshold.\n"
        "\n"
        "V5.2 Directive tools: directive_add creates a user directive (banned topic, preference, "
        "behavior rule) with configurable TTL. directive_list returns active directives for a "
        "character. directive_deactivate disables a directive by ID. directive_stats returns "
        "total/active directive counts."
    ),
)


# Inline docs: docs/scripts/core/v5/v5/mcp_server.md
from v5.tools import (  # noqa: E402
    v5_analyze_emotion, v5_emotion_status, v5_emotion_label,
    v5_memory_store, v5_memory_search, v5_memory_get, v5_memory_delete,
    v5_memory_stats,
    v5_self_model, v5_self_reflect, v5_latest_thought,
    v5_curiosity_check, v5_subconscious,
    v5_care_check, v5_care_status,
    v5_vitality, v5_vitality_tick,
    v5_relationship, v5_relationship_tick,
    v5_narrative_generate, v5_dissonance_check, v5_proactive_check,
    v5_self_discover, v5_reflect_run_op,
    # V5.2: neko migration tools
    v5_reflection_synthesize, v5_reflection_read,
    v5_reflection_apply_evidence, v5_reflection_promote,
    v5_reflection_stats,
    v5_anti_repeat_record, v5_anti_repeat_check,
    v5_anti_repeat_penalty, v5_anti_repeat_clear, v5_anti_repeat_stats,
    v5_directive_add, v5_directive_list, v5_directive_deactivate,
    v5_directive_stats,
    # V5.3: activity perception + context compression engine
    v5_activity_status, v5_context_compression_stats,
    # V5.4: project track
    v5_project_note, v5_project_retrieve, v5_project_stats,
)

_NEW_V5_TOOLS = [
    v5_analyze_emotion, v5_emotion_status, v5_emotion_label,
    v5_memory_store, v5_memory_search, v5_memory_get, v5_memory_delete,
    v5_memory_stats,
    v5_self_model, v5_self_reflect, v5_latest_thought,
    v5_curiosity_check, v5_subconscious,
    v5_care_check, v5_care_status,
    v5_vitality, v5_vitality_tick,
    v5_relationship, v5_relationship_tick,
    v5_narrative_generate, v5_dissonance_check, v5_proactive_check,
    v5_self_discover, v5_reflect_run_op,
    # V5.2: neko migration tools
    v5_reflection_synthesize, v5_reflection_read,
    v5_reflection_apply_evidence, v5_reflection_promote,
    v5_reflection_stats,
    v5_anti_repeat_record, v5_anti_repeat_check,
    v5_anti_repeat_penalty, v5_anti_repeat_clear, v5_anti_repeat_stats,
    v5_directive_add, v5_directive_list, v5_directive_deactivate,
    v5_directive_stats,
    # V5.3: activity perception + context compression engine
    v5_activity_status, v5_context_compression_stats,
    # V5.4: project track
    v5_project_note, v5_project_retrieve, v5_project_stats,
]
for _tool_fn in _NEW_V5_TOOLS:
    try:
        mcp.add_tool(_tool_fn)
    except Exception as _e:  # noqa: BLE001
        logger.warning("failed to register tool %s: %s",
                       getattr(_tool_fn, "__name__", _tool_fn), _e)


# ── Entry point ────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        # Hermes Studio transport: SSE on :9877.
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = 9877
        logger.info("v5 MCP server starting (sse) on 127.0.0.1:9877 ...")
        mcp.run(transport="sse")
    else:
        logger.info("v5 MCP server starting (stdio)...")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()