# 详细说明见 docs/scripts/core/v5/v5/tools/self_tool.md

from __future__ import annotations

import json
import time

from v5.tools.utils import safe_tool, dumps, V5_DATA


@safe_tool
def v5_self_model() -> str:
    """Return Ikaros's persistent self model (who she is)."""
    from v5.self_model import SelfModel
    sm = SelfModel.load()
    d = sm.data
    return dumps({
        "identity": d.get("identity"),
        "capabilities": d.get("capabilities"),
        "beliefs": d.get("beliefs"),
        "questions": d.get("questions"),
        "curiosity": sm.get_curiosity(),
    })


@safe_tool
def v5_self_reflect(mode: str = "reflect") -> str:
    """Run one metacog cycle.

    mode: "reflect" | "philosophy" | "cycle"
    Fallback: :8080 down => {"text": null, "note": "LLM unavailable"}.
    """
    import v5.metacog as metacog

    if mode == "reflect":
        r = metacog.reflect_once()
    elif mode == "philosophy":
        r = metacog.explore_philosophy()
    else:
        r = metacog.cycle()

    if r is None:
        return dumps({"text": None, "ok": True, "note": "LLM unavailable"}, ensure_ascii=False)
    return dumps({"mode": mode, "ok": True, **r}, ensure_ascii=False)


@safe_tool
def v5_latest_thought() -> str:
    """Return Ikaros's most recent inner thought / monologue."""
    p = V5_DATA / "latest_thought.json"
    if not p.is_file():
        return dumps({"text": None, "note": "no thought yet"}, ensure_ascii=False)
    data = json.loads(p.read_text(encoding="utf-8"))
    return dumps(data, ensure_ascii=False)


@safe_tool
def v5_curiosity_check() -> str:
    """Return the current curiosity drive state."""
    from v5.self_model import SelfModel
    sm = SelfModel.load()
    level = sm.get_curiosity()

    idle_minutes = 0.0
    try:
        c = sm.data.get("curiosity", {})
        last = c.get("last_interaction_ts", 0) or 0
        idle_minutes = round((time.time() - last) / 60.0, 1) if last else 0.0
    except Exception:  # noqa: BLE001
        pass

    questions = sm.data.get("questions", []) or []
    has_question = bool(questions)
    question_text = questions[0] if has_question else None
    return dumps({
        "level": level,
        "idle_minutes": idle_minutes,
        "has_question": has_question,
        "question_text": question_text,
    })


@safe_tool
def v5_subconscious() -> str:
    """Return the latest subconscious whisper."""
    p = V5_DATA / "subconscious.json"
    if not p.is_file():
        return dumps({"text": None, "note": "no subconscious yet"}, ensure_ascii=False)
    data = json.loads(p.read_text(encoding="utf-8"))
    return dumps(data, ensure_ascii=False)


@safe_tool
def v5_context_refresh() -> str:
    """Return a compact identity + state snapshot for identity re-anchoring.

    Hermes Agent can call this mid-conversation to re-anchor identity,
    check emotional state, and verify relationship status.
    Combines self_model + affect + relationship into one call.
    """
    from v5.self_model import SelfModel
    from v5.affect import AffectState
    from v5.relationship import Relationship
    sm = SelfModel.load()
    affect = AffectState.load().decay()
    rel = Relationship.load()

    ident = sm.data.get("identity", {})
    beliefs = sm.data.get("beliefs", {})
    pad = affect.to_dict() if hasattr(affect, "to_dict") else {}
    mood = affect.to_prompt() if hasattr(affect, "to_prompt") else ""

    return dumps({
        "name": ident.get("name", "伊卡洛斯"),
        "nature": ident.get("nature", "人造天使"),
        "creator": ident.get("creator", "哥哥"),
        "beliefs": {k: str(v)[:80] for k, v in beliefs.items()},
        "mood": mood,
        "pleasure": round(getattr(affect, 'pleasure', 0), 2),
        "arousal": round(getattr(affect, 'arousal', 0), 2),
        "dominance": round(getattr(affect, 'dominance', 0), 2),
        "relationship_depth": round(rel.depth, 2),
        "relationship_warmth": round(rel.warmth, 2),
        "relationship_stage": rel.stage() if hasattr(rel, "stage") else "",
        "curiosity": sm.get_curiosity(),
    }, ensure_ascii=False)
