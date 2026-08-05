# 详细说明见 docs/scripts/core/v5/v5/tools/extra_tool.md

from __future__ import annotations

from v5.tools.utils import safe_tool, dumps


@safe_tool
def v5_narrative_generate(days: int = 30, use_llm: bool = True) -> str:
    """Generate a monthly self-narrative from recent memories.

    Falls back to a rule-based narrative when :8080 / DeepSeek is down.
    """
    from v5.narrative import generate_narrative
    r = generate_narrative(days=days, use_llm=use_llm)
    return dumps(r, ensure_ascii=False)


@safe_tool
def v5_dissonance_check(content: str, mem_type: str = "fact") -> str:
    """Detect whether `content` contradicts an existing memory.

    Fallback: :8080 down => no NLI performed, returns {conflicts: []}.
    """
    from v5.dissonance import detect_dissonance
    r = detect_dissonance(content, mem_type)
    return dumps(r, ensure_ascii=False)


@safe_tool
def v5_proactive_check() -> str:
    """Decide whether Ikaros should proactively speak right now.

    Fallback: runs the gate checks locally (no LLM needed).
    """
    from v5.proactive import should_speak, try_proactive

    ok, reason = should_speak()
    text = None
    if ok:
        try:
            text = try_proactive()
        except Exception:  # noqa: BLE001
            text = None
    return dumps({"should_speak": ok, "reason": reason, "text": text}, ensure_ascii=False)


@safe_tool
def v5_self_discover() -> str:
    """Run one self-architecture discovery pass (reads project files).

    Fallback: :8080 down => returns {written: 0}.
    """
    from v5.self_discovery import self_discover
    n = self_discover()
    return dumps({"written": n, "ok": n > 0})


@safe_tool
def v5_reflect_run_op(op_name: str = "", force: bool = False) -> str:
    """Run one (or all due) reflection ops from the registry.

    op_name: name of a registered op (consolidate / distill / reflect /
             cleanup / narrative / self_discovery / vector_sync / promote /
             dedup).  Empty => run all currently-due ops.
    Per-op fallback: a failing op is reported, run_all continues on error.
    """
    from v5.reflect.registry import make_default_scheduler
    from v5.reflect.scheduler import load_state, save_state

    sched = make_default_scheduler(load_state())
    force = bool(force)

    if op_name:
        op = sched.get_op(op_name)
        if op is None:
            return dumps({"ok": False, "error": f"unknown op: {op_name}"}, ensure_ascii=False)
        n = sched.run_one(op, force=force)
        save_state(sched.state)
        return dumps({"op": op_name, "processed": n, "ok": True}, ensure_ascii=False)

    results = sched.run_all(force=force, continue_on_error=True)
    return dumps({"results": results, "ok": True}, ensure_ascii=False)


@safe_tool
def v5_activity_status() -> str:
    """Return Ikaros's real-time activity perception (foreground window + LLM inference).

    Uses Windows API to detect the active window title, then matches against
    known patterns or uses local LLM for inference. 60s cache, silent fallback
    to time-based inference on failure.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parent)
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    import cogno_5d
    title = cogno_5d._get_foreground_window_title()
    narrative = cogno_5d._get_activity_narrative()
    return dumps({
        "narrative": narrative,
        "window_title": title,
        "cached": True,
    }, ensure_ascii=False)


@safe_tool
def v5_context_compression_stats() -> str:
    """Return Ikaros's context compression engine status for Hermes Dashboard.

    Includes: activity perception, rhythm state, summary cache, user profile
    stats, and memory type breakdown.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parent)
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    import cogno_5d
    from v5.rhythm import build_rhythm_block, last_interaction_ts
    from v5.summary import _load_cache as _load_summary_cache
    from v5.profile import load_dislikes, load_preferences
    from v5 import store as _store

    title = cogno_5d._get_foreground_window_title()
    narrative = cogno_5d._get_activity_narrative()
    rhythm = build_rhythm_block()
    sc = _load_summary_cache()
    dislikes = load_dislikes()
    prefs = load_preferences()
    _s = _store.stats()
    total = _s.get("total", 0)
    by_type = {k: v["count"] for k, v in _s.get("by_type", {}).items()}

    return dumps({
        "activity": {
            "narrative": narrative,
            "window_title": title,
        },
        "rhythm": rhythm,
        "summary": {
            "cached": bool(sc.get("last_summary")),
            "last_round": sc.get("last_round", -1),
            "preview": (sc.get("last_summary") or "")[:80],
        },
        "profile": {
            "dislikes": len(dislikes),
            "preferences": len(prefs),
        },
        "memory": {
            "total": total,
            "by_type": by_type,
        },
    }, ensure_ascii=False)
