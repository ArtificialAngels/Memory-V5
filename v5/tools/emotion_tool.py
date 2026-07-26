"""Emotion tools for Ikaros V5.

Since the local :8080 small-model was removed from V5 (2026-07-26), emotion
is computed by the lightweight rule-based estimator (``state.update_from_text``).
This keeps the tools dependency-free and offline-safe — the same behavior the
old code fell back to whenever :8080 was unreachable.
"""
from __future__ import annotations

import math

from v5.tools.utils import safe_tool, dumps, answer


@safe_tool
def v5_analyze_emotion(text: str, *, force_update: bool = False) -> str:
    """Update Ikaros's PAD emotion state from a piece of text.

    Uses the rule-based estimator (no external model). Returns the new PAD
    state as JSON with a mood label, the PAD delta this text caused, and an
    intensity scalar (magnitude of the shift).
    """
    from v5.affect import AffectState

    prev = AffectState.load().decay()      # 应用前的衰减态
    state = prev.apply_event(text)         # 应用事件 (内部再次 decay 近似 no-op)
    state.save()
    dp = state.pleasure - prev.pleasure
    da = state.arousal - prev.arousal
    dd = state.dominance - prev.dominance
    return answer(
        f"情感已更新：愉悦{state.pleasure:.2f} 激活{state.arousal:.2f} 掌控{state.dominance:.2f}",
        {
            "pleasure": state.pleasure,
            "arousal": state.arousal,
            "dominance": state.dominance,
            "mood_label": state.to_prompt(),
            "delta": {
                "pleasure": round(dp, 4),
                "arousal": round(da, 4),
                "dominance": round(dd, 4),
            },
            "intensity": round(math.hypot(dp, da, dd), 4),
        }
    )


@safe_tool
def v5_emotion_status() -> str:
    """Return the current PAD emotion state (no external dependency).

    Includes a brief mood label generated from the PAD values.
    """
    from v5.affect import AffectState

    state = AffectState.load()
    return answer(
        f"当前情感：愉悦{state.pleasure:.2f} 激活{state.arousal:.2f} 掌控{state.dominance:.2f}",
        {
            "pleasure": state.pleasure,
            "arousal": state.arousal,
            "dominance": state.dominance,
            "mood_label": state.to_prompt(),
            "last_updated": getattr(state, "last_updated", None),
        }
    )


@safe_tool
def v5_emotion_label(text: str, *, fallback: str = "平静") -> str:
    """Return 1-2 emotion tags for the text.

    Rule-based keyword matcher only (the local LLM path was removed from V5
    along with the small model). Always reports ``method="rule"``.
    """
    tags = [fallback]
    method = "rule"
    return dumps({"tags": tags, "method": method})
