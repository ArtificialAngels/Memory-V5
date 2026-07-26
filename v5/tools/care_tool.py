# 详细说明见 docs/scripts/core/v5/v5/tools/care_tool.md

from __future__ import annotations

from v5.tools.utils import safe_tool, dumps


@safe_tool
def v5_care_check() -> str:
    """Check whether哥哥 needs active care (rest / water / sleep).

    Call chain: care.check_and_care().  Falls back to template care when
    :8080 is down.
    """
    from v5.care import check_and_care
    text = check_and_care()
    if text:
        return dumps({"needs_care": True, "suggestion": text}, ensure_ascii=False)
    return dumps({"needs_care": False})


@safe_tool
def v5_care_status() -> str:
    """Return the care monitor's cumulative activity counters."""
    from v5.care import CareMonitor
    m = CareMonitor.load()
    return dumps({
        "cumulative_coding_sec": m.cumulative_coding_sec,
        "cumulative_gaming_sec": m.cumulative_gaming_sec,
        "cumulative_focused_sec": m.cumulative_focused_sec,
        "last_activity": m.last_activity,
        "last_remind_time": m.last_remind_time,
        "total_reminders": m.total_reminders,
    })
