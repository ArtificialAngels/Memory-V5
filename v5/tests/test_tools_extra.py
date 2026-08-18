"""v5.tests.test_tools_extra — P1/P2 tool contract tests.

Covers narrative / dissonance / proactive / self-discovery / reflect-op tools.
Offline-safe: narrative uses use_llm=False (rule path); dissonance and
reflect-run use deterministic local logic; self-discover degrades to
{written:0} when :8080 is down.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))

from v5.tools.extra_tool import (
    v5_dissonance_check,
    v5_narrative_generate,
    v5_proactive_check,
    v5_reflect_run_op,
    v5_self_discover,
)


def _parse(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    assert isinstance(out, str), "every tool must return a JSON string"
    return json.loads(out)


def test_narrative_rule_path():
    d = _parse(v5_narrative_generate, 7, False)
    assert isinstance(d, dict)


def test_dissonance_shape():
    d = _parse(v5_dissonance_check, "猫是哺乳动物", "fact")
    assert isinstance(d, dict) and "conflicts" in d


def test_proactive_shape():
    d = _parse(v5_proactive_check)
    assert {"should_speak", "reason", "text"} <= d.keys()


def test_self_discover_shape():
    d = _parse(v5_self_discover)
    assert "written" in d


def test_reflect_unknown_op_is_safe():
    d = _parse(v5_reflect_run_op, "__nonexistent_op__")
    assert d.get("ok") is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
