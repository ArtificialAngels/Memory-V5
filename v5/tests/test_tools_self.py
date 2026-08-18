"""v5.tests.test_tools_self — self-cognition tool contract tests.

Offline-safe: v5_self_reflect degrades to {ok, text:None} when :8080 is down;
we only assert the guaranteed JSON shape, not the LLM text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))

from v5.tools.self_tool import (
    v5_curiosity_check,
    v5_latest_thought,
    v5_self_model,
    v5_self_reflect,
    v5_subconscious,
)


def _parse(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    assert isinstance(out, str), "every tool must return a JSON string"
    return json.loads(out)


def test_self_model_shape():
    d = _parse(v5_self_model)
    assert "identity" in d and "curiosity" in d


def test_self_reflect_shape():
    d = _parse(v5_self_reflect, "reflect")
    assert d.get("ok") is True  # present in both LLM-up and LLM-down paths


def test_latest_thought_shape():
    d = _parse(v5_latest_thought)
    assert isinstance(d, dict)


def test_curiosity_check_shape():
    d = _parse(v5_curiosity_check)
    assert "level" in d and "has_question" in d


def test_subconscious_shape():
    d = _parse(v5_subconscious)
    assert isinstance(d, dict)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
