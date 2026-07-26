"""v5.tests.test_tools_emotion — emotion tool contract tests.

Offline-safe: never asserts on LLM output content, only on the JSON shape
the tools guarantee.  With :8080 down, v5_emotion_label reports method "rule".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(V5_ROOT))

from v5.tools.emotion_tool import (
    v5_analyze_emotion,
    v5_emotion_label,
    v5_emotion_status,
)


def _parse(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    assert isinstance(out, str), "every tool must return a JSON string"
    # Tools may emit a one-line natural-language preamble before the JSON
    # (the ``answer()`` helper returns ``"<summary>\n<json>"``). Find the
    # first ``{`` and parse from there so the JSON shape is what we verify.
    start = out.find("{")
    if start != -1:
        out = out[start:]
    return json.loads(out)


def test_emotion_status_shape():
    d = _parse(v5_emotion_status)
    assert {"pleasure", "arousal", "dominance", "mood_label"} <= d.keys()


def test_analyze_emotion_shape():
    d = _parse(v5_analyze_emotion, "我今天写完了所有测试，很开心")
    assert "mood_label" in d and "delta" in d and "intensity" in d


def test_emotion_label_fallback():
    d = _parse(v5_emotion_label, "这消息让我有点难过")
    assert "tags" in d and "method" in d
    assert isinstance(d["tags"], list)
    # No :8080 in test env => rule-based path.
    assert d["method"] == "rule"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
