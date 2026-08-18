"""v5.tests.test_tools_care — care tool contract tests.

Offline-safe: v5_care_check falls back to template care when :8080 is down,
and always returns a dict with a boolean needs_care flag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))

from v5.tools.care_tool import v5_care_check, v5_care_status


def _parse(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    assert isinstance(out, str), "every tool must return a JSON string"
    return json.loads(out)


def test_care_check_shape():
    d = _parse(v5_care_check)
    assert "needs_care" in d and isinstance(d["needs_care"], bool)


def test_care_status_shape():
    d = _parse(v5_care_status)
    assert "cumulative_coding_sec" in d
    assert "total_reminders" in d


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
