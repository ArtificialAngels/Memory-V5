"""v5.tests.test_tools_relationship — relationship tool contract tests.

Offline-safe: relationship state is persisted JSON; tick records one
interaction without needing any external service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))

from v5.tools.relationship_tool import v5_relationship, v5_relationship_tick


def _parse(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    assert isinstance(out, str), "every tool must return a JSON string"
    return json.loads(out)


def test_relationship_shape():
    d = _parse(v5_relationship)
    assert {"depth", "warmth", "stage", "closeness", "days_known"} <= d.keys()


def test_relationship_tick_shape():
    d = _parse(v5_relationship_tick, 0.5)
    assert {"depth", "warmth", "stage", "closeness"} <= d.keys()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
