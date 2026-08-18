"""v5.tests.test_tools_vitality — vitality tool contract tests.

Offline-safe: v5.vitality imports psutil lazily; these tools must still
return a valid energy-state dict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))

from v5.tools.vitality_tool import v5_vitality, v5_vitality_tick


def _parse(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    assert isinstance(out, str), "every tool must return a JSON string"
    return json.loads(out)


def test_vitality_shape():
    d = _parse(v5_vitality)
    assert {"vitality", "label", "emoji"} <= d.keys()


def test_vitality_tick_shape():
    d = _parse(v5_vitality_tick, True)
    assert {"vitality", "label", "conversation"} <= d.keys()
    assert d["conversation"] is True


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
