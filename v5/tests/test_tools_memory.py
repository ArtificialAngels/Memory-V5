"""v5.tests.test_tools_memory — memory tool contract + roundtrip tests.

Offline-safe: the default search path falls back to FTS5 when ChromaDB /
:8587 are unavailable, so we only assert that a stored item is retrievable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(V5_ROOT))

from v5.tools.memory_tool import (
    v5_memory_delete,
    v5_memory_get,
    v5_memory_search,
    v5_memory_stats,
    v5_memory_store,
)

_MARK = f"__v5test_memory_{int(time.time())}__"


def _parse(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    assert isinstance(out, str), "every tool must return a JSON string"
    return json.loads(out)


def test_store_returns_id():
    d = _parse(v5_memory_store, f"{_MARK} hello world", type="fact", importance=0.7)
    assert d.get("ok") is True and d.get("id")


def test_search_roundtrip():
    _parse(v5_memory_store, f"{_MARK} roundtrip entry", type="fact")
    results = _parse(v5_memory_search, _MARK)
    assert isinstance(results, list) and len(results) > 0
    assert any(_MARK in str(r.get("content", "")) for r in results)


def test_get_and_delete():
    created = _parse(v5_memory_store, f"{_MARK} deletable", type="fact")["id"]
    got = _parse(v5_memory_get, created)
    assert got.get("id") == created
    deleted = _parse(v5_memory_delete, created)
    assert deleted.get("ok") is True


def test_stats_shape():
    d = _parse(v5_memory_stats)
    assert "total" in d


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
