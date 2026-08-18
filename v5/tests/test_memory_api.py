"""v5.tests.test_memory_api — unified dual-addressing interface tests.

Covers V5MemoryAPI.store / search / get / delete / stats, including the
Ekko-style structured filters (domain / category_path / key) that resolve
via exact tag matching (no ChromaDB required).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))

from v5.memory_api import V5MemoryAPI

_MARK = f"__v5test_api_{int(time.time())}__"


def test_dual_addressing():
    api = V5MemoryAPI()
    mid = api.store(
        f"{_MARK} structured memory",
        domain="test_dom",
        category_path="self/values",
        key="api_key_1",
        importance=0.8,
    )
    assert mid > 0
    try:
        assert len(api.search(domain="test_dom")) > 0
        assert len(api.search(key="api_key_1")) > 0
        assert len(api.search(category_path="self/values")) > 0
    finally:
        # 2026-08-14: 测试隔离修复 —— 此前本测试写入真实 v5.db 且不清理,
        # 每次 pytest 运行都给生产记忆库留下 1 条垃圾行 (v5.db id 递增).
        assert api.delete(mid) is True


def test_get_delete_stats():
    api = V5MemoryAPI()
    mid = api.store(f"{_MARK} removable", domain="del_dom", key="del_k")
    got = api.get(mid)
    assert got is not None and got["id"] == mid
    assert api.delete(mid) is True
    assert isinstance(api.stats(), dict)


def test_combined_filters():
    """domain + key + type combined must narrow to exactly the matching row."""
    api = V5MemoryAPI()
    m1 = api.store(f"{_MARK} a", domain="cf_d1", key="cf_k1",
                   memory_type="fact", tags=["cf_t1"])
    m2 = api.store(f"{_MARK} b", domain="cf_d1", key="cf_k2",
                   memory_type="fact", tags=["cf_t2"])
    m3 = api.store(f"{_MARK} c", domain="cf_d2", key="cf_k1",
                   memory_type="preference", tags=["cf_t1"])
    try:
        r = api.search(domain="cf_d1", type="fact")
        assert len(r) == 2, f"d1+fact expected 2, got {len(r)}"
        r = api.search(key="cf_k1")
        assert len(r) == 2, f"key cf_k1 expected 2, got {len(r)}"
        r = api.search(domain="cf_d1", key="cf_k1", type="fact")
        assert len(r) == 1, f"d1+k1+fact expected 1, got {len(r)}"
    finally:
        for mid in (m1, m2, m3):
            try:
                api.delete(mid)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
