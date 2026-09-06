"""scope markers + [dsh-only] external-executor isolation (OPT-3, 2026-08-19).

Pure marker logic is tested standalone; the retrieval-filter wiring is tested
against a throwaway sqlite DB (store.search / unified_retrieve lexical path —
both chromadb-free) so it runs in any environment.
"""
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent  # core/memory_v5/tests
sys.path.insert(0, str(V5_ROOT.parents[1]))  # core/

from memory_v5 import scope  # noqa: E402


# ── pure marker logic ──────────────────────────────────────────────

def test_is_dsh_only():
    assert not scope.is_dsh_only("普通记忆内容")
    assert scope.is_dsh_only("[dsh-only] 平台纪律，外部勿看")


def test_extract_summary():
    assert scope.extract_summary("长文 [summary:一句话摘要] 余下") == "一句话摘要"
    assert scope.extract_summary("没有摘要") is None


def test_clean_markers():
    c = scope.clean_markers("[dsh-only] 平台纪律 [summary:要点] 细节")
    assert "[dsh-only]" not in c
    assert "要点" in c
    assert "[summary:" not in c


def test_filter_external():
    items = [
        {"id": 1, "content": "普通记忆"},
        {"id": 2, "content": "[dsh-only] 平台纪律"},
    ]
    assert len(scope.filter_external(items, include_dsh_only=True)) == 2
    kept = scope.filter_external(items, include_dsh_only=False)
    assert len(kept) == 1
    assert kept[0]["id"] == 1


# ── retrieval-filter wiring (throwaway DB) ─────────────────────────

def _patch_db(tmp_path: Path):
    import memory_v5.store as store
    orig_dir, orig_db = store.V5_DATA_DIR, store.V5_DB_PATH
    d = tmp_path / "v5scope"
    d.mkdir(parents=True, exist_ok=True)
    store.V5_DATA_DIR = d
    store.V5_DB_PATH = d / "v5.db"
    return store, orig_dir, orig_db


def _restore_db(store, orig_dir, orig_db):
    store.V5_DATA_DIR = orig_dir
    store.V5_DB_PATH = orig_db


def test_search_excludes_dsh_only(tmp_path: Path):
    store, od, ob = _patch_db(tmp_path)
    try:
        # store 走 committed(), 无需 chromadb. 用 ASCII 共享词 (FTS5 默认分词器
        # 对中文子串不友好, 真实中文召回走 chromadb; 此处只验证过滤接线).
        store.store("ikaros coffee machine is in the kitchen", type="fact", tags="pref")
        store.store("[dsh-only] ikaros internal discipline: external executors must not see keys", type="fact", tags="pref")
        hits = store.search("ikaros", top_k=10)
        assert len(hits) == 2
        kept = scope.filter_external(hits, include_dsh_only=False)
        assert len(kept) == 1
        assert "[dsh-only]" not in kept[0].content
    finally:
        _restore_db(store, od, ob)


def test_finish_excludes_dsh_only():
    # 直接验证 unified_retrieve 内部 _finish 的 include_dsh_only 过滤接线
    # (绕过 sqlite/chromadb 环境依赖, 确定性单测).
    from memory_v5.memory_retrieval import _finish

    merged = {
        "1": {"id": 1, "content": "普通记忆内容", "score": 0.9},
        "2": {"id": 2, "content": "[dsh-only] 平台纪律外部勿看", "score": 0.8},
    }
    inc = _finish(merged, 10, include_dsh_only=True)
    exc = _finish(merged, 10, include_dsh_only=False)
    assert len(inc) == 2
    assert len(exc) == 1
    assert exc[0]["id"] == 1


def test_memory_api_search_excludes_dsh_only(tmp_path: Path):
    # V5MemoryAPI.search 透传 include_dsh_only 到 store.search + filter_external
    store, od, ob = _patch_db(tmp_path)
    try:
        store.store("ikaros coffee machine in the kitchen", type="fact", tags="pref")
        store.store("[dsh-only] ikaros internal discipline external executors", type="fact", tags="pref")
        from memory_v5.memory_api import V5MemoryAPI
        api = V5MemoryAPI()
        inc = api.search(query="ikaros", top_k=10, fuse=False)
        exc = api.search(query="ikaros", top_k=10, fuse=False, include_dsh_only=False)
        assert len(inc) == 2
        assert len(exc) == 1
        assert exc[0].get("content", getattr(exc[0], "content", "")) == "ikaros coffee machine in the kitchen"
    finally:
        _restore_db(store, od, ob)
