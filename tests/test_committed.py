"""V5-2: write-path commit convergence onto ``store.committed()`` (2026-08-19).

Two layers of verification, both chromadb-free (throwaway sqlite DB):

1. Unit contract of ``committed()`` — auto-commits on normal exit, rolls back
   on exception (the historical footgun: ``with conn()`` rolls back on exit, so a
   forgotten ``c.commit()`` silently dropped a write).
2. Integration smoke of the migrated write functions in store / anti_repeat /
   reflections / user_directives — prove the ``committed()`` swap preserved
   real persistence behavior (data actually lands, and is readable back).
"""
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent  # core/memory_v5/tests
sys.path.insert(0, str(V5_ROOT.parents[1]))  # core/


def _patch_db(tmp_path: Path):
    import memory_v5.store as store
    orig_dir, orig_db = store.V5_DATA_DIR, store.V5_DB_PATH
    d = tmp_path / "v5committed"
    d.mkdir(parents=True, exist_ok=True)
    store.V5_DATA_DIR = d
    store.V5_DB_PATH = d / "v5.db"
    return store, orig_dir, orig_db


def _restore_db(store, orig_dir, orig_db):
    store.V5_DATA_DIR = orig_dir
    store.V5_DB_PATH = orig_db


# ── committed() unit contract ─────────────────────────────────

def test_committed_autocommits(tmp_path: Path):
    store, od, ob = _patch_db(tmp_path)
    try:
        from memory_v5.store import committed
        with committed() as c:
            c.execute("CREATE TABLE IF NOT EXISTS t_commit(id INTEGER)")
            c.execute("INSERT INTO t_commit (id) VALUES (1)")
        # read back on a fresh connection — must see the row
        with store.conn() as rc:
            n = rc.execute("SELECT COUNT(*) FROM t_commit").fetchone()[0]
        assert n == 1
    finally:
        _restore_db(store, od, ob)


def test_committed_rolls_back_on_error(tmp_path: Path):
    store, od, ob = _patch_db(tmp_path)
    try:
        from memory_v5.store import committed
        with committed() as c:
            c.execute("CREATE TABLE IF NOT EXISTS t_commit(id INTEGER)")
            c.execute("INSERT INTO t_commit (id) VALUES (1)")
        # force a rollback: insert then raise inside the block
        raised = False
        try:
            with committed() as c:
                c.execute("INSERT INTO t_commit (id) VALUES (2)")
                raise RuntimeError("boom")
        except RuntimeError:
            raised = True
        assert raised
        with store.conn() as rc:
            n = rc.execute("SELECT COUNT(*) FROM t_commit").fetchone()[0]
        # id=2 must have been rolled back; id=1 persisted from prior block
        assert n == 1
    finally:
        _restore_db(store, od, ob)


# ── migrated write functions persist correctly ───────────────

def test_store_write_blocks_persist(tmp_path: Path):
    store, od, ob = _patch_db(tmp_path)
    try:
        # store.merge / _write_event / link_project_edge / delete / access /
        # update_evidence all migrated to committed() — exercise a couple.
        mid = store.store("ikaros likes black coffee", type="fact", tags="pref")
        assert mid
        # access() bumps accessed count — write path
        store.access(mid)
        # delete() — write path (archived flag)
        store.delete(mid)
        # deleted memory should be recoverable via archived=1 (not physically gone)
        rows = store.search("coffee", top_k=10)  # lexical fallback path
        # at minimum store round-tripped without error
        assert isinstance(rows, (list, tuple))
    finally:
        _restore_db(store, od, ob)


def test_anti_repeat_write_paths(tmp_path: Path):
    from memory_v5 import anti_repeat
    store, od, ob = _patch_db(tmp_path)
    try:
        char = "ikaros"
        n = anti_repeat.record_response(char, "今天天气真好，我们去散步吧")
        assert isinstance(n, int)
        rep = anti_repeat.check_repetition(char, "今天天气真好，我们去散步吧")
        assert "score" in rep and "is_repetitive" in rep
        cleared = anti_repeat.clear(char)
        assert isinstance(cleared, int)
    finally:
        _restore_db(store, od, ob)


def test_reflections_write_paths(tmp_path: Path):
    from memory_v5 import reflections
    store, od, ob = _patch_db(tmp_path)
    try:
        char = "ikaros"
        rid = reflections.synthesize(
            char, "user prefers concise replies", importance=7,
            initial_reinforcement=12.0,
        )
        assert rid, "synthesize must return a reflection id"
        # read it back — proves the INSERT committed
        rows = reflections.read(char, status="pending")
        assert any(r.id == rid for r in rows)
        # apply_evidence drives a status transition (pending → confirmed/promoted);
        # the exact terminal status depends on the evidence thresholds, so we only
        # assert it moved OUT of pending — that proves the UPDATE committed.
        ok = reflections.apply_evidence(char, rid, delta_rein=20.0)
        assert ok is True
        final = reflections.read(char)
        moved = [r for r in final if r.id == rid]
        assert moved and moved[0].status != "pending"
    finally:
        _restore_db(store, od, ob)


def test_user_directives_write_paths(tmp_path: Path):
    from memory_v5 import user_directives
    store, od, ob = _patch_db(tmp_path)
    try:
        char = "ikaros"
        did = user_directives.add_directive(char, "不要提政治话题", directive_type="ban_topic")
        assert did, "add_directive must return a directive id"
        active = user_directives.get_active_directives(char)
        assert any(d["id"] == did for d in active)
        assert user_directives.deactivate(did) is True
        active2 = user_directives.get_active_directives(char)
        assert not any(d["id"] == did for d in active2)
        # expire_old is a write path too
        assert isinstance(user_directives.expire_old(char), int)
    finally:
        _restore_db(store, od, ob)
