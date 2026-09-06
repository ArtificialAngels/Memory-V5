"""file_store drift-guard / atomic-write regression tests (OPT-2, 2026-08-19).

Pure stdlib — no chromadb/numpy. Mirrors the graft from dsh-memory-evolve's
drift guard (docs/v5-vs-dsh-memory-evolve-20260819.md §②).
"""
import json
import re
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent  # core/memory_v5/tests
sys.path.insert(0, str(V5_ROOT.parents[1]))  # core/

from memory_v5.file_store import (  # noqa: E402
    atomic_write_text, atomic_write_json, safe_backup,
    DriftDetected, DEFAULT_MAX_BACKUPS,
)


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_atomic_write_text_basic(tmp_path: Path):
    p = tmp_path / "a.json"
    atomic_write_text(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"
    # no stray .tmp left behind
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_creates_backup_on_overwrite(tmp_path: Path):
    p = tmp_path / "a.json"
    _write(p, "v1")
    atomic_write_text(p, "v2")
    assert p.read_text(encoding="utf-8") == "v2"
    backups = list(tmp_path.glob("a.json.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "v1"


def test_rolling_backups_pruned(tmp_path: Path):
    p = tmp_path / "a.json"
    for i in range(DEFAULT_MAX_BACKUPS + 3):
        atomic_write_text(p, f"v{i}")
    backups = sorted(tmp_path.glob("a.json.bak.*"), key=lambda x: x.stat().st_mtime)
    assert len(backups) == DEFAULT_MAX_BACKUPS


def test_drift_guard_refuses_and_backs_up(tmp_path: Path):
    p = tmp_path / "a.json"
    # existing file is hand-corrupted (not valid JSON)
    _write(p, "{ this is not json")
    # validator: existing content must be valid JSON
    try:
        atomic_write_text(p, '{"ok": true}', validator=lambda s: _json_ok(s))
        raise AssertionError("expected DriftDetected")
    except DriftDetected as exc:
        assert exc.backup is not None
        # original (corrupted but meaningful) content preserved, not overwritten
        assert p.read_text(encoding="utf-8") == "{ this is not json"
        # and it was backed up
        assert exc.backup.read_text(encoding="utf-8") == "{ this is not json"


def test_drift_guard_passes_when_valid(tmp_path: Path):
    p = tmp_path / "a.json"
    _write(p, '{"x": 1}')
    atomic_write_text(p, '{"x": 2}', validator=lambda s: _json_ok(s))
    assert json.loads(p.read_text(encoding="utf-8")) == {"x": 2}


def test_atomic_write_json_roundtrip(tmp_path: Path):
    p = tmp_path / "s.json"
    atomic_write_json(p, {"name": "ikaros", "n": 5})
    assert json.loads(p.read_text(encoding="utf-8")) == {"name": "ikaros", "n": 5}


def test_atomic_write_json_makes_backup(tmp_path: Path):
    p = tmp_path / "s.json"
    atomic_write_json(p, {"v": 1})
    atomic_write_json(p, {"v": 2})
    backups = list(tmp_path.glob("s.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == {"v": 1}


def test_no_backup_when_disabled(tmp_path: Path):
    p = tmp_path / "s.json"
    _write(p, "x")
    atomic_write_text(p, "y", make_backup=False)
    assert not list(tmp_path.glob("s.json.bak.*"))


def test_safe_backup_nonexistent(tmp_path: Path):
    assert safe_backup(tmp_path / "nope.json") is None


def _json_ok(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (ValueError, TypeError):
        return False
