"""Safe file-side storage for Ikaros V5.

Grafted from ``dsh-memory-evolve``'s drift-guard pattern (see
``docs/v5-vs-dsh-memory-evolve-20260819.md`` §②). V5's primary fact source is
the SQLite ``v5.db`` (WAL), but several *side* stores are plain files that a
human or a parallel agent (pi / herdr) may edit by hand:

  * conversation-tree topology JSON (``data/v5/<persist_key>.json``)
  * affect / care / relationship / emotional / proactive state JSON
  * ``model_config.json``

A naive ``Path.write_text(...)`` can silently clobber an external edit or lose
content on a half-written write. This module replaces those calls with:

  1. **atomic write** — write to a unique ``*.tmp`` then ``os.replace``
     (Windows-safe, last-writer-wins, never a truncated target).
  2. **rolling ``.bak`` backup** — the pre-overwrite copy is kept
     (``<file>.bak.<timestamp>``, last ``max_backups`` retained) so any bad
     write is recoverable.
  3. **optional drift guard** — before overwriting, the *existing on-disk*
     content is passed to a ``validator``; if it no longer round-trips (e.g. a
     human edit introduced a syntax error), the drifted file is backed up and
     the overwrite is **refused** (``DriftDetected``) instead of destroying the
     meaningful (if malformed) content.

Pure stdlib — safe to import anywhere in V5 without pulling chromadb/numpy.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

logger = logging.getLogger("ikaros.v5.file_store")

# How many rolling backups to keep per file.
DEFAULT_MAX_BACKUPS = 5


class DriftDetected(Exception):
    """Raised when a drift guard refuses to overwrite a non-round-tripping file.

    The drifted (possibly hand-edited, malformed-but-meaningful) file has
    already been backed up to ``<path>.bak.<ts>`` before this is raised, so the
    caller can recover it.
    """

    def __init__(self, path: Path, backup: Path | None, reason: str) -> None:
        self.path = path
        self.backup = backup
        self.reason = reason
        super().__init__(f"drift guard refused overwrite of {path}: {reason}")


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}"


def safe_backup(path: Path, *, max_backups: int = DEFAULT_MAX_BACKUPS) -> Path | None:
    """Copy ``path`` to ``<path>.bak.<timestamp>``; keep only the last N.

    Returns the backup Path, or ``None`` if ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak.{_now_ts()}")
    try:
        shutil.copy2(path, backup)
    except OSError as exc:
        logger.warning("file_store: backup of %s failed: %s", path, exc)
        return None
    _prune_backups(path, max_backups=max_backups)
    return backup


def _prune_backups(path: Path, *, max_backups: int = DEFAULT_MAX_BACKUPS) -> None:
    """Delete oldest ``.bak.*`` files beyond ``max_backups``."""
    if max_backups <= 0:
        return
    cands = sorted(
        path.parent.glob(f"{path.name}.bak.*"),
        key=lambda p: p.stat().st_mtime,
    )
    excess = len(cands) - max_backups
    if excess <= 0:
        return
    for old in cands[:excess]:
        try:
            old.unlink()
        except OSError:
            pass


def _atomic_replace(tmp: Path, target: Path, *, retries: int = 4) -> None:
    """Move ``tmp`` onto ``target`` atomically, with short backoff on locking.

    Mirrors conversation_tree.persist()'s R9 fix: a unique tmp name + retry
    absorbs Windows' transient file-lock / antivirus interception (WinError 5).
    """
    for attempt in range(retries):
        try:
            os.replace(tmp, target)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    make_backup: bool = True,
    max_backups: int = DEFAULT_MAX_BACKUPS,
    validator: Callable[[str], bool] | None = None,
    encoding: str = "utf-8",
) -> Path:
    """Write ``text`` to ``path`` atomically, with backup + optional drift guard.

    Args:
        path: target file.
        text: full content to write.
        make_backup: keep a ``.bak.<ts>`` of the pre-overwrite file.
        max_backups: rolling backup retention.
        validator: optional ``callable(existing_content_str) -> bool``. If the
            existing on-disk content fails validation (does not round-trip),
            it is backed up and the write is refused with ``DriftDetected``.
        encoding: text encoding.

    Returns the target ``Path``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = None
    if target.exists():
        try:
            existing = target.read_text(encoding=encoding)
        except OSError:
            existing = None

    if existing is not None and validator is not None:
        try:
            ok = bool(validator(existing))
        except Exception as exc:  # validator bug must not clobber the file
            logger.warning("file_store: validator raised (%s); skipping guard", exc)
            ok = True
        if not ok:
            backup = safe_backup(target, max_backups=max_backups) if make_backup else None
            raise DriftDetected(
                target, backup,
                "existing content failed round-trip validation; refusing to overwrite",
            )

    if make_backup and existing is not None:
        safe_backup(target, max_backups=max_backups)

    tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        _atomic_replace(tmp, target)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return target


def _json_roundtrip(text: str) -> bool:
    """Default drift-guard validator for JSON files: must parse."""
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def atomic_write_json(
    path: str | Path,
    obj: object,
    *,
    make_backup: bool = True,
    max_backups: int = DEFAULT_MAX_BACKUPS,
    validator: Callable[[str], bool] | None = _json_roundtrip,
    indent: int = 2,
    encoding: str = "utf-8",
) -> Path:
    """Serialize ``obj`` to JSON and write atomically (see :func:`atomic_write_text`).

    ``validator`` defaults to JSON round-trip (validates the *existing*
    on-disk content before overwrite). Pass ``validator=None`` to disable the
    drift guard entirely.
    """
    text = json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
    return atomic_write_text(
        path, text,
        make_backup=make_backup,
        max_backups=max_backups,
        validator=validator,
        encoding=encoding,
    )
