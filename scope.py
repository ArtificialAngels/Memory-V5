"""Scope markers for memory entries (grafted from dsh-memory-evolve §③).

Grafted from ``dsh-memory-evolve``'s scope markers
(``docs/v5-vs-dsh-memory-evolve-20260819.md`` §③). Markers may appear inline
in a memory ``content`` string and let V5 express *visibility* without a schema
change:

  * ``[dsh-only]`` — the entry is for DSH's own reasoning. When building
    context for an **external executor** (pi / herdr coding agents), it is
    skipped so platform/discipline facts never leak into a sub-agent.
  * ``[summary:...]`` — progressive disclosure: a short inline summary; the
    full body can be kept/elided and only the summary injected by default to
    save tokens.

Pure stdlib — safe to import anywhere without pulling chromadb/numpy.
"""

from __future__ import annotations

import re

DSH_ONLY_MARKER = "[dsh-only]"
_SUMMARY_RE = re.compile(r"\[summary:(.*?)\]", re.S)


def is_dsh_only(content: str | None) -> bool:
    """True if ``content`` is marked ``[dsh-only]`` (external-executor skip)."""
    return DSH_ONLY_MARKER in (content or "")


def extract_summary(content: str | None) -> str | None:
    """Return the ``[summary:...]`` text if present, else None."""
    m = _SUMMARY_RE.search(content or "")
    return m.group(1).strip() if m else None


def clean_markers(content: str | None) -> str:
    """Strip scope markers for clean storage/display.

    ``[dsh-only]`` is removed entirely; ``[summary:body]`` is replaced by its
    inner ``body``.
    """
    if not content:
        return content
    c = content.replace(DSH_ONLY_MARKER, "")
    c = _SUMMARY_RE.sub(lambda m: m.group(1), c)
    return c


def _content_of(item) -> str:
    """Best-effort content extraction from a memory dict or namedtuple."""
    if isinstance(item, dict):
        return item.get("content") or ""
    return getattr(item, "content", "") or ""


def filter_external(items: list, *, include_dsh_only: bool = True) -> list:
    """Drop ``[dsh-only]`` entries unless ``include_dsh_only`` is True.

    ``items`` are memory records (dict, as from ``unified_retrieve`` /
    ``V5MemoryAPI.search``, or a ``Memory`` namedtuple from ``store.search``).
    Used by external-executor context builders (pi / herdr) which should call
    with ``include_dsh_only=False``.
    """
    if include_dsh_only:
        return items
    return [v for v in items if not is_dsh_only(_content_of(v))]
