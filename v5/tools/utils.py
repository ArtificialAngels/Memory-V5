# 详细说明见 docs/scripts/core/v5/v5/tools/utils.md

from __future__ import annotations

import functools
import importlib
import json
import logging
import socket
import sys
import time
from pathlib import Path

logger = logging.getLogger("ikaros.v5.tools")

# Ikaros-memory/  (tools/ -> v5/ -> Ikaros-memory/)
V5_ROOT = Path(__file__).resolve().parent
if str(V5_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(V5_ROOT.parent))

# Shared data dir for the v5 tool layer (latest_thought.json, subconscious.json, ...).
# Centralized here so self_tool / __init__ / other submodules import one canonical path.
V5_DATA = V5_ROOT / "data" / "v5"

# Local LLM listens on :8080.  Used only as a *best-effort*
# availability indicator so tools can report which code path they took.
_LOCAL_LLM_HOST = "127.0.0.1"
_LOCAL_LLM_PORT = 8080


def require_module(module_path: str):
    """Safely import a module, return None on failure (instead of raising)."""
    try:
        return importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("require_module failed for %s: %s", module_path, exc)
        return None


def safe_tool(fn):
    """Decorator: wrap a tool call in try/except + audit logging.

    On success the decorated function's own return value is passed through
    (typically a JSON string).  On any unexpected exception we return a
    structured error JSON string instead of letting the exception escape
    (which would crash the MCP server / agent loop).

    Every call is timed and logged so a slow or failing tool is visible to
    the operator (e.g. v5_self_reflect when :8080 is down).  Success logs at
    INFO, failures at ERROR — with the host logger at WARNING (the MCP
    server default) only failures surface unless the level is lowered.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        fn_name = getattr(fn, "__name__", "?")
        logger.info("tool call: %s", fn_name)
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            logger.info("tool ok: %s (%.2fs)", fn_name, elapsed)
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            logger.error("tool failed: %s (%.2fs) -- %s", fn_name, elapsed, exc)
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "tool": fn_name,
                },
                ensure_ascii=False,
            )

    return wrapper


def dumps(obj, ensure_ascii: bool = False) -> str:
    """json.dumps with unicode preserved and a tolerant default serializer.

    Accepts an optional ``ensure_ascii`` flag (defaults to False so Chinese
    text stays readable in tool output) and a tolerant ``default`` serializer
    for objects json can't encode directly.
    """
    return json.dumps(obj, ensure_ascii=ensure_ascii, default=str)


def answer(nl_summary: str, data: dict | list) -> str:
    """Natural-language-first tool output: human text + JSON appendix.

    Args:
        nl_summary: Concise human-readable summary (Chinese).
        data: Structured data to append as JSON.
    Returns:
        f"{nl_summary}\n{json.dumps(data, ensure_ascii=False, default=str)}"
    """
    return f"{nl_summary}\n{json.dumps(data, ensure_ascii=False, default=str)}"


def local_llm_available(host: str = _LOCAL_LLM_HOST, port: int = _LOCAL_LLM_PORT,
                        timeout: float = 1.0) -> bool:
    """Best-effort TCP probe: is the local LLM server reachable?

    Used to decide whether a tool likely ran via the LLM ('llm') or fell
    back to a rule ('rule').  Cheap (no HTTP parse) and never raises.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False