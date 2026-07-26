"""v5.tests.test_orchestrator — agent runtime contract tests.

These run fully offline: the companion/agent paths are exercised with an
injected ``_fallback`` so the real cloud_chat pipeline (and :8080) is never
touched.  The key guarantee is graceful degradation — agent mode with no LLM
must collapse back to companion delegation without raising.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(V5_ROOT))

from v5 import orchestrator as orch


def _setup_module():
    os.environ.pop("V5_AGENT_MODE", None)


def test_default_mode_companion():
    os.environ.pop("V5_AGENT_MODE", None)
    assert orch.current_mode() == "companion"


def test_set_mode_agent_and_back():
    orch.set_mode("agent")
    assert orch.current_mode() == "agent"
    orch.set_mode("companion")
    assert orch.current_mode() == "companion"


def test_agent_offline_fallback():
    os.environ["V5_AGENT_MODE"] = "agent"
    try:
        out = orch.run("你好", _fallback=lambda t, **k: "AGENT_FALLBACK")
        assert out == "AGENT_FALLBACK"
    finally:
        os.environ.pop("V5_AGENT_MODE", None)


def test_companion_offline_fallback():
    out = orch.run("你好", _fallback=lambda t, **k: "COMP_FALLBACK")
    assert out == "COMP_FALLBACK"


def test_tool_registry_populated():
    assert len(orch.get_tools()) >= 24


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
