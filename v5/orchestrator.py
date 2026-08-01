# 详细说明见 docs/scripts/core/v5/v5/orchestrator.md

from __future__ import annotations

import json
import logging
import os
import socket
import re
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger("ikaros.v5.orchestrator")

# ── Path bootstrap ──────────────────────────────────────────────────────────
# orchestrator.py lives in Ikaros-memory/v5/  -> V5_ROOT = Ikaros-memory/
V5_ROOT = Path(__file__).resolve().parent
if str(V5_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(V5_ROOT.parent))

# IKAROS_ROOT (set by Ikaros-environment or inferred from directory structure)
# orchestrator.py is at core/memory_v5/v5/orchestrator.py → V5_ROOT = core/memory_v5/
# Ikaros project root = V5_ROOT.parent.parent (after 2026-07-24 normalization)
IKAROS_ROOT = Path(os.environ.get("IKAROS_ROOT", str(V5_ROOT.parent.parent)))
_BIN_DIR = IKAROS_ROOT / "bin"
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))


# ── Global runtime state ─────────────────────────────────────────────────────
class OrchestratorState:
    """Shared mutable state for the running orchestrator.

    Kept deliberately tiny.  ``mode`` mirrors V5_AGENT_MODE; ``session_id``
    is a per-conversation id; ``context_budget`` caps how many chars of a
    tool result we feed back into the observe step.
    """

    def __init__(self) -> None:
        self.mode: str = "companion"
        self.session_id: str = ""
        self.context_budget: int = 2000

    def refresh(self) -> None:
        self.mode = current_mode()


state = OrchestratorState()


def estimate_tokens(text: str) -> int:
    """Rough token count: CJK chars ~1 token, ASCII words ~1.3 tokens.

    Used to cap how much of a tool result we feed back into the observe
    step, so a huge JSON blob can't blow the LLM context window (the old
    ``context_budget`` was a raw char count and wildly off for CJK text).
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_words = len(text.encode("ascii", "ignore").split())
    return int(cjk + ascii_words * 1.3)


# ── Mode resolution ──────────────────────────────────────────────────────────
def current_mode() -> str:
    """Resolve the active runtime mode from V5_AGENT_MODE (default companion)."""
    m = os.environ.get("V5_AGENT_MODE", "companion").strip().lower()
    return "agent" if m == "agent" else "companion"


def set_mode(mode: str) -> None:
    """Override the mode at runtime (also reflected back into the env)."""
    mode = mode.strip().lower()
    state.mode = "agent" if mode == "agent" else "companion"
    os.environ["V5_AGENT_MODE"] = state.mode


# ── Lazy imports ─────────────────────────────────────────────────────────────
def _require_module(module_path: str):
    try:
        return __import__(module_path, fromlist=["__all__"])
    except Exception as exc:  # noqa: BLE001
        logger.debug("require_module failed for %s: %s", module_path, exc)
        return None


# Cache of name -> callable for all v5_* tools.
_TOOLS: dict | None = None


def get_tools() -> dict:
    """Return a {name: callable} map of every registered v5_* tool.

    Loaded lazily and memoized so importing the orchestrator is cheap and
    never raises even if the tool layer has issues.
    """
    global _TOOLS
    if _TOOLS is not None:
        return _TOOLS
    tools: dict = {}
    mod = _require_module("v5.tools")
    if mod is not None:
        for name in getattr(mod, "__all__", []):
            if name == "__all__":
                continue
            fn = getattr(mod, name, None)
            if callable(fn):
                tools[name] = fn
    _TOOLS = tools
    return tools


# ── Local LLM helper (OpenAI-compatible, stdlib only) ───────────────────────
def local_llm_chat(
    system_text: str,
    user_text: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: float = 60,
) -> str | None:
    """Best-effort chat completion.

    Model source (after the local :8080 small-model was removed from V5,
    2026-07-26):
    - ``V5_LLM_OVERRIDE_BASE_URL`` set → use the forwarded Studio model config
      (cloud or any OpenAI-compatible endpoint).
    - Otherwise → no model available; return ``None`` so callers degrade
      gracefully (the agent loop falls back to companion delegation).
    Returns the assistant content, or ``None`` on any failure. Never raises.
    """
    try:
        base_url = os.environ.get("V5_LLM_OVERRIDE_BASE_URL", "").strip()
        if base_url:
            # Studio model mode: use the forwarded config
            api_key = os.environ.get("V5_LLM_OVERRIDE_API_KEY", "").strip()
            model = os.environ.get("V5_LLM_OVERRIDE_MODEL", "local")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            url = base_url.rstrip("/")
            if not url.endswith("/chat/completions"):
                url = url.rstrip("/v1") + "/v1/chat/completions"
        else:
            # 本地小模型已从 V5 剔除 (2026-07-26): 无 override 即无可用模型
            return None

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        logger.debug("local_llm_chat failed: %s", exc)
        return None


def _parse_json_obj(text: str | None) -> dict | None:
    """Extract the first JSON object from an LLM reply, tolerantly."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to finding the first balanced {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ── Agent loop stages ───────────────────────────────────────────────────────
def _think(user_text: str) -> tuple[str | None, dict]:
    """Ask the LLM to pick a single tool + args. Returns (tool_name, args)."""
    tools = get_tools()
    if not tools:
        return None, {}
    tool_list = "\n".join(f"  - {name}" for name in sorted(tools))
    system = (
        "You are the Ikaros V5 cognitive agent. You may call exactly ONE of "
        "these tools to help answer the user:\n"
        f"{tool_list}\n\n"
        "Decide the single most relevant tool and its keyword arguments. "
        "Respond with ONLY a JSON object of the form "
        '{"tool": "v5_xxx", "args": {...}}. '
        'If no tool is relevant, respond {"tool": null}. '
        "Do not add any explanation or markdown fences."
    )
    # Inject contextually relevant operation rules (semantic retrieval via :8587)
    try:
        from v5.rules_retriever import retrieve_relevant_rules
        rules_block = retrieve_relevant_rules(user_text)
        if rules_block:
            system += "\n\n" + rules_block
    except Exception:
        pass
    raw = local_llm_chat(system, user_text, max_tokens=300, temperature=0.1)
    obj = _parse_json_obj(raw)
    if not obj:
        return None, {}
    tool_name = obj.get("tool")
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    if tool_name not in tools:
        return None, {}
    return tool_name, args


def _observe(user_text: str, tool_result: str, *, max_tokens: int = 200) -> str | None:
    """Synthesize a natural reply from the user message + tool result."""
    if not tool_result:
        return None
    # Token-aware truncation: keep the most useful slice under ~1500 tokens
    # (CJK-aware) instead of a raw char budget.  Halve until under the limit.
    TOKEN_LIMIT = 1500
    full = tool_result
    while estimate_tokens(full) > TOKEN_LIMIT and len(full) > 100:
        full = full[: len(full) // 2]
    capped = full
    system = (
        "You are Ikaros, a warm companion AI. Given the user's message and a "
        "tool result (JSON), write a concise, natural-language reply that "
        "weaves the useful information in. Do not mention 'the tool'."
    )
    user = f"User said: {user_text}\n\nTool result:\n{capped}"
    return local_llm_chat(system, user, max_tokens=max_tokens, temperature=0.7)


def agent_loop(
    user_text: str,
    *,
    history=None,
    session_id: str = "",
    max_tokens: int = 200,
    _fallback=None,
) -> str:
    """think -> tool_call -> observe.

    Any failure in the agent path degrades to the companion pipeline
    (or to ``_fallback`` when supplied — used by tests to stay offline).
    """
    state.session_id = session_id or state.session_id

    # Input validation: check user input before processing
    from v5.validation import validate_input, check_and_log
    if not check_and_log(user_text, validate_input, context="agent_loop"):
        logger.warning("agent_loop: input validation failed for session=%s", session_id)

    # 1) think: LLM chooses a tool.
    tool_name, args = _think(user_text)
    if tool_name is None:
        return _companion(
            user_text, history=history, session_id=session_id,
            max_tokens=max_tokens, _fallback=_fallback,
        )

    # 2) tool_call: invoke the selected v5_* tool.
    tool_fn = get_tools().get(tool_name)
    if tool_fn is None:
        return _companion(
            user_text, history=history, session_id=session_id,
            max_tokens=max_tokens, _fallback=_fallback,
        )
    try:
        raw_result = tool_fn(**args)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent tool %s raised: %s", tool_name, exc)
        raw_result = None

    # 3) observe: synthesize a reply from the tool result.
    reply = _observe(user_text, raw_result or "", max_tokens=max_tokens)
    if reply is None:
        return _companion(
            user_text, history=history, session_id=session_id,
            max_tokens=max_tokens, _fallback=_fallback,
        )
    return reply.strip()


# ── Companion delegation (unchanged cloud_chat pipeline) ────────────────────
def _companion(
    user_text: str,
    *,
    history=None,
    session_id: str = "",
    max_tokens: int = 200,
    _fallback=None,
) -> str:
    """Delegate to cloud_chat. If ``_fallback`` is provided, use it instead
    (test hook so offline tests never touch the network)."""
    if _fallback is not None:
        return _fallback(user_text, history=history, session_id=session_id)
    try:
        from cloud_chat import cloud_chat_sync

        return cloud_chat_sync(
            user_text,
            history=history,
            session_id=session_id,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("companion delegation failed: %s", exc)
        reason = (str(exc) or "").strip()
        if reason:
            # Surface the underlying cause so the failure is self-diagnosing
            # instead of a silent "zoned out" message.
            return (
                "（伊卡洛斯走神了片刻…本地对话管线暂时不可用："
                f"{reason}。但我的记忆与状态都还在。）"
            )
        return (
            "（伊卡洛斯走神了片刻…本地对话管线暂时不可用，"
            "但我的记忆与状态都还在。）"
        )


# ── Public entry point ──────────────────────────────────────────────────────
def run(
    user_text: str,
    *,
    history=None,
    session_id: str = "",
    max_tokens: int = 200,
    mode: str | None = None,
    _fallback=None,
) -> str:
    """Run one turn through the orchestrator.

    ``mode`` overrides the env-resolved mode for this call only.
    ``_fallback`` is a test hook: when set, companion delegation uses it
    instead of the real cloud_chat pipeline.
    """
    active = (mode or current_mode()).strip().lower()
    state.mode = "agent" if active == "agent" else "companion"
    state.session_id = session_id or state.session_id

    if state.mode == "agent":
        try:
            return agent_loop(
                user_text, history=history, session_id=session_id,
                max_tokens=max_tokens, _fallback=_fallback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent loop failed, degrading to companion: %s", exc)
    return _companion(
        user_text, history=history, session_id=session_id,
        max_tokens=max_tokens, _fallback=_fallback,
    )


# Convenience async variant (companion delegates to the async cloud_chat).
async def run_async(
    user_text: str,
    *,
    history=None,
    session_id: str = "",
    max_tokens: int = 200,
    mode: str | None = None,
) -> str:
    active = (mode or current_mode()).strip().lower()
    if active == "agent":
        # Agent loop is sync (LLM calls are blocking); run it directly.
        return run(
            user_text, history=history, session_id=session_id,
            max_tokens=max_tokens, mode="agent",
        )
    try:
        from cloud_chat import cloud_chat

        return await cloud_chat(
            user_text, history=history, session_id=session_id,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("async companion delegation failed: %s", exc)
        return run(
            user_text, history=history, session_id=session_id,
            max_tokens=max_tokens, mode="companion",
        )


if __name__ == "__main__":
    # Tiny REPL for manual smoke testing (companion mode by default).
    print(f"[orchestrator] mode={current_mode()}  session tools={len(get_tools())}")
    try:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            print("ikaros>", run(line, session_id="cli"))
    except Exception:
        pass
