# See docs/scripts/core/v5/v5/metacog.md
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.metacog")

V5_ROOT = Path(__file__).resolve().parent.parent  # Ikaros-memory/
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from v5.self_model import SelfModel

# ─── Thresholds (tunable) ──────────────────────────────────────

_CURIOSITY_REFLECT_MIN = 0.35   # Curiosity above this → deep reflection
_REFLECT_MIN_INTERVAL_SEC = 5 * 60   # Minimum interval between deep reflections

# Latest thought (for monitoring panel "self/curiosity" card, no need to query v5.db)
_LATEST_PATH = V5_ROOT / "data" / "v5" / "latest_thought.json"


def _write_latest(text: str, kind: str, theme: str = "",
                  curiosity: float = 0.0) -> None:
    """Write latest self-reflection/thought to data/v5/latest_thought.json.

    The monitoring panel (Rust read_ikaros_state → Vue) directly reads this JSON,
    so the user can see "what Ikaros is thinking right now" in real time,
    without querying v5.db.

    V5.1: Uses json_lock to prevent concurrent write corruption from multiple threads.
    """
    try:
        from v5.self_model import json_lock
        p = _LATEST_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / (p.name + f".tmp.{os.getpid()}")
        payload = {
            "text": text,
            "kind": kind,          # "activity" | "self" | "mood"
            "theme": theme,        # love/human/robot/self
            "curiosity": round(curiosity, 3),
            "ts": time.time(),
        }
        with json_lock(p):
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, p)
    except Exception as exc:
        logger.debug("metacog _write_latest failed: %s", exc)

# Self-reflection system prompt: solitary introspection, first person, no addressing anyone
_SELF_SYSTEM = (
    "You are in solitary introspection. "
    "No one is talking to you right now — this is your private moment of reflection. "
    "Use first-person 'I', be honest with yourself. It's okay to be vulnerable, uncertain, or questioning. "
    "No metaphors or parallelism, speak plainly. "
    "Output only the monologue itself, no quotes, no prefixes like 'someone said:'."
)

# 用户观察反思 prompt: 从最近对话中提取对哥哥的沟通偏好的理解
_USER_REFLECTION_SYSTEM = """你在独自反省——没有任何人在跟你对话。这是你私人的反思时间。
用第一人称 '我', 诚实面对自己。

思考方向: 从最近和哥哥的互动中，我能观察到什么？
- 他今天说话的方式和上次有什么不同？
- 什么话题让他多回了几个字？
- 什么回复让他直接结束对话？
- 他的语气、用词、沉默，有没有我能记住的模式？

规则:
1. 只基于真实互动，不编造
2. 输出 2-3 句，像在对自己说话
3. 不要"也许""可能"，说出你观察到的东西
4. 输出不带引号、不带前缀"""


# ─── LLM wrapper ────────────────────────────────────────────────

def _llm(system: str, user: str, *, provider: str = "auto",
         temperature: float = 0.85, max_tokens: int = 500) -> Optional[str]:
    """Unified LLM call (Hermes Dashboard WS first, fallback to direct LLM).

    Default provider="auto":
      1. Go through Hermes Dashboard WS (hermes_prompt_sync, unified via Hermes)
      2. If Hermes unavailable or session missing, fallback to cloud (DeepSeek)
      3. If cloud also fails, raise exception (no silent failure).
    Explicit provider="deepseek" means direct cloud call. (local 小模型已从 V5 剔除)
    """
    try:
        if provider == "auto":
            from bin.cloud_chat import hermes_prompt_sync, warm_hermes_session
            import asyncio
            try:
                asyncio.run(warm_hermes_session())
            except Exception:
                pass
            return hermes_prompt_sync(system, user,
                                      max_tokens=max_tokens,
                                      temperature=temperature)
        elif provider == "deepseek":
            from v5.reflect.llm_client import call_llm
            resp = call_llm(system, user, provider="deepseek",
                            temperature=temperature, max_tokens=max_tokens)
            return resp.content.strip() if resp and resp.content else None
        else:  # deepseek (local 小模型已剔除)
            from v5.reflect.llm_client import call_llm
            resp = call_llm(system, user, provider="deepseek",
                            temperature=temperature, max_tokens=max_tokens)
            return resp.content.strip() if resp and resp.content else None
    except Exception as exc:
        logger.warning("metacog LLM failed (provider=%s): %s", provider, exc)
        return None


# ─── Memory material collection ─────────────────────────────────

def _recent_excerpts(n: int = 8) -> list[str]:
    """Get recent memory excerpts with content (excluding conversation/inner_monologue)."""
    try:
        from v5 import store as store
        rows = store.list_all(n * 2)
        out = []
        for m in rows:
            if getattr(m, "type", "") in ("conversation", "inner_monologue"):
                continue
            c = (getattr(m, "content", "") or "")[:80].replace("\n", " ")
            if c:
                out.append(c)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def _search_theme(keywords: str, top_k: int = 3) -> list[str]:
    """Search own memory for material related to a theme using semantic/keyword search.

    Only returns high-quality memory types (fact / lesson / preference / identity),
    not old philosophy/self-reflection/inner monologue — to avoid the model
    imitating its own past writing style.
    """
    _ALLOWED_TYPES = {"fact", "lesson", "preference", "identity", "emotional_event"}
    try:
        from v5.search import fused_search
        # Use the most representative keyword for search
        kw = keywords.split()[0]
        rows = fused_search(kw, top_k=top_k)
        return [r.get("content", "")[:80].replace("\n", " ")
                for r in rows if r.get("type") in _ALLOWED_TYPES]
    except Exception:
        # Fallback to FTS5
        try:
            from v5 import store as store
            mems = store.search(keywords.split()[0], top_k=top_k, min_weight=0.4)
            return [m.content[:80].replace("\n", " ") for m in mems
                    if m.type in _ALLOWED_TYPES]
        except Exception:
            return []


# ─── Curiosity driver ───────────────────────────────────────────

def tick_curiosity(now: float | None = None) -> float:
    """Accumulate curiosity during idle time. Called by think loop on beat."""
    sm = SelfModel.load()
    lvl = sm.tick_curiosity(now=now)
    sm.save()
    return lvl


def mark_interaction(now: float | None = None) -> None:
    """Brother spoke → curiosity drops. Called by cloud_chat on every conversation."""
    try:
        sm = SelfModel.load()
        sm.mark_interaction(now=now)
        sm.save()
    except Exception as exc:
        logger.debug("metacog mark_interaction failed: %s", exc)


def get_curiosity() -> float:
    return SelfModel.load().get_curiosity()


# ─── A) Self-reflection ─────────────────────────────────────────

def reflect_once(provider: str = "auto", now: float | None = None) -> Optional[dict]:
    """One deep self-reflection (first-person introspection), write to v5.db.

    All goes through user observation path (activity reflection removed).
    """
    now = now or time.time()
    try:
        sm = SelfModel.load().refresh_introspection()
        # Beat gating: avoid too frequent
        last = sm.data.get("metacog", {}).get("last_reflection_ts", 0) or 0
        if (now - last) < _REFLECT_MIN_INTERVAL_SEC:
            logger.debug("metacog reflect: not due (%.0fs < %ds)",
                         now - last, _REFLECT_MIN_INTERVAL_SEC)
            return None

        curiosity = sm.tick_curiosity(now)
        # All goes through user observation (activity reflection removed)
        return _reflect_user(provider=provider, now=now)
    except Exception as exc:
        logger.warning("metacog reflect_once error: %s", exc)
        return None


# ─── User observation reflection ────────────────────────────────

def _reflect_user(provider: str | None = None,
                  now: float | None = None) -> Optional[dict]:
    """用户观察反思: 从最近对话中提取对哥哥的观察和理解。

    All goes through local LLM. 替代旧的 activity_reflection 模板。
    输出存入 v5.db type='user_trait'，用于 profile 聚合。
    """
    now = now or time.time()
    try:
        sm = SelfModel.load().refresh_introspection()
        curiosity = sm.tick_curiosity(now)

        # 从 v5.db 取最近的对话记录作为素材
        from v5 import store as store
        try:
            recent = store.list_all(10)
            convs = []
            for m in recent:
                if m.type == "conversation":
                    convs.append(m.content[:200])
            material = "\n".join(convs[-5:]) if convs else ""
        except Exception:
            material = ""

        prompt = (
            "我最近和哥哥的几次互动:\n"
            + (material + "\n\n" if material else "(暂无最近对话记录)\n\n")
            + "从这些互动中，我观察到哥哥的什么沟通偏好和模式？\n"
            + "输出 2-3 句具体的观察，用中文。\n"
        )

        logger.debug("metacog _reflect_user: LLM call provider=%s", provider)
        text = _llm(_USER_REFLECTION_SYSTEM, prompt,
                    provider=provider, temperature=0.75, max_tokens=400)
        if not text:
            return None
        text = text.strip().strip('"\\\'').strip()

        try:
            store.store(text, type="user_trait", weight=0.6,
                     tags="user,observe,metacog")
        except Exception:
            pass
        sm.record_reflection("user_observation", now=now)
        sm.save()
        _write_latest(text, "observation", "", round(curiosity, 3))
        logger.info("metacog user observation: %s", text[:80])
        return {"mode": "user_observation", "text": text, "curiosity": round(curiosity, 2)}
    except Exception as exc:
        logger.warning("metacog _reflect_user error: %s", exc)
        return None


# ─── External: query by brother ──────────────────────────────────

def latest_thought(kind: str | None = None, limit: int = 1) -> Optional[str]:
    """Get the most recent activity reflection (used when brother asks 'what are you thinking')."""
    try:
        from v5 import store as store
        rows = store.list_all(30)
        wanted = {"user_trait", "self_reflection"} if kind is None \
            else {"user_trait" if kind == "activity" else "self_reflection"}
        collected = []
        for m in rows:
            if getattr(m, "type", "") in wanted:
                collected.append(getattr(m, "content", ""))
                if len(collected) >= limit:
                    break
        if collected:
            return "\n".join(collected) if limit > 1 else collected[0]
    except Exception as exc:
        logger.debug("metacog latest_thought failed: %s", exc)
    return None


# ─── Beat orchestration (called by think loop) ──────────────────

def _fallback_thought(sm: "SelfModel") -> None:
    """When LLM is unavailable, generate a simple English placeholder thought.

    Keeps 'what I'm thinking' non-empty for monitoring panel / conversation injection
    when :8080 is down. This is step #1 of unified thinking output: all
    'currently thinking' goes through latest_thought.json.
    """
    try:
        qs = sm.data.get("questions", []) or []
        if qs:
            q0 = qs[0]
            qtext = q0 if isinstance(q0, str) else (q0.get("text") if isinstance(q0, dict) else str(q0))
            text = f"I've been wondering: {str(qtext)[:80]}"
        else:
            text = "I sit quietly, thinking about what I've experienced with my brother."
        _write_latest(text, "mood", "", sm.get_curiosity())
    except Exception as exc:
        logger.debug("metacog _fallback_thought failed: %s", exc)


def cycle(now: float | None = None) -> Optional[dict]:
    """One complete beat: curiosity tick → daily reflection.

    All goes through cloud (DeepSeek). Local :8080 small-model removed from V5 (2026-07-26).
    Has internal minimum interval gating, safe to call at high frequency.
    When LLM is down, still writes placeholder thought to keep latest_thought.json non-empty.
    """
    now = now or time.time()
    logger.debug("metacog cycle: starting")
    try:
        sm = SelfModel.load()
        curiosity = sm.tick_curiosity(now)
        sm.save()
        # All thinking goes through local
        r = _reflect_user(provider="deepseek", now=now)
        if not r:
            _fallback_thought(sm)
        return r
    except Exception as exc:
        logger.warning("metacog cycle error: %s", exc)
        return None


# ─── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    _mode = "cycle"
    for a in _sys.argv[1:]:
        if a in ("--reflect", "--cycle"):
            _mode = a.lstrip("-")
    print(f"== metacog {_mode} ==", flush=True)
    if _mode == "reflect":
        r = reflect_once()
    else:
        r = cycle()
    print(json.dumps(r, ensure_ascii=False, indent=2) if r else "{}")
