# 详细说明见 docs/scripts/core/v5/v5/summary.md
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.summary")

V5_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = V5_ROOT / "data" / "v5" / "summary_cache.json"

_COMPRESS_SYSTEM = (
    "你是记忆压缩器。把以下对话压缩为 {n} 句关键要点，直接叙述，像真实记忆一样。"
    "保留：人物偏好、决策、情感变化、学到的东西。剔除：寒暄、客套、重复。"
    "严禁出现\"用户说\"\"助手说\"等元描述词汇，直接陈述事实。"
    "禁止任何开场白——第一句必须直接是要点，不许出现\"好的/我需要/首先/让我\"等引导语。"
)


def _defaults() -> dict:
    return {"trigger_rounds": 20, "reuse_rounds": 10, "max_age_rounds": 30,
            "model": "local-llm", "max_sentences": 3, "timeout_s": 5}


def _cfg() -> dict:
    try:
        from v5 import preprocess_config as pc
        return pc.cfg()["summary"]
    except Exception:
        return _defaults()


def _load_cache() -> dict:
    try:
        if _CACHE_PATH.is_file():
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"last_summary": "", "last_round": -1}


def _save_cache(d: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _fmt_turn(turn: dict) -> str:
    role = (turn or {}).get("role", "")
    content = (turn or {}).get("content", "")
    if role == "user":
        return f"用户: {content}"
    if role == "assistant":
        return f"助手: {content}"
    return f"{role}: {content}"


def _compress(old_turns: list[str], max_sentences: int, timeout: int) -> Optional[str]:
    if not old_turns:
        return None
    text = "\n".join(old_turns)
    try:
        from v5.reflect.llm_client import call_llm
        resp = call_llm(
            _COMPRESS_SYSTEM.format(n=max_sentences),
            f"对话记录：\n{text}",
            provider="deepseek", max_tokens=200, temperature=0.2, timeout=timeout,
        )
        out = (resp.content or "").strip()
        return out or None
    except Exception as e:
        logger.warning("summary compress failed: %s", e)
        return None


def _block(summary: str) -> str:
    return f"\n---\n近期对话摘要：\n{summary}"


def build_summary_block(history: list[dict] | None = None,
                        *, round_index: int | None = None) -> str:
    """返回摘要块 (空字符串 = 跳过). 非阻塞, 失败静默."""
    cfg = _cfg()
    trigger = int(cfg["trigger_rounds"])
    reuse = int(cfg["reuse_rounds"])
    max_age = int(cfg["max_age_rounds"])
    max_sent = int(cfg["max_sentences"])
    to = int(cfg["timeout_s"])

    turns = history or []
    n = len(turns)
    cache = _load_cache()

    if round_index is None:
        round_index = n // 2  # 近似轮次 (每轮 ~2 条)

    # 复用 / 丢弃
    if cache.get("last_round", -1) >= 0:
        gap = round_index - cache["last_round"]
        if gap < reuse:
            return _block(cache["last_summary"]) if cache["last_summary"] else ""
        if gap > max_age:
            cache = {"last_summary": "", "last_round": -1}

    # 不足阈值: 若已有有效摘要(未超龄)仍注入, 否则不注入
    if n < trigger:
        if cache.get("last_summary"):
            return _block(cache["last_summary"])
        return ""

    # 生成: 保留最近 trigger 条, 压缩更早的旧轮
    keep = min(n, trigger)
    old = turns[: max(0, n - keep)]
    old_turns = [_fmt_turn(t) for t in old]
    try:
        compressed = _compress(old_turns, max_sent, to)
    except Exception as e:
        logger.warning("summary generation failed: %s", e)
        compressed = None
    if compressed:
        cache["last_summary"] = compressed
        cache["last_round"] = round_index
        _save_cache(cache)
        return _block(compressed)
    # 生成失败: 回退已有缓存
    if cache.get("last_summary"):
        return _block(cache["last_summary"])
    return ""


# 内联说明见 docs/scripts/core/v5/v5/summary.md（见“内联注释摘录”）

# 后台重算 inflight 锁 (非阻塞 try-acquire, 避免并发重算)
_SUMMARY_REGEN_LOCK = threading.Lock()


def build_summary_block_nb(history: list[dict] | None = None,
                           *, round_index: int | None = None) -> str:
    """非阻塞摘要块 (供 cloud_chat 热路径).

    与 build_summary_block 行为一致, 但达到触发阈值且缓存陈旧时,
    **不**原地等 LLM, 立即返回上次缓存摘要(无则空)并在后台异步重算写回缓存.

    Returns:
        str: 摘要块 (空字符串 = 跳过).
    """
    cfg = _cfg()
    trigger = int(cfg["trigger_rounds"])
    reuse = int(cfg["reuse_rounds"])
    max_age = int(cfg["max_age_rounds"])

    turns = history or []
    n = len(turns)
    cache = _load_cache()
    if round_index is None:
        round_index = n // 2

    # 复用窗口内: 直接返回缓存 (同步, 无 LLM)
    if cache.get("last_round", -1) >= 0:
        gap = round_index - cache["last_round"]
        if gap < reuse:
            return _block(cache["last_summary"]) if cache["last_summary"] else ""
        if gap > max_age:
            cache = {"last_summary": "", "last_round": -1}

    # 不足阈值: 有有效缓存仍注入
    if n < trigger:
        if cache.get("last_summary"):
            return _block(cache["last_summary"])
        return ""

    # 达到阈值且缓存陈旧 → 后台异步重算, 本次立即返回旧缓存(或空)
    _schedule_summary_regen(turns, round_index, cfg)
    return _block(cache["last_summary"]) if cache.get("last_summary") else ""


def _schedule_summary_regen(turns: list[dict], round_index: int, cfg: dict) -> None:
    """后台 daemon 线程: 压缩旧轮 + 写回缓存. 带 inflight 锁避免并发重算."""
    if not _SUMMARY_REGEN_LOCK.acquire(blocking=False):
        return  # 已有重算在飞, 跳过
    keep = min(len(turns), int(cfg["trigger_rounds"]))
    old = turns[: max(0, len(turns) - keep)]
    old_turns = [_fmt_turn(t) for t in old]
    max_sent = int(cfg["max_sentences"])
    to = int(cfg["timeout_s"])

    def _run() -> None:
        try:
            compressed = _compress(old_turns, max_sent, to)
            if compressed:
                c = _load_cache()
                c["last_summary"] = compressed
                c["last_round"] = round_index
                _save_cache(c)
        except Exception as exc:
            logger.debug("summary background regen failed: %s", exc)
        finally:
            _SUMMARY_REGEN_LOCK.release()

    threading.Thread(target=_run, daemon=True, name="summary-regen").start()


if __name__ == "__main__":
    demo = [{"role": "user", "content": f"消息{i}"} for i in range(25)]
    print(build_summary_block(demo, round_index=12))
