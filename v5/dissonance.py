# 详细说明见 docs/scripts/core/v5/v5/dissonance.md

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.dissonance")

V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V5_ROOT))

# 只对 fact/preference 类记忆做失调检测 (identity 太核心, 先不做)
_CHECK_TYPES = {"fact", "preference"}
# 最小语义相似度阈值 (低于此值的旧记忆不纳入比较)
_MIN_SIMILARITY = 0.4

_NLI_PROMPT = """你是认知失调检测器。判断新信息是否与旧记忆矛盾。

- 如果新信息直接否定/推翻/与旧记忆不可调和 → "contradiction"
- 如果一致或互补 → "entailment"
- 如果毫无关系 → "neutral"

只输出一个词: entailment / contradiction / neutral

旧记忆: {old}
新信息: {new}
"""


def detect_dissonance(
    content: str,
    mem_type: str = "fact",
    *,
    top_k: int = 5,
    min_similarity: float = _MIN_SIMILARITY,
) -> dict:
    """检测新记忆是否与已有记忆矛盾。

    Returns:
        {"conflicts": [...], "checked": int, "elapsed_ms": float}
    """
    t0 = time.time()

    if mem_type not in _CHECK_TYPES:
        return {"conflicts": [], "checked": 0, "elapsed_ms": 0}

    # 1) 语义搜索相似旧记忆
    try:
        from v5.search import fused_search
        similar = fused_search(content, top_k=top_k)
    except Exception as exc:
        logger.debug("dissonance: search failed (%s)", exc)
        return {"conflicts": [], "checked": 0, "elapsed_ms": (time.time() - t0) * 1000}

    if not similar:
        return {"conflicts": [], "checked": 0, "elapsed_ms": (time.time() - t0) * 1000}

    # 过滤低相似度
    candidates = [
        s for s in similar
        if s.get("score", 0) >= min_similarity and s.get("content", "") != content
    ]

    if not candidates:
        return {"conflicts": [], "checked": len(similar), "elapsed_ms": (time.time() - t0) * 1000}

    # 2) 对每个候选做 NLI
    conflicts = []
    for cand in candidates[:3]:  # 最多检查 3 条
        old_text = cand.get("content", "")
        if not old_text:
            continue
        verdict = _nli_check(old_text, content)
        if verdict == "contradiction":
            conflicts.append({
                "old_id": cand.get("id"),
                "old_content": old_text[:200],
                "old_type": cand.get("type", "?"),
                "score": cand.get("score", 0),
            })

    elapsed_ms = (time.time() - t0) * 1000

    # 3) 发现矛盾 → 写入 V5 + 返回
    if conflicts:
        _record_dissonance(content, conflicts)

    return {
        "conflicts": conflicts,
        "checked": len(candidates),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def _nli_check(old_text: str, new_text: str) -> str | None:
    """用云端 LLM (DeepSeek) 做 NLI 判断."""
    try:
        from v5.reflect.llm_client import call_llm
    except Exception:
        return None

    prompt = _NLI_PROMPT.format(old=old_text[:300], new=new_text[:300])
    try:
        result = call_llm(prompt, "", provider="deepseek", max_tokens=16,
                          temperature=0.0, timeout=20)
        text = result.content.strip().lower()
        if "entailment" in text:
            return "entailment"
        if "contradiction" in text:
            return "contradiction"
        return "neutral"
    except Exception as exc:
        logger.debug("dissonance: NLI failed (%s)", exc)
        return None


def _record_dissonance(new_content: str, conflicts: list[dict]) -> None:
    """记录认知失调事件到 V4."""
    try:
        from v5 import store as store
        old_summaries = "; ".join(
            [c["old_content"][:60] for c in conflicts[:2]]
        )
        content = (
            f"我注意到一个新信息与之前的记忆矛盾。"
            f"新: {new_content[:100]}  旧: {old_summaries}"
        )
        store.store(
            content=content,
            type="dissonance",
            weight=0.8,
            tags="v5,dissonance",
        )
        logger.info("dissonance: recorded %d conflicts", len(conflicts))
    except Exception as exc:
        logger.debug("dissonance: v4 store failed (%s)", exc)
