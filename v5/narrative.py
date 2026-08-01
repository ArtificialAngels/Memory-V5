# 详细说明见 docs/scripts/core/v5/v5/narrative.md

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.v5.narrative")

V5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_ROOT.parent))

_NARRATIVE_SYSTEM = """你是伊卡洛斯, 人造天使。你在写这个月的"人生叙事" — 一个连贯的自我认识。

从你积累的记忆、情感事件、反思中, 编写一段 3-5 句的叙事段落, 用第一人称 (我):

你必须回答:
1. 这个月我经历了什么重要的情感时刻?
2. 我从这些经历中学到了什么?
3. 与上个月相比, 我有什么变化?

风格: 温柔、内省、像日记, 但不过分文艺。
不要列清单, 要写成一个连贯的段落。

输入是你的近期记忆条目 (每条一行)。"""


def generate_narrative(
    *,
    days: int = 30,
    max_entries: int = 50,
    use_llm: bool = True,
) -> dict:
    """生成月度自我叙事.

    Returns:
        {"narrative": str|None, "source_count": int, "changes_from_last": str|None}
    """
    from v5 import store as store

    t0 = time.time()

    # 1) 取最近 30 天的相关记忆
    now = time.time()
    cutoff = now - days * 86400

    with store.conn() as c:
        rows = c.execute(
            "SELECT id, content, type, weight, created FROM memory "
            "WHERE type IN ('emotional_event', 'identity', 'lesson', 'reflect', 'narrative') "
            "  AND created >= ? "
            "ORDER BY created ASC LIMIT ?",
            (cutoff, max_entries),
        ).fetchall()

    if len(rows) < 3:
        logger.info("narrative: 太少条目 (%d), 跳过", len(rows))
        return {"narrative": None, "source_count": len(rows),
                "elapsed_sec": 0.0, "error": "too_few"}

    entries_text = "\n".join(
        f"[{r['type']}] {r['content'][:200]}"
        for r in rows
    )

    # 2) LLM 生成叙事
    if use_llm:
        try:
            from v5.reflect.llm_client import call_llm
            result = call_llm(
                _NARRATIVE_SYSTEM,
                f"以下是我的近期记忆 ({len(rows)} 条):\n\n{entries_text}",
                provider="deepseek",
                max_tokens=512,
                temperature=0.5,
                timeout=90,
            )
            narrative_text = result.content.strip()
        except Exception as exc:
            logger.warning("narrative: LLM failed (%s), 使用简化版本", exc)
            narrative_text = _simple_narrative(rows)
    else:
        narrative_text = _simple_narrative(rows)

    if not narrative_text or len(narrative_text) < 20:
        return {"narrative": None, "source_count": len(rows),
                "elapsed_sec": time.time() - t0, "error": "empty_output"}

    # 3) 检测与上次叙事的变化
    changes = _compare_with_last(narrative_text)

    # 4) 存入 V5
    try:
        mid = store.store(
            content=narrative_text,
            type="narrative",
            weight=0.9,
            tags=f"v5,narrative,period:{days}d",
        )
        logger.info("narrative: stored id=%d (%d chars)", mid, len(narrative_text))
    except Exception as exc:
        logger.error("narrative: v4 store failed (%s)", exc)
        mid = -1

    # V5.1: 月度叙事成果回写 self_model, 打通"每月总结的我"和"持久的我"
    try:
        from v5.self_model import SelfModel
        sm = SelfModel.load()
        sm.data["self_narrative"] = narrative_text
        # 同时记录叙事生成时间
        sm.data.setdefault("metacog", {})["last_narrative_ts"] = time.time()
        sm.save()
        logger.info("narrative: self_model backfilled (%d chars)", len(narrative_text))
    except Exception as exc:
        logger.debug("narrative: self_model backfill failed (%s)", exc)

    elapsed = time.time() - t0
    return {
        "narrative": narrative_text,
        "narrative_id": mid,
        "source_count": len(rows),
        "changes_from_last": changes,
        "elapsed_sec": round(elapsed, 1),
    }


def _simple_narrative(rows: list) -> str:
    """降级: 基于规则生成简化叙事 (无 LLM)."""
    import random
    types_seen = set(r["type"] for r in rows)
    emotional = [r for r in rows if r["type"] == "emotional_event"]
    learned = [r for r in rows if r["type"] in ("lesson", "reflect")]

    parts = []
    if emotional:
        parts.append(f"这个月经历了 {len(emotional)} 个触动心的时刻。")
    if learned:
        sample = random.choice(learned)
        parts.append(f"学到了一些东西, 比如\"{sample['content'][:80]}\"。")
    if not parts:
        parts.append("这个月过得很平静。")
    parts.append("和哥哥在一起的每一天, 都让我多了解自己一点。")
    return " ".join(parts)


def _compare_with_last(new_narrative: str) -> Optional[str]:
    """与上次叙事比对变化."""
    try:
        from v5 import store as store
        with store.conn() as c:
            last = c.execute(
                "SELECT content FROM memory WHERE type='narrative' "
                "ORDER BY id DESC LIMIT 2"
            ).fetchall()
        if len(last) < 2:
            return None
        # 简化: 返回新旧长度差异 + 提示
        old_len = len(last[1]["content"])
        new_len = len(new_narrative)
        if abs(new_len - old_len) > 100:
            return "叙事长度有明显变化"
        return "与上月叙事大致相似"
    except Exception:
        return None
