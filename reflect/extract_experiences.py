"""结构化经验抽取 op (OpenViking extract_loop + merge_op 借鉴) — F3.

背景: 2026-08-14 决策 A 砍掉了 consolidate/distill/reflect/narrative 等 LLM 生成 op,
因为它们"无 schema + 无合并 + 无去重"产生 579 条雷同 user_trait / 哲学叙事 / 思维链
泄漏。OpenViking 的解法是: (a) 严格 Pydantic schema (LLM 不能自由发挥哲学叙事),
(b) PATCH merge_op 编辑既有行而非新建, (c) page_id 去重。

Ikaros 适配 (更简): 抽取走严格 JSON schema (禁止哲学叙事), 写入走 store.upsert()
——upsert 本身已做"同类型相似记忆合并强化 (content 取长/weight 取高/reinforcement+)"
(P1 写策略)。所以雷同的根因在写入边界被挡掉, 而非靠 LLM 自律。

只抽 lesson/decision/preference/fact 四类可复用经验; identity/axiom 不抽 (灵魂核心
不靠 LLM 臆测)。24h 一次, 幂等 (重叠窗口的重复抽取 → upsert 合并 → reinforcement+,
不会堆积)。LLM 不可用 → fail-open 跳过。
"""

from __future__ import annotations

import json
import logging
import time

from memory_v5.reflect.scheduler import (
    DEFAULT_REFLECT_INTERVAL,
    ReflectOp,
)

logger = logging.getLogger("ikaros.v5.reflect.extract_experiences")

_EXTRACT_INTERVAL = 24 * 3600  # 24h
_MAX_CONVERSATIONS = 40
_MAX_EXTRACTS = 10
_ALLOWED_TYPES = ("lesson", "decision", "preference", "fact")

_EXTRACT_PROMPT = """从以下对话中抽取"可复用的经验性记忆"——明确的教训、决策、偏好、事实。
只抽有明确信息量、可复用的条目; 禁止哲学反思/叙事/人格臆测/思维链。

输出严格 JSON 数组, 每条:
{"type": "lesson|decision|preference|fact", "content": "<一句话, 具体可复用, 不超过80字>", "weight": <0.5-0.9>}

规则:
- 只输出 JSON 数组, 不要任何其他文字、不要 markdown 代码块标记
- content 必须具体可复用 (如 "用户偏好简洁直接的沟通, 少修辞" 而非 "用户是个务实的人")
- 禁止 identity/axiom 类臆测 (如 "我是什么样的人")
- 无可抽取条目时输出 []

对话:
{conversations}
"""


def _gather_recent_conversations() -> list[str]:
    """取近 24h 的 conversation 记忆 (archived=0)."""
    from memory_v5 import store
    now = time.time()
    try:
        with store.conn() as c:
            rows = c.execute(
                "SELECT content FROM memory "
                "WHERE type = 'conversation' AND archived = 0 "
                "  AND created > ? "
                "ORDER BY created DESC LIMIT ?",
                (now - _EXTRACT_INTERVAL, _MAX_CONVERSATIONS),
            ).fetchall()
        return [r["content"] for r in rows if (r["content"] or "").strip()]
    except Exception as exc:
        logger.debug("extract_experiences: gather failed (%s)", exc)
        return []


def _call_extract_llm(conversations: list[str]) -> list[dict]:
    """调云端 LLM 抽取, 返回结构化 ops 列表. 失败 → []."""
    if not conversations:
        return []
    from memory_v5.reflect.llm_client import call_llm
    joined = "\n---\n".join(c[:300] for c in conversations[:_MAX_CONVERSATIONS])
    prompt = _EXTRACT_PROMPT.format(conversations=joined[:4000])
    try:
        resp = call_llm(prompt, "", provider="deepseek", max_tokens=800,
                        temperature=0.0, timeout=30)
        text = (resp.content or "").strip()
    except Exception as exc:
        logger.debug("extract_experiences: LLM failed (%s)", exc)
        return []
    # 容错: 去掉可能的 ```json 代码块标记
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        ops = json.loads(text)
    except Exception as exc:
        logger.warning("extract_experiences: parse failed (%s); raw=%r", exc, text[:200])
        return []
    if not isinstance(ops, list):
        return []
    return ops


def _apply_extracts(ops: list[dict]) -> int:
    """把抽取的结构化经验走 upsert 写入 (合并强化, 不堆积雷同)."""
    from memory_v5 import store
    written = 0
    for op in ops[:_MAX_EXTRACTS]:
        if not isinstance(op, dict):
            continue
        mtype = str(op.get("type", "")).strip().lower()
        if mtype not in _ALLOWED_TYPES:
            continue
        content = str(op.get("content", "")).strip()
        if not content or len(content) > 200:
            continue
        try:
            weight = float(op.get("weight", 0.6))
        except (TypeError, ValueError):
            weight = 0.6
        weight = max(0.5, min(0.9, weight))
        try:
            # upsert: 同类型相似记忆合并强化 (雷同在写入边界挡掉, 非靠 LLM 自律)
            # tags 标记来源, 便于追溯; 不带 v5_key: → 走合并路径 (相似则 _merge_into)
            store.upsert(
                content=content,
                type=mtype,
                weight=weight,
                tags="v5_extracted",
                reinforcement=0.0,
            )
            written += 1
        except Exception as exc:
            logger.debug("extract_experiences: upsert failed for %r (%s)", content[:60], exc)
    return written


def make_extract_experiences_op() -> ReflectOp:
    """结构化经验抽取 op (24h): conversation → lesson/decision/preference/fact."""

    def _fn() -> int:
        convs = _gather_recent_conversations()
        if not convs:
            return 0
        ops = _call_extract_llm(convs)
        if not ops:
            return 0
        written = _apply_extracts(ops)
        if written:
            logger.info("extract_experiences: wrote %d structured memories (via upsert merge)", written)
        return written

    return ReflectOp(
        name="extract_experiences",
        fn=_fn,
        interval_sec=_EXTRACT_INTERVAL,
        last_run_key="last_extract_experiences",
    )
