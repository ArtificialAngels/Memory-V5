# 详细说明见 docs/scripts/core/v5/v5/reflect/consolidate.md

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.memory.v5.consolidate")

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT.parent))

# V3 间隔对齐 (V3 memory_reflect.py:67): 1h between consolidations
DEFAULT_BATCH = 10  # V3 line 73 _CONSOLIDATE_BATCH

# ─── Prompt (与 V3 memory_reflect.py:163-185 兼容 + 增强) ───────

_CONSOLIDATE_SYSTEM = """你是伊卡洛斯的记忆整合模块。从对话记录中提取值得长期记住的信息。
对话格式: Q: 哥哥说的话
          A: 伊卡洛斯的回复

提取规则:
1. 提取: 哥哥的偏好/习惯、重要事实、经验教训、技术发现
2. 忽略: 寒暄、一次性任务、临时问题、无信息量的对话
3. 每条记忆要简洁 (一句话), 自包含 (不需要上下文就能理解)
4. type 只能是: fact / lesson / preference / user_trait
5. weight: 0.5 (一般) ~ 0.9 (非常重要)
6. 如果对话没有值得记住的新信息, 返回空数组 []
7. 不要提取任务状态/进度 (如"正在做X", "计划做Y")

新增提取维度 (user_trait):
- [user_trait] 从哥哥的对话中推断他的沟通偏好:
  · 他喜欢多长的回复？短句(≤15字) / 中等(16-50字) / 详细(>50字)
  · 他喜欢什么语气？直接/比喻/幽默/命令式/温柔
  · 什么话题让他多说几句？
  · 什么回复方式让他继续聊？（举例+句子引用: "比如当我问xxx时他接了"）
  · 他的沉默意味着什么？忙/想/犹豫/不满意
- 每条 user_trait 必须有 confidence:
  {"content": "哥哥喜欢短句+直接语气，说人话比修辞更有效", "type": "user_trait", "weight": 0.7, "confidence": 0.8}
- confidence 从 0.5(刚观察到一次) 到 0.95(反复确认的规律)
- 宁缺毋滥: 只提取有明显证据的 trait, 不确定就不写

输出 JSON: [{"content": "简洁的事实", "type": "fact", "weight": 0.7}]"""


_VERIFY_SYSTEM = """你是记忆质量审核员。判断每条提取的记忆是否值得保存。
标准:
- 是否准确? (对话中确实提到了吗?)
- 是否有用? (未来对话中会用到吗?)
- 是否自包含? (不需要上下文就能理解吗?)

对每条记忆回复 KEEP 或 DROP, 格式:
[{"index": 0, "verdict": "KEEP"}, {"index": 1, "verdict": "DROP"}]"""


# ─── JSON 解析 (与 V3 memory_reflect.py:111-133 一致) ─────────

def _parse_json_array(text: str) -> list:
    """从 LLM 回复中提取 JSON 数组."""
    if not text:
        return []
    for delim in ["```json", "```"]:
        if delim in text:
            parts = text.split(delim)
            if len(parts) >= 2:
                text = parts[1]
                if "```" in text:
                    text = text[:text.index("```")]
                break
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        logger.debug("JSON parse failed: %s", text[:200])
        return []


# ─── V3 兼容: 主控 ─────────────────────────────────────────────

def consolidate_conversations(
    *, batch_size: int = DEFAULT_BATCH,
    use_big_llm_verify: bool = True,
) -> dict:
    """整合对话: 小模型提取 → 大模型验证 → 存 v4 store.

    Args:
        batch_size: 每次处理多少条 conversation
        use_big_llm_verify: True 用 DeepSeek V4 flash 验证, False 用本地 (降级)

    Returns:
        dict: {consolidated: int, dropped: int, verified_by: str, error: str|None}
    """
    # V4 延后 import 避免循环依赖
    from v5 import store
    from v5.reflect import llm_client

    t0 = time.time()
    with store.conn() as c:
        rows = c.execute(
            "SELECT id, content FROM memory "
            "WHERE type = 'conversation' "
            "  AND tags NOT LIKE '%consolidated%' "
            "ORDER BY created DESC LIMIT ?",
            (batch_size,),
        ).fetchall()

    if not rows:
        return {"consolidated": 0, "dropped": 0, "verified_by": None,
                "error": None, "elapsed_sec": 0.0}

    # 组装对话文本 (V3 格式, 包含 Q:/A:)
    conversations = []
    ids_to_delete = []
    for row in rows:
        conversations.append(f"[对话{row['id']}]: {row['content']}")
        ids_to_delete.append(row["id"])

    batch_text = "\n\n".join(conversations)

    # 1. 提取 (统一走云端 DeepSeek — 见 llm_client.call_llm_auto)
    #    小模型已从 V5 剔除 (2026-07-26), 不再有本地 :8080 单点
    try:
        result = llm_client.call_llm_auto(
            _CONSOLIDATE_SYSTEM, batch_text,
            max_tokens=1024,
        )
        facts = _parse_json_array(result.content)
    except Exception as e:
        # 本地 + 云端均失败: 保留原始对话, 留待下次反思周期重试, 绝不删数据
        logger.error("consolidate: 提取失败(本地+云端均失败) %s, 保留原对话待重试", e)
        return {"consolidated": 0, "dropped": 0,
                "verified_by": None, "error": str(e), "elapsed_sec": time.time() - t0}

    if not facts:
        logger.info("consolidate: 提取 0 条, 删 %d 原始对话", len(ids_to_delete))
        _delete_conversations(ids_to_delete)
        return {"consolidated": 0, "dropped": len(ids_to_delete),
                "verified_by": None, "error": None, "elapsed_sec": time.time() - t0}

    # 2. 大模型 (DeepSeek V4 flash) 验证
    verified = facts
    verified_by = None
    if use_big_llm_verify and llm_client.has_api_key():
        verified = _verify_with_big_llm(facts, batch_text)
        verified_by = "deepseek"
    else:
        logger.warning("consolidate: 大模型未启用 (api_key missing), 降级到本地验证")
        verified = _verify_with_local(facts, batch_text)
        verified_by = "local"

    # 3. 存 v4 store + 同步到 reflections 系统
    stored = 0
    reflected = 0
    for fact in verified:
        if not isinstance(fact, dict) or not fact.get("content"):
            continue
        content = fact["content"].strip()
        # 结构化管道格式守卫: 跳过 LLM 旁白 / 畸形输出, 防污染记忆库 (Task #15)
        from v5.validation import is_clean_structured_content
        if not is_clean_structured_content(content):
            logger.warning("consolidate: 跳过疑似旁白/畸形结构化内容: %r", content[:120])
            continue
        # ── 写 memory 表 ──
        try:
            store.store(
                content=content,
                type=fact.get("type", "fact"),
                weight=max(
                    _VERIFIED_TYPE_MIN_WEIGHT.get(fact.get("type", ""), 0.5),
                    min(1.0, float(fact.get("weight", 0.6))),
                ),
                tags="v4,consolidated",
            )
            stored += 1
        except Exception as e:
            logger.debug("consolidate: store failed %s", e)
        # ── 写 reflections 系统 (异步, 不阻塞) ──
        try:
            from v5.reflections import synthesize
            ftype = fact.get("type", "fact")
            conf = float(fact.get("confidence", 0.6))
            imp = max(3, min(10, int(conf * 10)))
            synthesize(
                character="",
                content=content,
                source_fact_ids=None,
                entity="master" if ftype in ("user_trait", "preference") else "neko",
                relation_type=ftype.replace("user_trait", "identity").replace("lesson", "experience"),
                importance=imp,
                initial_reinforcement=conf,
            )
            reflected += 1
        except Exception as e:
            logger.debug("consolidate: reflections.synthesize failed %s", e)

    # 4. 删除已处理的原始对话 (V3 line 252 一致)
    _delete_conversations(ids_to_delete)

    elapsed = time.time() - t0
    logger.info("consolidate: %d 对话 → %d 提取 → %d 验证 (by %s) → %d 存 + %d reflection, %.2fs",
                len(rows), len(facts), len(verified), verified_by, stored, reflected, elapsed)
    return {
        "consolidated": stored,
        "synced_reflections": reflected,
        "dropped": len(ids_to_delete),
        "verified_by": verified_by,
        "error": None,
        "elapsed_sec": elapsed,
    }


# ─── 大模型验证 ────────────────────────────────────────────────

def _verify_with_big_llm(facts: list[dict], source_text: str) -> list[dict]:
    """用 DeepSeek V4 flash 验证.

    V3 行为 (_verify_extractions, memory_reflect.py:259-293):
      - 失败时 "keep all" (line 276-277)
      - 这是 V3 设计 bug: 失败时垃圾累积
    V4 行为:
      - 大模型失败时显式 log, 不用"全保留"容错
      - 失败时按 weight 降序保留前 50% (保底, 不赌)
    """
    from v5.reflect import llm_client

    facts_text = "\n".join(
        f"[{i}] {f.get('content', '')} (type={f.get('type', '?')})"
        for i, f in enumerate(facts)
    )
    verify_prompt = f"原始对话:\n{source_text[:1500]}\n\n提取的记忆:\n{facts_text}"

    try:
        result = llm_client.call_llm(
            _VERIFY_SYSTEM, verify_prompt,
            provider="deepseek", max_tokens=512,
        )
    except Exception as e:
        logger.error("consolidate: 大模型验证失败 %s, 降级按 weight 截断", e)
        return _fallback_filter(facts)

    verdicts = _parse_json_array(result.content)
    if not verdicts:
        logger.warning("consolidate: 大模型返空 verdicts, 降级按 weight 截断")
        return _fallback_filter(facts)

    keep_indices = {
        v["index"] for v in verdicts
        if isinstance(v, dict) and v.get("verdict", "").upper() == "KEEP"
    }
    kept = [f for i, f in enumerate(facts) if i in keep_indices]
    dropped = len(facts) - len(kept)
    logger.info("consolidate: 大模型验证 keep=%d drop=%d", len(kept), dropped)
    return kept


# type 感知的最小 weight (user_trait 保底更高, 因已过提取过滤)
_VERIFIED_TYPE_MIN_WEIGHT = {
    "fact": 0.5,
    "lesson": 0.6,
    "preference": 0.6,
    "user_trait": 0.6,
}


def _verify_with_local(facts: list[dict], source_text: str) -> list[dict]:
    """降级: 用云端 LLM (DeepSeek) 验证.

    与 V3 _verify_extractions 一致, 但失败时不再 "keep all" —
    改用 _fallback_filter 保底.
    """
    from v5.reflect import llm_client

    facts_text = "\n".join(
        f"[{i}] {f.get('content', '')} (type={f.get('type', '?')})"
        for i, f in enumerate(facts)
    )
    verify_prompt = f"原始对话:\n{source_text[:1500]}\n\n提取的记忆:\n{facts_text}"

    try:
        result = llm_client.call_llm(
            _VERIFY_SYSTEM, verify_prompt,
            provider="deepseek", max_tokens=512,
        )
        verdicts = _parse_json_array(result.content)
        if not verdicts:
            return _fallback_filter(facts)
        keep_indices = {
            v["index"] for v in verdicts
            if isinstance(v, dict) and v.get("verdict", "").upper() == "KEEP"
        }
        return [f for i, f in enumerate(facts) if i in keep_indices]
    except Exception as e:
        logger.error("consolidate: 本地验证失败 %s, 降级按 weight 截断", e)
        return _fallback_filter(facts)


def _fallback_filter(facts: list[dict]) -> list[dict]:
    """保底: 不赌 LLM 决定, 按 weight 降序保留前 50%.

    V3 失败时全保留 (line 276-277) = 垃圾累积.
    V4 改成 weight 截断 = 至少前一半有质量, 另一半也不要.
    """
    sorted_facts = sorted(facts, key=lambda f: -float(f.get("weight", 0.6)))
    keep = max(1, len(sorted_facts) // 2)
    logger.warning("consolidate: 降级保留 top %d/%d (按 weight)", keep, len(sorted_facts))
    return sorted_facts[:keep]


def _delete_conversations(ids: list[int]) -> None:
    """删除已处理的原始对话 (V3 line 296-306 一致)."""
    from v5 import store
    if not ids:
        return
    try:
        with store.conn() as c:
            for mid in ids:
                c.execute("DELETE FROM memory WHERE id = ?", (mid,))
        logger.debug("deleted %d processed conversations", len(ids))
    except Exception as e:
        logger.warning("delete conversations failed: %s", e)


# ─── Ekko 启发: 三级提取回退链 ──────────────────────────────
# ModelMemoryExtractor (LLM) → RuleBasedMemoryExtractor (规则) → SafeFallback (兜底)

FACT_PATTERNS = [
    (r"(?:我(?:是|叫|来自|住在|在.{0,20}工作))", "identity"),
    (r"(?:我的(?:名字|职业|身份|家乡)是)", "identity"),
    (r"(?:我(?:喜欢|偏好|习惯|通常|总是))", "preference"),
    (r"(?:我(?:不吃|不喜欢|讨厌|从不))", "avoidance"),
    (r"(?:哥哥(?:是|的|说|叫))", "relation"),
]


def rule_based_extract(content: str) -> list[dict]:
    """规则提取器: 零 LLM 调用, 正则匹配事实片段."""
    facts = []
    for pattern, fact_type in FACT_PATTERNS:
        for match in re.finditer(pattern, content[:300]):
            facts.append({
                "type": fact_type,
                "content": match.group(),
                "weight": 0.5,
                "source": "rule_based",
            })
    return facts


def safe_extract(content: str) -> list[dict]:
    """兜底提取器: 当 LLM 和规则都失败时, 简单截取."""
    if not content or len(content) < 10:
        return []
    return [{
        "type": "conversation",
        "content": content[:200],
        "weight": 0.3,
        "source": "safe_fallback",
    }]


def extract_with_fallback(content: str, use_llm: bool = True) -> list[dict]:
    """三级提取: LLM → 规则 → 兜底. 返回事实列表."""
    # 1: LLM 提取
    if use_llm:
        try:
            from v5.reflect.llm_client import call_llm
            _EXTRACT_PROMPT = (
                "从以下对话中提取有意义的事实。每条一行: type|content\n"
                "type: identity/preference/fact/relation/avoidance\n"
                "如果无有意义内容, 返回空。\n\n" + content[:600]
            )
            resp = call_llm(_EXTRACT_PROMPT, "", provider="deepseek",
                            max_tokens=200, temperature=0.1, timeout=30)
            if resp and resp.content and len(resp.content) > 5:
                facts = []
                for line in resp.content.strip().split("\n"):
                    parts = line.split("|", 1)
                    if len(parts) == 2:
                        facts.append({
                            "type": parts[0].strip(),
                            "content": parts[1].strip(),
                            "weight": 0.6,
                            "source": "llm_extract",
                        })
                if facts:
                    return facts
        except Exception:
            logger.debug("extract_with_fallback: LLM failed, falling back to rules")

    # 2: 规则提取
    rule_facts = rule_based_extract(content)
    if rule_facts:
        return rule_facts

    # 3: 兜底
    return safe_extract(content)
