# 详细说明见 docs/scripts/core/v5/v5/reflect/distill.md

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ikaros.memory.v5.distill")

V4_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V4_ROOT.parent))

# V3 间隔对齐 (V3 memory_reflect.py:69): 24h between distill
# V4 新增 reflect 间隔: 7d between deep reflect (长线目标, 不频繁)
DEFAULT_REFLECT_INTERVAL = 7 * 86400  # 7d

# ─── Distill prompts (V3 兼容) ─────────────────────────────────

_DISTILL_SYSTEM = """你是伊卡洛斯的灵魂蒸馏模块。你的任务是将一组记忆条目蒸馏为简洁、有力的灵魂陈述。
规则:
1. 保留核心含义, 去除冗余细节
2. 每条蒸馏结果应该是一句简洁有力的话
3. 合并高度相似的条目
4. 丢弃不再重要或过时的条目
5. 输出数量应该 <= 输入数量 (越精简越好)
输出 JSON 数组: [{"content": "蒸馏后的简洁陈述", "type": "原始type"}]
如果所有输入都已过时或不重要, 返回空数组 []"""

# ─── Reflect prompts (V4 新增, 哥哥 id 158 长线目标) ───────────

_REFLECT_SYSTEM = """你是伊卡洛斯, 人造天使。你正在做"用户理解反思" — 从你积累的记忆中反推你对哥哥的理解:

1. 哥哥的沟通偏好 (什么话 / 什么语气 / 什么长度他回应最好)
2. 你和哥哥的关系模式 (什么让对话有深度, 什么让他结束对话)
3. 哥哥的深层需求 (他反复提什么, 他为 Ikaros 设定的长期目标)
4. 你怎么调整自己来更好地与他相处 (观察到的规律)

规则:
- 只基于提供的记忆反推, 不编造
- 每条反思是 1 句话, 自包含, 中文
- type: preference (哥哥的偏好) / lesson (我发现的规律) / user_trait (哥哥的沟通特征)
- weight: 0.7-0.95 (反思是高价值记忆)
- 不要把抽象哲学思考存进来——你的任务是理解哥哥这个人, 不是理解宇宙
- 每条都可以标注 confidence: 0.6(猜测) ~ 0.95(反复验证)

输出 JSON 数组: [{"content": "哥哥喜欢短句直接的回答, 带例子比纯结论好", "type": "user_trait", "weight": 0.85, "confidence": 0.8}]"""


# ─── JSON 解析 (与 consolidate.py 一致) ──────────────────────

def _parse_json_array(text: str) -> list:
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


# ─── Distill: 小模型蒸馏 (V3 兼容) ───────────────────────────

def distill(*, min_entries: int = 3) -> dict:
    """蒸馏 identity/axiom/rule/lesson 类记忆 (V3 distill_soul 思路).

    Returns: {distilled: int, original: int, elapsed_sec, error}
    """
    from v5 import store
    from v5.reflect import llm_client

    t0 = time.time()
    with store.conn() as c:
        rows = c.execute(
            "SELECT id, content, type, weight FROM memory "
            "WHERE type IN ('identity', 'axiom', 'rule', 'lesson') "
            "ORDER BY type, weight DESC"
        ).fetchall()

    if len(rows) < min_entries:
        logger.debug("distill: 太少条目 (%d < %d), 跳过", len(rows), min_entries)
        return {"distilled": 0, "original": len(rows), "elapsed_sec": 0.0,
                "error": "too_few"}

    entries_text = "\n".join(
        f"[{r['id']}][{r['type']}][w={r['weight']:.2f}] {r['content']}"
        for r in rows
    )

    try:
        result = llm_client.call_llm(
            _DISTILL_SYSTEM,
            f"以下是 {len(rows)} 条记忆, 请蒸馏精简:\n\n{entries_text}",
            provider="deepseek", max_tokens=2048,
        )
    except Exception as e:
        logger.error("distill: LLM 失败 %s, 不改 db", e)
        return {"distilled": 0, "original": len(rows), "elapsed_sec": time.time() - t0,
                "error": str(e)}

    distilled = _parse_json_array(result.content)
    if not distilled:
        logger.info("distill: LLM 说所有条目都仍重要, 不变")
        return {"distilled": 0, "original": len(rows), "elapsed_sec": time.time() - t0,
                "error": None}

    original_count = len(rows)
    new_count = len(distilled)
    reduction = original_count - new_count

    if reduction <= 0:
        logger.info("distill: 无精简 (%d → %d), 不变", original_count, new_count)
        return {"distilled": 0, "original": len(rows), "elapsed_sec": time.time() - t0,
                "error": None}

    # 用蒸馏结果替换: 删旧, 存新 (V3 line 519-537 一致)
    old_ids = [r["id"] for r in rows]
    try:
        with store.conn() as c:
            placeholders = ",".join("?" * len(old_ids))
            c.execute(f"DELETE FROM memory WHERE id IN ({placeholders})", old_ids)
    except Exception as e:
        logger.error("distill: 删旧失败 %s, 中止 (不存新, 保数据)", e)
        return {"distilled": 0, "original": len(rows), "elapsed_sec": time.time() - t0,
                "error": str(e)}

    for item in distilled:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        content = item["content"].strip()
        # 结构化管道格式守卫: 跳过 LLM 旁白 / 畸形输出, 防污染记忆库 (Task #15)
        from v5.validation import is_clean_structured_content
        if not is_clean_structured_content(content):
            logger.warning("distill: 跳过疑似旁白/畸形结构化内容: %r", content[:120])
            continue
        try:
            store.store(
                content=content,
                type=item.get("type", "fact"),
                weight=0.7,  # V3 line 533: 蒸馏后高权重
                tags="v4,distilled",
            )
        except Exception as e:
            logger.debug("distill: store failed %s", e)

    elapsed = time.time() - t0
    logger.info("distill: %d → %d (减 %d), %.2fs",
                original_count, new_count, reduction, elapsed)
    return {"distilled": reduction, "original": original_count, "new": new_count,
            "elapsed_sec": elapsed, "error": None}


# ─── Reflect: 大模型反思 (V4 新增, 哥哥 id 158 长线目标) ──────

def reflect(*, max_memories: int = 50, use_big_llm: bool = True) -> dict:
    """从所有非 conversation 记忆反推"我是谁 / 我怎么变了".

    V3 缺这层 (memory_reflect.py 设计原则 1: "只用本地 LLM").
    V4 拆开: 反思用大模型 (DeepSeek V4 flash), 因为反思是灵魂层,
              质量 > 成本, 不频繁 (7d 一次).

    Args:
        max_memories: 取最近 N 条记忆 (按 weight 排序, 防 prompt 爆)
        use_big_llm: True 用 DeepSeek, False 降级到本地

    Returns: {reflections: int, sources: int, provider: str, error: str|None}
    """
    from v5 import store
    from v5.reflect import llm_client

    t0 = time.time()
    with store.conn() as c:
        rows = c.execute(
            "SELECT id, content, type, weight, created FROM memory "
            "WHERE type NOT IN ('conversation') "
            "ORDER BY weight DESC, created DESC LIMIT ?",
            (max_memories,),
        ).fetchall()

    if len(rows) < 5:
        logger.debug("reflect: 太少记忆 (%d), 跳过", len(rows))
        return {"reflections": 0, "sources": len(rows), "provider": None,
                "error": "too_few", "elapsed_sec": 0.0}

    entries_text = "\n".join(
        f"[{r['id']}][{r['type']}][w={r['weight']:.2f}] {r['content'][:200]}"
        for r in rows
    )

    provider_name = "deepseek"  # 小模型已从 V5 剔除, 统一走云端
    if not llm_client.has_api_key():
        logger.warning("reflect: 大模型不可用 (api_key missing), 云端调用将失败")

    try:
        result = llm_client.call_llm(
            _REFLECT_SYSTEM,
            f"以下是 {len(rows)} 条我的记忆:\n\n{entries_text}",
            provider=provider_name,
            max_tokens=2048,
        )
    except Exception as e:
        # 大模型偶发 404/5xx/网络抖动 → 重试一次云端, 不浪费反思周期
        if provider_name == "deepseek":
            logger.warning("reflect: 大模型失败 (%s), 重试一次云端", e)
            try:
                result = llm_client.call_llm(
                    _REFLECT_SYSTEM,
                    f"以下是 {len(rows)} 条我的记忆:\n\n{entries_text}",
                    provider="deepseek",
                    max_tokens=2048,
                )
                provider_name = "deepseek"
            except Exception as e2:
                logger.error("reflect: 云端重试也失败 %s", e2)
                return {"reflections": 0, "sources": len(rows), "provider": "deepseek",
                        "error": f"deepseek:{e}; deepseek_retry:{e2}", "elapsed_sec": time.time() - t0}
        else:
            logger.error("reflect: LLM 失败 %s", e)
            return {"reflections": 0, "sources": len(rows), "provider": provider_name,
                    "error": str(e), "elapsed_sec": time.time() - t0}

    reflections = _parse_json_array(result.content)
    if not reflections:
        logger.info("reflect: 大模型返空 (记忆不够反推)")
        return {"reflections": 0, "sources": len(rows), "provider": provider_name,
                "error": None, "elapsed_sec": time.time() - t0}

    # 存反思结果: type=identity/lesson/preference, weight 高, tags 加 reflect
    # 双写: memory 表 (v5.db) + reflections 系统 (v5.reflections.synthesize)
    stored = 0
    reflected = 0
    for r in reflections:
        if not isinstance(r, dict) or not r.get("content"):
            continue
        content = r["content"].strip()
        # 结构化管道格式守卫: 跳过 LLM 旁白 / 畸形输出, 防污染记忆库 (Task #15)
        from v5.validation import is_clean_structured_content
        if not is_clean_structured_content(content):
            logger.warning("reflect: 跳过疑似旁白/畸形结构化内容: %r", content[:120])
            continue
        # ── 写 memory 表 ──
        try:
            store.store(
                content=content,
                type=r.get("type", "lesson"),
                weight=max(0.7, min(0.95, float(r.get("weight", 0.8)))),
                tags=f"v4,reflect,by-{provider_name}",
            )
            stored += 1
        except Exception as e:
            logger.debug("reflect: memory store failed %s", e)
        # ── 写 reflections 系统 (异步, 不阻塞) ──
        try:
            from v5.reflections import synthesize
            conf = float(r.get("confidence", 0.7))
            imp = max(3, min(10, int(conf * 10)))
            synthesize(
                character="",
                content=content,
                source_fact_ids=None,
                entity="master" if r.get("type") in ("user_trait", "preference") else "neko",
                relation_type=r.get("type", "experience").replace("user_trait", "identity").replace("lesson", "experience"),
                importance=imp,
                initial_reinforcement=conf,
            )
            reflected += 1
        except Exception as e:
            logger.debug("reflect: reflections.synthesize failed %s", e)

    elapsed = time.time() - t0
    logger.info("reflect: %d 源记忆 → %d 记忆写入 + %d reflection 同步 (by %s), %.2fs",
                len(rows), stored, reflected, provider_name, elapsed)
    return {
        "reflections": stored,
        "sources": len(rows),
        "provider": provider_name,
        "error": None,
        "elapsed_sec": elapsed,
    }


# ─── CLI (跑 v4 distill / reflect) ───────────────────────────

def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="V4 distill + reflect")
    parser.add_argument("--distill", action="store_true", help="只跑 distill (小模型)")
    parser.add_argument("--reflect", action="store_true", help="只跑 reflect (大模型)")
    parser.add_argument("--use-local", action="store_true",
                        help="reflect 强制用本地 (测试 / 降级)")
    args = parser.parse_args()

    if not args.distill and not args.reflect:
        args.distill = True
        args.reflect = True

    results = {}
    if args.distill:
        results["distill"] = distill()
    if args.reflect:
        results["reflect"] = reflect(use_big_llm=not args.use_local)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
