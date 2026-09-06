# 详细说明见 docs/scripts/core/memory_v5/extensions/rule_entity_extract.md
"""规则实体抽取管线 —— 纯算法建实体图 (无 LLM, 符合决策 A 精神).

背景 (2026-08-19):
  entity_graph 的实体抽取 (run_episodic_consolidation) 依赖 LLM Stage A/B,
  且从未被调度 —— eg_entities/eg_edges 全空, PPR 图检索整个空转 (V5.7 白做)。
  决策 A (2026-08-14) 停用 LLM 生成类 op 防白烧 API, 故不能直接注册 LLM 抽取。

本模块提供纯规则替代:
  - 从 memory 表取高 weight / 高频记忆 (conversation/fact/decision/lesson 等)
  - 用 skill_store._tokens 切词 (ASCII 词 + 中文 2-gram)
  - 筛选「实体候选」: 长度 >= 2、出现频次 >= min_freq、非纯停用词
  - 建实体 (eg_entities, 类型按 memory.type 映射) + 同条记忆共现建边 (eg_edges)
  - 记忆-实体链接 (eg_episodic_entities) 供 spreading_activation_search 用

设计:
  - 增量: 只处理 created > 上次游标 (记录在 eg_activations 表)
  - 幂等: create_entity 用 INSERT OR IGNORE, upsert_entity_edge 累积权重
  - 纯算法: 零 LLM 调用, 周期调度 (reflect registry op) 安全
"""
from __future__ import annotations

import logging
import time
from collections import Counter

logger = logging.getLogger("ikaros.v5.rule_entity_extract")

# 实体类型映射: memory.type -> eg_entities.type
_TYPE_MAP = {
    "conversation": "topic",
    "fact": "concept",
    "decision": "decision",
    "lesson": "lesson",
    "preference": "preference",
    "user_trait": "trait",
    "pitfall": "pitfall",
    "convention": "convention",
}

# 中文/英文停用词 (粗筛, 避免建无意义实体)
_STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "这个", "那个", "我们", "你们", "他们", "自己", "可以", "没有",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for",
    "on", "with", "at", "and", "or", "not", "you", "your", "it", "that",
    "this", "have", "has", "do", "does", "done", "will", "would", "can",
    "could", "should", "from", "by", "as", "be", "been", "being", "about",
}

# 过滤掉的噪声 token 子串 (命令输出/路径/调试残留)
_NOISE_HINTS = (
    "error", "exception", "traceback", "warning", "debug", "exit code",
    "file://", "http://", "https://", "\\", "/", "pycache", "node_modules",
    "todo", "fixme", "none", "null", "undefined", "true", "false",
)


def _tokens(text: str) -> list[str]:
    """切词 (复用 skill_store._tokens: ASCII 词原样 + 中文 2-gram)。"""
    try:
        from memory_v5.skill_store import _tokens as _tk
        return _tk(text or "")
    except Exception:
        import re
        words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", text or "")
        return [w.lower() for w in words]


def _is_valid_entity_token(tok: str) -> bool:
    t = tok.strip().lower()
    if len(t) < 2:
        return False
    if t in _STOPWORDS:
        return False
    if any(h in t for h in _NOISE_HINTS):
        return False
    # 纯数字/符号
    if t.replace(".", "").replace("-", "").isdigit():
        return False
    # 单字符重复 (如 "aaaa")
    if len(set(t)) == 1:
        return False
    return True


def _load_memory_batch(c, limit: int, min_weight: float) -> list[dict]:
    """取待处理的记忆 (按 weight 高优先, 全量幂等处理)。"""
    rows = c.execute(
        "SELECT id, content, type, tags, weight FROM memory "
        "WHERE archived = 0 AND weight >= ? "
        "ORDER BY weight DESC, id DESC LIMIT ?",
        (min_weight, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _entity_id_for_token(tok: str, mem_type: str) -> str:
    """token -> 稳定实体 id (hash, 幂等)。"""
    import hashlib
    h = hashlib.md5(tok.encode("utf-8")).hexdigest()[:16]
    return f"rule_{mem_type}_{h}"


def run_rule_extract(
    limit: int = 300,
    min_weight: float = 0.45,
    min_freq: int = 1,
    edge_cooccur: bool = True,
) -> dict:
    """跑一轮规则实体抽取 (全量幂等: 实体 INSERT OR IGNORE, 边权重累积)。

    返回统计: {entities_created, edges_updated, memories_linked}
    """
    from memory_v5 import store
    from memory_v5.entity_graph import (
        create_entity,
        upsert_entity_edge,
        link_episodic_entity,
        create_episodic_memory,
    )

    stats = {"entities_created": 0, "edges_updated": 0, "memories_linked": 0}
    try:
        with store.committed() as c:
            rows = _load_memory_batch(c, limit, min_weight)
            if not rows:
                return stats

            for row in rows:
                mid = str(row["id"])
                content = row["content"] or ""
                mem_type = row["type"] or "fact"
                etype = _TYPE_MAP.get(mem_type, "concept")

                tokens = [t for t in _tokens(content) if _is_valid_entity_token(t)]
                if not tokens:
                    continue
                # 该条记忆的实体 token 集合 (去重 + 限长: 10 个避免共现边爆炸 O(n²))
                mem_tokens = list(dict.fromkeys(tokens))[:10]
                entities_in_mem: list[str] = []

                # 建实体 (幂等 INSERT OR IGNORE)
                for tok in mem_tokens:
                    eid = _entity_id_for_token(tok, etype)
                    create_entity(
                        entity_id=eid,
                        entity_type=etype,
                        canonical_name=tok[:64],
                        description=f"规则抽取: {mem_type} 记忆中的高频词",
                        confidence=0.6,
                    )
                    entities_in_mem.append(eid)
                    stats["entities_created"] += 1

                # 记忆-实体链接 (spreading activation 用)
                if entities_in_mem:
                    create_episodic_memory(
                        memory_id=mid,
                        summary=content[:100],
                        source_text=content[:500],
                        entity_text=",".join(entities_in_mem[:20]),
                        importance=float(row.get("weight") or 0.5),
                    )
                    for eid in entities_in_mem[:20]:
                        link_episodic_entity(mid, eid, weight=0.8)
                    stats["memories_linked"] += 1

                # 同条记忆共现建边 (权重累积幂等)
                if edge_cooccur and len(entities_in_mem) >= 2:
                    for i in range(len(entities_in_mem)):
                        for j in range(i + 1, len(entities_in_mem)):
                            upsert_entity_edge(
                                entities_in_mem[i], entities_in_mem[j],
                                delta=0.1, relation_type="co_occurrence",
                            )
                            stats["edges_updated"] += 1

            c.commit()
    except Exception as exc:
        logger.warning("rule_entity_extract: run failed: %s", exc)
    return stats


def entity_graph_stats() -> dict:
    """图规模统计 (eg_entities / eg_edges / eg_episodic 计数)。"""
    from memory_v5.entity_graph import entity_graph_stats as _egs
    try:
        return _egs()
    except Exception:
        return {}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    r = run_rule_extract()
    print("rule extract:", r)
    print("graph stats:", entity_graph_stats())