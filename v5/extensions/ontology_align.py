"""
ontology_align.py — V5 轻量本体对齐 (EXPERIMENTAL / 零 LLM 成本)
================================================================

借鉴 cognee modules/ontology 的 FuzzyMatchingStrategy (difflib + cutoff) 模式,
但用 V5 自己的资源 (eg_entities / eg_aliases), 不引 RDFLib:

  - align_entity(surface, threshold): 对实体规范名 + 别名做 difflib 模糊匹配,
    低于阈值返回 None (不误配) —— 对齐 cognee cutoff 0.8 并略保守取 0.82。
  - find_entity_candidates_fuzzy(): exact/contains 优先 (复用 entity_graph),
    不足时 difflib 补召回 —— 可替换 entity_graph.find_entity_candidates。
  - alias_extract(text): 规则抽取 "X（又称Y/也叫Y/aka Y）" 别名对写 eg_aliases。
  - add_alias(entity_id, alias): 幂等写别名。

全部标准库 (difflib / re), fail-open, 不阻塞主线。
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Optional

logger = logging.getLogger("ikaros.v5.ext.ontology_align")

# 对齐 cognee FuzzyMatchingStrategy cutoff 0.8, 略保守 (防误配)
DEFAULT_THRESHOLD = 0.82


# ── 归一化: 全角→半角 / 大小写 / 去空白与装饰 ─────────────────
def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # 全角 → 半角 (ASCII 区)
    s = "".join(
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
        for c in s
    )
    # 去掉括号内容 (如 "伊卡洛斯 (Ikaros)" → "伊卡洛斯") 与常见装饰
    s = re.sub(r"[（(].*?[)）]", "", s)
    s = re.sub(r"[\s_\-—]+", "", s)
    return s


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ── 1) 实体模糊对齐 ───────────────────────────────────────────
def align_entity(
    surface: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[dict]:
    """对 surface 做实体模糊对齐, 返回最佳匹配 dict 或 None (低于阈值不猜测).

    Returns: {"entity_id", "canonical_name", "similarity"} 或 None.
    """
    ns = _norm(surface)
    if not ns:
        return None
    try:
        from v5.entity_graph import eg_conn
    except Exception:
        return None
    best: Optional[dict] = None
    best_score = 0.0
    try:
        with eg_conn() as c:
            rows = c.execute(
                "SELECT id, canonical_name FROM eg_entities"
            ).fetchall()
            alias_rows = c.execute(
                "SELECT a.entity_id, a.alias, e.canonical_name "
                "FROM eg_aliases a JOIN eg_entities e ON e.id = a.entity_id"
            ).fetchall()
    except Exception as e:
        logger.debug("ontology_align: load failed (%s)", e)
        return None
    for r in rows:
        sc = _ratio(ns, _norm(r["canonical_name"]))
        if sc > best_score:
            best_score = sc
            best = {"entity_id": r["id"], "canonical_name": r["canonical_name"],
                    "similarity": sc}
    for r in alias_rows:
        sc = _ratio(ns, _norm(r["alias"]))
        if sc > best_score:
            best_score = sc
            best = {"entity_id": r["entity_id"],
                    "canonical_name": r["canonical_name"], "similarity": sc}
    if best_score >= threshold:
        return best
    return None


# ── 2) 模糊候选 (替代 entity_graph.find_entity_candidates) ─────
def find_entity_candidates_fuzzy(
    surface: str,
    top_k: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
) -> list:
    """exact/contains 优先 (复用 entity_graph), 不足时 difflib 补召回.

    返回与 entity_graph.EntityCandidate 同构的对象列表 (可直接喂
    spreading_activation_search). 纯 additive: 不改变原函数行为.
    """
    out: list = []
    try:
        from v5.entity_graph import (
            find_entity_candidates, EntityCandidate,
        )
        out = list(find_entity_candidates(surface))
    except Exception as e:
        logger.debug("ontology_align: base candidates failed (%s)", e)
        out = []
    if len(out) >= top_k:
        return out[:top_k]
    # 补路: difflib 匹配 (排除已命中的 entity_id)
    seen = {getattr(x, "entity_id", None) for x in out}
    hit = align_entity(surface, threshold=threshold)
    if hit and hit["entity_id"] not in seen:
        try:
            from v5.entity_graph import EntityCandidate
            out.append(EntityCandidate(
                entity_id=hit["entity_id"],
                canonical_name=hit["canonical_name"],
                entity_type="", description="",
                episodic_count=0, similarity=hit["similarity"],
            ))
            seen.add(hit["entity_id"])
        except Exception:
            pass
    return out[:top_k]


# ── 3) 别名抽取 (规则, 零 LLM) ────────────────────────────────
# 只支持括号形式: "X（Y）" / "X（又称Y）" / "X (aka Y)" —— 要求闭合括号,
# 避免 "X, 系统Y" 这类句法片段被误配 (别名组不含空白, 防把整句抓进来)
_ALIAS_RE = re.compile(
    r"(?P<main>[A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff\-·]{0,40})"
    r"\s*[（(]\s*(?:又称|也叫|又名|aka|a\.k\.a\.)?\s*"
    r"(?P<alias>[A-Za-z0-9\u4e00-\u9fff\-·]{1,40})"
    r"\s*[)）]"
)


def alias_extract(text: str) -> list[tuple[str, str]]:
    """从文本抽 "X（又称Y）" 别名对, 返回 [(main, alias), ...] 去重."""
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    for m in _ALIAS_RE.finditer(text):
        main = (m.group("main") or "").strip()
        alias = (m.group("alias") or "").strip()
        if main and alias and len(alias) <= 40:
            pair = (main, alias)
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def add_alias(entity_id: str, alias: str) -> bool:
    """幂等写别名到 eg_aliases."""
    alias = alias.strip()
    if not alias or not entity_id:
        return False
    try:
        from v5.entity_graph import eg_conn
        with eg_conn() as c:
            exists = c.execute(
                "SELECT 1 FROM eg_aliases WHERE entity_id = ? AND alias = ?",
                (entity_id, alias),
            ).fetchone()
            if not exists:
                c.execute(
                    "INSERT INTO eg_aliases (entity_id, alias) VALUES (?, ?)",
                    (entity_id, alias),
                )
                # eg_conn 退出默认 rollback → 显式提交
                c.commit()
        return True
    except Exception as e:
        logger.debug("ontology_align: add_alias failed (%s)", e)
        return False
