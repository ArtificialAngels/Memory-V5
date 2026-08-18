"""类型化项目知识边 (graph-memory 借鉴) — V5.7 (2026-08-14).

在 v5_project_note 写入时自动为项目笔记建**类型化关系边**，让 pi 检索时可沿
"这个坑怎么解的"（pitfall → SOLVES ← decision）、"这个约定避免哪个坑"
（convention → PREVENTS → pitfall）扩散。

与实体图 (eg_edges, 实体↔实体共现) 不同，这里是**笔记↔笔记**的类型化关系，
落在 project_edges 表 (见 store.PROJECT_EDGES_SCHEMA)。纯规则建边，零 LLM 成本：

  - 关系类型: SOLVES (决策/思路→坑) / PREVENTS (约定→坑) / CAUSED_BY (坑→决策)
              / RELATES_TO (同项目通用关联)
  - 建边条件: 同项目 + kind 规则命中 + 关键词重叠 >= MIN_OVERLAP (共享 token 数)
"""

from __future__ import annotations

import logging

logger = logging.getLogger("ikaros.v5.project_edges")

RELATIONS = ("SOLVES", "PREVENTS", "CAUSED_BY", "RELATES_TO")

# (新笔记 kind, 既有笔记 kind) → 有向关系 (source=新笔记, target=既有笔记)
KIND_RULES: dict[tuple[str, str], str] = {
    ("decision", "pitfall"): "SOLVES",      # 决策解决坑
    ("idea", "pitfall"): "SOLVES",          # 思路解决坑
    ("convention", "pitfall"): "PREVENTS",  # 约定避免坑
    ("pitfall", "decision"): "CAUSED_BY",   # 坑由决策导致
}

MIN_OVERLAP = 2          # 共享 token (中文 2-gram / ASCII 词) 数下限
                         # 2 而非 3: 技术笔记常靠 1-2 个英文关键词(llama-server/
                         # SIGSEGV)关联, 3 会漏掉这类真关联
MAX_LINKS_PER_WRITE = 5  # 每次写入最多建边数 (防一个热门主题刷爆)


def _tokens(text: str) -> list[str]:
    """切词 (复用 skill_store._tokens: ASCII 词原样 + 中文 2-gram)。"""
    try:
        from v5.skill_store import _tokens as _tk
        return _tk(text)
    except Exception:
        return [t for t in (text or "").lower().split() if len(t) >= 2]


def _kind_from_tags(tags: str) -> str:
    """从 tags 还原 kind (v5_kind:<kind>)。"""
    for t in (tags or "").split(","):
        t = t.strip()
        if t.startswith("v5_kind:"):
            return t.split(":", 1)[1]
    return ""


def auto_link_project_note(memory_id: int, content: str, kind: str,
                           project: str) -> int:
    """为新写入的项目笔记自动建类型化边 (纯规则, 无 LLM)。返回建边数。

    扫描同项目既有笔记, 命中 kind 规则且关键词重叠 >= MIN_OVERLAP 则建边。
    失败/无命中静默返回 0 (不阻塞写入)。
    """
    from v5 import store

    new_tokens = set(_tokens(content))
    if len(new_tokens) < MIN_OVERLAP:
        return 0
    try:
        with store.conn() as c:
            rows = c.execute(
                "SELECT id, content, tags FROM memory "
                "WHERE archived = 0 AND tags LIKE ? AND id != ? "
                "ORDER BY weight DESC, id DESC LIMIT 200",
                (f"%v5_project:{project}%", int(memory_id)),
            ).fetchall()
    except Exception as exc:
        logger.debug("project_edges: scan failed (%s)", exc)
        return 0

    created = 0
    for r in rows:
        if created >= MAX_LINKS_PER_WRITE:
            break
        other_kind = _kind_from_tags(r["tags"])
        if not other_kind or other_kind == kind:
            continue
        relation = KIND_RULES.get((kind, other_kind), "RELATES_TO")
        overlap = len(new_tokens & set(_tokens(r["content"] or "")))
        if overlap < MIN_OVERLAP:
            continue
        weight = min(0.9, 0.4 + 0.1 * overlap)
        if store.link_project_edge(memory_id, r["id"], relation, weight=weight):
            created += 1
    if created:
        logger.info("project_edges: auto-linked %d edge(s) for memory %s",
                    created, memory_id)
    return created


def traverse(memory_id: int, depth: int = 1,
             relation: str | None = None) -> list[dict]:
    """沿类型化项目边扩散 (1 跳 BFS), 返回邻居记忆 + 关系。

    返回 [{id, relation, direction(out/in), content, kind, weight}]。
    direction: out = 本记忆是 source (如 decision→pitfall), in = 是 target。
    """
    from v5 import store

    if depth < 1:
        return []
    edges = store.get_project_edges(memory_id)
    if relation:
        edges = [e for e in edges if e["relation"] == relation]
    out: list[dict] = []
    for e in edges:
        is_out = int(e["source_id"]) == int(memory_id)
        neighbor_id = e["target_id"] if is_out else e["source_id"]
        content, kind = "", ""
        try:
            m = store.get(neighbor_id)
            content = getattr(m, "content", "") or ""
            kind = _kind_from_tags(getattr(m, "tags", "") or "")
        except Exception:
            pass
        out.append({
            "id": neighbor_id,
            "relation": e["relation"],
            "direction": "out" if is_out else "in",
            "content": (content or "")[:200],
            "kind": kind,
            "weight": e["weight"],
        })
    return out


def project_graph_search(query: str, top_k: int = 5,
                         project: str | None = None) -> list[dict]:
    """项目知识图检索: 找项目笔记 + 沿类型化边扩散邻居。

    供 unified_retrieve(graph scope) 复用, 让通用检索也能沿"坑↔决策"边扩散
    (推荐 4 补充: 项目知识图与实体图并轨进 graph scope)。
    返回 [{id, content, type, weight, score, source, relation, kind}]。
    """
    from v5.memory_api import V5MemoryAPI
    try:
        api = V5MemoryAPI()
        tags = [f"v5_project:{project}"] if project else None
        results = api.search(query=query, domain="project", tags=tags,
                             top_k=max(1, top_k), min_score=0.0)
    except Exception as exc:
        logger.debug("project_graph_search: search failed (%s)", exc)
        return []
    if not results:
        return []

    out: list[dict] = []
    seen: set = set()
    for r in results:
        rid = r.get("id")
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        kind = _kind_from_tags(r.get("tags") or "")
        out.append({
            "id": str(rid), "content": r.get("content") or "",
            "type": r.get("type") or "fact",
            "weight": float(r.get("weight") or 0.7),
            "score": 0.8, "source": "project_graph",
            "relation": "self", "kind": kind,
        })
        for nb in traverse(int(rid), depth=1):
            nid = nb["id"]
            if nid in seen:
                continue
            seen.add(nid)
            out.append({
                "id": str(nid), "content": nb["content"],
                "type": "fact", "weight": float(nb.get("weight") or 0.5),
                "score": 0.5, "source": "project_graph",
                "relation": nb["relation"], "kind": nb["kind"],
            })
    return out[: max(1, top_k * 3)]
