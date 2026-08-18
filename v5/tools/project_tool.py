"""project_tool — V5「项目轨」记忆工具。

在 V5 人格/情感记忆之上，增加一层「项目记忆」：架构决策、踩坑、约定、
待办思路。它们与情感记忆同库（v5.db），但通过专用 tag 域区分：

  v5_domain:project       项目轨总域
  v5_kind:decision        架构决策（"为什么这么做"）
  v5_kind:pitfall         踩过的坑（"别这么干"）
  v5_kind:convention      协作/代码约定（"必须这么做"）
  v5_kind:idea            待验证的思路
  v5_project:<name>       具体项目名（如 ikaros / website-x）

用途（兼顾情感做超级项目）：
- 存：做项目时把决策/坑/约定即时写入，带情感上下文（pad 三轴可选）
- 查：新任务开始或中途用 v5_project_retrieve 检索相关项目经验，
     自动带回"哥哥偏好 + 历史决策 + 踩过的坑"，让记忆连续。

实现：复用 V5MemoryAPI.store/search（结构化 tag 精确匹配，无 ChromaDB
依赖），不新增数据库、不碰情感轨。
"""
from __future__ import annotations

from v5.tools.utils import safe_tool, dumps, answer


# ── 写入 ─────────────────────────────────────────────────────────────

@safe_tool
def v5_project_note(
    content: str,
    kind: str = "decision",
    project: str = "ikaros",
    tags: str = "",
    weight: float = 0.7,
    pad_p: float = 0.0,
    pad_a: float = 0.0,
    pad_d: float = 0.0,
) -> str:
    """Store a project memory (decision / pitfall / convention / idea).

    Project memory rides on the same v5.db as emotional memory but is scoped
    by project tags, so it never pollutes the persona track and can be
    retrieved exactly later.

    Args:
        content: the memory text, e.g. "V5 永远留在 SQLite，不迁移图数据库（用户拍板）"
        kind: decision | pitfall | convention | idea
        project: project name (default ikaros)
        tags: extra comma-separated tags (e.g. "hermes,mcp")
        weight: importance 0-1 (decisions/pitfalls default higher)
        pad_p/pad_a/pad_d: optional emotional valence if the memory carries affect
    """
    kind = (kind or "decision").strip().lower()
    valid = {"decision", "pitfall", "convention", "idea"}
    if kind not in valid:
        return answer(
            error=f"kind 必须是 {sorted(valid)} 之一，收到: {kind!r}",
            hint="decision=架构决策 / pitfall=踩坑 / convention=约定 / idea=待验证思路",
        )

    tag_set = [t for t in (tags or "").split(",") if t.strip()]
    tag_set += [f"v5_project:{project}", f"v5_kind:{kind}"]

    # V5 memory type 白名单不含 pitfall/convention；用合法类型存储，
    # kind 语义靠 v5_kind:<kind> tag 保留（检索时按 tag 精确命中）。
    _TYPE_MAP = {
        "decision": "decision",
        "pitfall": "lesson",      # 踩坑≈教训
        "convention": "fact",     # 约定≈事实性约束
        "idea": "thought",
    }

    from v5.memory_api import V5MemoryAPI

    api = V5MemoryAPI()
    mid = api.store(
        content=content,
        memory_type=_TYPE_MAP[kind],
        domain="project",
        tags=tag_set,
        importance=max(0.0, min(1.0, weight)),
        pad_p=float(pad_p),
        pad_a=float(pad_a),
        pad_d=float(pad_d),
    )
    # V5.7: 自动建类型化项目边 (fire-and-forget, 纯规则无 LLM)
    try:
        import threading
        from v5.project_edges import auto_link_project_note
        threading.Thread(
            target=auto_link_project_note,
            args=(mid, content, kind, project),
            daemon=True,
        ).start()
    except Exception:
        pass
    return dumps({"ok": True, "memory_id": mid, "kind": kind, "project": project})


# ── 检索 ─────────────────────────────────────────────────────────────

@safe_tool
def v5_project_retrieve(
    project: str = "ikaros",
    kind: str = None,
    query: str = None,
    top_k: int = 8,
    with_links: bool = False,
) -> str:
    """Retrieve project memory: decisions / pitfalls / conventions / ideas.

    Returns the most relevant project memories (optionally filtered by kind)
    so an agent starting or resuming work on a project can immediately see
    the accumulated "why", "don't", and "must" — without re-deriving them.

    Args:
        project: project name (default ikaros)
        kind: optional filter — decision | pitfall | convention | idea | None(全部)
        query: optional keyword to narrow results
        top_k: max memories to return (default 8)
        with_links: True 时每条附带类型化邻居 (SOLVES/PREVENTS/CAUSED_BY/RELATES_TO)
    """
    from v5.memory_api import V5MemoryAPI

    api = V5MemoryAPI()
    tags = [f"v5_project:{project}"]
    if kind:
        kind = kind.strip().lower()
        valid = {"decision", "pitfall", "convention", "idea"}
        if kind not in valid:
            return answer(
                error=f"kind 必须是 {sorted(valid)} 之一，收到: {kind!r}",
                hint="留空则返回全部项目记忆",
            )
        tags.append(f"v5_kind:{kind}")

    results = api.search(
        query=query,
        domain="project",
        tags=tags,
        top_k=top_k,
        min_score=0.0,
    )
    if not results:
        return dumps({"ok": True, "count": 0, "items": [],
                      "note": "该项目暂无记忆，可先用 v5_project_note 记录第一条"})

    items = []
    for r in results:
        # kind 从 v5_kind:<kind> tag 还原（存储时 pitfall→lesson 等已映射）
        _tags = [t for t in (r.get("tags") or "").split(",") if t.strip()]
        _kind = next((t.split(":", 1)[1] for t in _tags
                      if t.startswith("v5_kind:")), r.get("type"))
        _item = {
            "id": r.get("id"),
            "kind": _kind,
            "content": (r.get("content") or "")[:200],
            "weight": r.get("weight") or r.get("importance"),
            "created": r.get("created"),
        }
        # P8 可观测性: why 说明 (结构化精确命中 + kind)
        try:
            from v5.memory_retrieval import explain_result
            _item["why"] = explain_result({**r, "kind": _kind, "source": "structured"})
        except Exception:
            pass
        if with_links:
            try:
                from v5.project_edges import traverse
                _item["links"] = traverse(r.get("id"), depth=1)
            except Exception:
                _item["links"] = []
        items.append(_item)
    return dumps({"ok": True, "count": len(items), "items": items})


# ── 项目概览 ─────────────────────────────────────────────────────────

@safe_tool
def v5_project_stats(project: str = "ikaros") -> str:
    """Summarize project memory by kind (decision/pitfall/convention/idea counts)."""
    from v5.memory_api import V5MemoryAPI

    api = V5MemoryAPI()
    out = {}
    for kind in ("decision", "pitfall", "convention", "idea"):
        res = api.search(
            domain="project",
            tags=[f"v5_project:{project}", f"v5_kind:{kind}"],
            top_k=100,
            min_score=0.0,
        )
        out[kind] = len(res)
    return dumps({"ok": True, "project": project, "counts": out})
