from __future__ import annotations

# 详细说明见 docs/scripts/core/v5/v5/tools/memory_tool.md

import json

from v5.tools.utils import safe_tool, dumps, answer


@safe_tool
def v5_memory_store(
    content: str,
    type: str = "fact",
    weight: float = 0.6,
    tags: str = "",
    domain: str = None,
    category_path: str = None,
    key: str = None,
    importance: float = 0.5,
    pad_p: float = 0.0,
    pad_a: float = 0.0,
    pad_d: float = 0.0,
) -> str:
    """Store a memory (V5 native + optional Ekko-style structured fields).

    Structured fields are encoded into tags:
      v5_domain:<domain>  v5_cat:<category_path>  v5_key:<key>
    so they can be retrieved exactly later (see memory_api / v5_memory_search).

    Optional: runs dissonance.detect_dissonance() for fact/preference types.
    """
    tag_set = [t for t in (tags or "").split(",") if t]
    if domain:
        tag_set.append(f"v5_domain:{domain}")
    if category_path:
        tag_set.append(f"v5_cat:{category_path}")
    if key:
        tag_set.append(f"v5_key:{key}")
    combined_tags = ",".join(dict.fromkeys(tag_set))

    # Route through the unified memory API so store-side bugfixes / features
    # (tag encoding, ChromaDB fallbacks, ...) are inherited automatically.
    from v5.memory_api import V5MemoryAPI

    api = V5MemoryAPI()
    mid = api.store(
        content=content,
        memory_type=type,
        domain=domain,
        category_path=category_path,
        key=key,
        tags=tag_set,
        importance=max(0.0, min(1.0, weight)),
        pad_p=float(pad_p),
        pad_a=float(pad_a),
        pad_d=float(pad_d),
    )

    dissonance = None
    try:
        from v5.dissonance import detect_dissonance
        dr = detect_dissonance(content, type)
        if dr and dr.get("conflicts"):
            dissonance = dr
    except Exception:  # noqa: BLE001
        pass

    return answer(f"记忆已存储 #{mid}", {"id": mid, "ok": True, "dissonance": dissonance})


@safe_tool
def v5_memory_search(
    query: str,
    top_k: int = 5,
    min_weight: float = 0.0,
    time_start: float = None,
    time_end: float = None,
    exclude: str = None,
    emotion_tag: str = None,
) -> str:
    """Search long-term memory.

    Paths (first match wins):
      1. emotion_tag given  -> emotional_memory.search_by_emotion()
      2. time/exclude given  -> memory_retrieval.retrieve() (3-way fusion)
      3. default            -> search.fused_search() (FTS5 + vector)
      4. any failure        -> FTS5 only (store.search)
    Always returns a JSON array string; never raises.
    """
    # 1. emotion-tag retrieval (kept on its own path — emotional_memory specific)
    if emotion_tag:
        from v5.emotional_memory import search_by_emotion
        return answer(f"根据情感标签找到 {len(search_by_emotion(emotion_tag, top_k=top_k))} 条记忆", search_by_emotion(emotion_tag, top_k=top_k))

# 内联说明见 docs/scripts/core/v5/v5/tools/memory_tool.md（见“内联注释摘录”）
    from v5.memory_api import V5MemoryAPI

    api = V5MemoryAPI()
    time_range = (time_start, time_end) if (time_start and time_end) else None
    results = api.search(query=query, fuse=True, top_k=top_k, time_range=time_range)

    if min_weight > 0:
        results = [r for r in results if float(r.get("weight", 0)) >= min_weight]

    if exclude:
        exclude_list = [e.strip() for e in exclude.split(",") if e.strip()]
        if exclude_list:
            results = [
                r for r in results
                if not any(e in r.get("content", "") for e in exclude_list)
            ]

    return answer(f"找到 {len(results)} 条记忆", results)


@safe_tool
def v5_memory_get(memory_id: int) -> str:
    """Fetch a single memory by id."""
    from v5.memory_api import V5MemoryAPI
    api = V5MemoryAPI()
    m = api.get(int(memory_id))
    if m is None:
        return dumps({"ok": False, "error": "not_found", "id": memory_id})
        return dumps({"ok": False, "error": "not_found", "id": memory_id})


@safe_tool
def v5_memory_delete(memory_id: int) -> str:
    """Delete a single memory by id."""
    from v5.memory_api import V5MemoryAPI
    api = V5MemoryAPI()
    ok = bool(api.delete(int(memory_id)))
    return answer(f"记忆 #{memory_id} 已删除", {"ok": ok, "id": memory_id})


@safe_tool
def v5_memory_stats() -> str:
    """Return storage statistics."""
    from v5 import store as store
    return answer(f"共 {store.stats()["total"]} 条记忆", store.stats())
