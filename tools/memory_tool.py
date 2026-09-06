"""V5 记忆读写工具 — v5_memory_store / v5_memory_search 等 MCP 暴露入口。

基于 memory_v5.store 与 memory_retrieval, 提供存储/检索/删除记忆的工具注册。
详细说明见 docs/scripts/core/memory_v5/v5/tools/memory_tool.md
"""

from __future__ import annotations

import json

from memory_v5.tools.utils import safe_tool, dumps, answer


@safe_tool
def v5_memory_store(
    content: str,
    type: str = "fact",
    weight: float = 0.6,
    tags: str = "",
    domain: str = None,
    category_path: str = None,
    key: str = None,
    importance: float = None,
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
    from memory_v5.memory_api import V5MemoryAPI

    # 2026-08-24 P2-14: importance (Ekko-style) 优先, 未提供时回退 weight (V5-native)。
    # 之前 importance 参数被静默丢弃 (恒用 weight)。
    effective = importance if importance is not None else weight
    api = V5MemoryAPI()
    mid = api.store(
        content=content,
        memory_type=type,
        domain=domain,
        category_path=category_path,
        key=key,
        tags=tag_set,
        importance=max(0.0, min(1.0, effective)),
        pad_p=float(pad_p),
        pad_a=float(pad_a),
        pad_d=float(pad_d),
    )

    # 认知失调检测由 store.store() 内部异步执行（store.py _run_dissonance_detection），
    # 这里不再同步调 detect_dissonance——它会对候选逐条调云端 NLI（最多 3×20s），
    # 曾导致 MCP store 调用 300s 超时。保留 dissonance=None 维持返回结构兼容。
    return answer(f"记忆已存储 #{mid}", {"id": mid, "ok": True, "dissonance": None})


@safe_tool
def v5_memory_search(
    query: str,
    top_k: int = 5,
    min_weight: float = 0.0,
    time_start: float = None,
    time_end: float = None,
    exclude: str = None,
    emotion_tag: str = None,
    include_dsh_only: bool = True,
) -> str:
    """Search long-term memory.

    Paths (first match wins):
      1. emotion_tag given  -> emotional_memory.search_by_emotion()
      2. 默认               -> V5MemoryAPI.search(fuse=True) = unified_retrieve(scope="auto")
                              (语义三路融合 + 图补路 + Vault; 失败降级 FTS5)
      3. any failure        -> FTS5 only (store.search)
    Always returns a JSON array string; never raises.
    """
    # 1. emotion-tag retrieval (kept on its own path — emotional_memory specific)
    if emotion_tag:
        from memory_v5.emotional_memory import search_by_emotion
        hits = search_by_emotion(emotion_tag, top_k=top_k)
        return answer(f"根据情感标签找到 {len(hits)} 条记忆", hits)

# 内联说明见 docs/scripts/core/memory_v5/v5/tools/memory_tool.md（见“内联注释摘录”）
    from memory_v5.memory_api import V5MemoryAPI

    api = V5MemoryAPI()
    time_range = (time_start, time_end) if (time_start and time_end) else None
    results = api.search(query=query, fuse=True, top_k=top_k, time_range=time_range,
                         include_dsh_only=include_dsh_only)

    if min_weight > 0:
        results = [r for r in results if float(r.get("weight", 0)) >= min_weight]

    if exclude:
        exclude_list = [e.strip() for e in exclude.split(",") if e.strip()]
        if exclude_list:
            results = [
                r for r in results
                if not any(e in r.get("content", "") for e in exclude_list)
            ]

    # P8 可观测性: 每条附 why (召回依据: 向量/关键词分量 + 意图 + EI + 图 relation)
    try:
        from memory_v5.memory_retrieval import explain_result
        for r in results:
            r["why"] = explain_result(r)
    except Exception:
        pass

    return answer(f"找到 {len(results)} 条记忆", results)


@safe_tool
def v5_memory_get(memory_id: int) -> str:
    """Fetch a single memory by id."""
    from memory_v5.memory_api import V5MemoryAPI
    api = V5MemoryAPI()
    m = api.get(int(memory_id))
    if m is None:
        return dumps({"ok": False, "error": "not_found", "id": memory_id})
    return answer(f"记忆 #{memory_id} 读取成功", m)


@safe_tool
def v5_memory_delete(memory_id: int) -> str:
    """Delete a single memory by id."""
    from memory_v5.memory_api import V5MemoryAPI
    api = V5MemoryAPI()
    ok = bool(api.delete(int(memory_id)))
    return answer(f"记忆 #{memory_id} 已删除", {"ok": ok, "id": memory_id})


@safe_tool
def v5_memory_stats() -> str:
    """Return storage statistics."""
    from memory_v5 import store as store
    # NOTE: 2026-08-20 — PEP 701 嵌套同引号 f-string (Python 3.12+) 不能在 3.11 编译。
    # 这里把 f-string 拆成普通字符串 + .format(), 既兼容 3.11 又 3.12 安全。
    stats = store.stats()
    return answer("共 {} 条记忆".format(stats["total"]), stats)
