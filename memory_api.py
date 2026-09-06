# 详细说明见 docs/scripts/core/memory_v5/v5/memory_api.md

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

V5_ROOT = Path(__file__).resolve().parent
if str(V5_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(V5_ROOT.parent))

from memory_v5 import store as _store


def _row_to_dict(row) -> dict:
    """行 → 统一结果字典 (P6 收敛: 委托 memory_retrieval._norm, 结果形状唯一).

    结构化精确匹配 (tag/domain/key) 的输出标记 source="structured" (非语义融合)。
    """
    from memory_v5.memory_retrieval import _norm
    d = _norm(row)
    d["source"] = "structured"
    return d


class V5MemoryAPI:
    """Single entry point for all V5 memory operations.

    Every method has a fallback: if the semantic layer (ChromaDB / bge-m3 :8587)
    is unavailable, search() degrades to FTS5 keyword matching and the
    structured (tag-based) path always works against SQLite directly.
    (:8080 本地 LLM 已退役 2026-08-18, 注释里原本提到它作为 fallback 之一已移除。)
    """

    def store(
        self,
        content,
        *,
        memory_type: str = "fact",
        domain: str = None,
        category_path: str = None,
        key: str = None,
        tags: list = None,
        importance: float = 0.5,
        pad_p: float = 0.0,
        pad_a: float = 0.0,
        pad_d: float = 0.0,
    ) -> int:
        """Store a memory; return its integer id.

        Combines V5-native + Ekko-style structured fields into the tag set.
        """
        tag_set: list[str] = []
        if tags:
            tag_set.extend([t for t in tags if t])
        if domain:
            tag_set.append(f"v5_domain:{domain}")
        if category_path:
            tag_set.append(f"v5_cat:{category_path}")
        if key:
            tag_set.append(f"v5_key:{key}")
        combined_tags = ",".join(dict.fromkeys(tag_set))

        # 2026-08-14 Phase 1: 走 upsert 写策略 (同类相似 → 合并强化, 否则新建),
        # 根治"永远 INSERT"的雷同膨胀; 对话/事实/笔记均可受益。
        return _store.upsert(
            content=content,
            type=memory_type,
            weight=max(0.0, min(1.0, float(importance))),
            tags=combined_tags,
            pad_p=float(pad_p),
            pad_a=float(pad_a),
            pad_d=float(pad_d),
        )

    def search(
        self,
        query: str = None,
        *,
        domain: str = None,
        type: str = None,
        tags: list = None,
        key: str = None,
        category_path: str = None,
        fuse: bool = True,
        top_k: int = 5,
        time_range: tuple = None,
        min_score: Optional[float] = None,
        include_dsh_only: bool = True,
    ) -> list[dict]:
        """Search memory.

        Structured filters (domain / category_path / key / tags / type) take precedence and
        do an exact tag match against SQLite (works without ChromaDB).
        Otherwise, when `fuse=True`, runs the 3-way fused semantic retrieval;
        on any failure it falls back to FTS5 keyword search.

        R8 (M6): min_score 现在是真正的融合分下限 —— 对 unified_retrieve 返回的
        score 字段做后置过滤 (过滤后为空则如实返回 []); None = 不过滤, 沿用配置层
        min_fused_score (线上 0.3)。历史默认 0.6 会把 FTS-only 命中 (融合分
        0.3~0.4, 见 preprocess_config.yaml 2026-07-26 标定注释) 全部滤掉, 导致
        语义召回打空, 故默认改为 None, 由调用方按需收紧。
        """
        # 1) exact / structured path
        if domain or key or tags or type or category_path:
            clauses: list[str] = []
            params: list = []
            if domain:
                clauses.append("tags LIKE ?")
                params.append(f"%v5_domain:{domain}%")
            if key:
                clauses.append("tags LIKE ?")
                params.append(f"%v5_key:{key}%")
            if category_path:
                clauses.append("tags LIKE ?")
                params.append(f"%v5_cat:{category_path}%")
            if tags:
                for t in tags:
                    clauses.append("tags LIKE ?")
                    params.append(f"%{t}%")
            if type:
                clauses.append("type = ?")
                params.append(type)
            if query:
                clauses.append("(content LIKE ? OR tags LIKE ?)")
                params.append(f"%{query}%")
                params.append(f"%{query}%")
            if time_range:
                clauses.append("created >= ? AND created <= ?")
                params.extend(time_range)
            # archived=1 是 V5 的软删语义（retention/dedup 产物），lifecycle、
            # freshness、project_edges、reflect/* 每条读路径都过滤它，唯独本
            # 结构化检索路径漏了 —— 导致 v5_project_retrieve / 按 type 筛选
            # 会把已归档记忆当活记忆返回。2026-08-30 补上。
            # 列不存在（老迁移未跑）时的兼容见下方 except → 退回不带该条件。
            clauses.append("archived = 0")
            where = " AND ".join(clauses) if clauses else "1=1"
            try:
                with _store.conn() as c:
                    try:
                        rows = c.execute(
                            f"SELECT * FROM memory WHERE {where} "
                            f"ORDER BY weight DESC, id DESC LIMIT ?",
                            params + [int(top_k)],
                        ).fetchall()
                    except Exception:  # noqa: BLE001
                        # 极老库没有 archived 列（store 迁移未跑）→ 去掉该条件重试，
                        # 保证检索不因一次迁移缺失而整体失灵（fail-open）。
                        where2 = " AND ".join(
                            cl for cl in clauses if cl != "archived = 0") or "1=1"
                        rows = c.execute(
                            f"SELECT * FROM memory WHERE {where2} "
                            f"ORDER BY weight DESC, id DESC LIMIT ?",
                            params + [int(top_k)],
                        ).fetchall()
                results = [_row_to_dict(r) for r in rows]
                # 与 FTS5 fallback 一致: 外部执行器 (pi/herdr) 调用时 include_dsh_only=False
                # 必须过滤掉 [dsh-only] 平台纪律/密钥内容 (GH audit P0-4).
                if not include_dsh_only:
                    from memory_v5.scope import is_dsh_only
                    results = [r for r in results if not is_dsh_only(r.get("content"))]
                return results
            except Exception:  # noqa: BLE001
                return []

        # 2) semantic fuse path (统一路由层: auto scope = 三路融合 + 图补路 + Vault)
        if fuse:
            try:
                from memory_v5.memory_retrieval import unified_retrieve
                tr = tuple(time_range) if time_range else None
                fused = unified_retrieve(query, top_k=top_k, scope="auto",
                                         time_range=tr, min_weight=0.0,
                                         include_dsh_only=include_dsh_only)
# 内联说明见 docs/scripts/core/memory_v5/v5/memory_api.md（见“内联注释摘录”）
                if fused:
                    if min_score is not None and min_score > 0:
                        # R8 (M6): min_score 后置过滤生效 —— 过滤后为空则如实
                        # 返回空列表 (尊重调用方下限, 不回退到未过滤的 FTS5 路径)。
                        fused = [r for r in fused
                                 if float(r.get("score", 0.0)) >= min_score]
                        if not fused:
                            return []
                    return fused
            except Exception:  # noqa: BLE001
                pass

        # 3) FTS5 fallback (reachable both when fusion raises AND when it returns [])
        if query:
            try:
                rows = [_row_to_dict(m) for m in _store.search(query, top_k=top_k)]
                if not include_dsh_only:
                    from memory_v5.scope import is_dsh_only
                    rows = [r for r in rows if not is_dsh_only(r.get("content"))]
                return rows
            except Exception:  # noqa: BLE001
                return []
        return []

    def get(self, memory_id) -> Optional[dict]:
        """Fetch one memory by id, or None if missing."""
        m = _store.get(int(memory_id))
        if m is None:
            return None
        return _row_to_dict(m)

    def delete(self, memory_id) -> bool:
        """Delete one memory by id; return True if something was removed."""
        return bool(_store.delete(int(memory_id)))

    def stats(self) -> dict:
        """Return storage statistics."""
        return _store.stats()
