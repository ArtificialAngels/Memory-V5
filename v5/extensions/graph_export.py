"""graph_export — V5 记忆图谱导出层（graphify 兼容）。

把 v5.db 里的记忆数据（memory 表为主）导出为 graphify 兼容的 graph.json，
让注册在 Hermes 里的 graphify MCP server 能直接查询 V5 记忆图谱。

设计原则（对照 2026-07-30 用户拍板）：
- V5 永远留在 SQLite，不迁移图数据库后端。
- 本模块是"只读图出口"：SQLite 是唯一事实源，图 JSON 只是导出视图。
- 导出不写回 v5.db，不改动任何现有表 / 40 个 v5_* MCP 工具。

导出内容：
- 节点 = 记忆条目（memory 表，按 type 分：conversation/fact/user_trait/
  emotion_label/lesson/preference/identity/...）
- 边 = 三种关系：
  1. SAME_TAG   — 两条记忆共享 tag（weight = 共享 tag 数）
  2. SAME_TYPE  — 同 type（weight = 1）
  3. REFLECTED  — lesson/identity 引用其来源（source_memory_id）
- community = 记忆 type（graphify 社区检测所需）

用法：
    python -m v5.extensions.graph_export [--out PATH] [--min-weight N]
    # 默认输出 E:\\Ikaros\\core\\v5\\data\\v5\\graphify-out\\graph.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# ── 路径解析（与 store.py 一致，不硬编码盘符） ──────────────────────────
V5_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "v5"
V5_DB_PATH = V5_DATA_DIR / "v5.db"
DEFAULT_OUT = V5_DATA_DIR / "graphify-out" / "graph.json"

# memory.type 的中文可读名（graphify community_name 用）
TYPE_LABELS = {
    "conversation": "对话",
    "fact": "事实",
    "user_trait": "用户特质",
    "emotion_label": "情感标签",
    "emotional_event": "情感事件",
    "lesson": "教训",
    "preference": "偏好",
    "identity": "身份",
    "narrative": "叙事",
    "self_discovery": "自我发现",
}


def _load_memories(db_path: Path) -> list[dict]:
    """读 memory 表，返回规范化条目列表。"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, content, type, tags, weight, created
           FROM memory ORDER BY id"""
    ).fetchall()
    conn.close()
    memories = []
    for r in rows:
        tags = []
        if r["tags"]:
            try:
                tags = [t.strip() for t in str(r["tags"]).split(",") if t.strip()]
            except Exception:
                tags = []
        # 只导出有内容的记忆（跳过空壳）
        content = (r["content"] or "").strip()
        if not content:
            continue
        memories.append(
            {
                "id": r["id"],
                "content": content,
                "type": r["type"] or "unknown",
                "tags": tags,
                "weight": r["weight"] or 0.5,
                "created": r["created"],
                "source_memory_id": None,  # 由二次查询补
            }
        )
    return memories


def _load_sources(db_path: Path) -> dict[int, int]:
    """source_memory_id 映射（lesson/identity 引用来源）。"""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT id, source_memory_id FROM memory WHERE source_memory_id IS NOT NULL"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def build_graph(memories: list[dict], sources: dict[int, int],
                tag_max_group: int = 20, tag_max_edges: int = 50) -> dict:
    """从记忆条目构建 graphify 兼容图。

    tag_max_group: SAME_TAG 边的 tag 出现次数上限（高区分度 tag 才有意义，
                   通用 tag 如 user/observe 出现几百次会生成 O(n^2) 爆炸边）。
    tag_max_edges: 单个 tag 组最多生成的边数（防大组失控）。
    """
    nodes: list[dict] = []
    links: list[dict] = []
    seen: set[int] = set()

    # tag → [memory_id]
    tag_index: dict[str, list[int]] = defaultdict(list)
    # 项目轨: v5_project:<name> → [memory_id]（图里建成独立项目节点）
    project_index: dict[str, list[int]] = defaultdict(list)
    kind_index: dict[int, str] = {}  # memory_id → v5_kind:<kind>

    for m in memories:
        mid = m["id"]
        nid = f"mem_{mid}"
        nodes.append(
            {
                "id": nid,
                "label": f"{TYPE_LABELS.get(m['type'], m['type'])}#{mid}",
                "source_file": f"v5.db/memory#{mid}",
                "source_location": f"L{mid}",
                "community": _type_community(m["type"]),
                "community_name": TYPE_LABELS.get(m["type"], m["type"]),
                "norm_label": m["content"][:60],
                "content": m["content"],
                "type": m["type"],
                "weight": m["weight"],
            }
        )
        seen.add(mid)
        for t in m["tags"]:
            tag_index[t].append(mid)
            if t.startswith("v5_project:"):
                project_index[t.split(":", 1)[1]].append(mid)
            elif t.startswith("v5_kind:"):
                kind_index[mid] = t.split(":", 1)[1]

    # 项目节点 + BELONGS_TO 边（项目 → 它的决策/坑/约定）
    for proj, ids in project_index.items():
        pnid = f"proj_{proj}"
        nodes.append(
            {
                "id": pnid,
                "label": f"项目:{proj}",
                "source_file": "v5.db/project",
                "source_location": f"L0",
                "community": len(TYPE_LABELS),
                "community_name": "项目",
                "norm_label": f"项目 {proj} 的记忆",
                "content": f"项目 {proj}",
                "type": "project",
                "weight": 1.0,
            }
        )
        for mid in ids:
            links.append(
                {
                    "source": pnid,
                    "target": f"mem_{mid}",
                    "relation": "BELONGS_TO",
                    "confidence": "EXTRACTED",
                    "weight": 1.0,
                    "kind": kind_index.get(mid),
                }
            )

    # SAME_TAG 边：共享高区分度 tag 的记忆互连（受 tag_max_group/tag_max_edges 约束）
    for tag, ids in tag_index.items():
        if len(ids) < 2 or len(ids) > tag_max_group:
            continue
        edges = 0
        for i in range(len(ids)):
            if edges >= tag_max_edges:
                break
            for j in range(i + 1, len(ids)):
                if edges >= tag_max_edges:
                    break
                links.append(
                    {
                        "source": f"mem_{ids[i]}",
                        "target": f"mem_{ids[j]}",
                        "relation": "SAME_TAG",
                        "confidence": "EXTRACTED",
                        "weight": 1.0,
                        "tag": tag,
                    }
                )
                edges += 1

    # REFLECTED 边：lesson/identity → 来源记忆
    for mid, src in sources.items():
        if mid in seen and src in seen:
            links.append(
                {
                    "source": f"mem_{mid}",
                    "target": f"mem_{src}",
                    "relation": "REFLECTED",
                    "confidence": "EXTRACTED",
                    "weight": 1.0,
                }
            )

    return {
        "directed": False,
        "multigraph": False,
        "graph": {"name": "ikaros-v5-memory", "source": "v5.db"},
        "nodes": nodes,
        "links": links,
        "hyperedges": [],
    }


def _type_community(t: str) -> int:
    """type → 社区 id（与 TYPE_LABELS 顺序一致，稳定）。"""
    order = list(TYPE_LABELS.keys())
    return order.index(t) if t in order else len(order)


def export(db_path: Path = V5_DB_PATH, out_path: Path = DEFAULT_OUT,
           min_weight: float = 0.0, tag_max_group: int = 20,
           tag_max_edges: int = 50) -> dict:
    """导出 v5.db 记忆为 graphify 兼容图。返回构建的图 dict。"""
    memories = _load_memories(db_path)
    if min_weight > 0:
        memories = [m for m in memories if m["weight"] >= min_weight]
    sources = _load_sources(db_path)
    graph = build_graph(memories, sources, tag_max_group, tag_max_edges)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m v5.extensions.graph_export",
        description="Export V5 memory (v5.db) to a graphify-compatible graph.json",
    )
    parser.add_argument("--db", default=str(V5_DB_PATH), help="path to v5.db")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output graph.json path")
    parser.add_argument(
        "--min-weight", type=float, default=0.0, help="only export memories with weight >= N"
    )
    parser.add_argument(
        "--tag-max-group", type=int, default=20,
        help="only create SAME_TAG edges for tags seen <= N times (default 20)",
    )
    parser.add_argument(
        "--tag-max-edges", type=int, default=50,
        help="max SAME_TAG edges per tag group (default 50)",
    )
    args = parser.parse_args()

    graph = export(Path(args.db), Path(args.out), args.min_weight,
                   args.tag_max_group, args.tag_max_edges)
    n_edges = len(graph["links"])
    n_nodes = len(graph["nodes"])
    print(f"OK: {n_nodes} nodes, {n_edges} edges -> {args.out}")
    # 边统计
    rels: dict[str, int] = defaultdict(int)
    for l in graph["links"]:
        rels[l["relation"]] += 1
    for rel, cnt in sorted(rels.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {cnt}")


if __name__ == "__main__":
    main()
