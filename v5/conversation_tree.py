"""v5 对话树引擎 —— Explore.poker 风格的树形对话管理 (上下文 / 定向记忆 / 状态同步).

四大能力:
  1. 树形对话管理  (ConversationTree)   节点创建 / 遍历 / 跳转 / 分支 / 剪枝
  2. 上下文匹配    (get_context)        沿祖先链重建从根到目标节点的连贯历史
  3. 定向记忆拉取  (MemoryRetriever)    路径内记忆 + 跨分支关联记忆 (相关性过滤)
  4. 节点状态同步  (state + persist)    每节点独立 state/config, JSON 持久化 + 快速恢复

设计原则:
  - 框架无关, 仅依赖标准库, 不耦合 V5 其它模块 (记忆后端可注入).
  - 所有结构性写操作受 RLock 保护, 支持多分支并发场景下的上下文隔离与同步.
  - 持久化落盘到 data/v5/{persist_key}.json, 与 V5 既有 JSON 状态文件同目录.

后续优化方向 (见对话):
  - 将 MemoryRetriever.store 替换为 v5.store (SQLite + FTS5) 作为记忆后端.
  - 相关性评分升级为 embedding 余弦相似度.
  - 与 orchestrator.agent_loop / cloud_chat 的上下文注入对接.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ikaros.memory.v5.conversation_tree")

# 数据目录: 与 store.py 的 V5_DATA_DIR 同公式
V5_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "v5"

# 分词正则 (中英数), 可整体替换为 embedding 检索
_TOKEN_RE = re.compile(r"[^a-z0-9\u4e00-\u9fa5]+")


# ───────────────────────────── 工具 ─────────────────────────────
def uid(prefix: str = "n") -> str:
    """生成短唯一 id (uuid + 时间尾数)."""
    return f"{prefix}_{uuid.uuid4().hex[:9]}{int(time.time()) % 100000:05d}"


def tokenize(text: str) -> List[str]:
    """简易分词 (生产可换 embedding). 中英文 + 数字, 过滤单字."""
    return [t for t in _TOKEN_RE.split((text or "").lower()) if len(t) > 1]


def _clone(obj: Any) -> Any:
    """深拷贝 (等价于 JS 的 JSON round-trip)."""
    return json.loads(json.dumps(obj, ensure_ascii=False))


# ───────────────────────────── 节点 ─────────────────────────────
@dataclass
class ConvNode:
    """对话树节点: 承载一回合消息 + 独立状态/配置."""
    id: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0
    branch_label: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "depth": self.depth,
            "branch_label": self.branch_label,
            "messages": list(self.messages),
            "state": self.state,
            "config": self.config,
            "meta": self.meta,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConvNode":
        return cls(
            id=d["id"],
            parent_id=d.get("parent_id"),
            children=list(d.get("children", [])),
            depth=d.get("depth", 0),
            branch_label=d.get("branch_label"),
            messages=list(d.get("messages", [])),
            state=d.get("state", {}) or {},
            config=d.get("config", {}) or {},
            meta=d.get("meta", {}) or {},
            created_at=d.get("created_at", 0.0),
        )


# ─────────────────────────── 对话树 ───────────────────────────
class ConversationTree:
    """树形对话管理: 支持节点创建 / 遍历 / 跳转 / 分支 / 剪枝 + 持久化."""

    def __init__(
        self,
        persist_key: str = "conversation_tree",
        data_dir: Optional[str | Path] = None,
        onChange: Optional[Any] = None,
    ) -> None:
        self.nodes: Dict[str, ConvNode] = {}
        self.root_id: Optional[str] = None
        self.current_id: Optional[str] = None
        self.version: int = 0
        self.persist_key = persist_key
        self.data_dir = Path(data_dir) if data_dir else V5_DATA_DIR
        self.onChange = onChange
        self._lock = threading.RLock()  # 多分支并发安全

    # ── 回调 ──
    def _emit(self) -> None:
        if self.onChange:
            try:
                self.onChange(self)
            except Exception as exc:  # 回调不应破坏树操作
                logger.warning("onChange callback failed: %s", exc)

    # ── 初始化 ──
    def init(self, seed_messages: Optional[List[Dict[str, Any]]] = None) -> ConvNode:
        with self._lock:
            root = ConvNode(
                id=uid("root"),
                parent_id=None,
                depth=0,
                messages=seed_messages or [],
                created_at=time.time(),
            )
            self.nodes[root.id] = root
            self.root_id = root.id
            self.current_id = root.id
            self.version += 1
        self._emit()
        self.persist()
        return root

    # ── 访问 ──
    @property
    def current(self) -> Optional[ConvNode]:
        return self.nodes.get(self.current_id) if self.current_id else None

    def get_node(self, node_id: str) -> Optional[ConvNode]:
        return self.nodes.get(node_id)

    # ── 路径 / 上下文 ──
    def get_path(self, node_id: Optional[str] = None) -> List[ConvNode]:
        """根 -> 目标 (含): 祖先链."""
        with self._lock:
            target = node_id or self.current_id
            path: List[ConvNode] = []
            cur = self.nodes.get(target) if target else None
            while cur:
                path.insert(0, cur)
                cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
            return path

    def get_context(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """重建连贯历史: 沿路径拼合所有消息."""
        with self._lock:
            ctx: List[Dict[str, Any]] = []
            for n in self.get_path(node_id):
                ctx.extend(n.messages)
            return ctx

    def get_context_with_meta(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """带结构元信息的上下文 (供需要结构的 provider)."""
        with self._lock:
            return [
                {
                    "node_id": n.id,
                    "depth": n.depth,
                    "branch_label": n.branch_label,
                    "messages": n.messages,
                }
                for n in self.get_path(node_id)
            ]

    # ── 新增回合 (子节点) ──
    def add_turn(
        self,
        messages: List[Dict[str, Any]],
        parent_id: Optional[str] = None,
        branch_label: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
    ) -> ConvNode:
        with self._lock:
            pid = parent_id or self.current_id
            parent = self.nodes.get(pid)
            if not parent:
                raise ValueError(f"parent not found: {pid}")
            node = ConvNode(
                id=uid("n"),
                parent_id=pid,
                depth=parent.depth + 1,
                branch_label=branch_label,
                messages=messages,
                # 状态/配置从父节点深拷贝继承, 保证上下文连贯
                state=_clone(state) if state is not None else _clone(parent.state),
                config=_clone(config) if config is not None else _clone(parent.config),
                meta={"created_at": time.time(), "title": title},
                created_at=time.time(),
            )
            self.nodes[node.id] = node
            parent.children.append(node.id)
            self.current_id = node.id
            self.version += 1
        self._emit()
        self.persist()
        return node

    # ── 分支: 从目标节点派生新路径 ──
    def branch_from(
        self,
        node_id: str,
        messages: List[Dict[str, Any]],
        branch_label: Optional[str] = None,
        **kwargs: Any,
    ) -> ConvNode:
        """在该节点下创建子节点 (与已有子节点成兄弟 = 真正意义的从节点分叉)."""
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"node not found: {node_id}")
        return self.add_turn(
            messages,
            parent_id=node.id,
            branch_label=branch_label or "branch",
            **kwargs,
        )

    # ── 跳转: 恢复上下文 ──
    def jump_to(self, node_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                raise ValueError(f"node not found: {node_id}")
            self.current_id = node_id
        self._emit()
        self.persist()
        return self.get_context(node_id)

    # ── 子树 / 兄弟 ──
    def subtree(self, node_id: str) -> List[ConvNode]:
        with self._lock:
            res: List[ConvNode] = []
            stack = [node_id]
            while stack:
                cur = stack.pop()
                n = self.nodes.get(cur)
                if not n:
                    continue
                res.append(n)
                stack.extend(n.children)
            return res

    def siblings(self, node_id: str) -> List[str]:
        with self._lock:
            n = self.nodes.get(node_id)
            if not n or not n.parent_id:
                return []
            p = self.nodes.get(n.parent_id)
            return [c for c in p.children if c != node_id]

    # ── 剪枝 ──
    def prune(self, node_id: str) -> None:
        with self._lock:
            del_ids = {n.id for n in self.subtree(node_id)}
            target = self.nodes.get(node_id)
            if target and target.parent_id:
                p = self.nodes.get(target.parent_id)
                if p:
                    p.children = [c for c in p.children if c != node_id]
            for nid in del_ids:
                self.nodes.pop(nid, None)
            # 若当前节点在被删子树内, 回退到最近的现存祖先
            if self.current_id in del_ids:
                anc = self.nodes.get(target.parent_id) if target else None
                while anc and anc.id in del_ids:
                    anc = self.nodes.get(anc.parent_id) if anc.parent_id else None
                self.current_id = anc.id if anc else self.root_id
            self.version += 1
        self._emit()
        self.persist()

    # ── 持久化 ──
    def serialize(self) -> str:
        with self._lock:
            payload = {
                "v": self.version,
                "root_id": self.root_id,
                "current_id": self.current_id,
                "nodes": [n.to_dict() for n in self.nodes.values()],
            }
        return json.dumps(payload, ensure_ascii=False)

    def persist(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / f"{self.persist_key}.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(self.serialize(), encoding="utf-8")
            tmp.replace(path)  # 原子替换, 避免半写
        except Exception as exc:  # 隐私模式 / 只读目录等
            logger.debug("persist skipped: %s", exc)

    @classmethod
    def deserialize(cls, raw: str | dict, **kwargs: Any) -> "ConversationTree":
        data = json.loads(raw) if isinstance(raw, str) else raw
        t = cls(**kwargs)
        t.version = data.get("v", 0)
        t.root_id = data.get("root_id")
        t.current_id = data.get("current_id")
        t.nodes = {n["id"]: ConvNode.from_dict(n) for n in data.get("nodes", [])}
        return t

    @classmethod
    def load(
        cls,
        persist_key: str = "conversation_tree",
        data_dir: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> Optional["ConversationTree"]:
        d = Path(data_dir) if data_dir else V5_DATA_DIR
        path = d / f"{persist_key}.json"
        if not path.exists():
            return None
        try:
            return cls.deserialize(path.read_text(encoding="utf-8"),
                                   persist_key=persist_key, data_dir=data_dir, **kwargs)
        except Exception as exc:
            logger.warning("load failed (%s): %s", path, exc)
            return None


# ────────────────────────── 定向记忆拉取 ──────────────────────────
class MemoryRetriever:
    """根据当前节点及其路径动态拉取记忆.

    path  —— 锚定在当前路径节点上的记忆 (确凿相关, 直接展示).
    cross —— 不在当前路径、但共享 branch_label 或概念重叠的记忆 (跨分支关联),
            经相关性评分 + 标签 boost + 阈值过滤, 避免无关信息干扰.
    """

    def __init__(
        self,
        tree: ConversationTree,
        store: Optional[List[Dict[str, Any]]] = None,
        cross_branch_enabled: bool = True,
        max_results: int = 6,
        min_relevance: float = 0.15,
    ) -> None:
        self.tree = tree
        self.store: List[Dict[str, Any]] = store or []
        self.cross_branch_enabled = cross_branch_enabled
        self.max_results = max_results
        self.min_relevance = min_relevance

    def _relevance(self, query_text: str, mem: Dict[str, Any]) -> float:
        q = set(tokenize(query_text))
        if not q:
            return 0.0
        m = set(tokenize(f"{mem.get('text', '')} {' '.join(mem.get('tags', []))}"))
        if not m:
            return 0.0
        hits = sum(1 for t in q if t in m)
        return hits / len(q)  # recall 比例

    def retrieve(
        self,
        node_id: str,
        include_path: bool = True,
        include_cross: bool = True,
    ) -> Dict[str, Any]:
        node = self.tree.get_node(node_id)
        if not node:
            return {"path": [], "cross": [], "path_text": "", "branch_labels": []}

        path_nodes = self.tree.get_path(node_id)
        path_text = " ".join(
            m.get("content", "") for n in path_nodes for m in n.messages
        )
        path_set = {n.id for n in path_nodes}
        branch_labels = {n.branch_label for n in path_nodes if n.branch_label}

        # 路径内记忆: 锚定在当前路径节点上 -> 直接展示 (不阈值过滤)
        path = (
            [
                {
                    "mem": m,
                    "relevance": self._relevance(path_text, m),
                    "source": "path",
                }
                for m in self.store
                if m.get("node_id") and m["node_id"] in path_set
            ]
            if include_path
            else []
        )
        path.sort(key=lambda x: x["relevance"], reverse=True)
        path = path[: self.max_results]

        # 跨分支记忆: 排除路径内, 共享 label/概念 -> 阈值过滤
        cross = []
        if include_cross and self.cross_branch_enabled:
            for m in self.store:
                if m.get("node_id") and m["node_id"] in path_set:
                    continue  # 排除路径内
                label_match = bool(m.get("branch") and m["branch"] in branch_labels)
                rel = self._relevance(path_text, m) + (0.3 if label_match else 0)
                if rel >= self.min_relevance:
                    cross.append(
                        {
                            "mem": m,
                            "relevance": rel,
                            "label_match": label_match,
                            "source": "cross",
                        }
                    )
            cross.sort(key=lambda x: x["relevance"], reverse=True)
            cross = cross[: self.max_results]

        return {
            "path": path,
            "cross": cross,
            "path_text": path_text,
            "branch_labels": list(branch_labels),
        }

    def add_memory(self, mem: Dict[str, Any]) -> Dict[str, Any]:
        item = {"id": uid("m"), "ts": time.time(), **mem}
        self.store.append(item)
        return item


__all__ = [
    "ConversationTree",
    "ConvNode",
    "MemoryRetriever",
    "uid",
    "tokenize",
    "V5_DATA_DIR",
]
