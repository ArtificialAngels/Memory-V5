"""v5 对话树引擎 —— 拓扑与内容分离, 深度集成 V5 store.

四大能力:
  1. 树形对话管理  (ConversationTree)   节点创建 / 遍历 / 跳转 / 分支 / 剪枝
  2. 上下文匹配    (get_context)        沿祖先链从 V5 store 批量读取重建连贯历史
  3. 定向记忆拉取  (MemoryRetriever)    路径记忆走 FTS5, 跨分支走向量语义
  4. 节点状态同步  (state + persist)    每节点独立 state/config, JSON 持久化 + 快速恢复

架构:
  - 拓扑 JSON (data/v5/{persist_key}.json) 只存节点关系 + v5_memory_id + summary
  - 对话本体通过 store.store() 写入 V5 SQLite, 通过 store.get_batch() 批量回读
  - MemoryRetriever 不再维护独立记忆列表, 路径记忆走 store.search (FTS5),
    跨分支联想走 v5.memory_retrieval.retrieve() (向量 + FTS5 + 时间 三路融合)
  - store 依赖可注入 (ConversationTree._store / _load / _search), 方便测试与替换
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ikaros.memory.v5.conversation_tree")

# 数据目录: 与 store.py 的 V5_DATA_DIR 同公式
V5_DATA_DIR = Path(__file__).resolve().parent / "data" / "v5"

# 分词正则 (中英数)
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


def _esc(text: str) -> str:
    """mermaid 标签转义 (双引号 / 方括号 / 换行)."""
    return (text.replace('"', "'").replace("[", "(").replace("]", ")")
            .replace("\n", " ").strip())


def _extract_summary(messages: List[Dict[str, Any]], max_chars: int = 80) -> str:
    """从消息列表中提取摘要: 取第一条 user 消息的前 max_chars 字."""
    for m in messages:
        if m.get("role") == "user":
            content = (m.get("content") or "").strip()
            if content:
                return content[:max_chars]
    return ""


# ───────────────────────────── v2 新实体 ─────────────────────────────

@dataclass
class ToolCall:
    """单次工具调用记录"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    success: bool = True
    duration_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "result_summary": self.result_summary,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=d.get("name", ""),
            params=d.get("params", {}) or {},
            result_summary=d.get("result_summary", ""),
            success=d.get("success", True),
            duration_ms=d.get("duration_ms", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class NodeInsight:
    """节点级结论/洞察"""
    text: str
    confidence: float = 0.5
    source_ids: List[str] = field(default_factory=list)
    extracted_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "source_ids": list(self.source_ids),
            "extracted_at": self.extracted_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeInsight":
        return cls(
            text=d.get("text", ""),
            confidence=d.get("confidence", 0.5),
            source_ids=list(d.get("source_ids", [])),
            extracted_at=d.get("extracted_at", 0.0),
        )


# ───────────────────────────── 节点 ─────────────────────────────
# B2: 节点执行状态 (对齐 herdr idle/working/blocked/done, 扩展 pending/done/unknown)
# 供 conversation-tree 节点徽标 + CodingAgentSupervisor 状态机共用。
EXEC_STATES = ("idle", "pending", "working", "blocked", "done", "unknown")
# 引擎发布的状态变更事件类型 (与 core/taskbus.py 中 EVENT_NODE_EXEC_STATE 同值)
EVENT_EXEC_STATE = "node.exec_state_changed"


@dataclass
class ConvNode:
    """对话树节点: 拓扑元数据 + v5_memory_id 引用 V5 store 中的对话内容.

    对话本体不再内联存储 (消息走 store), 拓扑 JSON 体积约缩小 80%.
    """
    id: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0
    branch_label: Optional[str] = None
    v5_memory_id: int = 0        # 引用 V5 store 中 type="conversation" 的记忆 id
    summary: str = ""            # 摘要 (前 80 字), 供 UI 节点标签
    state: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    # v2.2: 代理归属 (ekko-agent 模式借鉴) —— 该分支由哪个 runtime 作答。
    #   "ikaros" = Ikaros 伴侣人格 (SOUL/公理/心绪); "hermes" = Hermes 任务代理 (runtime+tools)。
    #   等价于 ekko 的 MemoryRuntimeIdentity/session 的"运行身份", 决定 system prompt 与路由。
    agent: str = "ikaros"
    created_at: float = 0.0

    # ── v2 新增字段 ──
    node_type: str = "trunk"           # trunk | branch | merge_point | conclusion | insight
    is_valid: bool = True              # 有效分支（路径可达主干且未被废弃）
    merge_target: Optional[str] = None  # 合并目标节点 ID
    merged_from: List[str] = field(default_factory=list)  # 合并到此节点的分支
    skills_used: List[str] = field(default_factory=list)   # 调用的技能名
    tool_calls: List[ToolCall] = field(default_factory=list)  # 工具调用详情
    thinking: str = ""  # 模型思考过程 (reasoning), 供 chat 面板回显
    usage: Dict[str, Any] = field(default_factory=dict)  # 本轮 LLM 用量(token/缓存), 供 chat 面板回显
    conclusions: List[NodeInsight] = field(default_factory=list)  # 提取的结论

    # ── 树域适配 (tree_adapter): node→fact 持久化绑定 ──
    # 修已知 bug: 原 fact 绑定仅存内存 _node_memories, 重启后路径检索变空。
    # 此处把绑定持久化进拓扑 JSON, 重启后由 MemoryRetriever.retrieve 优先读取。
    memory_ids: List[int] = field(default_factory=list)

    # ── v2.1 执行状态 (B2: herdr exec_state 内化) ──
    exec_state: str = "idle"        # idle | pending | working | blocked | done | unknown
    exec_progress: float = 0.0      # 0.0 ~ 1.0 进度 (仅 working/blocked 时有意义)
    exec_detail: str = ""           # 人类可读状态说明 (如 "blocked: 等待用户批准")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "depth": self.depth,
            "branch_label": self.branch_label,
            "v5_memory_id": self.v5_memory_id,
            "summary": self.summary,
            "state": self.state,
            "config": self.config,
            "meta": self.meta,
            "agent": self.agent,
            "created_at": self.created_at,
            # v2
            "node_type": self.node_type,
            "is_valid": self.is_valid,
            "merge_target": self.merge_target,
            "merged_from": list(self.merged_from),
            "skills_used": list(self.skills_used),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "thinking": self.thinking,
            "usage": self.usage,
            "conclusions": [c.to_dict() for c in self.conclusions],
            # 树域适配: node→fact 持久化绑定
            "memory_ids": list(self.memory_ids),
            # v2.1 exec_state
            "exec_state": self.exec_state,
            "exec_progress": self.exec_progress,
            "exec_detail": self.exec_detail,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConvNode":
        return cls(
            id=d["id"],
            parent_id=d.get("parent_id"),
            children=list(d.get("children", [])),
            depth=d.get("depth", 0),
            branch_label=d.get("branch_label"),
            v5_memory_id=d.get("v5_memory_id", 0),
            summary=d.get("summary", ""),
            state=d.get("state", {}) or {},
            config=d.get("config", {}) or {},
            meta=d.get("meta", {}) or {},
            agent=d.get("agent", "ikaros"),
            created_at=d.get("created_at", 0.0),
            # v2 字段：缺失时使用默认值（向后兼容 v1 JSON）
            node_type=d.get("node_type", "trunk"),
            is_valid=d.get("is_valid", True),
            merge_target=d.get("merge_target"),
            merged_from=list(d.get("merged_from", [])),
            skills_used=list(d.get("skills_used", [])),
            tool_calls=[ToolCall.from_dict(tc) for tc in d.get("tool_calls", [])],
            thinking=d.get("thinking", "") or "",
            usage=d.get("usage", {}) or {},
            conclusions=[NodeInsight.from_dict(c) for c in d.get("conclusions", [])],
            # 树域适配: node→fact 持久化绑定 (缺失时默认空, 向后兼容旧 JSON)
            memory_ids=list(d.get("memory_ids", [])),
            # v2.1 exec_state (缺失时默认 idle, 向后兼容 v1/v2 JSON)
            exec_state=d.get("exec_state", "idle"),
            exec_progress=d.get("exec_progress", 0.0),
            exec_detail=d.get("exec_detail", ""),
        )


# ───────────────────────── Store 接口 (可注入) ───────────────────────

# _store(content: str, content_type: str, tags: str) -> int
#   存储对话文本 (JSON), 返回 memory_id. content_type 默认 "conversation".
def _default_store(content: str, type: str = "conversation",
                   tags: str = "") -> int:
    try:
        from v5 import store as v5s
        return v5s.store(content, type=type, tags=tags)
    except Exception as e:
        logger.warning("default store provider failed: %s", e)
        return 0


def _tree_tag(node_id: str, branch_label: Optional[str] = None) -> str:
    """树域标签 (node:/branch:), 供 tree_adapter.tree_scoped_retrieve 做树域过滤。

    优先用 extensions.tree_adapter.tag_for_node 作为单一真源; 不可用时内联构造
    相同格式, 保证零硬依赖 (extensions 模块缺失/离线也不影响主线存储)。
    """
    try:
        from v5.extensions.tree_adapter import tag_for_node
        return tag_for_node(node_id, branch_label)
    except Exception:
        tags = [f"node:{node_id}"]
        if branch_label:
            tags.append(f"branch:{branch_label}")
        return " ".join(tags)


# _load(memory_ids: list[int]) -> dict[int, str]
#   批量读取记忆内容, 返回 {id: content_string}.
def _default_load(memory_ids: list[int]) -> dict[int, str]:
    if not memory_ids:
        return {}
    try:
        from v5 import store as v5s
        batch = v5s.get_batch(memory_ids)
        return {mid: m.content for mid, m in batch.items()}
    except Exception as e:
        logger.warning("default load provider failed: %s", e)
        return {}


# _search(query: str, top_k: int) -> list[dict]
#   FTS5 关键词检索, 返回 [{"id":..., "content":...}, ...].
def _default_search(query: str, top_k: int = 10) -> list[dict]:
    if not query or not query.strip():
        return []
    try:
        from v5 import store as v5s
        results = v5s.search(query, top_k=top_k)
        return [{"id": r.id, "content": r.content} for r in results]
    except Exception as e:
        logger.debug("default search provider failed: %s", e)
        return []


StoreFn = Callable[[str, str, str], int]          # (content, type, tags) -> int
LoadFn = Callable[[list[int]], dict[int, str]]    # [ids] -> {id: content}
SearchFn = Callable[[str, int], list[dict]]       # (query, top_k) -> [{id, content}]


# ─────────────────────────── 对话树 ───────────────────────────
class ConversationTree:
    """树形对话管理: 支持节点创建 / 遍历 / 跳转 / 分支 / 剪枝 + 持久化.

    拓扑与内容分离: 树 JSON 只存节点关系 + v5_memory_id + summary,
    对话本体写入 V5 store (SQLite), 上下文通过 store.get_batch() 批量回读.
    """

    def __init__(
        self,
        persist_key: str = "conversation_tree",
        data_dir: Optional[str | Path] = None,
        onChange: Optional[Any] = None,
        # 可注入 store 依赖 (默认走 v5.store)
        _store: Optional[StoreFn] = None,
        _load: Optional[LoadFn] = None,
        _search: Optional[SearchFn] = None,
    ) -> None:
        self.nodes: Dict[str, ConvNode] = {}
        self.root_id: Optional[str] = None
        self.current_id: Optional[str] = None
        self.version: int = 0
        self.persist_key = persist_key
        self.data_dir = Path(data_dir) if data_dir else V5_DATA_DIR
        self.onChange = onChange
        self._lock = threading.RLock()
        # 存储后端: 可注入, 默认走 V5 store
        self._store_fn = _store or _default_store
        self._load_fn = _load or _default_load
        self._search_fn = _search or _default_search
        # B2: 可选事件总线 (由服务/编排层注入); 设置后状态变更会向其发布类型化事件
        self.event_bus = None

    # ── 回调 ──
    def _emit(self) -> None:
        if self.onChange:
            try:
                self.onChange(self)
            except Exception as exc:
                logger.warning("onChange callback failed: %s", exc)

    # ── B2: 执行状态 (herdr exec_state 内化) ──
    def set_exec_state(
        self,
        node_id: str,
        state: str,
        progress: Optional[float] = None,
        detail: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        publish: bool = True,
        persist_on_progress: bool = False,
    ) -> "ConvNode":
        """设置节点执行状态并向事件总线广播 (对应 herdr pane.agent_status_changed).

        - ``state`` 取 EXEC_STATES, 未知值归为 ``unknown`` (容错)。
        - 状态**跳变** (state 值改变) 时持久化; 仅进度/细节更新 (progress-only)
          默认不落盘 (避免高频 tick 写盘), 可由 ``persist_on_progress=True`` 强制落盘。
        - 始终发布 ``node.exec_state_changed`` 事件 (若 ``event_bus`` 已注入)。
        - 不持有 ``self._lock`` 时调用事件总线, 避免回调内再取锁造成死锁。
        """
        if state not in EXEC_STATES:
            state = "unknown"
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise KeyError(f"node not found: {node_id}")
            prev = node.exec_state
            node.exec_state = state
            if progress is not None:
                try:
                    node.exec_progress = max(0.0, min(1.0, float(progress)))
                except (TypeError, ValueError):
                    pass
            if detail is not None:
                node.exec_detail = str(detail)
            if meta:
                node.meta.update(meta)
            self.version += 1
            transitioned = (state != prev)
        # 发布事件 (不持锁)
        if publish and self.event_bus is not None:
            self.event_bus.publish({
                "type": EVENT_EXEC_STATE,
                "tree": self.persist_key,
                "data": {
                    "node_id": node.id,
                    "exec_state": node.exec_state,
                    "progress": node.exec_progress,
                    "detail": node.exec_detail,
                    "prev_state": prev,
                },
            })
        # 落盘: 仅状态跳变, 或显式要求进度也落盘
        if transitioned or persist_on_progress:
            self.persist()
        self._emit()
        return node

    # ── 初始化 ──
    def init(self, seed_messages: Optional[List[Dict[str, Any]]] = None,
             seed_summary: str = "") -> ConvNode:
        with self._lock:
            mid = 0
            root_id = uid("root")
            if seed_messages:
                content = json.dumps(seed_messages, ensure_ascii=False)
                sm = seed_summary or _extract_summary(seed_messages)
                mid = self._store_fn(content, type="conversation", tags=_tree_tag(root_id))
            else:
                sm = seed_summary
            root = ConvNode(
                id=root_id,
                parent_id=None,
                depth=0,
                v5_memory_id=mid,
                summary=sm,
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

    # ── 路径 ──
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

    # ── 上下文 (从 V5 store 批量回读) ──
    def get_context(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """重建连贯历史: 沿路径从 V5 store 批量读取, 反序列化拼合消息."""
        with self._lock:
            path = self.get_path(node_id)
            ids = [n.v5_memory_id for n in path if n.v5_memory_id > 0]
            if not ids:
                return []
            batch = self._load_fn(ids)
            ctx: List[Dict[str, Any]] = []
            for n in path:
                raw = batch.get(n.v5_memory_id, "")
                if raw:
                    try:
                        msgs = json.loads(raw)
                        if isinstance(msgs, list):
                            ctx.extend(msgs)
                    except json.JSONDecodeError:
                        ctx.append({"role": "system", "content": raw})
            return ctx

    def get_context_with_meta(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """带结构元信息的上下文 (供需要结构的 provider)."""
        with self._lock:
            path = self.get_path(node_id)
            ids = [n.v5_memory_id for n in path if n.v5_memory_id > 0]
            batch = self._load_fn(ids)
            result: List[Dict[str, Any]] = []
            for n in path:
                raw = batch.get(n.v5_memory_id, "")
                msgs: List[Dict[str, Any]] = []
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            msgs = parsed
                    except json.JSONDecodeError:
                        pass
                result.append({
                    "node_id": n.id,
                    "depth": n.depth,
                    "branch_label": n.branch_label,
                    "v5_memory_id": n.v5_memory_id,
                    "summary": n.summary,
                    "messages": msgs,
                })
            return result

    # ── v2: 超级上下文 (祖先 L0 + 兄弟 L1 + 合并结论 L2) ──
    def build_context_v2(
        self,
        node_id: Optional[str] = None,
        include_siblings: bool = True,
        include_merged: bool = True,
        max_messages: int = 50,
    ) -> List[Dict[str, Any]]:
        """构建超级上下文消息列表。

        L0: 祖先对话历史（从 V5 store 回读）
        L1: 兄弟节点摘要（告知模型 "其他路径在做什么"）
        L2: 已合并分支结论（从祖先 merged_from 注入）

        返回: [system_msg, ...历史消息...] 格式。
        """
        target = node_id or self.current_id
        ancestors = self.get_path(target)
        messages: List[Dict[str, Any]] = []

        # ── L0: 祖先对话历史 ──
        ancestor_ids = [a.v5_memory_id for a in ancestors if a.v5_memory_id > 0]
        if ancestor_ids:
            batch = self._load_fn(ancestor_ids)
            for a in ancestors:
                raw = batch.get(a.v5_memory_id, "")
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            messages.extend(parsed)
                    except json.JSONDecodeError:
                        pass
        messages = messages[-max_messages:]

        # ── L1: 兄弟节点摘要 ──
        sibling_text = ""
        if include_siblings:
            siblings_by_depth: Dict[int, List[str]] = {}
            for a in ancestors:
                sibs = self.get_sibling_nodes(a.id)
                if sibs:
                    entries = []
                    for s in sibs:
                        label = s.branch_label or "branch"
                        summary = s.summary[:60] if s.summary else ""
                        conclusions = "; ".join(c.text[:40] for c in s.conclusions[:2])
                        entry = f"[{label}]: {summary}"
                        if conclusions:
                            entry += f" | conclusions: {conclusions}"
                        entries.append(entry)
                    siblings_by_depth[a.depth] = entries
            if siblings_by_depth:
                sibling_text = "Sibling branches at each level:\n"
                for depth, entries in sorted(siblings_by_depth.items()):
                    sibling_text += f"  depth {depth}: " + "\n    ".join(entries) + "\n"

        # ── L2: 已合并分支结论 ──
        merged_text = ""
        if include_merged:
            insights: List[str] = []
            for a in ancestors:
                for bf in a.merged_from:
                    branch = self.nodes.get(bf)
                    if branch:
                        for c in branch.conclusions:
                            insights.append(
                                f"[merged:{branch.branch_label}] {c.text}"
                            )
            if insights:
                merged_text = "Merged branch conclusions:\n" + "\n".join(
                    f"  - {i}" for i in insights
                )

        # ── 组装 system prompt ──
        system_parts = [
            "You are Ikaros, an AI with deep conversation memory and branching context."
        ]
        if merged_text:
            system_parts.append(merged_text)
        if sibling_text:
            system_parts.append(sibling_text)

        system_msg = {"role": "system", "content": "\n\n".join(system_parts)}
        return [system_msg] + messages

    # ── 新增回合 (子节点) ──
    def add_turn(
        self,
        messages: List[Dict[str, Any]],
        parent_id: Optional[str] = None,
        branch_label: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        tags: str = "",
        thinking: str = "",
        tool_calls: Optional[List["ToolCall"]] = None,
        usage: Optional[Dict[str, Any]] = None,
        skills_used: Optional[List[str]] = None,
    ) -> ConvNode:
        with self._lock:
            pid = parent_id or self.current_id
            parent = self.nodes.get(pid)
            if not parent:
                raise ValueError(f"parent not found: {pid}")

            # 存储对话内容到 V5 store (带 node/branch 树域标签, 供 tree_scoped_retrieve)
            content = json.dumps(messages, ensure_ascii=False)
            sm = _extract_summary(messages)
            node_id = uid("n")
            new_tags = f"{tags} {_tree_tag(node_id, branch_label)}".strip()
            mid = self._store_fn(content, type="conversation", tags=new_tags)

            node = ConvNode(
                id=node_id,
                parent_id=pid,
                agent=parent.agent,
                depth=parent.depth + 1,
                branch_label=branch_label,
                v5_memory_id=mid,
                summary=sm,
                state=_clone(state) if state is not None else _clone(parent.state),
                config=_clone(config) if config is not None else _clone(parent.config),
                meta={"created_at": time.time(), "title": title},
                thinking=thinking,
                tool_calls=list(tool_calls) if tool_calls else [],
                usage=usage or {},
                skills_used=list(skills_used) if skills_used else [],
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
        """在该节点下创建子节点 (与已有子节点成兄弟)."""
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

    def get_sibling_nodes(self, node_id: str) -> List["ConvNode"]:
        """返回兄弟节点完整对象列表（含摘要/结论，不含自身）。"""
        return [self.nodes[c] for c in self.siblings(node_id) if c in self.nodes]

    # ── v2: 分叉 ──
    def fork_branch(
        self,
        fork_point_id: str,
        branch_label: str,
        messages: List[Dict[str, Any]],
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        tags: str = "",
    ) -> ConvNode:
        """从任意节点分叉出新分支。

        与 add_turn 的区别: fork_branch 显式标记 node_type="branch",
        用于分支管理场景。add_turn 保持 node_type 继承父节点。
        """
        fork_node = self.nodes.get(fork_point_id)
        if not fork_node:
            raise ValueError(f"fork point not found: {fork_point_id}")

        content = json.dumps(messages, ensure_ascii=False)
        sm = _extract_summary(messages)
        node_id = uid("br")
        new_tags = f"{tags} {_tree_tag(node_id, branch_label)}".strip()
        mid = self._store_fn(content, type="conversation", tags=new_tags)

        node = ConvNode(
            id=node_id,
            parent_id=fork_point_id,
            agent=fork_node.agent,
            depth=fork_node.depth + 1,
            branch_label=branch_label,
            node_type="branch",
            v5_memory_id=mid,
            summary=sm,
            state=_clone(state) if state is not None else _clone(fork_node.state),
            config=_clone(config) if config is not None else _clone(fork_node.config),
            meta={"created_at": time.time(), "title": title},
            created_at=time.time(),
        )
        self.nodes[node.id] = node
        fork_node.children.append(node.id)
        self.current_id = node.id
        self.version += 1
        self._emit()
        self.persist()
        return node

    # ── v2: 结论化 ──
    def conclude_branch(
        self,
        node_id: str,
        conclusions: List[str],
    ) -> ConvNode:
        """将分支节点标记为已结论化。"""
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"node not found: {node_id}")
        node.node_type = "conclusion"
        for text in conclusions:
            node.conclusions.append(NodeInsight(
                text=text,
                confidence=0.8,
                source_ids=[node_id],
                extracted_at=time.time(),
            ))
        self.version += 1
        self._emit()
        self.persist()
        return node

    # ── v2: 合并分支结论到主干 ──
    def merge_branch(
        self,
        branch_node_id: str,
        trunk_target_id: str,
    ) -> None:
        """将分支结论合并到主干节点。

        建立 DAG merge 边: branch.merge_target → trunk_target_id,
        trunk.merged_from += branch_node_id。
        分支结论注入主干时 confidence × 0.9 并标记来源。
        """
        branch = self.nodes.get(branch_node_id)
        trunk = self.nodes.get(trunk_target_id)
        if not branch:
            raise ValueError(f"branch node not found: {branch_node_id}")
        if not trunk:
            raise ValueError(f"trunk node not found: {trunk_target_id}")
        if trunk.node_type != "trunk":
            raise ValueError(f"merge target must be trunk, got: {trunk.node_type}")

        # 1. 建立 merge 边
        branch.merge_target = trunk_target_id
        if branch_node_id not in trunk.merged_from:
            trunk.merged_from.append(branch_node_id)

        # 2. 注入结论（confidence × 0.9）
        for insight in branch.conclusions:
            trunk.conclusions.append(NodeInsight(
                text=f"[merged from {branch.branch_label}] {insight.text}",
                confidence=insight.confidence * 0.9,
                source_ids=list(insight.source_ids) + [branch_node_id],
                extracted_at=time.time(),
            ))

        # 3. 更新主干 state
        trunk.state.setdefault("merged_insights", [])
        trunk.state["merged_insights"].append({
            "branch_id": branch_node_id,
            "branch_label": branch.branch_label,
            "conclusions": [i.text for i in branch.conclusions],
            "merged_at": time.time(),
        })

        self.version += 1
        self._emit()
        self.persist()

    # ── v2: 撤销合并 ──
    def unmerge_branch(self, branch_node_id: str) -> None:
        """撤销分支合并: 断开 merge 边, 从主干移除注入的结论."""
        branch = self.nodes.get(branch_node_id)
        if not branch or not branch.merge_target:
            return
        trunk = self.nodes.get(branch.merge_target)
        if trunk:
            if branch_node_id in trunk.merged_from:
                trunk.merged_from.remove(branch_node_id)
            trunk.conclusions = [
                c for c in trunk.conclusions
                if branch_node_id not in c.source_ids
            ]
            if "merged_insights" in trunk.state:
                trunk.state["merged_insights"] = [
                    m for m in trunk.state["merged_insights"]
                    if m.get("branch_id") != branch_node_id
                ]
        branch.merge_target = None
        self.version += 1
        self._emit()
        self.persist()

    # ── v2: 废弃分支 ──
    def abandon_branch(self, node_id: str) -> None:
        """废弃分支: 子树全部标记 is_valid=False, 不移除节点."""
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"node not found: {node_id}")
        for n in self.subtree(node_id):
            n.is_valid = False
            n.meta["abandoned_at"] = time.time()
        self.version += 1
        self._emit()
        self.persist()

    # ── v2: 有效分支判定 ──
    def is_valid_branch(self, node_id: str) -> bool:
        """判定分支是否有效: 路径可达主干 且 未被废弃。

        规则:
        1. is_valid=False → 无效（已废弃）
        2. trunk 类型 → 有效
        3. 已合并（有 merge_target）→ 追踪 merge_target 递归判定
        4. 未合并 → 沿祖先链找是否有 trunk 节点
        5. 既无 trunk 也无 merge_target → 无效
        6. 检测到环 → 无效
        """
        visited: set[str] = set()
        current_id: Optional[str] = node_id

        while current_id and current_id not in visited:
            visited.add(current_id)
            node = self.nodes.get(current_id)
            if not node:
                return False

            # 已废弃
            if not node.is_valid:
                return False

            # trunk 节点 → 主干可达
            if node.node_type == "trunk":
                return True

            # 已合并 → 追踪 merge_target
            if node.merge_target:
                current_id = node.merge_target
                continue

            # 未合并 → 沿祖先链上溯
            if node.parent_id:
                current_id = node.parent_id
                continue

            # 无父节点且非 trunk → 孤立节点
            return False

        # 检测到环
        return False

    # ── 深度重算 (delete_node 重挂子树后用) ──
    def _recompute_depth(self, node_id: str, depth: int) -> None:
        """递归重设节点及其子树深度 (depth 以父节点深度 +1 推算)."""
        node = self.nodes.get(node_id)
        if not node:
            return
        node.depth = depth
        for child_id in node.children:
            self._recompute_depth(child_id, depth + 1)

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
        if self.current_id in del_ids:
            anc = self.nodes.get(target.parent_id) if target else None
            while anc and anc.id in del_ids:
                anc = self.nodes.get(anc.parent_id) if anc.parent_id else None
            self.current_id = anc.id if anc else self.root_id
        self.version += 1
        self._emit()
        self.persist()

    # ── v2.2: 设置节点代理归属 (ekko-agent 模式) ──
    def set_agent(self, node_id: str, agent: str, cascade: bool = False) -> "ConvNode":
        """设置节点由哪个 runtime 作答: 'ikaros' (伴侣人格) 或 'hermes' (任务代理).

        非法值归默认 'ikaros'。不影响对话内容与记忆, 仅决定 chat 的 system prompt
        与 LLM 路由 (镜像 rename_node 的轻量元字段语义)。
        cascade=True 时一并同步其所有后代 (子节点集成父节点代理状态)。
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                raise ValueError(f"node not found: {node_id}")
            clean = (agent or "ikaros").strip().lower()
            if clean not in ("ikaros", "hermes"):
                clean = "ikaros"
            node.agent = clean
            if cascade:
                for child_id in list(node.children):
                    self._apply_agent_recursive(child_id, clean)
            self.version += 1
        self._emit()
        self.persist()
        return node

    def _apply_agent_recursive(self, node_id: str, agent: str) -> None:
        """已持有 _lock 时递归同步子树代理 (不可重入加锁, 避免与 set_agent 死锁)."""
        n = self.nodes.get(node_id)
        if not n:
            return
        n.agent = agent
        for child_id in n.children:
            self._apply_agent_recursive(child_id, agent)

    # ── v2: 重命名节点（UI 标签人工覆盖）──
    def rename_node(self, node_id: str, title: str) -> "ConvNode":
        """重命名节点: 写入 ``meta.title`` 作为卡片标签的人工覆盖.

        不影响存储的对话内容 (v5 store 不变); 前端 ``nodeText`` 优先用 ``meta.title``。
        传入空串视为清除覆盖, 回退到自动摘要。
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                raise ValueError(f"node not found: {node_id}")
            clean = (title or "").strip()
            if not clean:
                node.meta.pop("title", None)
            else:
                node.meta["title"] = clean[:120]
            self.version += 1
        self._emit()
        self.persist()
        return node

    # ── v2: 删除单节点（子树重挂父节点）──
    def delete_node(self, node_id: str) -> None:
        """删除单个节点, 将其所有子节点重挂到父节点 (保留子树其余部分).

        区别于 ``prune``: prune 删除整棵子树; 本方法只删本节点, 子节点上提一级。
        不删除关联 v5 记忆 (与 prune 一致, 留孤儿记忆, 避免误删对话内容)。
        根节点不可删。若当前节点被删, current_id 重指父节点 (或 root)。
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                raise ValueError(f"node not found: {node_id}")
            if node.depth == 0:
                raise ValueError("cannot delete root node")
            parent = self.nodes.get(node.parent_id) if node.parent_id else None
            # 从父节点的 children 移除本节点
            if parent:
                parent.children = [c for c in parent.children if c != node_id]
                # 子节点重挂父节点
                for child_id in node.children:
                    c = self.nodes.get(child_id)
                    if c:
                        c.parent_id = parent.id
                        if child_id not in parent.children:
                            parent.children.append(child_id)
                        # 重挂后递归重算子节点及其子树深度 (避免错层)
                        self._recompute_depth(child_id, parent.depth + 1)
            # 删除本节点
            self.nodes.pop(node_id, None)
            # current 指向被删节点 → 重指父节点 (或 root)
            if self.current_id == node_id:
                self.current_id = parent.id if parent else self.root_id
            self.version += 1
        self._emit()
        self.persist()

    # ── 检索 (走 V5 store FTS5 关键词) ──
    def search(
        self,
        query: str,
        limit: int = 10,
        scope_node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按消息内容检索, 返回 [{"node_id":..., "content":..., "score":...}].

        走 V5 store FTS5 关键词搜索; 限定 scope_node_id 时在子树范围内过滤.
        """
        if not query or not query.strip():
            return []
        results = self._search_fn(query, limit)
        if not results:
            return []
        # 限定子树范围
        if scope_node_id:
            scope_ids = {n.id for n in self.subtree(scope_node_id)}
        else:
            scope_ids = None
        out = []
        for r in results:
            # 关联 memory_id → node_id (反查节点)
            nid = self._resolve_memory_node(r["id"])
            if nid and (scope_ids is None or nid in scope_ids):
                out.append({
                    "node_id": nid,
                    "content": r.get("content", ""),
                    "v5_memory_id": r["id"],
                })
        return out[:limit]

    def _resolve_memory_node(self, memory_id: int) -> Optional[str]:
        """反查: 给定 v5_memory_id, 找到所属 node.id. O(n), 调用不频繁."""
        for n in self.nodes.values():
            if n.v5_memory_id == memory_id:
                return n.id
        return None

    # ── 可视化 ──
    def to_mermaid(self) -> str:
        """导出 mermaid 流程图 (用 summary 作节点标签)."""
        lines = ["graph TD"]
        for n in self.nodes.values():
            label = _esc((n.branch_label or n.summary or n.id)[:28])
            lines.append(f'  {n.id}["{label}"]')
        for n in self.nodes.values():
            for c in n.children:
                lines.append(f"  {n.id} --> {c}")
        return "\n".join(lines)

    @contextmanager
    def lock(self):
        """显式锁上下文, 便于多分支并发下做跨步骤的原子操作."""
        with self._lock:
            yield self

    # ── 持久化 (拓扑 + v5_memory_id + summary, 不含对话本体) ──
    def serialize(self) -> str:
        with self._lock:
            payload = {
                "v": self.version,
                "schema": "super-conv-2.0",
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
            tmp.replace(path)
        except Exception as exc:
            logger.debug("persist skipped: %s", exc)

    @classmethod
    def deserialize(cls, raw: str | dict, **kwargs: Any) -> "ConversationTree":
        data = json.loads(raw) if isinstance(raw, str) else raw
        t = cls(**kwargs)
        t.version = data.get("v", 0)
        t.root_id = data.get("root_id")
        t.current_id = data.get("current_id")
        t.nodes = {n["id"]: ConvNode.from_dict(n) for n in data.get("nodes", [])}

        # v1 → v2 自动迁移: 没有 schema 字段 → 所有节点 node_type 推断
        schema = data.get("schema", "")
        if not schema:
            migrated = 0
            for n in t.nodes.values():
                # v1 节点: 有 branch_label 的是 branch, 否则是 trunk
                if n.branch_label:
                    n.node_type = "branch"
                else:
                    n.node_type = "trunk"
                migrated += 1
            if migrated:
                logger.info("v1→v2 migration: %d nodes", migrated)
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

    path  —— 锚定在当前路径节点上的记忆 (走 V5 store FTS5 关键词, 确定性匹配).
    cross —— 不在当前路径、但语义相关的记忆 (走 v5.memory_retrieval.retrieve()
             三路融合: FTS5 + 向量 + 时间), 经相关性评分 + 标签 boost + 阈值过滤.
    """

    def __init__(
        self,
        tree: ConversationTree,
        cross_branch_enabled: bool = True,
        max_results: int = 6,
        min_relevance: float = 0.15,
        # 可注入检索后端 (默认走 V5 三路融合)
        _cross_retriever: Optional[Callable[..., list[dict]]] = None,
    ) -> None:
        self.tree = tree
        self.cross_branch_enabled = cross_branch_enabled
        self.max_results = max_results
        self.min_relevance = min_relevance
        self._cross_retriever = _cross_retriever or self._default_cross_retrieve
        # 节点 → 记忆映射: node_id → [memory_id, ...] (add_memory 时记录)
        self._node_memories: dict[str, list[int]] = {}

    @staticmethod
    def _default_cross_retrieve(query: str, top_k: int = 10,
                                **kwargs: Any) -> list[dict]:
        """默认跨分支检索: 走 v5.memory_retrieval 三路融合."""
        try:
            from v5 import memory_retrieval
            return memory_retrieval.retrieve(query, top_k=top_k, **kwargs)
        except Exception as e:
            logger.debug("cross-branch retrieve failed: %s", e)
            return []

    def set_cross_retriever(self, fn: Callable[..., list[dict]]) -> None:
        self._cross_retriever = fn

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
        path_set = {n.id for n in path_nodes}
        branch_labels = {n.branch_label for n in path_nodes if n.branch_label}

        # 路径文本 (供跨分支检索的 query)
        # 取路径上每个节点的 summary 拼接
        path_text = " ".join(n.summary for n in path_nodes if n.summary)
        if not path_text:
            path_text = " ".join(branch_labels)

        # ── 路径内记忆: 从节点映射表精确查找 ──
        # 优先读持久化的 node.memory_ids (重启后仍可用), 回退内存 _node_memories
        path: list[dict] = []
        if include_path:
            for nid in path_set:
                source_ids = list(self._node_memories.get(nid, []))
                cn = self.tree.get_node(nid)
                if cn is not None:
                    for mid in cn.memory_ids:
                        if mid not in source_ids:
                            source_ids.append(mid)
                for mid in source_ids:
                    batch = self.tree._load_fn([mid])
                    content = batch.get(mid, "")
                    if content:
                        path.append({
                            "mem": {"id": mid, "text": content, "node_id": nid},
                            "relevance": 1.0,
                            "source": "path",
                        })
            # 去重 + 截断
            seen = set()
            uniq: list[dict] = []
            for p in path:
                if p["mem"]["id"] not in seen:
                    seen.add(p["mem"]["id"])
                    uniq.append(p)
            path = uniq[:self.max_results]

        # ── 跨分支记忆: 三路融合向量语义检索 ──
        cross: list[dict] = []
        if include_cross and self.cross_branch_enabled and path_text.strip():
            try:
                results = self._cross_retriever(
                    path_text, top_k=self.max_results * 2,
                )
                for r in results:
                    content = r.get("content", "")
                    nid = self.tree._resolve_memory_node(int(r.get("id", 0)))
                    if not nid or nid in path_set:
                        continue  # 排除路径内
                    label_match = False
                    if nid:
                        cn = self.tree.get_node(nid)
                        if cn and cn.branch_label in branch_labels:
                            label_match = True
                    score = float(r.get("score", r.get("raw", 0.5)))
                    if label_match:
                        score += 0.3
                    if score >= self.min_relevance:
                        cross.append({
                            "mem": {"id": r.get("id"), "text": content, "node_id": nid},
                            "relevance": score,
                            "label_match": label_match,
                            "source": "cross",
                        })
                cross.sort(key=lambda x: x["relevance"], reverse=True)
                cross = cross[:self.max_results]
            except Exception as e:
                logger.debug("cross-branch retrieve failed: %s", e)

        return {
            "path": path,
            "cross": cross,
            "path_text": path_text,
            "branch_labels": list(branch_labels),
        }

    def add_memory(self, mem: Dict[str, Any]) -> Dict[str, Any]:
        """将记忆写入 V5 store (type=fact), 记入节点映射供路径检索."""
        text = mem.get("text", mem.get("content", ""))
        node_id = mem.get("node_id", "")
        node = self.tree.get_node(node_id) if node_id else None
        branch_label = node.branch_label if node else None
        base_tags = " ".join(mem.get("tags", [])) if isinstance(mem.get("tags"), list) else mem.get("tags", "")
        # 带 node/branch 树域标签, 供 tree_scoped_retrieve 做树域过滤 (fact 同样适用)
        new_tags = f"{base_tags} {_tree_tag(node_id, branch_label)}".strip() if (node_id or base_tags) else base_tags
        try:
            mid = self.tree._store_fn(text, type="fact", tags=new_tags)
        except Exception as e:
            logger.warning("add_memory store failed: %s", e)
            mid = 0
        # 双写: 内存 _node_memories + 持久化 node.memory_ids (修重启后绑定丢失的 bug)
        if node_id and mid > 0:
            self._node_memories.setdefault(node_id, []).append(mid)
            if node is not None:
                if mid not in node.memory_ids:
                    node.memory_ids.append(mid)
                self.tree.persist()  # 落盘 node→fact 绑定
        return {"id": mid, "text": text, "node_id": node_id, "ts": time.time()}


__all__ = [
    "ConversationTree",
    "ConvNode",
    "MemoryRetriever",
    "ToolCall",
    "NodeInsight",
    "uid",
    "tokenize",
    "V5_DATA_DIR",
]
