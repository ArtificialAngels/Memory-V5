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


def _ensure_message_ids(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """给消息补稳定 id (2026-08-15 poker 对齐).

    分支点 / 发散点用消息 id 定位 (parallel_source_message_id / branching_source_message_id)。
    已带 id 的保持不变; 缺失 (历史消息) 补 msg_<ts>_<rand>。返回新列表, 不修改入参。
    """
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        m2 = dict(m)
        if not m2.get("id"):
            m2["id"] = uid("msg")
        out.append(m2)
    return out


def _strip_msg_ids(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """剥离消息 id 字段 (LLM 上下文用).

    OpenAI 兼容 API 对 messages 中的未知字段可能严格报 400, 故上下文输出不携带 id。
    前端定位 (分支点/发散点) 用 state 内联 messages 的 id, 不走上下文路径。
    """
    return [{k: v for k, v in m.items() if k != "id"} for m in messages]


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
    # S3: 工具调用 id (OpenAI toolCallId) —— 持久化需要, 否则有工具的 chat
    # 落库时 ToolCall(**tc) 因多余关键字抛 TypeError 被吞, 节点内容静默丢失.
    id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
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
            id=d.get("id", ""),
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
        node = cls(
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
        # 整包导入: 序列化节点可能内联 messages (对话本体)。拓扑 JSON 只存指针,
        # 故此处仅作**迁移暂存** (from_dict 后由 _migrate_tree 写入 V5 store),
        # 不参与 to_dict/persist 序列化, 保证拓扑文件保持精简。
        # (旧代码直接丢弃 messages → import 后消息历史丢失, 导出-导入无法闭环)
        msgs = d.get("messages")
        if msgs is not None:
            node.messages = msgs
        return node


# ───────────────────────────── 卡片 (poker 对齐, 2026-08-15) ─────────────────────────────
# 数据层对齐 ai.explore.poker: **卡片 = 一段多轮会话** (messages 数组), 不是单回合。
# 底层仍保留 node=回合 为事实源 (V5 store 记忆 / supervisor / 上下文构建零改动),
# 卡片由节点链按"分叉点切分"自动聚合; 手动建卡(子/发散/分支卡)与分支点/未读经
# cards_meta 持久化合并进自动卡。

@dataclass
class ConvCard:
    """卡片: 一段多轮会话 (对齐 ai.explore.poker 的 Card).

    - 自动聚合: 卡片头 = ROOT + 每个分叉点(≥2 子节点)的每个孩子;
      卡片 = 从卡片头向下收连续子链, 到下一个卡片头为止。
      messages = 链上所有节点的消息串联 (一段连续对话)。
    - 手动建卡: fork 出的新节点天然成为卡片头; parallel/branching 来源标记、
      分支点 (来源消息 id)、标题、未读经 cards_meta 持久化后合并。
    """
    id: str                                       # "card_" + 卡片头节点 id (稳定可反查)
    title: str = ""                               # 无标题卡片 / 头节点 meta.title
    messages: List[Dict[str, Any]] = field(default_factory=list)   # 一段多轮会话
    parent_id: Optional[str] = None               # 父卡 id (卡片头 parent 所在卡)
    children: List[str] = field(default_factory=list)              # 子卡 id 列表
    depth: int = 0
    # → 发散卡: 从来源卡某条消息发散 (关联主题)
    parallel_source_id: Optional[str] = None
    parallel_source_message_id: Optional[str] = None
    # ↓ 分支卡: 从来源卡某条消息 (分支点) 开始的分支
    branching_source_id: Optional[str] = None
    branching_source_message_id: Optional[str] = None
    source_focus: str = ""                        # 触发建卡的选中文本
    is_unread: bool = False
    node_ids: List[str] = field(default_factory=list)     # 组成卡片的底层节点
    v5_memory_ids: List[int] = field(default_factory=list)
    summary: str = ""
    agent: str = "ikaros"
    branch_label: Optional[str] = None
    created_at: float = 0.0
    is_valid: bool = True
    kind: str = "auto"                            # auto | child | parallel | branching | manual
    parent_override: Optional[str] = None         # (旧模型保留兼容) 手动指定父卡
    # 2026-08-16 (显式连接图重构): 卡片独立存在, 关系 = 显式 links (多对多, 可断开)。
    # inputs  = 本卡接收的结论 (左/上锚点): [{id, from_card, from_port, to_port, kind}]
    # outputs = 本卡输出的结论 (右/下锚点): [{id, to_card, to_port, from_port, kind}]
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": self.messages,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "depth": self.depth,
            "parallel_source_id": self.parallel_source_id,
            "parallel_source_message_id": self.parallel_source_message_id,
            "branching_source_id": self.branching_source_id,
            "branching_source_message_id": self.branching_source_message_id,
            "source_focus": self.source_focus,
            "is_unread": self.is_unread,
            "node_ids": list(self.node_ids),
            "v5_memory_ids": list(self.v5_memory_ids),
            "summary": self.summary,
            "agent": self.agent,
            "branch_label": self.branch_label,
            "created_at": self.created_at,
            "is_valid": self.is_valid,
            "kind": self.kind,
            "parent_override": self.parent_override,
            # 2026-08-16 (显式连接图): 入/出连接 (前端锚点连线直接消费)
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConvCard":
        return cls(
            id=d["id"],
            title=d.get("title", ""),
            messages=list(d.get("messages", [])),
            parent_id=d.get("parent_id"),
            children=list(d.get("children", [])),
            depth=d.get("depth", 0),
            parallel_source_id=d.get("parallel_source_id"),
            parallel_source_message_id=d.get("parallel_source_message_id"),
            branching_source_id=d.get("branching_source_id"),
            branching_source_message_id=d.get("branching_source_message_id"),
            source_focus=d.get("source_focus", ""),
            is_unread=bool(d.get("is_unread", False)),
            node_ids=list(d.get("node_ids", [])),
            v5_memory_ids=list(d.get("v5_memory_ids", [])),
            summary=d.get("summary", ""),
            agent=d.get("agent", "ikaros"),
            branch_label=d.get("branch_label"),
            created_at=d.get("created_at", 0.0),
            is_valid=d.get("is_valid", True),
            kind=d.get("kind", "auto"),
            parent_override=d.get("parent_override"),
            inputs=list(d.get("inputs", [])),
            outputs=list(d.get("outputs", [])),
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
        # S1 (结构性修复): 显式主线终点 (trunk_id)。
        #   旧设计靠 node_type + "父节点有无 children" 的时序快照判定主线, 导致:
        #   - branch 下继续对话被误标 trunk (F1 已修继承)
        #   - 主线延续受"先 fork 后继续"影响, 主线身份随创建顺序漂移
        #   - 无法显式把分支提升为主线
        #   现在 trunk_id 是唯一真源: add_turn 在 trunk_id 节点下 → 主线延续;
        #   其余 → branch。is_valid_branch/__trunk__ 合并查找直接沿 trunk_id。
        self.trunk_id: Optional[str] = None
        self.version: int = 0
        self.persist_key = persist_key
        # 2026-08-15 (poker 对齐): 手动卡片元数据 —— card_id -> {kind, source_card_id,
        # source_message_id, source_focus, title, is_unread}。自动聚合卡无需登记,
        # 仅手动建卡(子/发散/分支卡)的附加标记与未读状态落盘 (重启不丢)。
        self.cards_meta: Dict[str, Dict[str, Any]] = {}
        # 2026-08-16 (显式连接图重构): 卡片独立存在, 关系 = 显式 links (多对多, 可断开)。
        # links 元素: {id, from_card, from_port(right|bottom), to_card, to_port(left|top), kind}
        #   from 出 → to 入; kind: auto(旧关系迁移) | manual(手连) | branching | parallel | child
        self.links: List[Dict[str, Any]] = []
        self._links_pending_migration: bool = False   # 旧 JSON 无 links → build_cards 首次迁移
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
                seed_msgs = _ensure_message_ids(seed_messages)
                content = json.dumps(seed_msgs, ensure_ascii=False)
                sm = seed_summary or _extract_summary(seed_msgs)
                mid = self._store_fn(content, type="conversation",
                                     tags=f"{_tree_tag(root_id)} session:{self.persist_key}")
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
            # S1: 根不是主线; 主线从根的第一个子节点开始 (add_turn 时设置)
            self.trunk_id = None
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
                            ctx.extend(_strip_msg_ids(msgs))   # LLM 上下文: 剥离 id (OpenAI 兼容 API 严格字段)
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
                    "messages": _strip_msg_ids(msgs),   # 树感知压缩输出: 剥离 id
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
                            messages.extend(_strip_msg_ids(parsed))   # LLM 上下文: 剥离 id
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
                        if not s.is_valid:
                            continue  # F3: 废弃分支不注入兄弟上下文
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
                    if branch and branch.is_valid:  # F3: 废弃分支的结论不注入
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
        force_branch: bool = False,
    ) -> ConvNode:
        with self._lock:
            pid = parent_id or self.current_id
            parent = self.nodes.get(pid)
            if not parent:
                raise ValueError(f"parent not found: {pid}")

            # 存储对话内容到 V5 store (带 node/branch/session 树域标签, 供 tree_scoped_retrieve
            # 做会话隔离过滤 —— session:<persist_key> 保证多会话记忆不串台, H1 修复)
            msgs = _ensure_message_ids(messages)
            content = json.dumps(msgs, ensure_ascii=False)
            sm = _extract_summary(msgs)
            node_id = uid("n")
            new_tags = f"{tags} {_tree_tag(node_id, branch_label)} session:{self.persist_key}".strip()
            mid = self._store_fn(content, type="conversation", tags=new_tags)
            if mid == 0:
                # F5: store 失败 (返回 0) 时显式告警, 不再无声丢内容 (节点仍创建, UI 可见)
                logger.warning("add_turn: V5 store failed for node %s (content not persisted)", node_id)

            # S1: node_type 由 trunk_id 唯一决定 (不再依赖时序快照):
            #   - force_branch=True (显式分叉: branch_from/前端 fork) → branch
            #   - parent 是当前主线终点 (trunk_id) → trunk (主线延续)
            #   - 树还没有主线 (trunk_id=None) 且 parent 是根 → trunk (主线起点)
            #   - 其余 (分支内继续/主线中间节点分叉/其他类型节点下) → branch
            if force_branch:
                node_type = "branch"
            elif self.trunk_id is None and parent.id == self.root_id:
                node_type = "trunk"
            elif parent.id == self.trunk_id and parent.node_type == "trunk":
                # S1: 主线延续 (conclusion 节点虽曾是 trunk_id, 但收尾后不再延续主线)
                node_type = "trunk"
            else:
                node_type = "branch"

            node = ConvNode(
                id=node_id,
                parent_id=pid,
                agent=parent.agent,
                depth=parent.depth + 1,
                branch_label=branch_label,
                # S1: node_type 已由上方 trunk_id 判定逻辑决定 (F1 继承语义并入 trunk_id)
                node_type=node_type,
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
            # S1: 主线延续时 trunk_id 前进到新节点 (主线终点 = 最近的主线对话)
            if node_type == "trunk":
                self.trunk_id = node.id
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
        """在该节点下创建子节点 (与已有子节点成兄弟).

        S1: 显式分叉语义 —— 无论目标节点是不是 trunk_id, 都强制 node_type="branch"
        (不再依赖"父节点是否已有子节点"的时序判定)。
        """
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"node not found: {node_id}")
        kwargs.setdefault("force_branch", True)
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
        # F4: 与 add_turn 对齐的元数据参数 (fork 节点也应能携带思考/工具/用量)
        thinking: str = "",
        tool_calls: Optional[List["ToolCall"]] = None,
        usage: Optional[Dict[str, Any]] = None,
        skills_used: Optional[List[str]] = None,
    ) -> ConvNode:
        """从任意节点分叉出新分支。

        与 add_turn 的区别: fork_branch 显式标记 node_type="branch",
        用于分支管理场景。add_turn 保持 node_type 继承父节点。
        """
        with self._lock:
            fork_node = self.nodes.get(fork_point_id)
            if not fork_node:
                raise ValueError(f"fork point not found: {fork_point_id}")

            msgs = _ensure_message_ids(messages)
            content = json.dumps(msgs, ensure_ascii=False)
            sm = _extract_summary(msgs)
            node_id = uid("br")
            new_tags = f"{tags} {_tree_tag(node_id, branch_label)} session:{self.persist_key}".strip()
            mid = self._store_fn(content, type="conversation", tags=new_tags)
            if mid == 0:
                # F5: store 失败 (返回 0) 时显式告警, 不再无声丢内容
                logger.warning("fork_branch: V5 store failed for node %s (content not persisted)", node_id)

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
                thinking=thinking,
                tool_calls=list(tool_calls) if tool_calls else [],
                usage=usage or {},
                skills_used=list(skills_used) if skills_used else [],
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
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                raise ValueError(f"node not found: {node_id}")
            node.node_type = "conclusion"
            # S1: 主线终点被收尾 → 回退到最近的 trunk 祖先 (conclusion 不再延续主线)
            if self.trunk_id == node_id:
                anc = self.nodes.get(node.parent_id) if node.parent_id else None
                self.trunk_id = None
                while anc:
                    if anc.node_type == "trunk" and anc.id != self.root_id:
                        self.trunk_id = anc.id
                        break
                    anc = self.nodes.get(anc.parent_id) if anc.parent_id else None
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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

        S1 重写: 主干判定沿 trunk_id (唯一真源), 不再依赖 node_type 时序快照。
        规则:
        1. is_valid=False → 无效（已废弃）
        2. 节点在主线路径上 (沿祖先链可达 trunk_id) → 有效
        3. 已合并（有 merge_target）→ 追踪 merge_target 递归判定
        4. 树未初始化主线 (trunk_id=None) 且节点沿祖先链可达根 → 有效 (兼容旧树)
        5. 检测到环 → 无效
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

            # 主线可达 → 有效:
            #   - 主线终点 (trunk_id) 本身
            #   - 任何 trunk 类型节点 (主线链成员; node_type 已由 trunk_id 维护, 可靠)
            #   - 从主线中间节点 fork 的分支, 其祖先链含 trunk 节点即有效
            if current_id == self.trunk_id or (node.node_type == "trunk" and node.id != self.root_id):
                return True

            # 旧树兼容 (trunk_id=None): 节点沿祖先链可达根 → 有效
            if self.trunk_id is None and current_id == self.root_id:
                return True

            # 已合并 → 追踪 merge_target
            if node.merge_target:
                current_id = node.merge_target
                continue

            # 未合并 → 沿祖先链上溯
            if node.parent_id:
                current_id = node.parent_id
                continue

            # 无父节点且非主线 → 孤立节点
            return False

        # 检测到环
        return False

    # ── S1: 显式主线管理 ──
    def set_trunk(self, node_id: str, cascade: bool = False) -> "ConvNode":
        """把节点提升为新的主线终点 (trunk_id)。

        用户显式把某个分支节点设为主线: 该节点成为主线终点,
        后续在其下继续对话 = 主线延续 (trunk)。
        cascade=True 时把该节点祖先路径上所有节点标记为 trunk (重建主线链)。
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                raise ValueError(f"node not found: {node_id}")
            # F14: 废弃分支禁止提升为主线 (语义上废弃=已放弃, 不应成为主线锚点)
            if not node.is_valid:
                raise ValueError(f"cannot set trunk: node {node_id} is abandoned (is_valid=False)")
            self.trunk_id = node.id
            # 节点自身必为 trunk (主线终点), 否则 add_turn 的
            # parent.node_type == "trunk" 判定会拒绝主线延续
            if node.id != self.root_id:
                node.node_type = "trunk"
            if cascade:
                # 沿祖先链 (不含自身, 已标) 全部标 trunk, 重建主线链
                cur = self.nodes.get(node.parent_id) if node.parent_id else None
                while cur:
                    if cur.id != self.root_id:
                        cur.node_type = "trunk"
                    cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
            self.version += 1
        self._emit()
        self.persist()
        return node

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
        if node_id == self.root_id:
            # 禁止剪根: 会删光整棵树且 current_id 指向已删 root, 使树彻底失效.
            raise ValueError("prune(root) is not allowed; the root node cannot be pruned")
        with self._lock:
            del_ids = {n.id for n in self.subtree(node_id)}
            target = self.nodes.get(node_id)
            if target and target.parent_id:
                p = self.nodes.get(target.parent_id)
                if p:
                    p.children = [c for c in p.children if c != node_id]
            for nid in del_ids:
                self.nodes.pop(nid, None)
            # F2: 清理其它节点对已删节点的 merge 引用 (merged_from/merge_target/
            # merged_insights/conclusions.source_ids), 避免 unmerge 静默失效 + 数据残留
            self._cleanup_merge_refs(del_ids)
            # 2026-08-15 (poker 对齐): 清理被删节点为头的卡片元数据 (防残留脏键)
            self._cleanup_cards_meta(del_ids)
            # S1: 主线终点被删 → 重指最近的 trunk 祖先 (或根), 保持主线语义
            if self.trunk_id in del_ids:
                self.trunk_id = None
                anc = self.nodes.get(target.parent_id) if target else None
                while anc:
                    if anc.id != self.root_id:
                        self.trunk_id = anc.id
                        break
                    anc = self.nodes.get(anc.parent_id) if anc.parent_id else None
        if self.current_id in del_ids:
            anc = self.nodes.get(target.parent_id) if target else None
            while anc and anc.id in del_ids:
                anc = self.nodes.get(anc.parent_id) if anc.parent_id else None
            self.current_id = anc.id if anc else self.root_id
        self.version += 1
        self._emit()
        self.persist()

    def _cleanup_merge_refs(self, deleted_ids: "set[str]") -> None:
        """删除节点后清理 DAG merge 引用 (需已持有 _lock)。

        - trunk.merged_from 移除已删 branch id
        - 其他节点的 merge_target 指向已删节点 → 置 None
        - trunk.state.merged_insights 移除对应条目
        - trunk.conclusions 移除 source_ids 含已删节点的注入结论 (仅限注入的)
        """
        if not deleted_ids:
            return
        for other in self.nodes.values():
            if deleted_ids & set(other.merged_from):
                other.merged_from = [b for b in other.merged_from if b not in deleted_ids]
            if other.merge_target in deleted_ids:
                other.merge_target = None
            insights = other.state.get("merged_insights")
            if isinstance(insights, list):
                other.state["merged_insights"] = [
                    m for m in insights if m.get("branch_id") not in deleted_ids
                ]
            # 只清"注入的"结论: source_ids 含已删节点且 text 带 [merged from 标记
            other.conclusions = [
                c for c in other.conclusions
                if not (deleted_ids & set(c.source_ids) and c.text.startswith("[merged from"))
            ]

    def _cleanup_cards_meta(self, deleted_ids: "set[str]") -> None:
        """删除节点后清理以该节点为头的卡片元数据 (需已持有 _lock)."""
        if not deleted_ids or not self.cards_meta:
            return
        gone = {f"card_{nid}" for nid in deleted_ids}
        for cid in list(self.cards_meta):
            if cid in gone:
                self.cards_meta.pop(cid, None)

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
            # F2: 清理其它节点对已删节点的 merge 引用
            self._cleanup_merge_refs({node_id})
            # 2026-08-15 (poker 对齐): 清理被删节点为头的卡片元数据
            self._cleanup_cards_meta({node_id})
            # S1: 主线终点被删 → 重指父节点 (保持主线语义)
            if self.trunk_id == node_id:
                self.trunk_id = parent.id if parent and parent.id != self.root_id else None
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

    # ── 卡片视图 (2026-08-15: 数据层对齐 poker; 底层节点=回合 保留为事实源) ──
    def _card_heads(self) -> set:
        """卡片头集合 (2026-08-16 主线聚合修复): ROOT + 分支起点.

        旧规则 (分叉点孩子 = 卡头) 会把分叉点自身孤立成单回合薄卡 —— 真实树
        分叉密集时画布退化成"每回合一张卡", 与"一段会话一张卡"的目标相悖。
        新规则: 主线 (trunk 路径) **连续聚合不切分**, 主线整条 = 一张卡;
        仅"从分叉点岔出去的分支起点" (父是分叉点且不在主线路径) 成为新卡头。
        trunk_id 缺失时沿"首个子节点链"推断主线 (首条对话路径)。
        """
        heads: set = set()
        if self.root_id is not None:
            heads.add(self.root_id)
        # 主线终点: trunk_id 优先; 缺失时沿首个子节点链下探 (首条路径作主线)
        trunk_end: Optional[str] = self.trunk_id
        if trunk_end is None and self.root_id is not None:
            n = self.nodes.get(self.root_id)
            while n:
                trunk_end = n.id
                first = next((c for c in n.children if c in self.nodes), None)
                n = self.nodes.get(first) if first else None
        # 主线路径 = trunk_end 及其祖先链 (沿 parent 可达 trunk_end 的节点)
        trunk_path: set = set()
        cur = trunk_end
        guard = 0
        while cur and guard < 512:
            if cur in trunk_path:
                break
            trunk_path.add(cur)
            p = self.nodes.get(cur)
            cur = p.parent_id if p else None
            guard += 1
        for n in self.nodes.values():
            parent = self.nodes.get(n.parent_id) if n.parent_id else None
            if not parent:
                continue
            kids = [c for c in parent.children if c in self.nodes]
            if len(kids) >= 2 and n.id not in trunk_path:
                heads.add(n.id)          # 分支起点 (非主线延续) → 新卡头
        return heads

    def _collect_card(self, head_id: str, heads: set) -> "ConvCard":
        """从卡片头向下收连续子链, 遇到下一个卡片头停止 → 一张卡片."""
        node_ids: List[str] = []
        v5_ids: List[int] = []
        child_card_ids: List[str] = []
        summary = ""
        agent = "ikaros"
        branch_label: Optional[str] = None
        created_at = 0.0
        is_valid = True
        max_depth = 0
        stack = [head_id]
        while stack:
            nid = stack.pop()
            n = self.nodes.get(nid)
            if not n:
                continue
            node_ids.append(nid)
            if n.v5_memory_id > 0:
                v5_ids.append(n.v5_memory_id)
            if not summary and n.summary:
                summary = n.summary
            agent = n.agent or agent
            if n.branch_label:
                branch_label = n.branch_label
            if n.created_at and (not created_at or n.created_at < created_at):
                created_at = n.created_at
            if not n.is_valid:
                is_valid = False
            max_depth = max(max_depth, n.depth)
            for c in n.children:
                if c not in self.nodes:
                    continue
                if c in heads:
                    child_card_ids.append("card_" + c)
                else:
                    stack.append(c)
        # 节点按 (深度, 创建序) 排列, 保证消息串联顺序稳定 (头在前)
        node_ids.sort(key=lambda nid: (
            self.nodes[nid].depth if nid in self.nodes else 0,
            self.nodes[nid].created_at if nid in self.nodes else 0,
        ))
        head = self.nodes.get(head_id)
        return ConvCard(
            id="card_" + head_id,
            parent_id=None,
            children=child_card_ids,
            depth=head.depth if head else 0,
            node_ids=node_ids,
            v5_memory_ids=v5_ids,
            summary=summary,
            agent=agent,
            branch_label=branch_label,
            created_at=created_at,
            is_valid=is_valid,
        )

    def build_cards(self, load_messages: bool = True) -> List["ConvCard"]:
        """构建卡片视图 (自动聚合 + cards_meta 手动覆盖合并).

        每张卡 = 一段连续会话; messages 从 V5 store 批量回读串联
        (load_messages=False 时跳过回读, 供轻量拓扑请求)。
        cards_meta 中手动建卡标记 (kind/source/title/is_unread) 合并到自动卡。
        """
        with self._lock:
            heads = self._card_heads()
            cards: List[ConvCard] = []
            for hid in heads:
                card = self._collect_card(hid, heads)
                meta = (self.cards_meta or {}).get(card.id, {})
                if meta:
                    card.kind = meta.get("kind", card.kind)
                    card.parallel_source_id = meta.get("parallel_source_id") or card.parallel_source_id
                    card.parallel_source_message_id = meta.get("parallel_source_message_id") or card.parallel_source_message_id
                    card.branching_source_id = meta.get("branching_source_id") or card.branching_source_id
                    card.branching_source_message_id = meta.get("branching_source_message_id") or card.branching_source_message_id
                    card.source_focus = meta.get("source_focus", card.source_focus)
                    if "is_unread" in meta:
                        card.is_unread = bool(meta["is_unread"])
                    if meta.get("title"):
                        card.title = meta["title"]
                # 标题 fallback: cards_meta.title → 头节点 meta.title → summary → 无标题卡片
                if not card.title:
                    head = self.nodes.get(hid)
                    if head and head.meta.get("title"):
                        card.title = head.meta["title"]
                    elif card.summary:
                        card.title = (card.summary or "")[:24]
                    else:
                        card.title = "无标题卡片"
                cards.append(card)
            # 卡片父链: 手动 parent_override 优先 (科技树编排), 否则卡片头 parent 所在卡 (自动派生)
            node2card: Dict[str, str] = {}
            for c in cards:
                for nid in c.node_ids:
                    node2card[nid] = c.id
            card_ids = {c.id for c in cards}
            for c in cards:
                meta2 = (self.cards_meta or {}).get(c.id, {})
                if "parent_override" in meta2:
                    ov = meta2.get("parent_override")
                    c.parent_override = ov
                    c.parent_id = ov if ov and ov in card_ids else None   # 目标卡不存在/空 → 无父 (根)
                else:
                    hid = c.id[len("card_"):]
                    head = self.nodes.get(hid)
                    c.parent_id = node2card.get(head.parent_id) if head and head.parent_id else None
            # children 重算 (基于最终 parent; 手动重排后自动跟随)
            for c in cards:
                c.children = []
            for c in cards:
                if c.parent_id:
                    pc = next((x for x in cards if x.id == c.parent_id), None)
                    if pc:
                        pc.children.append(c.id)
            # 卡片深度重算 (手动重排后跟随新父链), 供布局/折叠层级判断
            def _card_depth(cid: str, memo: Dict[str, int]) -> int:
                if cid in memo:
                    return memo[cid]
                cc = next((x for x in cards if x.id == cid), None)
                d = 0 if (not cc or not cc.parent_id) else _card_depth(cc.parent_id, memo) + 1
                memo[cid] = d
                return d
            _dmemo: Dict[str, int] = {}
            for c in cards:
                c.depth = _card_depth(c.id, _dmemo)
            cards.sort(key=lambda c: (c.depth, c.created_at))
            # 2026-08-16 (显式连接图重构): 旧 JSON 首次迁移 links + 派生每卡 inputs/outputs
            if self._links_pending_migration:
                self._migrate_links(cards)
                self._links_pending_migration = False
                self.persist()
            card_map = {c.id: c for c in cards}
            for lk in self.links or []:
                fc = lk.get("from_card"); tc = lk.get("to_card")
                if fc in card_map and tc in card_map:
                    card_map[tc].inputs.append({
                        "id": lk.get("id"), "from_card": fc,
                        "from_port": lk.get("from_port", "bottom"),
                        "to_port": lk.get("to_port", "top"), "kind": lk.get("kind", "manual"),
                    })
                    card_map[fc].outputs.append({
                        "id": lk.get("id"), "to_card": tc,
                        "to_port": lk.get("to_port", "top"),
                        "from_port": lk.get("from_port", "bottom"), "kind": lk.get("kind", "manual"),
                    })
            if not load_messages:
                return cards
            # 批量回读消息并串联 (一段连续会话)
            all_ids = [mid for c in cards for mid in c.v5_memory_ids]
            batch = self._load_fn(all_ids) if all_ids else {}
            for c in cards:
                msgs: List[Dict[str, Any]] = []
                for nid in c.node_ids:
                    n = self.nodes.get(nid)
                    if not n or n.v5_memory_id <= 0:
                        continue
                    raw = batch.get(n.v5_memory_id, "")
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, list):
                                msgs.extend(parsed)
                        except json.JSONDecodeError:
                            msgs.append({"role": "system", "content": raw})
                c.messages = msgs
            return cards

    def card_of_node(self, node_id: str) -> Optional["ConvCard"]:
        """返回某节点所属卡片 (无卡片视图时 None)."""
        for c in self.build_cards():
            if node_id in c.node_ids:
                return c
        return None

    # ── 2026-08-16 (显式连接图重构): 链接管理 ──
    def _migrate_links(self, cards: List["ConvCard"]) -> None:
        """旧模型关系迁移为显式连接 (每卡独立新模型, 一次性).

        优先级: parent_override (手动挂接) → branching/parallel 来源 (建卡来源)
        → 自动父链 (主线聚合)。统一转成 links (from 出 bottom → to 入 top)。
        迁移后 tree.links 即事实, 后续仅用户手动连接/断开。
        """
        card_ids = {c.id for c in cards}
        links: List[Dict[str, Any]] = []
        for c in cards:
            meta = (self.cards_meta or {}).get(c.id, {})
            ov = meta.get("parent_override") if meta and "parent_override" in meta else None
            if ov and ov in card_ids and ov != c.id:
                links.append({"id": uid("lk"), "from_card": ov, "from_port": "bottom",
                              "to_card": c.id, "to_port": "top", "kind": "manual"})
                continue
            src = (meta or {}).get("branching_source_id") or (meta or {}).get("parallel_source_id")
            if src and src in card_ids and src != c.id:
                links.append({"id": uid("lk"), "from_card": src, "from_port": "bottom",
                              "to_card": c.id, "to_port": "top", "kind": c.kind or "branching"})
                continue
            if c.parent_id and c.parent_id in card_ids and c.parent_id != c.id:
                links.append({"id": uid("lk"), "from_card": c.parent_id, "from_port": "bottom",
                              "to_card": c.id, "to_port": "top", "kind": "auto"})
        self.links = links

    def link_cards(self, from_card: str, to_card: str, from_port: str = "bottom",
                   to_port: str = "top", kind: str = "manual") -> Dict[str, Any]:
        """建立显式连接 (from 出锚点 → to 入锚点). 防自连/防重复.

        from_port: 'right'|'bottom' (出); to_port: 'left'|'top' (入)。
        """
        with self._lock:
            card_ids = {cc.id for cc in self.build_cards(load_messages=False)}
            if from_card not in card_ids or to_card not in card_ids:
                raise ValueError("source or target card not found")
            if from_card == to_card:
                raise ValueError("cannot link a card to itself")
            if from_port not in ("right", "bottom"):
                from_port = "bottom"
            if to_port not in ("left", "top"):
                to_port = "top"
            for lk in self.links:
                if lk.get("from_card") == from_card and lk.get("to_card") == to_card:
                    return lk                 # 已存在 → 幂等返回
            link = {"id": uid("lk"), "from_card": from_card, "from_port": from_port,
                    "to_card": to_card, "to_port": to_port, "kind": kind}
            self.links.append(link)
            self.version += 1
        self._emit()
        self.persist()
        return link

    def unlink_cards(self, link_id: Optional[str] = None,
                     from_card: Optional[str] = None, to_card: Optional[str] = None) -> bool:
        """断开连接 (按 link_id, 或按 from_card→to_card). 断开后卡片恢复独立."""
        with self._lock:
            before = len(self.links)
            if link_id:
                self.links = [lk for lk in self.links if lk.get("id") != link_id]
            else:
                self.links = [lk for lk in self.links
                              if not (lk.get("from_card") == from_card and lk.get("to_card") == to_card)]
            changed = len(self.links) != before
            if changed:
                self.version += 1
        if changed:
            self._emit()
            self.persist()
        return changed

    def _anchor_node_of_message(
        self,
        card: "ConvCard",
        source_message_id: Optional[str] = None,
        source_message_index: Optional[int] = None,
    ) -> "ConvNode":
        """定位源消息所属节点: 消息 id 优先, 缺 id 历史消息用卡片内全局索引回退.

        找不到具体消息 → 回退源卡头节点。
        """
        if card.node_ids:
            global_idx = 0
            for nid in card.node_ids:
                n = self.nodes.get(nid)
                if not n or n.v5_memory_id <= 0:
                    continue
                raw = self._load_fn([n.v5_memory_id]).get(n.v5_memory_id, "")
                try:
                    msgs = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    msgs = []
                if not isinstance(msgs, list):
                    msgs = []
                for m in msgs:
                    if source_message_id and m.get("id") == source_message_id:
                        return n
                    if source_message_index is not None and global_idx == source_message_index:
                        return n
                    global_idx += 1
        head_id = card.id[len("card_"):]
        return self.nodes.get(head_id) or next(iter(self.nodes.values()))

    def create_card_from_message(
        self,
        source_card_id: str,
        kind: str = "child",          # child | parallel | branching | manual
        messages: Optional[List[Dict[str, Any]]] = None,
        source_message_id: Optional[str] = None,
        source_message_index: Optional[int] = None,
        source_focus: str = "",
        title: Optional[str] = None,
        branch_label: Optional[str] = None,
    ) -> "ConvCard":
        """从源卡某条消息创建新卡 (2026-08-15 poker 对齐: 子卡/发散卡/分支卡).

        - 定位源消息所属节点 (消息 id 优先, 历史无 id 消息用卡片内索引回退);
        - 新建回合节点 (parent = 源消息所在节点, force_branch) → 父节点成为分叉点,
          新节点天然成为卡片头 (自动聚合); cards_meta 记录 kind/来源标记/title/未读。
        """
        with self._lock:
            src_card: Optional[ConvCard] = None
            for c in self.build_cards(load_messages=False):
                if c.id == source_card_id:
                    src_card = c
                    break
            if src_card is None:
                raise ValueError(f"source card not found: {source_card_id}")
            anchor = self._anchor_node_of_message(src_card, source_message_id, source_message_index)
            msgs = _ensure_message_ids(messages or [])
            node = self.add_turn(
                msgs,
                parent_id=anchor.id,
                branch_label=branch_label,
                force_branch=True,
                title=title,
            )
            card_id = "card_" + node.id
            meta: Dict[str, Any] = {
                "kind": kind if kind in ("child", "parallel", "branching", "manual") else "manual",
                "title": title or "",
                "source_focus": source_focus or "",
                "is_unread": True,
            }
            if kind == "parallel":
                meta["parallel_source_id"] = source_card_id
                meta["parallel_source_message_id"] = source_message_id
            elif kind == "branching":
                meta["branching_source_id"] = source_card_id
                meta["branching_source_message_id"] = source_message_id
            elif kind == "child":
                meta["parent_source_id"] = source_card_id
                meta["parent_source_message_id"] = source_message_id
            self.cards_meta[card_id] = meta
            self.version += 1
        self._emit()
        self.persist()
        return self.card_of_node(node.id)

    # ── 分支链回溯 (对齐 ai.explore.poker 的 t6) ──────────────────────────
    def get_branch_chain(self, card_id: str) -> List[Dict[str, Any]]:
        """沿 branching_source_id 递归回溯分支继承链 (poker t6 的同构移植).

        返回 [最古老祖先卡 ... 紧邻当前卡的祖先卡] 的继承消息列表:
            [{card_id, source_card_id, at_message_id, inherited_messages}]
        其中 inherited_messages 是"站在该链节视角, 应继承的源卡消息子集":
          - at_message_id == "START"  → 继承 [] (从源卡开头重新开始, 不继承源卡消息)
          - at_message_id 为具体消息 id → 继承源卡 [0 .. 该消息] (含)
          - at_message_id 为 None      → 继承源卡全部消息 (整卡来源)
        chat 上下文 = sum(inherited_messages, []) + 当前卡 messages。
        """
        with self._lock:
            cards = {c.id: c for c in self.build_cards(load_messages=True)}
            chain: List[Dict[str, Any]] = []
            cur = cards.get(card_id)
            seen: set = set()
            while cur and cur.branching_source_id and cur.id not in seen:
                seen.add(cur.id)
                src = cards.get(cur.branching_source_id)
                if src is None:
                    break
                at = cur.branching_source_message_id
                if at == "START":
                    inherited: List[Dict[str, Any]] = []
                elif at:
                    idx = next((i for i, m in enumerate(src.messages) if m.get("id") == at), -1)
                    inherited = src.messages[:idx + 1] if idx >= 0 else []
                else:
                    inherited = list(src.messages)
                chain.insert(0, {
                    "card_id": cur.id,
                    "source_card_id": src.id,
                    "at_message_id": at,
                    "inherited_messages": _strip_msg_ids(inherited),
                })
                cur = src
            return chain

    def get_branch_context(self, card_id: str) -> List[Dict[str, Any]]:
        """分支卡的完整上下文: 所有继承链消息 + 当前卡消息 (poker fetchLLMStream 的 p)."""
        with self._lock:
            cards = {c.id: c for c in self.build_cards(load_messages=True)}
            cur = cards.get(card_id)
            if cur is None:
                raise ValueError(f"card not found: {card_id}")
            inherited: List[Dict[str, Any]] = []
            for link in self.get_branch_chain(card_id):
                inherited.extend(link["inherited_messages"])
            return _strip_msg_ids(inherited) + _strip_msg_ids(list(cur.messages))

    def set_card_branch_point(self, card_id: str, at_message_id: Optional[str]) -> "ConvCard":
        """事后调整分支卡的分支点 (对齐 poker effectiveBranchPointId).

        at_message_id: "START" (从源卡开头重来) | 源卡内某消息 id | None (整卡来源 / 清除精确锚点)。
        校验卡片存在且为 branching 来源; 写入 cards_meta.branching_source_message_id。
        """
        with self._lock:
            cards = {c.id: c for c in self.build_cards(load_messages=False)}
            card = cards.get(card_id)
            if card is None:
                raise ValueError(f"card not found: {card_id}")
            if not card.branching_source_id:
                raise ValueError(f"card {card_id} is not a branching card (no branching_source_id)")
            if at_message_id and at_message_id != "START":
                src = cards.get(card.branching_source_id)
                if src is None:
                    raise ValueError(f"branching source card not found: {card.branching_source_id}")
                # 校验消息 id 属于源卡
                src_full = self.build_cards(load_messages=True)
                src2 = next((c for c in src_full if c.id == card.branching_source_id), None)
                if src2 is not None and not any(m.get("id") == at_message_id for m in src2.messages):
                    raise ValueError(f"message {at_message_id} not found in source card {card.branching_source_id}")
            meta = self.cards_meta.setdefault(card_id, {})
            meta["branching_source_message_id"] = at_message_id
            self.version += 1
        self._emit()
        self.persist()
        return self.card_of_node(card_id[len("card_"):])

    def set_card_read(self, card_id: str, is_unread: bool) -> None:
        """设置卡片未读状态 (cards_meta 持久化)."""
        with self._lock:
            meta = self.cards_meta.setdefault(card_id, {})
            meta["is_unread"] = bool(is_unread)
            self.version += 1
        self._emit()
        self.persist()

    def set_card_parent(self, card_id: str, parent_card_id: Optional[str] = None) -> "ConvCard":
        """手动指定卡片父关系 (2026-08-15 科技树编排: 拖拽挂接/解除).

        - parent_card_id: 目标父卡 id; None → 清除覆盖, 恢复自动派生父级。
        - 校验: 卡存在 / 不能挂到自身 / 不能成环 (目标卡不得是 card 的后代)。
        - 覆盖存 cards_meta["parent_override"], build_cards 应用 (节点树不动)。
        """
        with self._lock:
            cards = {c.id: c for c in self.build_cards(load_messages=False)}
            if card_id not in cards:
                raise ValueError(f"card not found: {card_id}")
            meta = self.cards_meta.setdefault(card_id, {})
            if not parent_card_id:
                meta.pop("parent_override", None)
            else:
                if parent_card_id == card_id:
                    raise ValueError("cannot parent a card to itself")
                if parent_card_id not in cards:
                    raise ValueError(f"target card not found: {parent_card_id}")
                # 成环检查: 沿目标卡的最终父链上溯, 不得遇到 card_id
                cur: Optional[str] = parent_card_id
                seen: set = set()
                while cur:
                    if cur == card_id:
                        raise ValueError("cannot create a cycle")
                    if cur in seen:
                        break
                    seen.add(cur)
                    m = self.cards_meta.get(cur, {})
                    if "parent_override" in m:
                        cur = m.get("parent_override")
                    else:
                        cc = cards.get(cur)
                        cur = cc.parent_id if cc else None
                meta["parent_override"] = parent_card_id
            self.version += 1
        self._emit()
        self.persist()
        head_id = card_id[len("card_"):]
        return self.card_of_node(head_id)

    # ── 持久化 (拓扑 + v5_memory_id + summary, 不含对话本体) ──
    def serialize(self) -> str:
        with self._lock:
            payload = {
                "v": self.version,
                "schema": "super-conv-2.0",
                "root_id": self.root_id,
                "current_id": self.current_id,
                # S1: 持久化主线终点, 重启后主线语义不丢
                "trunk_id": self.trunk_id,
                "nodes": [n.to_dict() for n in self.nodes.values()],
                # 2026-08-15 (poker 对齐): 手动卡片元数据 (建卡标记/分支点/未读),
                # 缺省空 dict, 旧 JSON 兼容
                "cards_meta": self.cards_meta or {},
                # 2026-08-16 (显式连接图): 卡片显式连接 (多对多, 可断开)
                "links": self.links or [],
            }
        return json.dumps(payload, ensure_ascii=False)

    def persist(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / f"{self.persist_key}.json"
            # R9: 并发持久化安全 —— persist() 在引擎锁外被调用 (R1 设计),
            # 多线程同时写同一 .json.tmp 会互相截断, Windows 下 rename 期间
            # 文件被占 → WinError 32, 写全部丢失。每次用唯一 tmp 名,
            # os.replace 原子替换目标, 并发写同一目标 = 最后一次赢, 不丢文件。
            tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:12]}.tmp")
            tmp.write_text(self.serialize(), encoding="utf-8")
            # 2026-08-14: Windows 下目标文件可能被并发读者/杀软瞬时锁定,
            # os.replace 抛 WinError 5 (access denied) → 短退避重试 (最多 4 次).
            for _attempt in range(4):
                try:
                    tmp.replace(path)
                    break
                except OSError:
                    if _attempt == 3:
                        raise
                    time.sleep(0.05 * (_attempt + 1))
        except Exception as exc:
            logger.warning("persist failed for %s: %s", self.persist_key, exc)

    @classmethod
    def deserialize(cls, raw: str | dict, **kwargs: Any) -> "ConversationTree":
        data = json.loads(raw) if isinstance(raw, str) else raw
        t = cls(**kwargs)
        t.version = data.get("v", 0)
        t.root_id = data.get("root_id")
        t.current_id = data.get("current_id")
        # S1: 读取主线终点; 旧 JSON (无 trunk_id) 时按最深的 trunk 节点推断:
        #   优先取"从根出发的最深 trunk 链末端" (旧树主线 ≈ 首条链)。
        t.trunk_id = data.get("trunk_id")
        t.nodes = {n["id"]: ConvNode.from_dict(n) for n in data.get("nodes", [])}
        # 2026-08-15 (poker 对齐): 手动卡片元数据 (旧 JSON 缺省空 → 纯自动聚合)
        t.cards_meta = data.get("cards_meta") or {}
        # 2026-08-16 (显式连接图): 显式连接; 旧 JSON 无 links → 标记待迁移 (build_cards 首次)
        t.links = data.get("links") or []
        t._links_pending_migration = "links" not in data

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

        # S1 兼容: trunk_id 缺失时按 node_type 推断主线终点 —— 找最深的 trunk 链末端
        # (优先从根的首条子链向下: 根 → 第一个 trunk 子 → 其第一个 trunk 子 …)
        if t.trunk_id is None and t.root_id is not None:
            root = t.nodes.get(t.root_id)
            if root:
                cur = root
                # 根的首个 trunk 子节点
                nxt = next((c for c in cur.children
                            if t.nodes.get(c) and t.nodes[c].node_type == "trunk"), None)
                if nxt is None:
                    nxt = next((c for c in cur.children if c in t.nodes), None)
                cur = t.nodes.get(nxt) if nxt else None
                while cur:
                    # 沿"首个子节点"链下探 (旧树主线 = 最左路径)
                    child = next((c for c in cur.children
                                  if t.nodes.get(c) and t.nodes[c].node_type == "trunk"), None)
                    if child is None:
                        break
                    cur = t.nodes.get(child)
                if cur is not None:
                    t.trunk_id = cur.id
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
        """默认跨分支检索: 统一走 v5.memory_retrieval.unified_retrieve (P6 收敛)."""
        try:
            from v5 import memory_retrieval
            return memory_retrieval.unified_retrieve(query, scope="semantic",
                                                     top_k=top_k, **kwargs)
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
        # 带 node/branch/session 树域标签, 供 tree_scoped_retrieve 做树域+会话过滤 (fact 同样适用)
        sess_tag = f" session:{self.tree.persist_key}" if hasattr(self.tree, "persist_key") else ""
        new_tags = f"{base_tags} {_tree_tag(node_id, branch_label)}{sess_tag}".strip() if (node_id or base_tags) else base_tags
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
    "ConvCard",
    "MemoryRetriever",
    "ToolCall",
    "NodeInsight",
    "uid",
    "tokenize",
    "V5_DATA_DIR",
]
