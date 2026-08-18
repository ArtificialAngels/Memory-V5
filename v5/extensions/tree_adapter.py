"""tree_adapter —— 树模式（chat tree）定向适配层。

背景
----
Ikaros 的对话是 **树** 而非线性流。但 V5 记忆架构与压缩算法最初都按「线性 transcript」
假设设计：全局语义检索不区分分支、压缩器把消息当扁平列表压中段、节点→记忆绑定只存在
内存里。本模块把记忆/压缩对树做定向适配，弥补四个具体缺口（均保持非侵入：包裹/新增，
不改动原函数体，便于回滚）：

  A. 存储打标 (tag_for_node)         —— 写入 V5 时附带 `node:<id>` / `branch:<label>` 标签，
                                         使检索能做树域过滤。
  B. 树域检索 (tree_scoped_retrieve)  —— 把全局语义检索按「当前路径节点 + 分支」重新加权/
                                         过滤，替代 hermes on_pre_compress 里的裸 `_v5_search`。
  C. 树感知压缩 (TreePathCompressor)  —— 按 *节点边界* 压缩：head(根+早期)/tail(近期) 保留完整
                                         消息，middle(远端祖先) 用节点自带 summary 替换（树拓扑即
                                         压缩结构），fork/merge 锚点始终可见。
  D. 树感知上下文组装 (build_tree_aware_context) —— L0 走 TreePathCompressor，L1 兄弟 + L2 合并
                                         结论照旧，产出可直接喂 LLM 的消息列表。

注意：本模块是 **骨架**。A/B 要真正生效，需在 core 存储入口补两行标签（见文件末尾
`CORE_HOOKS` 注释）；C/D 已自包含、可直接接 server.py 的 build_context_v2 替换。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("v5.extensions.tree_adapter")


# ───────────────────────────────────────────────────────────────────────────
# A. 存储打标
# ───────────────────────────────────────────────────────────────────────────
def tag_for_node(node_id: str, branch_label: Optional[str] = None) -> str:
    """生成应追加到 V5 记忆 tags 的树域标签。

    约定：`node:<id>` 绑定所属节点；`branch:<label>` 绑定所属分支。
    检索时按这些 tag 做路径/分支域过滤与加权（见 tree_scoped_retrieve）。
    """
    tags = [f"node:{node_id}"]
    if branch_label:
        tags.append(f"branch:{branch_label}")
    return " ".join(tags)


# ───────────────────────────────────────────────────────────────────────────
# B. 树域检索
# ───────────────────────────────────────────────────────────────────────────
def tree_scoped_retrieve(
    tree,
    node_id: str,
    query: str,
    top_k: int = 5,
    character: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """树感知语义检索：把全局相似度结果按「活动路径 / 分支」重新加权并限 top_k。

    权重规则（叠加在 V5 融合分上）：
      - 命中路径节点标签 `node:<path_id>`  → +0.40（确定相关，最高优先）
      - 命中分支标签 `branch:<label>`      → +0.20（同分支，次优先）
      - 其余全局命中                       → 原分（跨分支兜底）

    这替代了 hermes `on_pre_compress` 里裸的 `self._v5_search(query, top_k=5)`：
    在树模式下把记忆检索锁定到当前分支上下文，避免无关分支/全局噪声污染。

    要求：记忆条目已在存储时通过 `tag_for_node` 打过 `node:`/`branch:` 标签。
    """
    try:
        from v5.memory_retrieval import unified_retrieve as _v5_retrieve
    except Exception as exc:  # pragma: no cover - 防御性
        logger.debug("tree_scoped_retrieve: 无法导入 memory_retrieval (%s)", exc)
        return []

    path_nodes = tree.get_path(node_id)
    path_ids = {n.id for n in path_nodes}
    branch_labels = {n.branch_label for n in path_nodes if n.branch_label}

    # 会话隔离 (H1): 仅保留本会话(session:<persist_key>)或 legacy(无 session 标签)的记忆,
    # 杜绝其他会话/主 Ikaros 长期记忆串台. 记忆在存储时已带 session 标签 (conversation_tree).
    sess_tag = f"session:{getattr(tree, 'persist_key', '')}"

    try:
        # P6 收敛: 统一走 unified_retrieve(scope="semantic") (含时效过滤 + 统一形状)
        results = _v5_retrieve(query, scope="semantic", top_k=max(top_k * 3, 12),
                               character=character)
    except Exception as exc:
        logger.debug("tree_scoped_retrieve: _v5_retrieve 失败 (%s)", exc)
        return []

    scoped: List[tuple] = []
    for r in results:
        tagset = set((r.get("tags") or "").split())
        # 会话边界过滤: 有 session 标签但不属于本会话 → 排除
        tagged_sessions = {t for t in tagset if t.startswith("session:")}
        if tagged_sessions and sess_tag not in tagged_sessions:
            continue
        on_path = any(f"node:{pid}" in tagset for pid in path_ids)
        on_branch = any(f"branch:{b}" in tagset for b in branch_labels)
        score = float(r.get("score", r.get("raw", 0.5)))
        if on_path:
            score += 0.40
        elif on_branch:
            score += 0.20
        scoped.append((score, r, on_path, on_branch))

    scoped.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, r, on_path, on_branch in scoped[:top_k]:
        item = dict(r)
        item["tree_scope"] = "path" if on_path else ("branch" if on_branch else "global")
        item["tree_score"] = round(score, 4)
        out.append(item)
    return out


# ───────────────────────────────────────────────────────────────────────────
# C. 树感知压缩
# ───────────────────────────────────────────────────────────────────────────
class TreePathCompressor:
    """按 *节点边界* 压缩一条路径的对话历史（替代线性「压中段」）。

    输入：`get_context_with_meta(node_id)` 的结构化列表（root→leaf 有序），
    每项含 node_id / depth / branch_label / summary / messages。
    输出：扁平消息列表 [{role, content}, ...]，可直接喂 LLM。

    策略（与线性 ContextCompressor 的本质区别：压缩单元是「节点」而非「消息序号」）：
      - head_nodes：根 + 最早若干祖先 → 完整保留（对话的「种子/系统」区）。
      - tail_nodes：最近若干节点     → 完整保留（活动任务所在区，对应线性压缩的 tail 锚点）。
      - 中间节点                   → 用其自带 `summary` 替换整段 transcript。
                                    树拓扑本身就是压缩结构（每节点已维护 summary）。
      - fork/merge 锚点（branch_label 非空）→ 即便在中间也显式带出 `[branch:X]` 标记，
                                    因其是「为何分叉」的结构锚点，对应线性压缩需特判的
                                    last-user/last-assistant 锚点，但这里天然按节点保留。
      - 超预算 → 对最旧一端消息再用 token_compressor 瘦身（复用已验证的离线回退）。
    """

    def __init__(
        self,
        head_nodes: int = 2,
        tail_nodes: int = 3,
        budget_messages: int = 50,
    ):
        self.head_nodes = head_nodes
        self.tail_nodes = tail_nodes
        self.budget_messages = budget_messages

    def compress(
        self,
        meta_context: List[Dict[str, Any]],
        head_nodes: Optional[int] = None,
        tail_nodes: Optional[int] = None,
        budget_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        head_nodes = head_nodes if head_nodes is not None else self.head_nodes
        tail_nodes = tail_nodes if tail_nodes is not None else self.tail_nodes
        budget_messages = budget_messages if budget_messages is not None else self.budget_messages

        if not meta_context:
            return []

        n = len(meta_context)
        head_idx = set(range(min(head_nodes, n)))
        tail_start = max(0, n - min(tail_nodes, n))
        tail_idx = set(range(tail_start, n))

        out: List[Dict[str, Any]] = []
        for i, node in enumerate(meta_context):
            label = node.get("branch_label") or "trunk"
            summary = (node.get("summary") or "").strip()
            msgs = node.get("messages", []) or []
            is_head = i in head_idx
            is_tail = i in tail_idx

            if is_head or is_tail:
                # 完整保留；fork 锚点用一条轻量标记提示（不破坏消息本身）
                if node.get("branch_label"):
                    out.append({
                        "role": "system",
                        "content": f"[fork anchor · branch:{label}]",
                    })
                out.extend(msgs)
            else:
                # 中间节点：用 summary 替换整段（无 summary 则丢弃该节点冗余内容）
                if summary:
                    out.append({
                        "role": "system",
                        "content": f"(earlier branch '{label}' summary: {summary})",
                    })
        return self._enforce_budget(out, budget_messages)

    # ── 预算强制执行：对最旧一端再用 token_compressor 瘦身 ──────────────
    def _enforce_budget(
        self, messages: List[Dict[str, Any]], budget_messages: int
    ) -> List[Dict[str, Any]]:
        if len(messages) <= budget_messages:
            return messages
        try:
            from .token_compressor import compress_text
            has_tc = True
        except Exception:
            has_tc = False

        # 压缩最旧一端（列表前端），保留最近一端
        overflow = messages[: len(messages) - budget_messages]
        kept = messages[len(messages) - budget_messages:]
        compressed_overflow: List[Dict[str, Any]] = []
        for m in overflow:
            content = m.get("content", "") or ""
            if has_tc and len(content) > 80:
                content = compress_text(content, quality="auto")
            compressed_overflow.append({**m, "content": content})

        out = compressed_overflow + kept
        # 最终硬护栏：仍超预算则保留最近 budget_messages 并加压缩提示
        if len(out) > budget_messages + 4:
            out = [
                {"role": "system", "content": "(earlier branching history compressed)"}
            ] + out[-budget_messages:]
        return out


# ───────────────────────────────────────────────────────────────────────────
# D. 树感知上下文组装（L0 压缩 + L1 兄弟 + L2 合并，自包含，可直接替换 build_context_v2）
# ───────────────────────────────────────────────────────────────────────────
def build_tree_aware_context(
    tree,
    node_id: Optional[str] = None,
    include_siblings: bool = True,
    include_merged: bool = True,
    max_messages: int = 50,
    head_nodes: int = 2,
    tail_nodes: int = 3,
    system_prompt: Optional[str] = None,
    extra_memory: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """与 build_context_v2 对齐，但 L0 祖先历史改走 TreePathCompressor（树感知压缩）。

    返回 [system_msg, ...L0压缩后历史...]，可直接拼 [user_msg] 喂 LLM。

    system_prompt: 覆盖默认 system 身份（如注入 Ikaros 人格）。
    extra_memory:   额外的记忆文本块（如 tree_scoped_retrieve 的树域语义检索结果），
                    会作为独立段落拼进 system，使 V5 记忆引擎真正进入 chat。
    """
    ancestors = tree.get_path(node_id)
    meta = tree.get_context_with_meta(node_id)
    l0 = TreePathCompressor(
        head_nodes=head_nodes, tail_nodes=tail_nodes, budget_messages=max_messages
    ).compress(meta)

    # ── L1 兄弟节点摘要 ──
    sibling_text = ""
    if include_siblings:
        sibs_by_depth: Dict[int, List[str]] = {}
        for a in ancestors:
            for s in tree.get_sibling_nodes(a.id):
                label = s.branch_label or "branch"
                sm = s.summary[:60] if s.summary else ""
                sibs_by_depth.setdefault(a.depth, []).append(f"[{label}]: {sm}")
        if sibs_by_depth:
            sibling_text = "Sibling branches at each level:\n" + "\n".join(
                f"  depth {d}: " + "\n    ".join(e)
                for d, e in sorted(sibs_by_depth.items())
            )

    # ── L2 已合并分支结论 ──
    merged_text = ""
    if include_merged:
        insights: List[str] = []
        for a in ancestors:
            for bf in a.merged_from:
                br = tree.get_node(bf)
                if br:
                    for c in br.conclusions:
                        insights.append(f"[merged:{br.branch_label}] {c.text}")
        if insights:
            merged_text = "Merged branch conclusions:\n" + "\n".join(
                f"  - {i}" for i in insights
            )

    system_parts = [
        system_prompt or "You are Ikaros, an AI with deep conversation memory and branching context."
    ]
    if extra_memory and extra_memory.strip():
        system_parts.append("Relevant memories (V5):\n" + extra_memory.strip())
    if merged_text:
        system_parts.append(merged_text)
    if sibling_text:
        system_parts.append(sibling_text)

    return [{"role": "system", "content": "\n\n".join(system_parts)}] + l0


# ───────────────────────────────────────────────────────────────────────────
# CORE_HOOKS —— 要让 A/B 真正生效，需在 core 存储入口补的两处（非本模块内）
# ───────────────────────────────────────────────────────────────────────────
# 1) core/v5/conversation_tree.py :: add_turn (约 L572)
#      现有: mid = self._store_fn(content, type="conversation", tags=tags)
#      改为: mid = self._store_fn(content, type="conversation",
#                                 tags=f"{tags} {tag_for_node(node.id, branch_label)}".strip())
#    这样对话内容记忆带 node/branch 标签（当前仅靠 node.v5_memory_id 反查，重启可恢复，
#    但全局检索无法树域过滤；打标后 tree_scoped_retrieve 才能工作）。
#
# 2) core/v5/conversation_tree.py :: MemoryRetriever.add_memory (约 L1118)
#      现有: mid = self.tree._store_fn(text, type="fact", tags=tags)
#      改为: mid = self.tree._store_fn(text, type="fact",
#                                 tags=f"{tags} {tag_for_node(node_id, ...)}".strip())
#    并建议把 node→memory 绑定持久化（见下），否则重启后路径检索仍空。
#
# 3) 持久化 node→fact 绑定（修 _node_memories 重启丢失）：
#    在 ConvNode 增加字段 `memory_ids: List[int] = []`，persist/reload 时一并存读；
#    MemoryRetriever.retrieve 的 path 分支优先读 node.memory_ids，回退 _node_memories。
#    （对话内容绑定本就靠 node.v5_memory_id 持久化，无需改；仅 fact 绑定缺持久化。）
# ───────────────────────────────────────────────────────────────────────────
