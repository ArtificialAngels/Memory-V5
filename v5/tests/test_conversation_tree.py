"""v5 对话树引擎单元测试 (V5 store 集成版).

覆盖: 树形分叉语义 / 上下文重建 (从 store 批量回读) / 定向记忆 /
       持久化往返 / 可注入 store 后端.

所有测试使用 MockStore 模拟 V5 store 后端, 不依赖真实 SQLite/Chroma.
"""
import json
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # core  (v5 包所在)

import v5.conversation_tree as ct  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_v5_data_dir(tmp_path, monkeypatch):
    """隔离持久化: 树 JSON 写入 tmp 而非真实 data/v5 (防测试污染生产数据目录).

    2026-08-14: 此前各 persist_key (test_tree / st / test_ctx / test_schema ...)
    直接把 JSON 写到 core/v5/data/v5/, 污染生产数据目录.
    """
    monkeypatch.setattr(ct, "V5_DATA_DIR", tmp_path)


# ────────────────────────── Mock V5 Store ──────────────────────────
class MockStore:
    """内存 store: 模拟 V5 store.store / get_batch / search 接口."""

    def __init__(self):
        self.mem: dict[int, str] = {}  # {memory_id: json_content}
        self.counter = 0

    def store(self, content: str, type: str = "conversation",
              weight: float = 0.6, tags: str = "", **kwargs) -> int:
        self.counter += 1
        self.mem[self.counter] = content
        return self.counter

    def load(self, ids: list[int]) -> dict[int, str]:
        return {mid: self.mem[mid] for mid in ids if mid in self.mem}

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        results = []
        q = query.lower()
        for mid, content in self.mem.items():
            if q in content.lower():
                results.append({"id": mid, "content": content})
        return results[:top_k]

    def cross_retrieve(self, query: str, top_k: int = 10,
                       **kwargs) -> list[dict]:
        """模拟跨分支向量检索: 回退到 FTS5 结果, 附上 score."""
        results = self.search(query, top_k)
        return [
            {"id": str(r["id"]), "content": r["content"],
             "score": 0.7, "raw": 0.7}
            for r in results
        ]


# ── fixtures ──

def _build_demo_ms():
    """创建 MockStore + 注入树的便捷函数. 返回 (tree, nodes_dict)."""
    ms = MockStore()
    t = ct.ConversationTree(
        persist_key="test_tree",
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    t.init([{"role": "system", "content": "root"}])
    a = t.add_turn(
        [
            {"role": "user", "content": "What is GTO in poker?"},
            {"role": "assistant", "content": "Game theory optimal"},
        ],
        branch_label="main",
    )
    b = t.add_turn(
        [
            {"role": "user", "content": "Nash equilibrium?"},
            {"role": "assistant", "content": "no unilateral gain"},
        ],
        branch_label="main",
    )
    c = t.add_turn(
        [
            {"role": "user", "content": "river bluff code"},
            {"role": "assistant", "content": "bluff freq 29%"},
        ],
        branch_label="main",
    )
    d = t.branch_from(
        a.id,
        [
            {"role": "user", "content": "explain via neural nets"},
            {"role": "assistant", "content": "self-play converges"},
        ],
        branch_label="ml",
    )
    e = t.add_turn(
        [
            {"role": "user", "content": "attention relation"},
            {"role": "assistant", "content": "soft policy"},
        ],
        branch_label="ml",
    )
    t.jump_to(c.id)
    return t, dict(a=a, b=b, c=c, d=d, e=e), ms


def _build_demo():
    t, n, _ = _build_demo_ms()
    return t, n


def _seed_memories(retriever: ct.MemoryRetriever,
                   nodes: dict) -> list[dict]:
    """通过 retriever.add_memory 注入测试记忆 (绑定 node_id)."""
    memories = [
        {"text": "User exploring GTO poker strategy",
         "tags": ["poker", "gto"], "node_id": nodes["a"].id, "branch": "main"},
        {"text": "Nash equilibrium definition",
         "tags": ["nash", "game-theory"], "node_id": nodes["b"].id, "branch": "main"},
        {"text": "river bluff frequency formula",
         "tags": ["poker", "code", "bluff"], "node_id": nodes["c"].id, "branch": "main"},
        {"text": "bridged GTO to neural network self-play",
         "tags": ["ml", "neural", "game-theory"], "node_id": nodes["d"].id, "branch": "ml"},
        {"text": "attention as soft policy",
         "tags": ["ml", "attention", "game-theory"], "node_id": nodes["e"].id, "branch": "ml"},
    ]
    for mem in memories:
        retriever.add_memory(mem)
    return memories


# ── 1. 树形对话管理 ──

def test_branch_fork_semantics():
    t, n = _build_demo()
    assert n["a"].children == [n["b"].id, n["d"].id], n["a"].children
    assert len(t.nodes) == 6
    assert t.get_node(n["d"].id).parent_id == n["a"].id


def test_prune_removes_subtree_and_repoints_current():
    t, n = _build_demo()
    t.jump_to(n["d"].id)
    t.prune(n["d"].id)
    assert n["d"].id not in t.nodes
    assert n["e"].id not in t.nodes
    assert n["d"].id not in t.get_node(n["a"].id).children
    assert t.current_id == n["a"].id


def test_siblings_and_subtree():
    t, n = _build_demo()
    assert set(t.siblings(n["b"].id)) == {n["d"].id}
    sub = {x.id for x in t.subtree(n["a"].id)}
    assert sub == {n["a"].id, n["b"].id, n["c"].id, n["d"].id, n["e"].id}


# ── 2. 上下文匹配 (从 store 批量回读) ──

def test_context_reconstruction_main_path():
    t, n = _build_demo()
    ctx = t.get_context(n["c"].id)
    assert len(ctx) == 7   # system + 3 turns * 2
    depths = [x.depth for x in t.get_path(n["c"].id)]
    assert depths == [0, 1, 2, 3]
    assert ctx[1]["content"] == "What is GTO in poker?"


def test_context_reconstruction_cross_branch():
    t, n = _build_demo()
    ctx = t.get_context(n["e"].id)
    assert len(ctx) == 7   # root -> a -> d -> e
    assert ctx[-1]["content"] == "soft policy"
    assert not any("Nash" in m.get("content", "") for m in ctx)


def test_context_with_meta_carries_branch():
    t, n = _build_demo()
    meta = t.get_context_with_meta(n["c"].id)
    assert meta[-1]["branch_label"] == "main"
    assert all("messages" in m for m in meta)


# ── 3. 定向记忆拉取 (走 MockStore FTS5 + cross retriever) ──

def test_path_memories_anchored_and_shown():
    t, n, ms = _build_demo_ms()
    r = ct.MemoryRetriever(t, _cross_retriever=ms.cross_retrieve)
    _seed_memories(r, n)  # 用 retriever.add_memory 绑定 node_id
    res = r.retrieve(n["c"].id)
    # 路径 c: root→a→b→c, 应该有 a,b,c 的记忆(3 条)
    path_texts = [x["mem"]["text"] for x in res["path"]]
    assert any("poker" in x.lower() for x in path_texts), f"path_texts={path_texts}"
    assert any("nash" in x.lower() for x in path_texts), f"path_texts={path_texts}"
    assert any("bluff" in x.lower() for x in path_texts), f"path_texts={path_texts}"


def test_cross_branch_filtered_by_relevance():
    t, n, ms = _build_demo_ms()
    r = ct.MemoryRetriever(t, _cross_retriever=ms.cross_retrieve)
    _seed_memories(r, n)
    res = r.retrieve(n["c"].id)
    for x in res["cross"]:
        assert x["relevance"] >= r.min_relevance


def test_cross_relevance_boost_on_shared_label():
    t, n, ms = _build_demo_ms()
    r = ct.MemoryRetriever(t, _cross_retriever=ms.cross_retrieve)
    _seed_memories(r, n)
    res = r.retrieve(n["e"].id)
    path_texts = [x["mem"]["text"] for x in res["path"]]
    # path 含 ml 分支的记忆 (d 和 e)
    assert any("neural" in x.lower() for x in path_texts), path_texts
    # cross 可能出现 main 分支记忆 (概念重叠)
    for x in res["cross"]:
        assert x["source"] == "cross"


def test_cross_branch_disabled_returns_empty():
    t, n, ms = _build_demo_ms()
    r = ct.MemoryRetriever(t, cross_branch_enabled=False,
                           _cross_retriever=ms.cross_retrieve)
    _seed_memories(r, n)
    res = r.retrieve(n["c"].id)
    assert res["cross"] == []


# ── 4. 节点状态同步 / 持久化 ──

def test_persistence_roundtrip(tmp_path):
    ms = MockStore()
    data_dir = tmp_path / "data" / "v5"
    t2 = ct.ConversationTree(
        persist_key="rt", data_dir=data_dir,
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    t2.init([{"role": "system", "content": "root"}])
    aa = t2.add_turn(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        branch_label="main",
    )
    bb = t2.add_turn(
        [{"role": "user", "content": "more"}],
        branch_label="main",
    )
    t2.jump_to(bb.id)

    # 反序列化 (内容在 store 里, 树 JSON 只存拓扑)
    ser = t2.serialize()
    t3 = ct.ConversationTree.deserialize(
        ser, persist_key="rt", data_dir=data_dir,
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    assert t3.root_id == t2.root_id
    assert t3.current_id == bb.id
    assert len(t3.nodes) == 3
    assert len(t3.get_context(bb.id)) == 4  # root(system) + aa(2) + bb(1)

    # 从磁盘 load
    t4 = ct.ConversationTree.load(
        persist_key="rt", data_dir=data_dir,
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    assert t4 is not None
    assert t4.current_id == bb.id
    assert len(t4.get_context(aa.id)) == 3


def test_node_independent_state_inherited_then_isolated():
    ms = MockStore()
    t = ct.ConversationTree(
        persist_key="st",
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    t.init([])
    root = t.get_node(t.root_id)
    root.state["progress"] = 0.5
    child = t.add_turn([{"role": "user", "content": "x"}], branch_label="main")
    assert child.state["progress"] == 0.5
    child.state["progress"] = 0.9
    assert root.state["progress"] == 0.5
    assert child.state["progress"] == 0.9


def test_load_missing_returns_none(tmp_path):
    assert ct.ConversationTree.load(persist_key="nope", data_dir=tmp_path) is None


# ── 5. 检索 / 可视化 ──

def test_search_finds_node_by_content():
    t, n, ms = _build_demo_ms()
    # conversation 内容存在 MockStore 里, search 走 FTS5 子串匹配
    res = t.search("Nash")
    ids = [r["node_id"] for r in res]
    assert n["b"].id in ids


def test_to_mermaid_includes_nodes_and_edges():
    t, n = _build_demo()
    mm = t.to_mermaid()
    assert mm.startswith("graph TD")
    for nid in t.nodes:
        assert nid in mm
    assert "-->" in mm


# ── 6. 新增: store 集成 ──

def test_add_turn_stores_messages_via_provider():
    ms = MockStore()
    t = ct.ConversationTree(
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    t.init([])
    node = t.add_turn(
        [{"role": "user", "content": "hello"},
         {"role": "assistant", "content": "hi there"}],
    )
    assert node.v5_memory_id > 0
    # store 中存了 JSON
    stored = ms.mem[node.v5_memory_id]
    parsed = json.loads(stored)
    assert parsed[0]["content"] == "hello"


def test_get_context_reads_from_store():
    ms = MockStore()
    t = ct.ConversationTree(
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    t.init([])
    t.add_turn([{"role": "user", "content": "msg1"}])
    t.add_turn([{"role": "user", "content": "msg2"}])
    ctx = t.get_context()
    assert len(ctx) == 2
    assert ctx[0]["content"] == "msg1"
    assert ctx[1]["content"] == "msg2"


def test_summary_extraction():
    ms = MockStore()
    t = ct.ConversationTree(
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    t.init([])
    node = t.add_turn([
        {"role": "user", "content": "A" * 100},  # long message, truncated to 80
        {"role": "assistant", "content": "reply"},
    ])
    assert len(node.summary) == 80
    assert node.summary == "A" * 80


def test_memory_retriever_add_memory_goes_to_store():
    t, n, _ = _build_demo_ms()
    r = ct.MemoryRetriever(t)
    result = r.add_memory({"text": "test fact", "node_id": n["a"].id})
    assert result["id"] > 0  # 写入了 MockStore
    # 路径检索能找到
    res = r.retrieve(n["a"].id)
    path_texts = [x["mem"]["text"] for x in res["path"]]
    assert any("test fact" in x for x in path_texts)


# ────────────────────────── v2 新功能测试 ──────────────────────────

class TestConvNodeV2:
    """v2 数据模型测试."""

    def test_default_fields(self):
        n = ct.ConvNode(id="test")
        assert n.node_type == "trunk"
        assert n.is_valid is True
        assert n.merge_target is None
        assert n.merged_from == []
        assert n.skills_used == []
        assert n.tool_calls == []
        assert n.conclusions == []
        # B2: 执行状态默认 idle
        assert n.exec_state == "idle"
        assert n.exec_progress == 0.0
        assert n.exec_detail == ""

    def test_serialization_roundtrip(self):
        n = ct.ConvNode(
            id="n1", node_type="branch", is_valid=True,
            skills_used=["web_search"],
            tool_calls=[ct.ToolCall(name="web_search", params={"q": "test"},
                                     result_summary="result", duration_ms=100)],
            conclusions=[ct.NodeInsight(text="insight1", confidence=0.9)],
            # B2: exec_state 字段
            exec_state="working", exec_progress=0.5, exec_detail="agent running",
        )
        d = n.to_dict()
        n2 = ct.ConvNode.from_dict(d)
        assert n2.node_type == "branch"
        assert n2.skills_used == ["web_search"]
        assert len(n2.tool_calls) == 1
        assert n2.tool_calls[0].name == "web_search"
        assert len(n2.conclusions) == 1
        assert n2.conclusions[0].text == "insight1"
        # B2: exec_state 序列化往返
        assert n2.exec_state == "working"
        assert n2.exec_progress == 0.5
        assert n2.exec_detail == "agent running"

    def test_v1_backward_compat(self):
        """v1 JSON (无 v2 字段) 反序列化后应有默认值."""
        v1 = {"id": "old", "summary": "test", "branch_label": "alt"}
        n = ct.ConvNode.from_dict(v1)
        assert n.node_type == "trunk"  # 默认值
        assert n.is_valid is True


class StubBus:
    """最小事件总线桩: 记录 publish 的事件 (dict)."""
    def __init__(self):
        self.events = []
    def publish(self, e):
        self.events.append(e)


class TestExecState:
    """B2: set_exec_state 状态机 + 事件发布 + 持久化."""

    def test_transition_publishes_and_persists(self, tmp_path):
        ms = MockStore()
        bus = StubBus()
        t = ct.ConversationTree(
            persist_key="test_exec", data_dir=tmp_path,
            _store=ms.store, _load=ms.load, _search=ms.search,
        )
        t.init([{"role": "system", "content": "root"}])
        n1 = t.add_turn([{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "hello"}])
        t.event_bus = bus
        assert n1.exec_state == "idle"
        # idle -> working: 发布 + 持久化
        n = t.set_exec_state(n1.id, "working", progress=0.2, detail="agent running")
        assert n.exec_state == "working"
        assert n.exec_progress == 0.2
        assert n.exec_detail == "agent running"
        assert len(bus.events) == 1
        ev = bus.events[0]
        assert ev["type"] == "node.exec_state_changed"
        assert ev["data"]["node_id"] == n1.id
        assert ev["data"]["exec_state"] == "working"
        assert ev["data"]["prev_state"] == "idle"
        # 落盘 JSON 含 exec_state
        raw = (tmp_path / "test_exec.json").read_text(encoding="utf-8")
        assert '"exec_state": "working"' in raw

    def test_progress_only_publishes_not_persist(self, tmp_path):
        ms = MockStore()
        bus = StubBus()
        t = ct.ConversationTree(
            persist_key="test_exec2", data_dir=tmp_path,
            _store=ms.store, _load=ms.load, _search=ms.search,
        )
        t.init([{"role": "system", "content": "root"}])
        n1 = t.add_turn([{"role": "user", "content": "hi"}])
        t.event_bus = bus
        t.set_exec_state(n1.id, "working")
        bus.events.clear()
        # 同状态仅更新进度: 仍发布事件, 但不触发新一轮 persist (transitioned=False)
        n = t.set_exec_state(n1.id, "working", progress=0.6)
        assert n.exec_progress == 0.6
        assert len(bus.events) == 1

    def test_unknown_state_normalized(self, tmp_path):
        ms = MockStore()
        t = ct.ConversationTree(
            persist_key="test_exec3", data_dir=tmp_path,
            _store=ms.store, _load=ms.load, _search=ms.search,
        )
        t.init([{"role": "system", "content": "root"}])
        n1 = t.add_turn([{"role": "user", "content": "hi"}])
        n = t.set_exec_state(n1.id, "bogus")
        assert n.exec_state == "unknown"

    def test_missing_node_raises(self, tmp_path):
        ms = MockStore()
        t = ct.ConversationTree(
            persist_key="test_exec4", data_dir=tmp_path,
            _store=ms.store, _load=ms.load, _search=ms.search,
        )
        t.init([{"role": "system", "content": "root"}])
        import pytest as _pytest
        with _pytest.raises(KeyError):
            t.set_exec_state("nope", "working")


class TestForkMerge:
    """fork / merge / conclude / abandon / is_valid 测试."""

    def _build(self):
        ms = MockStore()
        t = ct.ConversationTree(
            persist_key="test_v2",
            _store=ms.store, _load=ms.load, _search=ms.search,
        )
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ])
        b = t.add_turn([
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ])
        return t, ms, a, b

    def test_fork_branch(self):
        t, ms, a, b = self._build()
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt Q"},
            {"role": "assistant", "content": "Alt A"},
        ])
        assert br.node_type == "branch"
        assert br.branch_label == "alt"
        assert br.parent_id == a.id
        assert br.id in a.children
        assert t.is_valid_branch(br.id) is True

    def test_conclude_branch(self):
        t, ms, a, b = self._build()
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt Q"},
            {"role": "assistant", "content": "Alt A"},
        ])
        t.conclude_branch(br.id, ["Python is faster"])
        assert br.node_type == "conclusion"
        assert len(br.conclusions) == 1
        assert br.conclusions[0].text == "Python is faster"

    def test_merge_branch(self):
        t, ms, a, b = self._build()
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt Q"},
            {"role": "assistant", "content": "Alt A"},
        ])
        t.conclude_branch(br.id, ["Python is faster"])
        t.merge_branch(br.id, b.id)
        assert br.merge_target == b.id
        assert br.id in b.merged_from
        assert any("Python is faster" in c.text for c in b.conclusions)
        assert "merged_insights" in b.state

    def test_merge_non_trunk_fails(self):
        t, ms, a, b = self._build()
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt Q"},
        ])
        br2 = t.fork_branch(b.id, "alt2", [
            {"role": "user", "content": "Alt2 Q"},
        ])
        with pytest.raises(ValueError, match="must be trunk"):
            t.merge_branch(br.id, br2.id)

    def test_unmerge_branch(self):
        t, ms, a, b = self._build()
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt Q"},
        ])
        t.conclude_branch(br.id, ["Insight"])
        t.merge_branch(br.id, b.id)
        t.unmerge_branch(br.id)
        assert br.merge_target is None
        assert br.id not in b.merged_from
        assert not any("Insight" in c.text for c in b.conclusions)

    def test_abandon_branch(self):
        t, ms, a, b = self._build()
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt Q"},
        ])
        t.abandon_branch(br.id)
        assert br.is_valid is False
        assert t.is_valid_branch(br.id) is False

    def test_is_valid_abandoned_returns_false(self):
        """已废弃分支即使祖先有 trunk 也应返回 False."""
        t, ms, a, b = self._build()
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt Q"},
        ])
        assert t.is_valid_branch(br.id) is True
        t.abandon_branch(br.id)
        assert t.is_valid_branch(br.id) is False


class TestContextV2:
    """build_context_v2 测试."""

    def test_includes_siblings(self):
        ms = MockStore()
        t = ct.ConversationTree(
            persist_key="test_ctx",
            _store=ms.store, _load=ms.load, _search=ms.search,
        )
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ])
        b = t.add_turn([
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ])
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt"},
            {"role": "assistant", "content": "AltA"},
        ])
        ctx = t.build_context_v2(b.id)
        sys_content = ctx[0]["content"]
        assert "Sibling" in sys_content or "sibling" in sys_content

    def test_includes_merged_conclusions(self):
        ms = MockStore()
        t = ct.ConversationTree(
            persist_key="test_ctx2",
            _store=ms.store, _load=ms.load, _search=ms.search,
        )
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ])
        b = t.add_turn([
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ])
        br = t.fork_branch(a.id, "alt", [
            {"role": "user", "content": "Alt"},
        ])
        t.conclude_branch(br.id, ["Python is faster"])
        t.merge_branch(br.id, b.id)
        ctx = t.build_context_v2(b.id)
        sys_content = ctx[0]["content"]
        assert "merged" in sys_content
        assert "Python is faster" in sys_content


class TestSerializationV2:
    """持久化 v2 格式测试."""

    def test_schema_marker(self):
        ms = MockStore()
        t = ct.ConversationTree(
            persist_key="test_schema",
            _store=ms.store, _load=ms.load, _search=ms.search,
        )
        t.init([{"role": "system", "content": "root"}])
        s = t.serialize()
        d = json.loads(s)
        assert d["schema"] == "super-conv-2.0"

    def test_v1_json_auto_migration(self):
        """v1 JSON (无 schema 字段) 反序列化后 node_type 推断."""
        v1 = json.dumps({
            "v": 1, "root_id": "r1", "current_id": "n1",
            "nodes": [
                {"id": "r1", "parent_id": None, "children": ["n1"],
                 "depth": 0, "branch_label": None, "v5_memory_id": 1,
                 "summary": "root", "state": {}, "config": {}, "meta": {},
                 "created_at": 1000},
                {"id": "n1", "parent_id": "r1", "children": [],
                 "depth": 1, "branch_label": "alt", "v5_memory_id": 2,
                 "summary": "branch", "state": {}, "config": {}, "meta": {},
                 "created_at": 1001},
            ]
        })
        t = ct.ConversationTree.deserialize(v1)
        assert t.nodes["r1"].node_type == "trunk"
        assert t.nodes["n1"].node_type == "branch"

# ────────────────────────── 卡片视图 (poker 对齐, 2026-08-15) ──────────────────────────

class TestCards:
    """build_cards: 卡片 = 一段多轮会话 (分叉点切分聚合)."""

    def _chain_tree(self):
        """ROOT -> A -> B -> C (无分叉): 应聚合为单张卡."""
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_cards", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}])
        b = t.add_turn([{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}])
        c = t.add_turn([{"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}])
        return t, {"a": a.id, "b": b.id, "c": c.id}

    def test_single_chain_one_card(self):
        t, ids = self._chain_tree()
        cards = t.build_cards()
        assert len(cards) == 1
        card = cards[0]
        assert card.id == "card_" + t.root_id
        assert card.node_ids == [t.root_id, ids["a"], ids["b"], ids["c"]]
        # 消息串联 = 一段连续会话 (root system + 3 回合)
        roles = [m["role"] for m in card.messages]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user", "assistant"]
        assert card.parent_id is None
        assert card.children == []

    def test_fork_splits_cards(self):
        """ROOT -> A -> B -> C, B -> D -> E: 主线连续聚合 (A/B/C 一张卡) + 分支卡 (D/E)."""
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_cards2", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([{"role": "user", "content": "q1"}])
        b = t.add_turn([{"role": "user", "content": "q2"}])
        c = t.add_turn([{"role": "user", "content": "q3"}])
        d = t.branch_from(b.id, [{"role": "user", "content": "f1"}], branch_label="fork")
        e = t.add_turn([{"role": "user", "content": "f2"}])
        cards = t.build_cards()
        by_id = {c.id: c for c in cards}
        # 主线 A/B/C 连续聚合为一张卡 (不再被分叉点切成薄卡); 分支 D/E 独立成卡
        assert len(cards) == 2, [c.id for c in cards]
        root_card = by_id["card_" + t.root_id]
        assert root_card.node_ids == [t.root_id, a.id, b.id, c.id]
        d_card = by_id["card_" + d.id]
        assert d_card.node_ids == [d.id, e.id]
        # 卡片父链: 分支卡挂回主线卡
        assert d_card.parent_id == root_card.id
        assert root_card.children == [d_card.id]
        # 分支卡带 branch_label
        assert d_card.branch_label == "fork"

    def test_cards_meta_merge_and_roundtrip(self):
        """cards_meta (标题/未读/分支点标记) 持久化往返并合并进自动卡."""
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_cards3", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([{"role": "user", "content": "q1"}])
        b = t.add_turn([{"role": "user", "content": "q2"}])
        d = t.branch_from(a.id, [{"role": "user", "content": "f1"}], branch_label="fork")
        card_id = "card_" + d.id
        t.cards_meta[card_id] = {
            "kind": "branching",
            "branching_source_id": "card_" + a.id,
            "branching_source_message_id": "msg_x",
            "title": "分支探索",
            "is_unread": True,
        }
        # 往返
        raw = t.serialize()
        t2 = ct.ConversationTree.deserialize(raw, persist_key="test_cards3",
                                             _store=ms.store, _load=ms.load, _search=ms.search)
        assert t2.cards_meta[card_id]["title"] == "分支探索"
        cards = {c.id: c for c in t2.build_cards()}
        card = cards[card_id]
        assert card.title == "分支探索"
        assert card.is_unread is True
        assert card.kind == "branching"
        assert card.branching_source_id == "card_" + a.id
        assert card.branching_source_message_id == "msg_x"
        # 旧 JSON 无 cards_meta → 兼容
        v1 = json.loads(raw); v1.pop("cards_meta")
        t3 = ct.ConversationTree.deserialize(json.dumps(v1))
        assert t3.cards_meta == {}
        # 主线 b 连续聚合 (root,a,b 一张卡) + 分支 d 独立 → 2 张卡
        assert len(t3.build_cards()) == 2

    def test_card_of_node(self):
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_cards4", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([{"role": "user", "content": "q1"}])
        b = t.add_turn([{"role": "user", "content": "q2"}])
        d = t.branch_from(a.id, [{"role": "user", "content": "f1"}])   # a 两孩子 → d 成卡头; b 主线并入 root 卡
        card = t.card_of_node(d.id)
        assert card is not None and card.id == "card_" + d.id
        # b 在主线上 → 属于 root 卡 (主线连续聚合)
        assert t.card_of_node(b.id).id == "card_" + t.root_id
        assert t.card_of_node("nope") is None

    def test_create_card_from_message(self):
        """从源卡某条消息建分支卡: 新节点成卡片头 + cards_meta 分支点标记 + 消息 id 补全."""
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_cards5", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([
            {"role": "user", "content": "q1", "id": "msg_q1"},
            {"role": "assistant", "content": "a1"},
        ])
        b = t.add_turn([{"role": "user", "content": "q2"}])
        src = "card_" + t.root_id          # a 在 root 卡内 (a 非卡片头)
        card = t.create_card_from_message(
            source_card_id=src, kind="branching",
            messages=[{"role": "user", "content": "继续深入 q1"}],
            source_message_id="msg_q1", source_focus="q1 选中文本",
            title="深入探索", branch_label="deep",
        )
        assert card.kind == "branching"
        meta = t.cards_meta[card.id]
        assert meta["kind"] == "branching"
        assert meta["branching_source_id"] == src
        assert meta["branching_source_message_id"] == "msg_q1"
        assert meta["source_focus"] == "q1 选中文本"
        # 新卡挂源消息所在节点 (a) 下 → a 成为分叉点, 新节点是卡片头
        assert card.node_ids[0] in t.get_node(a.id).children
        # 消息 id 自动补全 (add_turn 存库前); LLM 上下文剥离 id (见 test_context_strips_message_ids)
        card_msgs = t.build_cards()[0].messages or []
        assert any(m.get("id", "").startswith("msg_") for m in card_msgs)

    def test_set_card_read_roundtrip(self):
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_cards6", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        cid = "card_" + t.root_id
        t.set_card_read(cid, True)
        assert t.cards_meta[cid]["is_unread"] is True
        t2 = ct.ConversationTree.deserialize(t.serialize(),
                                             _store=ms.store, _load=ms.load, _search=ms.search)
        assert t2.cards_meta[cid]["is_unread"] is True
        assert t2.build_cards()[0].is_unread is True

    def test_context_strips_message_ids(self):
        """LLM 上下文剥离消息 id (OpenAI 兼容 API 严格字段); state 内联保留 id (前端定位)."""
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_cards7", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        t.add_turn([{"role": "user", "content": "q1", "id": "msg_keep"}])
        ctx = t.get_context(t.current_id)
        assert ctx and all("id" not in m for m in ctx)
        c = t.build_cards()
        assert any("id" in m for m in c[0].messages)
        assert t.get_context_with_meta(t.current_id)[-1]["messages"][0].get("id") is None

    def test_set_card_parent(self):
        """手动挂接 (科技树编排): parent_override 覆盖自动父链; 解除恢复; 成环/自挂拒绝."""
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_cards8", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([{"role": "user", "content": "q1"}])
        b = t.add_turn([{"role": "user", "content": "q2"}])
        d = t.branch_from(a.id, [{"role": "user", "content": "f1"}])   # b 主线并入 root 卡; d 分支卡
        cards = {c.id: c for c in t.build_cards()}
        root_card = cards["card_" + t.root_id]
        d_card = cards["card_" + d.id]
        assert b.id in root_card.node_ids and d_card.parent_id == root_card.id
        # 手动挂接: 卡 d 显式挂到 root 卡下 (override)
        t.set_card_parent(d_card.id, root_card.id)
        assert t.cards_meta[d_card.id]["parent_override"] == root_card.id
        cards2 = {c.id: c for c in t.build_cards()}
        assert cards2[d_card.id].parent_id == root_card.id
        assert d_card.id in cards2[root_card.id].children
        assert cards2[d_card.id].parent_override == root_card.id
        # 成环拒绝 (把 root 挂到 d 下)
        with pytest.raises(ValueError):
            t.set_card_parent(root_card.id, d_card.id)
        # 自挂拒绝
        with pytest.raises(ValueError):
            t.set_card_parent(d_card.id, d_card.id)
        # 解除 → 恢复自动
        t.set_card_parent(d_card.id, None)
        assert "parent_override" not in t.cards_meta[d_card.id]
        cards3 = {c.id: c for c in t.build_cards()}
        assert cards3[d_card.id].parent_id == root_card.id

    def test_links_migration(self):
        """旧 JSON (无 links) → build_cards 首次迁移默认链接 (父→子); inputs/outputs 派生."""
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_links_mig", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([{"role": "user", "content": "q1"}])
        b = t.add_turn([{"role": "user", "content": "q2"}])
        d = t.branch_from(a.id, [{"role": "user", "content": "f1"}])
        raw = t.serialize()
        data = json.loads(raw); data.pop("links", None)   # 模拟旧 JSON
        t2 = ct.ConversationTree.deserialize(json.dumps(data), _store=ms.store, _load=ms.load, _search=ms.search)
        assert t2._links_pending_migration is True
        cards = t2.build_cards()
        assert t2._links_pending_migration is False
        # 迁移: 主线 root{a,b} + 分支 d; link(root卡 → d卡, 自动)
        assert len(t2.links) == 1, t2.links
        lk = t2.links[0]
        assert lk["to_card"] == "card_" + d.id
        assert lk["from_card"] == "card_" + t.root_id
        assert lk["kind"] == "auto" or lk["kind"] == "manual"
        # inputs/outputs 派生
        cm = {c.id: c for c in t2.build_cards()}
        d_card = cm["card_" + d.id]
        root_card = cm["card_" + t.root_id]
        assert d_card.inputs and d_card.inputs[0]["from_card"] == root_card.id
        assert root_card.outputs and root_card.outputs[0]["to_card"] == d_card.id

    def test_link_unlink(self):
        """显式连接 (多对多, 可断开): 建立/幂等/自连拒绝/断开后独立."""
        ms = MockStore()
        t = ct.ConversationTree(persist_key="test_links2", _store=ms.store, _load=ms.load, _search=ms.search)
        t.init([{"role": "system", "content": "root"}])
        a = t.add_turn([{"role": "user", "content": "q1"}])
        b = t.add_turn([{"role": "user", "content": "q2"}])          # b parent a (主线)
        d = t.branch_from(a.id, [{"role": "user", "content": "f1"}]) # a 两孩子 → d 卡头
        cards = {c.id: c for c in t.build_cards()}
        root = "card_" + t.root_id
        d_card = "card_" + d.id
        # 建立
        lk = t.link_cards(root, d_card)
        assert any(x["from_card"] == root and x["to_card"] == d_card for x in t.links)
        # 幂等: 再次连接不新增
        lk2 = t.link_cards(root, d_card)
        assert lk2["id"] == lk["id"] and len(t.links) == 1
        # 自连拒绝
        import pytest
        with pytest.raises(ValueError):
            t.link_cards(root, root)
        # 断开 (按 from/to)
        assert t.unlink_cards(from_card=root, to_card=d_card) is True
        assert len(t.links) == 0
        # 断开后卡片恢复独立 (无 inputs/outputs)
        cards2 = {c.id: c for c in t.build_cards()}
        assert cards2[d_card].inputs == [] and cards2[root].outputs == []
