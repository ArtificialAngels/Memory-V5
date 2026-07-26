"""v5 对话树引擎单元测试.

覆盖: 树形分叉语义 / 上下文重建 / 定向记忆 (路径内 + 跨分支) / 持久化往返.
路径自举同仓库其它测试.
"""
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # core/v5  (v5 包所在)

import v5.conversation_tree as ct  # noqa: E402


# ── fixtures ──
def _build_demo() -> ct.ConversationTree:
    t = ct.ConversationTree(persist_key="test_tree")
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
    return t, dict(a=a, b=b, c=c, d=d, e=e)


def _mem_store(t, nodes) -> list:
    return [
        {"text": "User exploring GTO poker strategy", "tags": ["poker", "gto"], "node_id": nodes["a"].id, "branch": "main"},
        {"text": "Nash equilibrium definition", "tags": ["nash", "game-theory"], "node_id": nodes["b"].id, "branch": "main"},
        {"text": "river bluff frequency formula", "tags": ["poker", "code", "bluff"], "node_id": nodes["c"].id, "branch": "main"},
        {"text": "bridged GTO to neural network self-play", "tags": ["ml", "neural", "game-theory"], "node_id": nodes["d"].id, "branch": "ml"},
        {"text": "attention as soft policy", "tags": ["ml", "attention", "game-theory"], "node_id": nodes["e"].id, "branch": "ml"},
    ]


# ── 1. 树形对话管理 ──
def test_branch_fork_semantics():
    t, n = _build_demo()
    # branch_from(a) 应在 a 下创建子节点, 与 b 成兄弟
    assert n["a"].children == [n["b"].id, n["d"].id], n["a"].children
    assert len(t.nodes) == 6
    assert t.get_node(n["d"].id).parent_id == n["a"].id


def test_prune_removes_subtree_and_repoints_current():
    t, n = _build_demo()
    t.jump_to(n["d"].id)
    t.prune(n["d"].id)  # 删 d (及其子 e)
    assert n["d"].id not in t.nodes
    assert n["e"].id not in t.nodes
    # a 不再指向 d
    assert n["d"].id not in t.get_node(n["a"].id).children
    # 当前节点若在被删子树内, 应回退到祖先 a
    assert t.current_id == n["a"].id


def test_siblings_and_subtree():
    t, n = _build_demo()
    assert set(t.siblings(n["b"].id)) == {n["d"].id}
    sub = {x.id for x in t.subtree(n["a"].id)}
    assert sub == {n["a"].id, n["b"].id, n["c"].id, n["d"].id, n["e"].id}


# ── 2. 上下文匹配 ──
def test_context_reconstruction_main_path():
    t, n = _build_demo()
    ctx = t.get_context(n["c"].id)
    # system + 3 turns * 2 = 7
    assert len(ctx) == 7
    depths = [x.depth for x in t.get_path(n["c"].id)]
    assert depths == [0, 1, 2, 3]
    assert ctx[1]["content"] == "What is GTO in poker?"


def test_context_reconstruction_cross_branch():
    t, n = _build_demo()
    ctx = t.get_context(n["e"].id)
    # 路径 root -> a -> d -> e = 4 节点 => 7 条 (system + 3 turns)
    assert len(ctx) == 7
    assert ctx[-1]["content"] == "soft policy"
    # e 路径不含 b/c 的内容
    assert not any("Nash" in m.get("content", "") for m in ctx)


def test_context_with_meta_carries_branch():
    t, n = _build_demo()
    meta = t.get_context_with_meta(n["c"].id)
    assert meta[-1]["branch_label"] == "main"
    assert all("messages" in m for m in meta)


# ── 3. 定向记忆拉取 ──
def test_path_memories_anchored_and_shown():
    t, n = _build_demo()
    r = ct.MemoryRetriever(t, store=_mem_store(t, n))
    res = r.retrieve(n["c"].id)
    path_texts = [x["mem"]["text"] for x in res["path"]]
    # 路径内全部展示 (含 gto@a, nash@b, bluff@c)
    assert any("GTO poker" in x for x in path_texts)
    assert any("Nash" in x for x in path_texts)
    assert any("river bluff" in x for x in path_texts)


def test_cross_branch_filtered_by_relevance():
    t, n = _build_demo()
    r = ct.MemoryRetriever(t, store=_mem_store(t, n))
    res = r.retrieve(n["c"].id)
    cross_texts = [x["mem"]["text"] for x in res["cross"]]
    # 跨分支 ml 记忆 (与 main 路径共享 game-theory 概念) 浮现
    assert any("neural network" in x for x in cross_texts)
    # 但无 label_match=False 的项被无关信息淹没 (阈值已过滤弱相关)
    for x in res["cross"]:
        assert x["relevance"] >= r.min_relevance


def test_cross_relevance_boost_on_shared_label():
    t, n = _build_demo()
    # 跳到 ml 分支 e: 路径含 a(main) -> d(ml) -> e(ml)
    r = ct.MemoryRetriever(t, store=_mem_store(t, n))
    res = r.retrieve(n["e"].id)
    path_texts = [x["mem"]["text"] for x in res["path"]]
    # path 含 ml 记忆 + a 上的 gto(main, 仍在路径上)
    assert any("neural network" in x for x in path_texts)
    assert any("GTO poker" in x for x in path_texts)
    # cross 出现 main 分支的 nash/bluff (概念重叠)
    cross_texts = [x["mem"]["text"] for x in res["cross"]]
    assert any("Nash" in x for x in cross_texts)


def test_cross_branch_disabled_returns_empty():
    t, n = _build_demo()
    r = ct.MemoryRetriever(t, store=_mem_store(t, n), cross_branch_enabled=False)
    res = r.retrieve(n["c"].id)
    assert res["cross"] == []


# ── 4. 节点状态同步 / 持久化 ──
def test_persistence_roundtrip(tmp_path):
    t, n = _build_demo()
    data_dir = tmp_path / "data" / "v5"
    t2 = ct.ConversationTree(persist_key="rt", data_dir=data_dir)
    t2.init([{"role": "system", "content": "root"}])
    aa = t2.add_turn([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], branch_label="main")
    bb = t2.add_turn([{"role": "user", "content": "more"}], branch_label="main")
    t2.jump_to(bb.id)

    # 反序列化
    ser = t2.serialize()
    t3 = ct.ConversationTree.deserialize(ser, persist_key="rt", data_dir=data_dir)
    assert t3.root_id == t2.root_id
    assert t3.current_id == bb.id
    assert len(t3.nodes) == 3
    assert len(t3.get_context(bb.id)) == 5  # system + 2 turns

    # 从磁盘 load
    t4 = ct.ConversationTree.load(persist_key="rt", data_dir=data_dir)
    assert t4 is not None
    assert t4.current_id == bb.id
    assert len(t4.get_context(aa.id)) == 3


def test_node_independent_state_inherited_then_isolated():
    t = ct.ConversationTree(persist_key="st")
    t.init([])
    t.current.state["progress"] = 0.5
    child = t.add_turn([{"role": "user", "content": "x"}], branch_label="main")
    # 子节点继承父状态 (深拷贝)
    assert child.state["progress"] == 0.5
    # 修改子节点不影响父节点 (隔离)
    child.state["progress"] = 0.9
    assert t.current.state["progress"] == 0.5
    assert child.state["progress"] == 0.9


def test_load_missing_returns_none(tmp_path):
    assert ct.ConversationTree.load(persist_key="nope", data_dir=tmp_path) is None
