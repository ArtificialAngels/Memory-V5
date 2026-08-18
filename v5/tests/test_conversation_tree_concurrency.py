"""对话树多线程并发操作测试 (R9 测试缺口 #1 / R1 锁路径).

验证 fork_branch + add_turn + conclude_branch + merge_branch 在并发
调用下的拓扑一致性: 无异常、无孤儿节点、parent/children 双向一致、
trunk 合并记录完整、version 计数正确、持久化往返一致。

参考: conversation_tree.py fork_branch :776 / conclude_branch :823 /
      merge_branch :861 (R1 加锁路径, 变更体持锁, _emit/persist 在锁外)。

全部走注入 ThreadSafeMockStore + tmp data_dir, 不落盘生产目录。
"""
import json
import sys
import threading
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # core (v5 包所在)

import v5.conversation_tree as ct  # noqa: E402


class ThreadSafeMockStore:
    """内存 store (线程安全): 模拟 V5 store.store / load / search 接口."""

    def __init__(self):
        self._lock = threading.Lock()
        self.mem: dict[int, str] = {}
        self.counter = 0

    def store(self, content: str, type: str = "conversation",
              weight: float = 0.6, tags: str = "", **kwargs) -> int:
        with self._lock:
            self.counter += 1
            self.mem[self.counter] = content
            return self.counter

    def load(self, ids: list[int]) -> dict[int, str]:
        with self._lock:
            return {mid: self.mem[mid] for mid in ids if mid in self.mem}

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        with self._lock:
            q = query.lower()
            return [{"id": mid, "content": c}
                    for mid, c in self.mem.items() if q in c.lower()][:top_k]


def _assert_topology_consistent(t: ct.ConversationTree) -> None:
    """拓扑不变量: 根可达全部节点 / parent 回指 / children 边数."""
    nodes = t.nodes
    assert nodes, "tree 不应为空"
    # 1. 根可达全部节点 (无孤儿)
    visited: set[str] = set()
    stack = [t.root_id]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(nodes[cur].children)
    assert visited == set(nodes.keys()), (
        f"孤儿节点: {set(nodes.keys()) - visited}")
    # 2. parent 回指一致性
    for nid, n in nodes.items():
        if n.parent_id is not None:
            p = nodes.get(n.parent_id)
            assert p is not None, f"parent 缺失: {nid} -> {n.parent_id}"
            assert nid in p.children, f"parent.children 缺回指: {nid}"
    # 3. children 边数 = 节点数 - 1 (树形, 无额外边)
    assert sum(len(n.children) for n in nodes.values()) == len(nodes) - 1


def _run_fork_storm(t: ct.ConversationTree, trunk_id: str,
                    thread_idx: int, iters: int, errors: list,
                    barrier: threading.Barrier) -> None:
    """单线程负载: 反复 fork → add_turn → conclude → merge (R1 锁路径)."""
    try:
        barrier.wait(timeout=30)
    except threading.BrokenBarrierError:
        pass
    for i in range(iters):
        label = f"t{thread_idx}-{i}"
        branch = t.fork_branch(
            trunk_id, label, [{"role": "user", "content": f"分支 {label} 的问题"}])
        child = t.add_turn(
            [{"role": "assistant", "content": f"分支 {label} 的回复"}],
            parent_id=branch.id,
        )
        t.conclude_branch(branch.id, [f"结论 {label}"])
        t.merge_branch(branch.id, trunk_id)
        assert child.parent_id == branch.id  # 线程内自检
    errors.append(None)


def test_fork_add_merge_concurrency(tmp_path):
    """多线程并发 fork/add/conclude/merge: 无异常 + 拓扑一致."""
    ms = ThreadSafeMockStore()
    t = ct.ConversationTree(
        persist_key="conv_test",
        data_dir=str(tmp_path),
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    t.init([{"role": "system", "content": "root"}])
    trunk = t.add_turn([{"role": "user", "content": "主线问题"}],
                       branch_label="main")
    trunk_id = trunk.id
    assert t.trunk_id == trunk_id
    assert trunk.node_type == "trunk"

    threads = 6
    iters = 6
    barrier = threading.Barrier(threads)
    errors: list = []
    ts = [
        threading.Thread(target=_run_fork_storm,
                         args=(t, trunk_id, idx, iters, errors, barrier))
        for idx in range(threads)
    ]
    for th in ts:
        th.start()
    for th in ts:
        th.join(timeout=60)
    assert not any(th.is_alive() for th in ts), "线程未在超时内结束"
    assert errors == [None] * threads, f"并发异常: {errors}"

    k = threads * iters
    # 节点数: root + trunk + (fork 节点 + add_turn 子节点) * k
    assert len(t.nodes) == 2 + 2 * k, len(t.nodes)
    # version: init +1, 首次 add_turn +1, 之后每次操作 (fork/add/conclude/merge) +1
    assert t.version == 2 + 4 * k, t.version
    # trunk 合并记录完整
    assert len(trunk.merged_from) == k and len(set(trunk.merged_from)) == k
    assert len(trunk.conclusions) == k
    # 每个 fork 节点: 结论化 + merge_target 指向主线
    branch_nodes = [n for n in t.nodes.values()
                    if n.node_type == "conclusion" and n.branch_label]
    assert len(branch_nodes) == k
    assert all(n.merge_target == trunk_id for n in branch_nodes)
    assert all(len(n.conclusions) == 1 for n in branch_nodes)

    _assert_topology_consistent(t)

    # 持久化往返: 文件存在且反序列化后拓扑一致
    pfile = tmp_path / "conv_test.json"
    assert pfile.exists()
    payload = json.loads(pfile.read_text(encoding="utf-8"))
    assert payload["v"] == t.version
    assert payload["root_id"] == t.root_id
    assert payload["trunk_id"] == t.trunk_id
    assert len(payload["nodes"]) == len(t.nodes)
    t2 = ct.ConversationTree.deserialize(pfile.read_text(encoding="utf-8"),
                                         persist_key="conv_test",
                                         data_dir=str(tmp_path))
    _assert_topology_consistent(t2)
    assert t2.trunk_id == trunk_id


def test_sequential_baseline_versions(tmp_path):
    """串行基准: version / 拓扑计数与并发场景一致 (防并发测试自身偏差)."""
    ms = ThreadSafeMockStore()
    t = ct.ConversationTree(
        persist_key="conv_seq",
        data_dir=str(tmp_path),
        _store=ms.store, _load=ms.load, _search=ms.search,
    )
    t.init([{"role": "system", "content": "root"}])
    trunk = t.add_turn([{"role": "user", "content": "主线"}])
    for i in range(4):
        b = t.fork_branch(trunk.id, f"s{i}", [{"role": "user", "content": f"q{i}"}])
        t.add_turn([{"role": "assistant", "content": f"a{i}"}], parent_id=b.id)
        t.conclude_branch(b.id, [f"c{i}"])
        t.merge_branch(b.id, trunk.id)
    assert t.version == 2 + 4 * 4
    assert len(t.nodes) == 2 + 2 * 4
    _assert_topology_consistent(t)
