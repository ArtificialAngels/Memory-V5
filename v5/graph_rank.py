"""图排序算法 (graph-memory 借鉴) — V5.7 (2026-08-14).

个性化 PageRank + Label Propagation 社区检测, 纯函数无 DB/LLM 依赖, 可单测。
供实体图检索排序用 (推荐 4):

  - personalized_pagerank: 从 seed 实体出发的稳态分布, 比 1-hop 传播更稳的
    实体重要性排序 (graph-memory 的 PPR 排序同款思路)
  - label_propagation: 社区检测, 供"泛化召回"——精确实体匹配失败时按社区
    代表节点补召回 (graph-memory 的 generalized path)
"""

from __future__ import annotations

from collections import Counter


def personalized_pagerank(
    edges: list[tuple[str, str, float]],
    seeds: list[str],
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """个性化 PageRank (从 seeds 出发的稳态分布)。

    edges: [(source, target, weight)]。返回 {node: ppr_score}。
    纯 Python 幂迭代, 无 numpy 依赖 (便携环境)。悬空节点质量回注 seeds。
    """
    out_edges: dict[str, list[tuple[str, float]]] = {}
    nodes: set[str] = set(seeds)
    for s, t, w in edges:
        out_edges.setdefault(s, []).append((t, max(0.0, float(w))))
        nodes.add(s)
        nodes.add(t)
    if not nodes:
        return {}

    # 出度归一化
    for s in list(out_edges.keys()):
        total = sum(w for _, w in out_edges[s])
        if total > 0:
            out_edges[s] = [(t, w / total) for t, w in out_edges[s]]

    seeds = [s for s in seeds if s in nodes]
    n_seeds = len(seeds) or 1
    p: dict[str, float] = {n: (1.0 / n_seeds if n in seeds else 0.0) for n in nodes}

    for _ in range(iterations):
        new_p: dict[str, float] = {n: 0.0 for n in nodes}
        # 个性化跳跃项: 无出边/重启都回到 seeds
        for n in nodes:
            if n in seeds:
                new_p[n] += (1.0 - damping) * (1.0 / n_seeds)
        # 沿边传播
        for s, nbrs in out_edges.items():
            ps = p.get(s, 0.0)
            if ps <= 0:
                continue
            for t, w in nbrs:
                new_p[t] += damping * ps * w
        # 悬空节点 (无出边) 的质量回注 seeds
        dangling_mass = damping * sum(
            p.get(n, 0.0) for n in nodes if n not in out_edges
        )
        if dangling_mass > 0:
            for n in seeds:
                new_p[n] += dangling_mass / n_seeds
        p = new_p
    return p


def label_propagation(
    edges: list[tuple[str, str]],
    *,
    max_iterations: int = 50,
) -> dict[str, int]:
    """Label Propagation 社区检测 (graph-memory 同款, 纯 Python)。

    edges: [(source, target)] (无向)。返回 {node: community_id} (连续整数)。
    """
    adj: dict[str, set[str]] = {}
    for s, t in edges:
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    nodes = list(adj.keys())
    if not nodes:
        return {}
    labels = {n: i for i, n in enumerate(nodes)}
    for _ in range(max_iterations):
        changed = False
        for n in nodes:
            nbrs = adj.get(n)
            if not nbrs:
                continue
            cnt = Counter(labels[nb] for nb in nbrs)
            most = cnt.most_common(1)[0][0]
            if labels[n] != most:
                labels[n] = most
                changed = True
        if not changed:
            break
    # 重编号为连续整数
    mapping: dict[int, int] = {}
    for n in nodes:
        cid = labels[n]
        if cid not in mapping:
            mapping[cid] = len(mapping)
        labels[n] = mapping[cid]
    return labels
