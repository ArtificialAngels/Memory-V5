"""v5_recall — 预算感知 + 跨輪去重的召回工具 (OpenViking context_assembler 借鉴) — F1+F2+F4.

unified_retrieve 返回 top-k 原始结果, 本工具在它之上加三层:
  1. 跨輪去重 (recall_ledger): 近 dedup_turns 轮展示过正文的记忆跳过;
  2. token 预算装配 (recall_budget): 广度后深度 + per-entry cap + 降级不截断;
  3. 轨迹 stats (F4): candidates/placed/dropped/deduped/cooled/tier_counts/fill.

每次调用 = 一轮 (advance_turn), 记录本轮展示 (bare-URI 不计), 供下一轮去重.
"""

from __future__ import annotations

from memory_v5.tools.utils import safe_tool, dumps, answer


def _render_block(entries) -> str:
    """紧凑可读渲染 (每条: [id] type score detail)."""
    if not entries:
        return "(无相关记忆)"
    lines = []
    for e in entries:
        lines.append(f"[#{e.id}] ({e.type}, score={e.score:.2f}, {e.detail}) {e.text}")
    return "\n".join(lines)


@safe_tool
def v5_recall(
    query: str,
    *,
    max_tokens: int = 1600,
    top_k: int = 20,
    session_id: str = "default",
    dedup_turns: int = 5,
    include_dsh_only: bool = True,
) -> str:
    """Token-bounded, dedup-aware recall. Returns assembled context + trajectory stats.

    Paths:
      1. unified_retrieve (语义三路融合 + 图补路 + Vault) over-fetch top_k=20
      2. recall_ledger 排除近 dedup_turns 轮展示过正文的记忆 (bare-URI 不冷却)
      3. recall_budget 广度后深度装配到 max_tokens (降级不截断)
      4. 记录本轮展示 (bare-URI 不计) 供下一轮去重
    Always returns JSON; never raises.
    """
    if not query or not query.strip():
        return answer("空查询", {"context": "", "stats": {}})

    from memory_v5.memory_retrieval import unified_retrieve
    from memory_v5.recall_ledger import RecallLedger
    from memory_v5.recall_budget import Candidate, plan_entries

    # 0. 推进一轮 (本次召回 = 新一轮)
    ledger = RecallLedger(session_id=session_id, dedup_turns=dedup_turns)
    cur_turn = ledger.advance_turn()

    # 1. 检索 (over-fetch 给预算留头)
    try:
        results = unified_retrieve(query, top_k=top_k, include_dsh_only=include_dsh_only)
    except Exception as exc:  # noqa: BLE001
        return answer(f"召回失败: {exc}", {"context": "", "stats": {"error": str(exc)}})

    retrieved = len(results)

    # 2. 跨轮去重: 排除近 N 轮展示过正文的记忆
    cooled_set = ledger.cooled_ids()
    fresh = [r for r in results if str(r.get("id")) not in cooled_set]
    cooled_n = retrieved - len(fresh)

    # 2b. 兜底: 候选被冷却一空时放宽去重, 重新用全量候选。
    #
    # ⚠️ 为什么要这一步 (2026-08-30 实测):
    #     recall_ledger 的冷却窗口是「近 dedup_turns 轮」, 而 ledger **落盘持久化**
    #     (data/v5/recall_log_<sid>.json) 且 turn 单调递增、永不清零。插件侧
    #     session_id 又硬编码成 'dsh' (所有会话共用一本账)。
    #     于是「同一话题连着问几轮」或「重启后紧接着复问上一个话题」时,
    #     top-k 候选会**整批**处于冷却中 → fresh 为空 → 装配 0 条 →
    #     返回 "(无相关记忆)"。
    #     去重只是省 token、减少"又讲一遍"的噪声, 是**优化**; 交一张空纸条
    #     则是**功能性失败** —— 宁可重复, 不可失忆。
    #     实测不同 query 的 cooled 仅 0~2 条, 兜底极少触发, 不会抵消去重收益。
    dedup_relaxed = False
    if not fresh and results:
        fresh = list(results)
        dedup_relaxed = True

    # 3. 预算装配 (广度后深度 + 降级不截断)
    cands = [
        Candidate(
            id=str(r.get("id") or ""),
            content=r.get("content") or "",
            type=r.get("type") or "fact",
            score=float(r.get("score") or 0.0),
            tags=r.get("tags") or "",
        )
        for r in fresh
    ]
    plan = plan_entries(cands, max_tokens)

    # 4. 记录本轮展示 (bare-URI 不计 = 输给预算, 下轮可正常展示)
    served = [(e.id, e.detail != "uri") for e in plan.entries]
    ledger.record_served(served)

    # 5. 渲染 + stats
    block = _render_block(plan.entries)
    stats = {
        "retrieved": retrieved,
        "cooled": cooled_n,
        "dedup_relaxed": dedup_relaxed,  # True = 候选被冷却一空, 本次放宽了去重
        "turn": cur_turn,
        **plan.stats,
    }
    return answer(
        f"召回 {len(plan.entries)} 条 (检索 {retrieved}, 冷却 {cooled_n}, 预算 {max_tokens}t 用 {plan.stats['used_tokens']}t)",
        {"context": block, "stats": stats},
    )
