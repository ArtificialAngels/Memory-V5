"""Token 预算装配 (OpenViking context_assembler/budget 借鉴) — F2.

问题: unified_retrieve 返回 top-k 原始结果, 注入上下文时由 harness/agent 自己截断,
没有预算感知。OpenViking 的 plan_entries 在装配阶段做三件事:
  1. per_entry_cap = 2× 平均份额 — 防 top-1 长记忆吃满整个预算 (scores 聚集在窄带,
     把预算全押 top-1 是坏赌, budget.py:5-8);
  2. 广度后深度 — 每条先放默认层(广度, 多条可见), 剩余预算再加深最高分条(深度);
  3. 降级不截断 — 超限的 full 退回 abstract, 超限 abstract 退回 uri, 保留信号不丢.

Ikaros 三层 (记忆短, 默认 full):
  uri      = "#{id}"            裸指针 (~3 token)
  abstract = content[:120]      摘要预览
  full     = content            全文 (默认)

纯函数, 无 DB/LLM 依赖, 可单测。stats 透出装配轨迹 (F4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger("ikaros.v5.recall_budget")

SEPARATOR_TOKENS = 1

# 升序 detail: uri < abstract < full
TIER_ORDER: tuple[str, ...] = ("uri", "abstract", "full")
TIER_RANK: dict[str, int] = {t: i for i, t in enumerate(TIER_ORDER)}

_ABSTRACT_CHARS = 120


def estimate_tokens(text: str) -> int:
    """粗估 token (中文 ~1 token/字, ASCII ~0.5; 取 0.6 折中, 保底 1).

    用于预算上限判定; 估偏保守(偏大)更安全, 估偏小会塞超.
    """
    if not text:
        return 1
    # CJK 字符按 1 token, 其他按 0.5
    n = 0
    for ch in text:
        n += 1 if ord(ch) > 0x2E80 else 0.5
    return max(1, int(n) + 1)


def per_entry_cap(max_tokens: int, candidate_count: int) -> int:
    """2× 平均份额 — 单条上限 (OpenViking budget.py:44)."""
    return max(1, max_tokens // max(1, candidate_count) * 2)


@dataclass
class Candidate:
    """unified_retrieve 结果项的装配视图."""
    id: str
    content: str
    type: str
    score: float
    tags: str = ""


@dataclass
class AssembledEntry:
    """装配后的条目 (注入上下文用)."""
    id: str
    type: str
    score: float
    detail: str          # uri / abstract / full
    text: str            # 实际注入文本
    tokens: int = 0


@dataclass
class BudgetPlan:
    entries: list[AssembledEntry]
    stats: dict[str, Any] = field(default_factory=dict)


def _tier_text(c: Candidate, tier: str) -> str:
    if tier == "uri":
        return f"#{c.id}"
    if tier == "abstract":
        body = (c.content or "").strip()
        if len(body) <= _ABSTRACT_CHARS:
            return body
        return body[:_ABSTRACT_CHARS].rstrip() + "…"
    # full
    return (c.content or "").strip()


def _tiers_down_from(tier: str) -> list[str]:
    """从 tier 起逐层降级 (full→abstract→uri)."""
    rank = TIER_RANK[tier]
    return list(reversed(TIER_ORDER[: rank + 1]))


def _make_entry(c: Candidate, tier: str) -> AssembledEntry:
    text = _tier_text(c, tier)
    e = AssembledEntry(id=c.id, type=c.type, score=c.score, detail=tier, text=text)
    e.tokens = estimate_tokens(text)
    return e


def plan_entries(
    candidates: Sequence[Candidate],
    max_tokens: int,
    *,
    default_tier: str = "full",
) -> BudgetPlan:
    """按预算装配候选, 返回 entries + stats.

    广度: 每条先试 default_tier, 超限则降级 (full→abstract→uri), 都放不下才 drop.
    深度: 剩余预算按分数从高到低, 把 abstract/uri 升级回 full.
    dedup: body hash 相同 (都只展示裸 id) 视为重复, 保留首条.
    """
    max_tokens = max(1, int(max_tokens))
    n = len(candidates)
    cap = per_entry_cap(max_tokens, n)
    slots: list[AssembledEntry] = []
    used = 0
    dropped = 0
    deduped = 0
    seen_bodies: set[int] = set()
    tier_counts: dict[str, int] = {"uri": 0, "abstract": 0, "full": 0}

    def fits_cap(tier: str, tokens: int) -> bool:
        # 裸 uri 无正文, 不受 cap 约束 (OpenViking budget.py:105)
        return tier == "uri" or tokens <= cap

    # ── 广度: 每条试 default_tier, 降级直到放下; 命中 dedup 则整条跳过 ──
    for c in candidates:
        placed: AssembledEntry | None = None
        dup = False
        for tier in _tiers_down_from(default_tier):
            entry = _make_entry(c, tier)
            # body-hash dedup: 与已放下的条目渲染文本相同 → 整条跳过 (不再试更低层)
            body_key = hash(entry.text or c.id)
            if body_key in seen_bodies:
                deduped += 1
                dup = True
                break
            if not fits_cap(tier, entry.tokens):
                continue
            cost = entry.tokens + (SEPARATOR_TOKENS if slots else 0)
            if used + cost > max_tokens:
                continue
            placed = entry
            break
        if placed is None:
            if not dup:
                dropped += 1
            continue
        seen_bodies.add(hash(placed.text or c.id))
        slots.append(placed)
        used += placed.tokens + (SEPARATOR_TOKENS if len(slots) > 1 else 0)
        tier_counts[placed.detail] += 1

    # ── 深度: 剩余预算升级最高分的 abstract/uri → full ──
    # slots 是 candidates 的子集 (drop/dedup 后), 按 id 回溯 candidate
    full_upgrades = 0
    abstract_upgrades = 0
    cand_by_id = {c.id: c for c in candidates}
    # slots 顺序 = candidates 顺序 = score 降序 (调用方传入时已排)
    for i, e in enumerate(slots):
        if e.detail == "full":
            continue
        c = cand_by_id.get(e.id)
        if c is None:
            continue
        full_entry = _make_entry(c, "full")
        extra = full_entry.tokens - e.tokens
        if extra <= 0:
            continue
        if used + extra > max_tokens:
            continue
        if full_entry.tokens > cap:
            continue
        prev_detail = e.detail
        slots[i] = full_entry
        used += extra
        tier_counts[prev_detail] -= 1
        tier_counts["full"] += 1
        if prev_detail == "uri":
            abstract_upgrades += 1  # uri→full 跨层
        else:
            full_upgrades += 1

    spare = max_tokens - used
    stats = {
        "candidates": n,
        "placed": len(slots),
        "dropped": dropped,
        "deduped": deduped,
        "tier_counts": dict(tier_counts),
        "used_tokens": used,
        "max_tokens": max_tokens,
        "per_entry_cap": cap,
        "spare_tokens": spare,
        "fill": {
            "full_upgrades": full_upgrades,
            "abstract_upgrades": abstract_upgrades,
            "spare": spare,
        },
    }
    return BudgetPlan(entries=slots, stats=stats)
