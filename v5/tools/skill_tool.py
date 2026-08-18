"""Skill 记忆工具 —— v5_skill_* 系列 (自我进化: agent 自主决定 no-op/patch/create).

设计来源: memU 调研 (2026-08-10, 记忆 #2619).
核心哲学: 判断权在 agent —— 本工具不做自动蒸馏, 只提供
写/读/列/搜/删 五个原子操作; 是否值得沉淀、是新建还是修补现有技能,
由调用方 (agent) 自己决定. "什么都不做" 是完全正当的输出.

检索走渐进形状: v5_skill_search 只返回窄命中 (name/description/path/score),
需要全文时再调 v5_skill_get —— 给 agent 位置 + 摘要, 而不是全文.
"""

from __future__ import annotations

from v5.tools.utils import safe_tool, dumps, answer


@safe_tool
def v5_skill_write(name: str, description: str, content: str) -> str:
    """Create or update a reusable skill (Markdown file, kebab-case name).

    Skills are agent-distilled workflows: repeatable steps for a task family,
    including useful branches, edge cases, and pitfalls. The file lives under
    data/v5/skills/ with a `name` + `description` front-matter — human-readable,
    diffable, reviewable.

    Rules of thumb:
      - Prefer patching an existing skill (grow it with a new branch) over
        creating a near-duplicate one.
      - A no-op is a perfectly good outcome — do not invent a skill to
        justify a run.
    """
    from v5 import skill_store

    result = skill_store.write_skill(name=name, description=description, content=content)
    verb = "已更新" if not result["created"] else "已创建"
    return answer(f"技能 {verb}: {name}", result)


@safe_tool
def v5_skill_list() -> str:
    """List all skills: name / description / path (no full text)."""
    from v5 import skill_store

    skills = skill_store.list_skills()
    return answer(f"共 {len(skills)} 个技能", skills)


@safe_tool
def v5_skill_get(name: str) -> str:
    """Read a skill's full content by name (the "wide" layer of progressive retrieval)."""
    from v5 import skill_store

    skill = skill_store.get_skill(name)
    if skill is None:
        return dumps({"ok": False, "error": "not_found", "name": name})
    return answer(f"技能读取成功: {name}", skill)


@safe_tool
def v5_skill_search(query: str, top_k: int = 5) -> str:
    """Search skills — narrow hits only (name/description/path/score), no full text.

    Weighted keyword scoring: name hit > description hit > body hit (capped).
    Call v5_skill_get on a hit's `name` when you need the full workflow.
    """
    from v5 import skill_store

    hits = skill_store.search_skills(query, top_k=top_k)
    return answer(f"找到 {len(hits)} 个相关技能", hits)


@safe_tool
def v5_skill_remove(name: str) -> str:
    """Delete a skill by name (idempotent: missing skill is a clean no-op)."""
    from v5 import skill_store

    ok = skill_store.remove_skill(name)
    if not ok:
        return dumps({"ok": False, "error": "not_found", "name": name})
    return answer(f"技能已删除: {name}", {"ok": True, "name": name})
