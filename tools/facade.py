"""冷路径门面工具 — 一个资源一个 MCP 工具, 用 ``action`` 内部分发。

详见 docs/v5-mcp-consolidation.md。

## 为什么是门面而不是全合并

MCP 工具数直接影响模型的选择成本: 50 个工具里只有 5 个被 persona / SOUL / AGENTS
点名, 其余 45 个是"一个动词一个工具"的历史堆叠。但热路径 (memory_search /
memory_store / recall / project_note) **保留独立工具** —— 它们调用最频繁, 独立工具
免掉一层 action 选择, 参数表也窄。冷路径才合。

## 实现原则: 纯委托, 零行为变更

每个 action 直接委托给已被测试覆盖的旧函数, 本文件**不重写任何业务逻辑**。
旧函数在 Python 侧全部保留 (测试 / 脚本 / 桥接层继续 import), 只是 slim 模式下
不再注册为 MCP 工具。因此门面与旧工具的输出**逐字节一致**。

## 被 Loop 内化的工具

``v5_vitality_tick`` / ``v5_relationship_tick`` / ``v5_anti_repeat_record`` /
``v5_reflect_run_op`` 是**机器态维护动作**——不该由模型决定做不做。
它们进了 ``memory_v5/loop.py`` 的标准循环, 门面里不再重复暴露写动作
(读动作如 ``v5_state(action=vitality)`` 仍可读)。
"""

from __future__ import annotations

from memory_v5.tools.utils import safe_tool, dumps, answer


# ─── 分发助手 ────────────────────────────────────────────────────────

def _bad_action(tool: str, action: str, table: dict) -> str:
    """未知 action 的统一错误: 列出合法值, 帮模型一次改对。"""
    return dumps({
        "ok": False,
        "error": f"unknown action: {action!r}",
        "tool": tool,
        "valid_actions": sorted(table),
    }, ensure_ascii=False)


# ─── v5_self: 自我 / 内省 (6 -> 1) ───────────────────────────────────

@safe_tool
def v5_self(action: str = "model", mode: str = "reflect") -> str:
    """Ikaros 自我模型与内省。

    action:
        model       持久自我模型 (身份/能力/信念/问题/好奇心)
        reflect     跑一次元认知循环 —— mode: reflect | philosophy | cycle
        anchor      身份 + 情感 + 关系紧凑快照 (中途重新锚定自我用)

    mode 仅对 action=reflect 生效。

    2026-09-05 精简: 移除 thought/curiosity/subconscious/discover 四个零消费方
    action (dsh 插件层 + Python 生产代码均无调用)。底层函数仍在 legacy 模式保留。
    """
    if action == "model":
        from memory_v5.tools.self_tool import v5_self_model
        return v5_self_model()
    if action == "reflect":
        from memory_v5.tools.self_tool import v5_self_reflect
        return v5_self_reflect(mode=mode)
    if action == "anchor":
        from memory_v5.tools.self_tool import v5_context_refresh
        return v5_context_refresh()
    return _bad_action("v5_self", action, {
        "model": 1, "reflect": 1, "anchor": 1,
    })


# ─── v5_state: 情感 / 关怀 / 精力 / 关系 / 活动 (11 -> 1) ──────────────

@safe_tool
def v5_state(action: str = "emotion", text: str = "",
             intensity: float = 0.3, character: str = "") -> str:
    """Ikaros 与哥哥的实时状态。

    action:
        emotion         当前 PAD 情感状态 (愉悦/激活/掌控 + 心情标签)
        emotion_update  用 text 更新 PAD 情感状态, 返回 delta 与强度
        care            关怀监测累计计数 (编码/游戏/专注时长, 提醒次数)
        care_check      检查哥哥是否需要主动关怀 (休息/喝水/睡觉)
        vitality        当前精力值 + 标签 + 累计运行时长
        relationship    与哥哥的关系模型 (深度/温度/阶段/亲密/天数/互动数)

    写动作 (vitality_tick / relationship_tick) 已内化进标准 Loop 的 post 阶段,
    每轮自动推进 —— 由 loop.py 调用, 不再作为工具暴露。

    2026-09-05 精简: 移除 emotion_label/activity/compression 三个零消费方 action。
    """
    if action == "emotion":
        from memory_v5.tools.emotion_tool import v5_emotion_status
        return v5_emotion_status()
    if action == "emotion_update":
        from memory_v5.tools.emotion_tool import v5_analyze_emotion
        return v5_analyze_emotion(text)
    if action == "care":
        from memory_v5.tools.care_tool import v5_care_status
        return v5_care_status()
    if action == "care_check":
        from memory_v5.tools.care_tool import v5_care_check
        return v5_care_check()
    if action == "vitality":
        from memory_v5.tools.vitality_tool import v5_vitality
        return v5_vitality()
    if action == "relationship":
        from memory_v5.tools.relationship_tool import v5_relationship
        return v5_relationship()
    return _bad_action("v5_state", action, {
        "emotion": 1, "emotion_update": 1,
        "care": 1, "care_check": 1, "vitality": 1, "relationship": 1,
    })


# ─── v5_content 已删除 (2026-09-05) ────────────────────────────────────
# 三个 action 均零消费方:
#   narrative  → 内化进 reflect op (30d 周期, 由调度器驱动)
#   dissonance → store.py 已有异步 _run_dissonance_detection, 手动检测重复
#   proactive  → dsh 无主动触发机制消费此门控
# 底层函数 v5_narrative_generate / v5_dissonance_check / v5_proactive_check
# 仍在 legacy 模式保留 (extra_tool.py), 测试/脚本可继续 import。


# ─── v5_skill: 技能库 (5 -> 1) ───────────────────────────────────────

@safe_tool
def v5_skill(action: str = "list", name: str = "", description: str = "",
             content: str = "", query: str = "", top_k: int = 5) -> str:
    """可复用技能库 (Markdown 文件, agent 自主蒸馏的工作流)。

    检索走渐进形状: search 只返窄命中 (name/description/path/score),
    要全文再 get —— 先给位置与摘要, 而不是整篇塞进上下文。

    action:
        list     列出全部技能 (name/description/path, 无全文)
        search   关键词检索, 窄命中 —— query / top_k
        get      按 name 读全文 —— name
        write    创建或更新技能 —— name / description / content
        remove   按 name 删除 (幂等: 不存在是干净的 no-op) —— name

    判断权在 agent: 本工具不做自动蒸馏。优先 patch 现有技能而不是新建近似重复;
    "什么都不做"是完全正当的输出。
    """
    from memory_v5 import skill_store

    if action == "list":
        skills = skill_store.list_skills()
        return answer(f"共 {len(skills)} 个技能", skills)
    if action == "search":
        hits = skill_store.search_skills(query, top_k=top_k)
        return answer(f"找到 {len(hits)} 个相关技能", hits)
    if action == "get":
        skill = skill_store.get_skill(name)
        if skill is None:
            return dumps({"ok": False, "error": "not_found", "name": name})
        return answer(f"技能读取成功: {name}", skill)
    if action == "write":
        result = skill_store.write_skill(
            name=name, description=description, content=content)
        verb = "已更新" if not result["created"] else "已创建"
        return answer(f"技能 {verb}: {name}", result)
    if action == "remove":
        ok = skill_store.remove_skill(name)
        if not ok:
            return dumps({"ok": False, "error": "not_found", "name": name})
        return answer(f"技能已删除: {name}", {"ok": True, "name": name})
    return _bad_action("v5_skill", action,
                       {"list": 1, "search": 1, "get": 1, "write": 1, "remove": 1})


# ─── v5_reflection: 反思轨 (5 -> 1) ──────────────────────────────────

@safe_tool
def v5_reflection(action: str = "stats", content: str = "",
                  reflection_id: str = "", character: str = "",
                  status: str = "", entity: str = "master",
                  relation_type: str = "experience",
                  importance: int = 5, limit: int = 10,
                  source_fact_ids: str = "",
                  delta_rein: float = 0.0, delta_disp: float = 0.0) -> str:
    """反思条目: 从事实合成 -> 证据强化 -> 晋升为人格。

    action:
        stats     按状态聚合反思计数
        read      查询反思库 —— character / status / entity / limit
        synthesize 从事实合成新反思 —— content / character / source_fact_ids
                   (JSON 数组字符串) / entity / relation_type / importance(1-10)
        apply     施加证据信号, 触发状态迁移
                  (pending->confirmed->promoted->merged) ——
                  reflection_id / delta_rein / delta_disp
        promote   把反思合并进 self_model 人格 —— reflection_id / character
    """
    if action == "stats":
        from memory_v5.tools.reflection_tool import v5_reflection_stats
        return v5_reflection_stats(character)
    if action == "read":
        from memory_v5.tools.reflection_tool import v5_reflection_read
        return v5_reflection_read(character=character, status=status,
                                  limit=limit, entity=entity)
    if action == "synthesize":
        from memory_v5.tools.reflection_tool import v5_reflection_synthesize
        return v5_reflection_synthesize(
            content=content, character=character,
            source_fact_ids=source_fact_ids, entity=entity,
            relation_type=relation_type, importance=importance)
    if action == "apply":
        from memory_v5.tools.reflection_tool import v5_reflection_apply_evidence
        return v5_reflection_apply_evidence(
            reflection_id=reflection_id, character=character,
            delta_rein=delta_rein, delta_disp=delta_disp)
    if action == "promote":
        from memory_v5.tools.reflection_tool import v5_reflection_promote
        return v5_reflection_promote(reflection_id=reflection_id,
                                     character=character)
    return _bad_action("v5_reflection", action, {
        "stats": 1, "read": 1, "synthesize": 1, "apply": 1, "promote": 1,
    })


# ─── v5_directive: 用户指令 (4 -> 1) ─────────────────────────────────

@safe_tool
def v5_directive(action: str = "list", character: str = "",
                 directive_text: str = "",
                 directive_type: str = "ban_topic",
                 ttl_hours: float = 72.0, directive_id: int = 0) -> str:
    """哥哥下达的指令 (禁话题 / 偏好 / 行为规则), 带 TTL。

    action:
        list     列出生效指令 —— character / directive_type
        add      新增指令 —— character / directive_text / directive_type /
                 ttl_hours (0 = 永不过期)
        off      按 ID 停用指令 —— directive_id
        stats    指令总数 / 生效数 —— character
    """
    if action == "list":
        from memory_v5.tools.directive_tool import v5_directive_list
        return v5_directive_list(character, directive_type)
    if action == "add":
        from memory_v5.tools.directive_tool import v5_directive_add
        return v5_directive_add(character, directive_text,
                                directive_type, ttl_hours)
    if action == "off":
        from memory_v5.tools.directive_tool import v5_directive_deactivate
        return v5_directive_deactivate(directive_id)
    if action == "stats":
        from memory_v5.tools.directive_tool import v5_directive_stats
        return v5_directive_stats(character)
    return _bad_action("v5_directive", action,
                       {"list": 1, "add": 1, "off": 1, "stats": 1})


# ─── v5_repeat: 反重复语料 (5 -> 1, record 已内化进 Loop) ──────────────

@safe_tool
def v5_repeat(action: str = "stats", character: str = "",
              candidate_text: str = "") -> str:
    """反重复语料 (BM25 风格的 n-gram 重复检测)。

    写入已内化: 每轮回复会由标准 Loop 的 post 阶段自动记录, 不需要手动调。

    action:
        stats    语料统计 —— character
        check    评估 candidate_text 的重复风险 —— character / candidate_text
        penalty  重复风险超阈值时返回 system-prompt 惩罚提示 (干净则空串)
        clear    清空语料 —— character (留空 = 全清)
    """
    if action == "stats":
        from memory_v5.tools.repeat_tool import v5_anti_repeat_stats
        return v5_anti_repeat_stats(character)
    if action == "check":
        from memory_v5.tools.repeat_tool import v5_anti_repeat_check
        return v5_anti_repeat_check(character, candidate_text)
    if action == "penalty":
        from memory_v5.tools.repeat_tool import v5_anti_repeat_penalty
        return v5_anti_repeat_penalty(character, candidate_text)
    if action == "clear":
        from memory_v5.tools.repeat_tool import v5_anti_repeat_clear
        return v5_anti_repeat_clear(character)
    return _bad_action("v5_repeat", action,
                       {"stats": 1, "check": 1, "penalty": 1, "clear": 1})


# ─── v5_loop: 标准记忆循环的入口与观测 ─────────────────────────────────

@safe_tool
def v5_loop(action: str = "status", phase: str = "", query: str = "",
            response: str = "", session_id: str = "default",
            character: str = "", force: bool = False) -> str:
    """标准记忆循环 (Standard Memory Loop) 的入口与状态观测。

    一轮对话的记忆仪式原本散落在 3 个插件 hook + 若干手动调用里, 现在收敛成
    三个声明式阶段, 一个阶段一次调用跑完所有到期步骤:

        pre          身份锚定 + 记忆召回 + 项目经验召回   (需要 query)
        post         精力/关系推进 + 反重复语料记录       (需要 response)
        maintenance  反思管线 (6h 冷却)                   (无需参数)

    dsh 插件的 pre-step / turn-stopping / 定时器 hook 会自动驱动这三个阶段,
    正常对话**不需要手动调用**。本工具用于:
        - 手动补账 (某轮 hook 没跑成功)
        - 观测: action=status 看每个 step 的 last_run / due / next_run_in_sec

    action:
        status  返回三阶段全部 step 的状态快照 (只读, 不执行)
        run     跑指定 phase —— phase 必填 (pre/post/maintenance);
                force=True 忽略冷却全跑
    """
    from memory_v5 import loop as loop_mod

    if action == "status":
        return dumps({"ok": True, "phases": loop_mod.status()}, ensure_ascii=False)

    if action == "run":
        if phase not in loop_mod.PHASES:
            return dumps({
                "ok": False,
                "error": f"unknown phase: {phase!r}",
                "valid_phases": list(loop_mod.PHASES),
            }, ensure_ascii=False)
        result = loop_mod.run_phase(
            phase, query=query, response=response,
            session_id=session_id or "default",
            character=character, force=force,
        )
        ran = result.get("ran") or []
        return answer(
            f"Loop {phase}: 跑了 {len(ran)} 步 ({', '.join(ran) or '无'})"
            + (f", {len(result.get('errors') or {})} 步失败"
               if result.get("errors") else ""),
            result,
        )

    return _bad_action("v5_loop", action, {"status": 1, "run": 1})
